from cmath import log
import copy
import logging
import random
import standard_trainer
import numpy as np
import torch
import wandb
import os
from .client import Client
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
import logging
import pickle
from data_synthesizer import DataSynthesizer

class FedAvgAPI(object):
    def __init__(self, args, device, dataset, model, model_trainer=None):
        self.device = device
        self.args = args
        [
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            class_num,
        ] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.val_global = None
        self.train_data_num_in_total = train_data_num
        self.test_data_num_in_total = test_data_num
        self.fair_gradient = None
        self.synthetic_data_size = None
        self.acc_list_all_round = []
        self.eo_list_all_round = []
        self.dp_list_all_round = []
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict

        logging.info("model = {}".format(model))
        if model_trainer is None:
            if args.dataset == "stackoverflow_lr":
                model_trainer = MyModelTrainerTAG(model)
            elif args.dataset in ["fed_shakespeare", "stackoverflow_nwp"]:
                model_trainer = MyModelTrainerNWP(model)
            else:
                # default model trainer is for classification problem
                model_trainer = MyModelTrainerCLS(model)
        self.model_trainer = model_trainer
        logging.info("self.model_trainer = {}".format(self.model_trainer))

        self._setup_clients(
            train_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            self.model_trainer,
        )
        self.global_model_trajectory = []
        self.fair_gradient = None
        self.synthetic_data = None
        self.synthetic_labels = None
        self.synthetic_sensitive_attr = None

    def _setup_clients(
        self,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        model_trainer,
    ):
        logging.info("############setup_clients (START)#############")
        for client_idx in self.args.users:
            c = Client(
                client_idx,
                train_data_local_dict[client_idx],
                test_data_local_dict[client_idx],
                val_data_local_dict[client_idx],
                train_data_local_num_dict[client_idx],
                self.args,
                self.device,
                model_trainer,
            )
            self.client_list.append(c)
        logging.info("############setup_clients (END)#############")


    def calculate_eo_loss(self, model, data, labels, sensitive_attributes):
        outputs = model(data)
        # 获取每个参数的梯度
        """""
        for name, param in model.named_parameters():
            print(f"Parameter name: {name}")
            print(param)
        """""
        
        probs = torch.sigmoid(outputs)
        ##print('啦啦啦我是敏感属性'+str(sensitive_attributes))
        pos_mask = (labels == 1)
        group_0_mask = (sensitive_attributes == 0) & pos_mask
        group_1_mask = (sensitive_attributes == 1) & pos_mask
        ##print('啦啦啦我是概率'+str(probs))
        avg_prob_0 = probs[group_0_mask].mean()
        avg_prob_1 = probs[group_1_mask].mean()
        print('啦啦啦我是0的平均'+ str(avg_prob_0)+'啦啦啦我是1的平均'+str(avg_prob_1) )
        return torch.abs(avg_prob_0 - avg_prob_1)
    def generate_fair_gradient(self, model, synthetic_data, learning_rate=0.1):
        x, y, sensitive_attr = synthetic_data
        device = next(model.parameters()).device  # Get the device where the model resides

        # Ensure all data is on the correct device
        x = x.to(device)
        y = y.to(device)
        sensitive_attr = sensitive_attr.to(device)

        # Create a temporary model for a single gradient descent step
        temp_model = type(model)(
            input_dim=self.args.input_dim,
            hidden_outdim=self.args.num_hidden,
            output_dim=self.args.output_dim
        ).to(device)  # Move the temporary model to the correct device
        temp_model.load_state_dict(model.state_dict())

        # Define optimizer and calculate equal opportunity (eo) loss
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
    
        
        eo_loss = self.calculate_eo_loss(temp_model, x, y, sensitive_attr)
        ##print('啦啦啦啦我是eo的loss，看看我爆炸没+'f'EO Loss: {eo_loss.item()}')


        # Backward pass and update model
        optimizer.zero_grad()  # Clear gradients
        eo_loss.backward()
        optimizer.step()
        for name, param in temp_model.named_parameters():
            print('啦啦啦我是梯度更新关于公平的')
            print(f'Gradient - {name}:')
            print(param.grad)
        # Return gradients of updated model parameters as a dictionary
        return {name: param.clone().detach() for name, param in temp_model.named_parameters()}
    
    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            self.global_model_trajectory.append(copy.deepcopy(w_global))
            w_locals = []

            """
            for scalability: following the original FedAvg algorithm, we uniformly sample a fraction of clients in each round.
            Instead of changing the 'Client' instances, our implementation keeps the 'Client' instances and then updates their local dataset 
            """
            client_indexes = self._client_sampling(
                round_idx, self.args.client_num_in_total, self.args.client_num_per_round
            )
            logging.info("client_indexes = " + str(client_indexes))

            w_save = []
            for idx, client_idx in enumerate(client_indexes):
                client = self.client_list[idx]
                w = client.train(copy.deepcopy(w_global))
                w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
                w_save.append(copy.deepcopy(w))

            w_global = self._aggregate(w_locals, round_idx)
            self.model_trainer.set_model_params(w_global)

            # 在指定轮次生成合成数据和公平梯度
            if round_idx == self.args.synthetic_data_generation_round:
                synthesizer = DataSynthesizer(self.args)
                self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_model_trajectory)
                ##print('啦啦啦我是假数据'+str(self.synthetic_data))
            # 从生成公平梯度后的轮次开始使用
            if round_idx >= self.args.synthetic_data_generation_round:
                # 将公平梯度添加到 w_locals
                print('啦啦啦我是假梯度额外的步骤我运行了'+str(self.args.synthetic_data_generation_round))
                device = next(self.model_trainer.model.parameters()).device
                self.synthetic_data = self.synthetic_data.to(device)
                self.synthetic_labels = self.synthetic_labels.to(device)
                self.synthetic_sensitive_attr = self.synthetic_sensitive_attr.to(device)
            
                self.fair_gradient = self.generate_fair_gradient(
                    self.model_trainer.model,
                    (self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr),
                    learning_rate=self.args.learning_rate,
                )
                fair_gradient = {k: v.to(device) for k, v in self.fair_gradient.items()}
                ##print('啦啦啦我是公平梯度'+str(fair_gradient))
                self.model_trainer.set_model_params(fair_gradient)
        
            if round_idx % self.args.save_epoches == 0:
                torch.save(
                    self.model_trainer.model.state_dict(),
                    os.path.join(
                        self.args.run_folder,
                        "%s_at_%s.pt" % (self.args.save_model_name, round_idx),
                    ),
                )  # check the fedavg model name
                with open(
                    "%s/%s_locals_at_%s.pt"
                    % (self.args.run_folder, self.args.save_model_name, round_idx),
                    "wb",
                ) as f:
                    pickle.dump(w_save, f, protocol=pickle.HIGHEST_PROTOCOL)

            if (
                round_idx == self.args.comm_round - 1
                or round_idx % self.args.frequency_of_the_test == 0
            ):
                self._local_test_on_all_clients(round_idx)

    def train_one_round(self, round_idx, w_global):
        logging.info("################Communication round : {}".format(round_idx))
        print('啦啦啦我是生成假梯度的轮次。。。无语了'+str(self.args.synthetic_data_generation_round))
        w_locals = []
        w_save = []      

        client_indexes = self._client_sampling(
            round_idx, self.args.client_num_in_total, self.args.client_num_per_round
        )
        logging.info("client_indexes = " + str(client_indexes))

        for idx, client_idx in enumerate(client_indexes):
            client = self.client_list[idx]
            w = client.train(copy.deepcopy(w_global))
            w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
        ##print('啦啦啦我是local模型'+str(w_locals))
        1.##聚合生成新的全局模型 先生成W_c+1
        w_global = self._aggregate(w_locals, round_idx)
        self.model_trainer.set_model_params(w_global)

        # 存储全局模型参数
        self.global_model_trajectory.append(copy.deepcopy(w_global))
       
        # 在指定轮次生成合成数据和公平梯度
        if round_idx == self.args.synthetic_data_generation_round:
            synthesizer = DataSynthesizer(self.args)
            self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_model_trajectory)
            ##print('啦啦啦我是假数据'+str(self.synthetic_data))
        # 从生成公平梯度后的轮次开始使用
        if round_idx >= self.args.synthetic_data_generation_round:
            # 将公平梯度添加到 w_locals
            print('啦啦啦我是假梯度额外的步骤我运行了'+str(self.args.synthetic_data_generation_round))
            device = next(self.model_trainer.model.parameters()).device
            self.synthetic_data = self.synthetic_data.to(device)
            self.synthetic_labels = self.synthetic_labels.to(device)
            self.synthetic_sensitive_attr = self.synthetic_sensitive_attr.to(device)
        
            self.fair_gradient = self.generate_fair_gradient(
                self.model_trainer.model,
                (self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr),
                learning_rate=self.args.learning_rate,
            )
            fair_gradient = {k: v.to(device) for k, v in self.fair_gradient.items()}
            ##print('啦啦啦我是公平梯度'+str(fair_gradient))
            self.model_trainer.set_model_params(fair_gradient)
    
        if round_idx % self.args.save_epoches == 0:
            torch.save(
                self.model_trainer.model.state_dict(),
                os.path.join(
                    self.args.run_folder,
                    "%s_at_%s.pt" % (self.args.save_model_name, round_idx),
                ),
            )
            with open(
                "%s/%s_locals_at_%s.pt"
                % (self.args.run_folder, self.args.save_model_name, round_idx),
                "wb",
            ) as f:
                pickle.dump(w_save, f, protocol=pickle.HIGHEST_PROTOCOL)

        if (
            round_idx == self.args.comm_round - 1
            or round_idx % self.args.frequency_of_the_test == 0
        ):
            self._local_test_on_all_clients(round_idx)
        ##print('啦啦啦我是全局参数'+str(w_global))
        ##全局参数正常，在变化
        return w_global

    def _client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = self.args.users
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(
                round_idx
            )  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(
                self.args.users, num_clients, replace=False
            )
            np.random.seed(self.args.random_seed)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        return False

    def _aggregate(self, w_locals, round_idx):
        training_num = 0
        for idx in range(len(w_locals)):
            (sample_num, averaged_params) = w_locals[idx]
            training_num += sample_num

        (sample_num, averaged_params) = w_locals[0]
        for k in averaged_params.keys():
            for i in range(0, len(w_locals)):
                local_sample_number, local_model_params = w_locals[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        return averaged_params
        def _aggregate_noniid_avg(self, w_locals):
            # uniform aggregation
            """
            The old aggregate method will impact the model performance when it comes to Non-IID setting
            Args:
                w_locals:
            Returns:
            """
            (_, averaged_params) = w_locals[0]
            for k in averaged_params.keys():
                temp_w = []
                for _, local_w in w_locals:
                    temp_w.append(local_w[k])
                averaged_params[k] = sum(temp_w) / len(temp_w)
            return averaged_params

    def _local_test_on_all_clients(self, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
       
        train_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
            "eo_gap": [],
            "dp_gap": [],
        }

        test_metrics = {
            "num_samples": [],
            "num_correct": [],
            "losses": [],
            "eo_gap": [],
            "dp_gap": [],
        }

        for idx, client_idx in enumerate(self.args.users):
            """
            Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
            the training client number is larger than the testing client number
            """
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[idx]
            train_local_metrics = client.local_test(False)
            train_metrics["num_samples"].append(
                copy.deepcopy(train_local_metrics["test_total"])
            )
            train_metrics["num_correct"].append(
                copy.deepcopy(train_local_metrics["test_correct"])
            )
            train_metrics["losses"].append(
                copy.deepcopy(train_local_metrics["test_loss"])
            )
            train_metrics["eo_gap"].append(copy.deepcopy(train_local_metrics["eo_gap"]))
            train_metrics["dp_gap"].append(copy.deepcopy(train_local_metrics["dp_gap"]))

            test_local_metrics = client.local_test(True)
            test_metrics["num_samples"].append(
                copy.deepcopy(test_local_metrics["test_total"])
            )
            test_metrics["num_correct"].append(
                copy.deepcopy(test_local_metrics["test_correct"])
            )
            test_metrics["losses"].append(
                copy.deepcopy(test_local_metrics["test_loss"])
            )
            test_metrics["eo_gap"].append(copy.deepcopy(test_local_metrics["eo_gap"]))
            test_metrics["dp_gap"].append(copy.deepcopy(test_local_metrics["dp_gap"]))
        # test on training dataset
        train_acc = sum(train_metrics["num_correct"]) / sum(
            train_metrics["num_samples"]
        )
        train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])
        train_dp_gap = sum(train_metrics["dp_gap"]) / len(self.args.users)
        train_eo_gap = sum(train_metrics["eo_gap"]) / len(self.args.users)
        ##print(train_metrics)
        # test on test dataset
        test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
        test_dp_gap = sum(test_metrics["dp_gap"]) / len(self.args.users)
        test_eo_gap = sum(test_metrics["eo_gap"]) / len(self.args.users)
        logging.info(train_metrics["eo_gap"])
        logging.info(
            "Train acc: {} Train Loss: {}, Test acc: {} Test Loss: {}".format(
                train_acc, train_loss, test_acc, test_loss
            )
        )
        logging.info(
            "Train dp gap: {} Train eo gap: {}, Test dp gap: {} Test eo gap: {}".format(
                train_dp_gap, train_eo_gap, test_dp_gap, test_eo_gap
            )
        )

        if self.args.enable_wandb:
            wandb.log({"Test/Acc": test_acc, "round": round_idx})
            wandb.log({"Test/Loss": test_loss, "round": round_idx})
            wandb.log({"Train/Acc": train_acc, "round": round_idx})
            wandb.log({"Train/Loss": train_loss, "round": round_idx})
        self.acc_list_all_round.append(test_acc)
        self.eo_list_all_round.append(test_eo_gap)
        self.dp_list_all_round.append(test_dp_gap)
        logging.info( "准确性list: {} eo list: {}, dp list: {}".format(
            self.acc_list_all_round, self.eo_list_all_round, self.dp_list_all_round))

    def _local_test_on_validation_set(self, round_idx):
        logging.info(
            "################local_test_on_validation_set : {}".format(round_idx)
        )

        if self.val_global is None:
            self._generate_validation_set()

        client = self.client_list[0]
        client.update_local_dataset(0, None, self.val_global, None)
        # test data
        test_metrics = client.local_test(True)

        if self.args.dataset == "stackoverflow_nwp":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {"test_acc": test_acc, "test_loss": test_loss}
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        elif self.args.dataset == "stackoverflow_lr":
            test_acc = test_metrics["test_correct"] / test_metrics["test_total"]
            test_pre = test_metrics["test_precision"] / test_metrics["test_total"]
            test_rec = test_metrics["test_recall"] / test_metrics["test_total"]
            test_loss = test_metrics["test_loss"] / test_metrics["test_total"]
            stats = {
                "test_acc": test_acc,
                "test_pre": test_pre,
                "test_rec": test_rec,
                "test_loss": test_loss,
            }
            if self.args.enable_wandb:
                wandb.log({"Test/Acc": test_acc, "round": round_idx})
                wandb.log({"Test/Pre": test_pre, "round": round_idx})
                wandb.log({"Test/Rec": test_rec, "round": round_idx})
                wandb.log({"Test/Loss": test_loss, "round": round_idx})
        else:
            raise Exception(
                "Unknown format to log metrics for dataset {}!" % self.args.dataset
            )
        
        logging.info( "准确性list: {} eo list: {}, dp list: {}".format(
                self.acc_list_all_round, self.eo_list_all_round, self.dp_list_all_round))
        logging.info(stats)

    def save(self):
        torch.save(
            self.model_trainer.model.state_dict(),
            os.path.join(self.args.run_folder, "%s.pt" % (self.args.save_model_name)),
        )  # check the fedavg model name
