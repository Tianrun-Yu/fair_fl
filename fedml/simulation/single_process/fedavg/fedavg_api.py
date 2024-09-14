from cmath import log
import copy
import logging
import random
import standard_trainer
import numpy as np
import torch
import wandb
import os
import hdbscan
from .client import Client
from .my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
from .my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
from .my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG
import logging
import pickle
from data_synthesizer import DataSynthesizer
import torch.nn as nn
import torch.nn.functional as F
from adaptive_fairness_optimizer import AdaptiveFairnessOptimizer
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter



class FedAvgAPI(object):
    aggregation_method = 'fedavg' # median trimmed_mean fedavg flame multikrum foolsgold deepsight

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
        self.ba_list_all_round = []
        self.cal_list_all_round = []
        self.con_list_all_round = []
        self.client_list = []
        self.train_data_local_num_dict = train_data_local_num_dict
        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        self.val_data_local_dict = val_data_local_dict
        self.global_dataset = train_data_global.dataset  #  train_data_global 是一个 DataLoader
        self.fairness_optimizer = AdaptiveFairnessOptimizer(['eo', 'dp'], learning_rate=args.learning_rate, acc_threshold=0.01)
        self.metrics_history = {'acc': [], 'eo': [], 'dp': []}

        self.global_model = model  

        # self.aggregation_method = args.aggregation_method
        self.perturbation = self.initialize_perturbation(model)

        # 为 Flame 添加
        self.flame_memory = {}
        self.flame_m = args.flame_m  # Flame 参数，需要在 args 中定义
        self.args.eta = getattr(self.args, 'eta', 0.1)  # 默认学习率为 0.1
        self.args.noise_sigma = getattr(self.args, 'noise_sigma', 0.0)  # 默认不添加噪声
        
        # 为 Deepsight 添加
        self.deepsight_memory = []
        self.deepsight_window_size = args.deepsight_window_size  # Deepsight 参数，需要在 args 中定义

        
        # # 为 Foolsgold 添加
        # self.update_history = []
        # # 根据模型类型确定最后一层的名称
        # if hasattr(args, 'model'):
        #     if args.model == "two-layer":
        #         self.last_layer_name = "linear"  # 假设两层网络的最后一层是线性层
        #     else:
        #         # 对于其他模型类型，您可能需要根据实际情况进行调整
        #         self.last_layer_name = "linear"  # 默认使用 "linear"
        # else:
        #     # 如果 args 中没有 model 属性，我们使用一个默认值
        #     self.last_layer_name = "linear"

        # # 获取最后一层的权重和偏置
        # if self.last_layer_name in model.state_dict():
        #     last_weight = model.state_dict()[f"{self.last_layer_name}.weight"].detach().clone().view(-1)
        #     last_bias = model.state_dict()[f"{self.last_layer_name}.bias"].detach().clone().view(-1)
        #     last_params = torch.cat((last_weight, last_bias))
        # else:
        #     # 如果找不到指定的层，创建一个空的张量
        #     logging.warning(f"Layer {self.last_layer_name} not found in the model. Using empty tensor for Foolsgold initialization.")
        #     last_params = torch.zeros(1)

        # for _ in range(args.client_num_in_total):
        #     last_layer_params = torch.zeros_like(last_params)
        #     self.update_history.append(last_layer_params)

        # 为 Foolsgold 添加
        self.update_history = []

        # 根据模型类型确定最后一层的名称
        if hasattr(args, 'model'):
            self.model_type = args.model
            if args.model == "two-layer":
                self.last_layer_name = "fc2"  # 假设两层网络的最后一层是第二个全连接层
            else:
                # 对于其他模型类型，您可能需要根据实际情况进行调整
                self.last_layer_name = "fc"  # 默认使用 "fc"
        else:
            # 如果 args 中没有 model 属性，我们使用一个默认值
            self.model_type = "unknown"
            self.last_layer_name = "fc"

        # 输出并判断模型类型
        logging.info(f"Model type: {self.model_type}")
        logging.info(f"Last layer name: {self.last_layer_name}")

        # 获取最后一层的权重和偏置
        if self.last_layer_name in model.state_dict():
            last_weight = model.state_dict()[f"{self.last_layer_name}.weight"].detach().clone().view(-1)
            last_bias = model.state_dict()[f"{self.last_layer_name}.bias"].detach().clone().view(-1)
            last_params = torch.cat((last_weight, last_bias))
            logging.info(f"Last layer parameters shape: {last_params.shape}")
        else:
            # 如果找不到指定的层，创建一个空的张量
            logging.warning(f"Layer {self.last_layer_name} not found in the model. Using empty tensor for Foolsgold initialization.")
            last_params = torch.zeros(1)

        # 初始化 update_history
        for _ in range(args.client_num_in_total):
            self.update_history.append(torch.zeros_like(last_params))

        self.no_of_adversaries = args.no_of_adversaries  # 需要在 args 中定义

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


    def get_positive_class_probs(outputs, labels):
        probs = torch.softmax(outputs, dim=1)
        return probs[:, 0]
        
    def calculate_eo_loss(self, model, data, labels, sensitive_attributes):
        outputs = model(data)
        probs = torch.sigmoid(outputs).squeeze()
    
        pos_mask = (labels == 1)
        group_0_mask = (sensitive_attributes == 1) & pos_mask
        group_1_mask = (sensitive_attributes == 2) & pos_mask
    
        print(f"Positive samples: {pos_mask.sum().item()}")
        print(f"Group 0 positive samples: {group_0_mask.sum().item()}")
        print(f"Group 1 positive samples: {group_1_mask.sum().item()}")

        # 添加小的 epsilon 值以避免除以零
        epsilon = 1e-8
        avg_prob_0 = (probs[group_0_mask].sum() + epsilon) / (group_0_mask.sum() + epsilon)
        avg_prob_1 = (probs[group_1_mask].sum() + epsilon) / (group_1_mask.sum() + epsilon)
    
        print(f"Average probability for group 0: {avg_prob_0.item()}")
        print(f"Average probability for group 1: {avg_prob_1.item()}")

        eo_loss = torch.abs(avg_prob_0 - avg_prob_1)
        print(f"EO loss: {eo_loss.item()}")

        return eo_loss
    
    def calculate_dp_loss(self, model, data, labels, sensitive_attributes):
        outputs = model(data)
        probs = torch.sigmoid(outputs).squeeze()

        group_0_mask = (sensitive_attributes == 1)
        group_1_mask = (sensitive_attributes == 2)

        print(f"Total samples: {len(data)}")
        print(f"Group 0 samples: {group_0_mask.sum().item()}")
        print(f"Group 1 samples: {group_1_mask.sum().item()}")

        # 添加小的 epsilon 值以避免除以零
        epsilon = 1e-8
        avg_prob_0 = (probs[group_0_mask].sum() + epsilon) / (group_0_mask.sum() + epsilon)
        avg_prob_1 = (probs[group_1_mask].sum() + epsilon) / (group_1_mask.sum() + epsilon)

        print(f"Average probability for group 0: {avg_prob_0.item()}")
        print(f"Average probability for group 1: {avg_prob_1.item()}")

        dp_loss = torch.abs(avg_prob_0 - avg_prob_1)
        print(f"DP loss: {dp_loss.item()}")

        return dp_loss

    def calculate_ba_loss(self, model, data, labels, sensitive_attributes):
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        pos_probs = probs[:, 0]
        pred_labels = (pos_probs >= 0.5).float()
        
        group_1_mask = (sensitive_attributes == 1)
        group_2_mask = (sensitive_attributes == 2)
        
        ba_1 = 0.5 * (
            (pred_labels[group_1_mask & (labels == 1)] == 1).float().mean() +
            (pred_labels[group_1_mask & (labels == 0)] == 0).float().mean()
        )
        ba_2 = 0.5 * (
            (pred_labels[group_2_mask & (labels == 1)] == 1).float().mean() +
            (pred_labels[group_2_mask & (labels == 0)] == 0).float().mean()
        )
        
        ba_loss = torch.abs(ba_1 - ba_2)
        print(f"BA loss: {ba_loss.item()}")
        return ba_loss

    

    def calculate_cal_loss(self, model, data, labels, sensitive_attributes, num_bins=10):
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        pos_probs = probs[:, 0]

        group_1_mask = (sensitive_attributes == 1)
        group_2_mask = (sensitive_attributes == 2)
        
        def calc_group_cal(group_mask):
            group_probs = pos_probs[group_mask]
            group_labels = labels[group_mask].float()
            cal_sum = 0
            for i in range(num_bins):
                bin_mask = (group_probs >= i/num_bins) & (group_probs < (i+1)/num_bins)
                if bin_mask.sum() > 0:
                    bin_probs = group_probs[bin_mask]
                    bin_labels = group_labels[bin_mask]
                    cal_sum += torch.abs(bin_probs.mean() - bin_labels.mean())
            return cal_sum / num_bins

        cal_1 = calc_group_cal(group_1_mask)
        cal_2 = calc_group_cal(group_2_mask)
        
        cal_loss = torch.abs(cal_1 - cal_2)
        print(f"CAL loss: {cal_loss.item()}")
        return cal_loss

    def calculate_con_loss(self, model, data, labels, k=5):
        outputs = model(data)
        probs = torch.softmax(outputs, dim=1)
        pos_probs = probs[:, 0]
        
        distances = torch.cdist(data, data)
        _, indices = torch.topk(distances, k=k+1, largest=False)
        neighbor_preds = pos_probs[indices[:, 1:]]
        
        con_loss = torch.abs(pos_probs.unsqueeze(1) - neighbor_preds).mean()
        print(f"CON loss: {con_loss.item()}")
        return con_loss
    
    def calculate_accuracy_loss(self, model, data, labels):
        outputs = model(data)
        # 使用 softmax 而不是 sigmoid，因为输出是两个类别的 logits
        probs = F.softmax(outputs, dim=1)
        # 选择概率最高的类别作为预测
        pred_labels = torch.argmax(probs, dim=1)
        # 计算准确率
        accuracy = (pred_labels == labels).float().mean()
        # 我们希望最大化准确率，所以损失是准确率的负值
        accuracy_loss = -accuracy
        print(f"Accuracy: {accuracy.item():.4f}, Accuracy loss: {accuracy_loss.item():.4f}")
        return accuracy_loss

    def calculate_accuracy_loss(self, model, data, labels):
        outputs = model(data)
        loss = F.cross_entropy(outputs, labels)
        
        # 计算准确率（用于监控，不用于优化）
        with torch.no_grad():
            probs = F.softmax(outputs, dim=1)
            pred_labels = torch.argmax(probs, dim=1)
            accuracy = (pred_labels == labels).float().mean()
        
        print(f"Accuracy: {accuracy.item():.4f}, Loss: {loss.item():.4f}")
        
        return loss
    def _compute_gradient(self, loss, model):
        model.zero_grad()
        loss.backward(retain_graph=True)
        return {name: param.grad.clone() for name, param in model.named_parameters()}
    
    def generate_fair_gradient(self, model, synthetic_data, learning_rate=0.1):
        x, y, sensitive_attr = synthetic_data
        device = next(model.parameters()).device  # Get the device where the model resides
        # print('啦啦啦我是x'+str(x)+'啦啦啦我是y'+str(y))
        # print('啦啦啦我是x的shape:', x.shape)
        # print('啦啦啦我是y的shape:', y.shape)
        # print('啦啦啦我是sensitive_attr的shape:', sensitive_attr.shape)
        # print('啦啦啦我是x的前5个样本:', x[:5])
        # print('啦啦啦我是y的前20个标签:', y[:20])
        # print('啦啦啦我是sensitive_attr的前20个值:', sensitive_attr[:20])
        # Ensure all data is on the correct device
        x = x.to(device)
        y = y.to(device)
        sensitive_attr = sensitive_attr.to(device)

        print('数据是否包含NaN:')
        print('x:', torch.isnan(x).any().item())
        print('y:', torch.isnan(y).any().item())
        print('sensitive_attr:', torch.isnan(sensitive_attr).any().item())

        # Create a temporary model for a single gradient descent step
        temp_model = type(model)(
            input_dim=self.args.input_dim,
            hidden_outdim=self.args.num_hidden,
            output_dim=self.args.output_dim
        ).to(device)  # Move the temporary model to the correct device
        temp_model.load_state_dict(model.state_dict())

        for param in temp_model.parameters():
            param.requires_grad = True

        # Define optimizer and calculate equal opportunity (eo) loss
        optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)

        # # 使用近似的公平梯度更新模型
        # with torch.no_grad():
        #     for name, param in temp_model.named_parameters():
        #         if name in fair_gradient:
        #             param.add_(fair_gradient[name])
    
        
        accuracy_loss = self.calculate_accuracy_loss(temp_model,x,y)
        eo_loss = self.calculate_eo_loss(temp_model, x, y, sensitive_attr)
        dp_loss = self.calculate_dp_loss(temp_model, x, y, sensitive_attr)
        con_loss = self.calculate_con_loss(temp_model, x, y)
        ba_loss = self.calculate_ba_loss(temp_model, x, y, sensitive_attr)
        cal_loss = self.calculate_cal_loss(temp_model, x, y, sensitive_attr)

       
        
        combined_loss = 1.5 * eo_loss + (1) * dp_loss  + 0.4*(accuracy_loss)
        

        # Backward pass and update model
        optimizer.zero_grad()  # Clear gradients
        combined_loss.backward()
        optimizer.step()

        torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)


        for name, param in temp_model.named_parameters():
            print('啦啦啦我是梯度更新关于公平的')
            print(f'Gradient - {name}:')
            print(param.grad)

            if param.grad is not None:
                print(f'  Mean: {param.grad.mean().item()}, Std: {param.grad.std().item()}')
                print(f'  Min: {param.grad.min().item()}, Max: {param.grad.max().item()}')
                print(f'  Contains NaN: {torch.isnan(param.grad).any().item()}')
            else:
                print('  Gradient is None')

        # Return gradients of updated model parameters as a dictionary
        return {name: param.clone().detach() for name, param in temp_model.named_parameters()}

        
    def initialize_perturbation(self, model):
        return {name: torch.randn_like(param) for name, param in model.named_parameters()}

    def agr_update(self, w_locals, round_idx):
        # 使用设定的聚合方法聚合模型参数
        w_global = self._aggregate(w_locals, round_idx)

        if round_idx >= self.args.synthetic_data_generation_round:
            # 计算基准梯度 (∇^b)
            base_gradient = self._fedavg_aggregate(w_locals)  # 使用FedAvg来计算基准梯度

            # 优化 gamma
            gamma_range = np.linspace(0, 2, 100)  # 可以根据需要调整范围和精度
            best_gamma = 0
            max_objective = float('-inf')

            for gamma in gamma_range:
                w_temp = {}
                for k in w_global.keys():
                    w_temp[k] = base_gradient[k] + gamma * self.perturbation[k]

                # 使用当前聚合方法计算 f_agr
                w_locals_temp = [(1, w_temp)] + w_locals[1:]  # 替换第一个客户端的梯度
                aggregated_gradient = self._aggregate(w_locals_temp, round_idx)

                # 计算目标函数值
                objective = torch.norm(torch.cat([
                    (base_gradient[k] - aggregated_gradient[k]).flatten() 
                    for k in base_gradient.keys()
                ]), p=2)

                if objective > max_objective:
                    max_objective = objective
                    best_gamma = gamma

            # 计算近似的公平梯度
            fair_gradient = {k: best_gamma * self.perturbation[k] for k in w_global.keys()}

            return fair_gradient
        
        return None

    def calculate_objective(self, w_temp, w_global, x, y, sensitive_attr):
        # 计算目标函数值，这里使用 L2 范数作为示例
        diff = torch.cat([(w_temp[k] - w_global[k]).flatten() for k in w_global.keys()])
        return torch.norm(diff, p=2).item()


    # def train(self):
    #     logging.info(f"Using aggregation method: {self.aggregation_method}")
    #     logging.info("self.model_trainer = {}".format(self.model_trainer))
    #     w_global = self.model_trainer.get_model_params()
    #     for round_idx in range(self.args.comm_round):
    #         logging.info("################Communication round : {}".format(round_idx))
    #         w_locals = []

    #         client_indexes = self._client_sampling(
    #             round_idx, self.args.client_num_in_total, self.args.client_num_per_round
    #         )
    #         logging.info("client_indexes = " + str(client_indexes))

    #         w_save = []
    #         for idx, client_idx in enumerate(client_indexes):
    #             client = self.client_list[idx]
    #             w = client.train(copy.deepcopy(w_global))
    #             w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
    #             w_save.append(copy.deepcopy(w))

    #         # 使用设定的聚合方法进行聚合
    #         w_global = self._aggregate(w_locals, round_idx)

    #         # 在指定轮次生成合成数据
    #         if round_idx == self.args.synthetic_data_generation_round:
    #             synthesizer = DataSynthesizer(self.args)
    #             self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_dataset)
    #             print('Synthetic data generated')

    #         # 从生成公平梯度后的轮次开始使用
    #         if round_idx >= self.args.synthetic_data_generation_round:
    #             # 使用 AGR 更新方法生成近似公平梯度
    #             fair_gradient = self.agr_update(w_locals, round_idx)
                
    #             if fair_gradient is not None:
    #                 # 使用生成的公平梯度优化模型
    #                 self.fair_gradient = self.generate_fair_gradient(
    #                     self.model_trainer.model,
    #                     (self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr),
    #                     fair_gradient, learning_rate=self.args.learning_rate,
    #                 )
                    
    #                 if self.fair_gradient is not None:
    #                     print("Checking fair gradient before adding to aggregation:")
    #                     for name, grad in self.fair_gradient.items():
    #                         print(f"Gradient stats for {name}:")
    #                         print(f"  Mean: {grad.mean().item()}, Std: {grad.std().item()}")
    #                         print(f"  Min: {grad.min().item()}, Max: {grad.max().item()}")
    #                         print(f"  Contains NaN: {torch.isnan(grad).any().item()}")
                        
    #                     if any(torch.any(grad != 0) for grad in self.fair_gradient.values()):
    #                         logging.info("Adding fair gradient to aggregation")
    #                         w_locals.append((len(self.synthetic_data), self.fair_gradient))
    #                         # 再次使用设定的聚合方法进行聚合
    #                         w_global = self._aggregate(w_locals, round_idx)
    #                     else:
    #                         logging.warning("Fair gradient is all zeros, not adding to aggregation")

    #         self.model_trainer.set_model_params(w_global)

    #         if round_idx % self.args.save_epoches == 0:
    #             torch.save(
    #                 self.model_trainer.model.state_dict(),
    #                 os.path.join(
    #                     self.args.run_folder,
    #                     "%s_at_%s.pt" % (self.args.save_model_name, round_idx),
    #                 ),
    #             )
    #             with open(
    #                 "%s/%s_locals_at_%s.pt"
    #                 % (self.args.run_folder, self.args.save_model_name, round_idx),
    #                 "wb",
    #             ) as f:
    #                 pickle.dump(w_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    #         if (
    #             round_idx == self.args.comm_round - 1
    #             or round_idx % self.args.frequency_of_the_test == 0
    #         ):
    #             self._local_test_on_all_clients(round_idx)

    def train(self):
        logging.info("self.model_trainer = {}".format(self.model_trainer))
        w_global = self.model_trainer.get_model_params()
        for round_idx in range(self.args.comm_round):
            logging.info("################Communication round : {}".format(round_idx))
            # self.global_model_trajectory.append(copy.deepcopy(w_global))
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

            original_state_dict = self.model_trainer.get_model_params()
            for k, v in w_global.items():
                if k in original_state_dict:
                    if v.shape != original_state_dict[k].shape:
                        logging.warning(f"Shape mismatch for {k}: aggregated {v.shape}, original {original_state_dict[k].shape}")
                        try:
                            w_global[k] = v.view(original_state_dict[k].shape)
                        except RuntimeError:
                            logging.error(f"Cannot reshape {k} from {v.shape} to {original_state_dict[k].shape}")
                            w_global[k] = original_state_dict[k]


            self.model_trainer.set_model_params(w_global)

            # 在指定轮次生成合成数据和公平梯度
            if round_idx == self.args.synthetic_data_generation_round:
                synthesizer = DataSynthesizer(self.args)
                self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_dataset)
                print('啦啦啦我是假数据'+str(self.synthetic_data))
            # if round_idx >= self.args.synthetic_data_generation_round:
            #     self._calculate_and_log_metrics(round_idx)
            # 从生成公平梯度后的轮次开始使用
            if round_idx >= self.args.synthetic_data_generation_round:
                if self.fair_gradient is None:
                    print("Fair gradient is None. Skipping fair gradient aggregation.")
                else:
                    print("Checking fair gradient before adding to aggregation:")
        
                    for name, grad in self.fair_gradient.items():
                        print(f"Gradient stats for {name}:")
                        print(f"  Mean: {grad.mean().item()}, Std: {grad.std().item()}")
                        print(f"  Min: {grad.min().item()}, Max: {grad.max().item()}")
                        print(f"  Contains NaN: {torch.isnan(grad).any().item()}")
    
                # 如果梯度不全为零，则添加到聚合中
                    if any(torch.any(grad != 0) for grad in self.fair_gradient.values()):
                          # 设置公平梯度的权重
                        fair_gradient_times = 1
                        for _ in range(fair_gradient_times):
                            logging.info("Adding fair gradient to aggregation")
                            w_locals.append((len(self.synthetic_data), self.fair_gradient))
                    else:
                        logging.warning("Fair gradient is all zeros, not adding to aggregation")


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
                fair_gradient_weight = 0.1
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

    # def _aggregate(self, w_locals, round_idx):
        # training_num = 0
        # for idx in range(len(w_locals)):
        #     (sample_num, averaged_params) = w_locals[idx]
        #     training_num += sample_num

        # (sample_num, averaged_params) = w_locals[0]
        # for k in averaged_params.keys():
        #     for i in range(0, len(w_locals)):
        #         local_sample_number, local_model_params = w_locals[i]
        #         w = local_sample_number / training_num
        #         if i == 0:
        #             averaged_params[k] = local_model_params[k] * w
        #         else:
        #             averaged_params[k] += local_model_params[k] * w

        # return averaged_params
        # def _aggregate_noniid_avg(self, w_locals):
        #     # uniform aggregation
        #     """
        #     The old aggregate method will impact the model performance when it comes to Non-IID setting
        #     Args:
        #         w_locals:
        #     Returns:
        #     """
        #     (_, averaged_params) = w_locals[0]
        #     for k in averaged_params.keys():
        #         temp_w = []
        #         for _, local_w in w_locals:
        #             temp_w.append(local_w[k])
        #         averaged_params[k] = sum(temp_w) / len(temp_w)
        #     return averaged_params



    def _aggregate(self, w_locals, round_idx):
        logging.info(f"Aggregating using {self.aggregation_method} method")
        if self.aggregation_method == 'fedavg':
            return self._fedavg_aggregate(w_locals)
        elif self.aggregation_method == 'median':
            return self._median_aggregate(w_locals)
        elif self.aggregation_method == 'trimmed_mean':
            return self._trimmed_mean_aggregate(w_locals)
        elif self.aggregation_method == 'multikrum':
            return self._multikrum_aggregate(w_locals)
        elif self.aggregation_method == 'deepsight':
            return self._deepsight_aggregate(w_locals, round_idx)
        elif self.aggregation_method == 'foolsgold':
            return self._foolsgold_aggregate(w_locals)
        elif self.aggregation_method == 'flame':
            return self._flame_aggregate(w_locals, round_idx)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

    def _fedavg_aggregate(self, w_locals):
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

    def _deepsight_aggregate(self, w_locals, round_idx):
        w_global = {}
        for k in w_locals[0][1].keys():
            weights = torch.stack([w[1][k].float() for w in w_locals])
            
            # 确保所有权重张量具有相同的形状
            if len(self.deepsight_memory) > 0 and self.deepsight_memory[0][k].shape != weights.shape:
                # 如果形状不匹配，重新初始化 deepsight_memory
                self.deepsight_memory = []
            
            if len(self.deepsight_memory) == 0:
                self.deepsight_memory = [{k: torch.zeros_like(weights) for k in w_locals[0][1].keys()}]
            
            # 更新 deepsight_memory
            if len(self.deepsight_memory) >= self.deepsight_window_size:
                self.deepsight_memory.pop(0)
            self.deepsight_memory.append({k: weights.clone() for k in w_locals[0][1].keys()})
            
            # 计算历史权重
            historical_weights = torch.stack([mem[k] for mem in self.deepsight_memory])
            
            # 计算每个客户端的权重
            client_weights = torch.sum(historical_weights, dim=0)
            client_weights = torch.softmax(client_weights, dim=0)
            
            # 使用客户端权重进行加权平均
            w_global[k] = torch.sum(weights * client_weights.unsqueeze(1), dim=0)
        
        # 确保聚合后的权重与原始模型形状一致
        original_state_dict = self.model_trainer.get_model_params()
        for k, v in w_global.items():
            if k in original_state_dict:
                if v.shape != original_state_dict[k].shape:
                    logging.warning(f"Shape mismatch for {k}: aggregated {v.shape}, original {original_state_dict[k].shape}")
                    # 尝试重塑权重以匹配原始形状
                    try:
                        w_global[k] = v.view(original_state_dict[k].shape)
                    except RuntimeError:
                        logging.error(f"Cannot reshape {k} from {v.shape} to {original_state_dict[k].shape}")
                        # 如果无法重塑，保留原始权重
                        w_global[k] = original_state_dict[k]
        
        return w_global

    def _flame_aggregate(self, w_locals, round_idx):
        local_model_vector = []
        update_params = []
        weight_accumulator = {}
        
        # 初始化 weight_accumulator
        for name, data in self.model_trainer.get_model_params().items():
            weight_accumulator[name] = torch.zeros_like(data)

        # 准备数据
        for client_data in w_locals:
            local_model_vector_sub = []
            update_params_sub = []
            for name, param in client_data[1].items():
                local_model_vector_sub.append(param.view(-1))
                update_params_value = param - self.model_trainer.get_model_params()[name]
                update_params_sub.append(update_params_value.view(-1))
            
            local_model_vector.append(torch.cat(local_model_vector_sub).cuda())
            update_params.append(torch.cat(update_params_sub).cuda())

        # 执行 FLAME 聚合
        benign_client, clip_value = self._flame(local_model_vector=local_model_vector, update_params=update_params)
        
        # 聚合更新
        for ind in benign_client:
            client_weight, client_model = w_locals[ind]
            for name, param in client_model.items():
                if name in self.model_trainer.get_model_params():
                    update = param - self.model_trainer.get_model_params()[name]
                    # 应用裁剪
                    update_norm = torch.norm(update)
                    scale = min(1.0, clip_value / update_norm)
                    weight_accumulator[name].add_(update * scale)

        # 应用更新到全局模型
        updated_model = {}
        for name, data in self.model_trainer.get_model_params().items():
            if name in weight_accumulator:
                update = weight_accumulator[name] * (self.args.eta / len(benign_client))
                if self.args.noise_sigma > 0:
                    noise = torch.cuda.FloatTensor(data.shape).normal_(mean=0, std=self.args.noise_sigma * clip_value)
                    update.add_(noise)
                updated_model[name] = data + update
            else:
                updated_model[name] = data

        return updated_model

    def _flame(self, local_model_vector, update_params):
        cos = torch.nn.CosineSimilarity(dim=0, eps=1e-6).cuda()
        cos_list = []

        for i in range(len(local_model_vector)):
            cos_i = []
            for j in range(len(local_model_vector)):
                cos_ij = 1 - cos(local_model_vector[i], local_model_vector[j])
                cos_i.append(cos_ij.item())
            cos_list.append(cos_i)

        num_clients = len(local_model_vector)
        clusterer = hdbscan.HDBSCAN(min_cluster_size=num_clients//2 + 1, min_samples=1, allow_single_cluster=True).fit(cos_list)

        benign_client = []
        norm_list = []

        if clusterer.labels_.max() < 0:
            benign_client = list(range(len(local_model_vector)))
            norm_list = [torch.norm(update, p=2).item() for update in update_params]
        else:
            max_cluster_index = np.argmax(np.bincount(clusterer.labels_[clusterer.labels_ >= 0]))
            for i, label in enumerate(clusterer.labels_):
                if label == max_cluster_index:
                    benign_client.append(i)
                    norm_list.append(torch.norm(update_params[i], p=2).item())

        clip_value = np.median(norm_list)
        return benign_client, clip_value


    def _multikrum_aggregate(self, w_locals):
        candidates = []
        candidate_indices = []
        remaining_updates = [w for _, w in w_locals]
        all_indices = np.arange(len(w_locals))

        while len(remaining_updates) > 2 * self.args.no_of_adversaries + 2:
            distances = []
            for i, update in enumerate(remaining_updates):
                distance = []
                update_tensor = torch.cat([torch.flatten(x) for x in update.values()])
                for j, update_ in enumerate(remaining_updates):
                    if i != j:
                        update_tensor_ = torch.cat([torch.flatten(x) for x in update_.values()])
                        distance.append(torch.norm(update_tensor - update_tensor_).item())
                    else:
                        distance.append(0)
                distances.append(distance)
            distances = torch.tensor(distances)

            sorted_indices = torch.argsort(distances, dim=1)
            scores = torch.sum(sorted_indices[:, :len(remaining_updates) - 2 - self.args.no_of_adversaries], dim=1)
            best_index = torch.argmin(scores).item()

            candidate_indices.append(all_indices[best_index])
            all_indices = np.delete(all_indices, best_index)
            candidates.append(remaining_updates[best_index])
            del remaining_updates[best_index]

        # Convert the dictionaries in the candidates list to tensors
        candidate_tensors = [torch.cat([torch.flatten(x) for x in candidate.values()]) for candidate in candidates]

        # Aggregate the selected candidates
        aggregated_update = torch.mean(torch.stack(candidate_tensors), dim=0)

        # Reshape the aggregated update back to the original model structure
        aggregated_model = {}
        start = 0
        for key, value in w_locals[0][1].items():
            shape = value.shape
            numel = value.numel()
            aggregated_model[key] = aggregated_update[start:start+numel].reshape(shape)
            start += numel

        return aggregated_model

    def _foolsgold_aggregate(self, w_locals):
        selected_clients = list(range(len(w_locals)))
        cs = np.zeros((len(w_locals), len(w_locals)))

        # 更新 update_history
        for i, (_, update) in enumerate(w_locals):
            if f"{self.last_layer_name}.weight" in update and f"{self.last_layer_name}.bias" in update:
                last_weight = update[f"{self.last_layer_name}.weight"].detach().clone().view(-1)
                last_bias = update[f"{self.last_layer_name}.bias"].detach().clone().view(-1)
                last_params = torch.cat((last_weight, last_bias))
                self.update_history[i] += last_params.cpu()
            else:
                logging.warning(f"Last layer {self.last_layer_name} not found in client update. Skipping this client for FoolsGold.")

        # 计算余弦相似度
        for i in range(len(w_locals)):
            for j in range(len(w_locals)):
                if self.update_history[i].norm() > 0 and self.update_history[j].norm() > 0:
                    cs[i][j] = F.cosine_similarity(self.update_history[i].unsqueeze(0), 
                                                self.update_history[j].unsqueeze(0)).item()

        
        cs = cs - np.eye(len(w_locals))
        maxcs = np.max(cs, axis=1) + 1e-5
        for i in range(len(w_locals)):
            for j in range(len(w_locals)):
                if i == j:
                    continue
                if maxcs[i] < maxcs[j]:
                    cs[i][j] = cs[i][j] * maxcs[i] / maxcs[j]
        
        wv = 1 - np.max(cs, axis=1)
        wv[wv > 1] = 1
        wv[wv < 0] = 0
        wv = wv / np.max(wv)
        wv[wv == 1] = .99
        wv = (np.log((wv / (1 - wv)) + 1e-5) + 0.5)
        wv[np.isinf(wv) + wv > 1] = 1
        wv[wv < 0] = 0
        
        # 聚合更新
        aggregated_update = {}
        for name in w_locals[0][1].keys():
            aggregated_update[name] = torch.zeros_like(w_locals[0][1][name])
            for i, (_, update) in enumerate(w_locals):
                aggregated_update[name] += wv[i] * update[name]
        
        return aggregated_update

    def _median_aggregate(self, w_locals):
        averaged_params = {}
        for k in w_locals[0][1].keys():
            k_weights = torch.stack([w[1][k].float() for w in w_locals])
            averaged_params[k] = torch.median(k_weights, dim=0).values
        return averaged_params

    def _trimmed_mean_aggregate(self, w_locals, trim_ratio=0.1):
        averaged_params = {}
        for k in w_locals[0][1].keys():
            k_weights = torch.stack([w[1][k].float() for w in w_locals])
            k_sorted, _ = torch.sort(k_weights, dim=0)
            n = k_weights.size(0)
            trim = int(trim_ratio * n)
            if trim > 0:
                k_trimmed = k_sorted[trim:-trim]
            else:
                k_trimmed = k_sorted
            averaged_params[k] = torch.mean(k_trimmed, dim=0)
        return averaged_params

    def _local_test_on_all_clients(self, round_idx):
        logging.info("################local_test_on_all_clients : {}".format(round_idx))
    
        metrics = ["num_samples", "num_correct", "losses", "eo", "dp", "ba", "cal", "con"]
        train_metrics = {metric: [] for metric in metrics}
        test_metrics = {metric: [] for metric in metrics}

        for idx, client_idx in enumerate(self.args.users):
            if self.test_data_local_dict[client_idx] is None:
                continue
            client = self.client_list[idx]
        
            for is_test in [False, True]:
                local_metrics = client.local_test(is_test)
                target_metrics = test_metrics if is_test else train_metrics
            
                target_metrics["num_samples"].append(copy.deepcopy(local_metrics["test_total"]))
                target_metrics["num_correct"].append(copy.deepcopy(local_metrics["test_correct"]))
                target_metrics["losses"].append(copy.deepcopy(local_metrics["test_loss"]))
            
                for metric in ["eo", "dp", "ba", "cal", "con"]:
                    if metric in local_metrics:
                        target_metrics[metric].append(copy.deepcopy(local_metrics[metric]))

        # Calculate metrics
        results = {}
        for dataset in ["train", "test"]:
            dataset_metrics = train_metrics if dataset == "train" else test_metrics
        
            if sum(dataset_metrics["num_samples"]) > 0:
                results[f"{dataset}_acc"] = sum(dataset_metrics["num_correct"]) / sum(dataset_metrics["num_samples"])
                results[f"{dataset}_loss"] = sum(dataset_metrics["losses"]) / sum(dataset_metrics["num_samples"])
            else:
                results[f"{dataset}_acc"] = 0
                results[f"{dataset}_loss"] = 0

            for metric in ["eo", "dp", "ba", "cal", "con"]:
                if dataset_metrics[metric]:
                    results[f"{dataset}_{metric}"] = sum(dataset_metrics[metric]) / len(self.args.users)
                else:
                    results[f"{dataset}_{metric}"] = 0

        # Log results
        logging.info(f"Train acc: {results['train_acc']:.4f}, Train Loss: {results['train_loss']:.4f}")
        logging.info(f"Test acc: {results['test_acc']:.4f}, Test Loss: {results['test_loss']:.4f}")

        for metric in ["eo", "dp", "ba", "cal", "con"]:
            if f"train_{metric}" in results and f"test_{metric}" in results:
                logging.info(f"Train {metric}: {results[f'train_{metric}']:.4f}, Test {metric}: {results[f'test_{metric}']:.4f}")

        # Log to wandb
        if self.args.enable_wandb:
            for dataset in ["Train", "Test"]:
                wandb.log({f"{dataset}/Acc": results[f"{dataset.lower()}_acc"], "round": round_idx})
                wandb.log({f"{dataset}/Loss": results[f"{dataset.lower()}_loss"], "round": round_idx})
                for metric in ["eo", "dp", "ba", "cal", "con"]:
                    if f"{dataset.lower()}_{metric}" in results:
                        wandb.log({f"{dataset}/{metric.upper()}": results[f"{dataset.lower()}_{metric}"], "round": round_idx})

        # Update all_round lists
        self.acc_list_all_round.append(results["test_acc"])
        for metric in ["eo", "dp", "ba", "cal", "con"]:
            if f"test_{metric}" in results:
                getattr(self, f"{metric}_list_all_round").append(results[f"test_{metric}"])

        # Log all_round lists
        logging.info("Metrics over all rounds:")
        for metric in ["acc"] + ["eo", "dp", "ba", "cal", "con"]:
            if hasattr(self, f"{metric}_list_all_round"):
                logging.info(f"{metric}_list = {getattr(self, f'{metric}_list_all_round')}")

        # Save metrics to file
        metrics_data = {
            metric: getattr(self, f"{metric}_list_all_round")
            for metric in ["acc"] + ["eo", "dp", "ba", "cal", "con"]
            if hasattr(self, f"{metric}_list_all_round")
        }
        with open(os.path.join(self.args.run_folder, "metrics.pkl"), "wb") as f:
            pickle.dump(metrics_data, f)

        # Plot metrics
        self.plot_metrics(metrics_data)

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
        
        logging.info( "Acc list: {} eo list: {}, dp list: {}".format(
                self.acc_list_all_round, self.eo_list_all_round, self.dp_list_all_round))
        logging.info(stats)

    def save(self):
        torch.save(
            self.model_trainer.model.state_dict(),
            os.path.join(self.args.run_folder, "%s.pt" % (self.args.save_model_name)),
        )  # check the fedavg model name

    def plot_metrics(self, metrics_data):
        rounds = list(range(1, len(metrics_data["acc"]) + 1))

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

        # Plot accuracy
        ax1.plot(rounds, metrics_data["acc"], label="Accuracy")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy Trend")
        ax1.legend()
        ax1.grid(True)

        # Plot fairness metrics
        fairness_metrics = ["eo", "dp", "ba", "cal", "con"]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(fairness_metrics)))
        
        for metric, color in zip(fairness_metrics, colors):
            line, = ax2.plot(rounds, metrics_data[metric], label=metric.upper(), color=color)
            ax2.text(rounds[-1], metrics_data[metric][-1], metric.upper(), 
                    color=color, fontweight='bold', ha='left', va='center')

        ax2.set_xlabel("Rounds")
        ax2.set_ylabel("Fairness Metric Value")
        ax2.set_title("Fairness Metrics Trends")
        
        # Create custom y-axis for fairness metrics
        y_ticks = list(np.arange(0, 0.11, 0.02)) + list(np.arange(0.2, 1.1, 0.2))
        ax2.set_yticks(y_ticks)
        ax2.set_ylim(0, 0.4)  # Adjust the upper limit as needed
        
        # Use symlog scale for y-axis of fairness metrics
        ax2.set_yscale('symlog', linthresh=0.1)
        
        ax2.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(self.args.run_folder, "metrics_trends.png"), dpi=300)
        plt.close()