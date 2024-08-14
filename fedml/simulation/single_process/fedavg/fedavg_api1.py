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
import torch.nn as nn
import torch.nn.functional as F
from adaptive_fairness_optimizer import AdaptiveFairnessOptimizer
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# def calculate_gradients(fedavg_api, model, x, y, sensitive_attr):
#     gradients = {}
#     losses = {}

#     # Accuracy loss
#     accuracy_loss = fedavg_api.calculate_accuracy_loss(model, x, y)
#     accuracy_loss.backward(retain_graph=True)
#     gradients['accuracy'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['accuracy'] = accuracy_loss.item()
#     model.zero_grad()

#     # EO loss
#     eo_loss = fedavg_api.calculate_eo_loss(model, x, y, sensitive_attr)
#     eo_loss.backward(retain_graph=True)
#     gradients['eo'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['eo'] = eo_loss.item()
#     model.zero_grad()

#     # DP loss
#     dp_loss = fedavg_api.calculate_dp_loss(model, x, y, sensitive_attr)
#     dp_loss.backward(retain_graph=True)
#     gradients['dp'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['dp'] = dp_loss.item()
#     model.zero_grad()

#     # CON loss
#     con_loss = fedavg_api.calculate_con_loss(model, x, y)
#     con_loss.backward(retain_graph=True)
#     gradients['con'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['con'] = con_loss.item()
#     model.zero_grad()

#     # BA loss
#     ba_loss = fedavg_api.calculate_ba_loss(model, x, y, sensitive_attr)
#     ba_loss.backward(retain_graph=True)
#     gradients['ba'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['ba'] = ba_loss.item()
#     model.zero_grad()

#     # CAL loss
#     cal_loss = fedavg_api.calculate_cal_loss(model, x, y, sensitive_attr)
#     cal_loss.backward(retain_graph=True)
#     gradients['cal'] = [p.grad.clone() for p in model.parameters() if p.grad is not None]
#     losses['cal'] = cal_loss.item()
#     model.zero_grad()

#     return gradients, losses

# def mgda_update(gradients, losses, model, optimizer, prev_losses=None):
#     n_tasks = len(gradients)
#     scale = np.zeros(n_tasks)
    
#     # Frank-Wolfe algorithm to find the optimal scaling
#     for _ in range(20):  # number of iterations for Frank-Wolfe
#         grad_prod = np.zeros(n_tasks)
#         for i in range(n_tasks):
#             for j in range(n_tasks):
#                 grad_prod[i] += scale[j] * sum(
#                     torch.sum(g_i * g_j) for g_i, g_j in zip(gradients[list(gradients.keys())[i]], gradients[list(gradients.keys())[j]])
#                 ).item()
        
#         idx = np.argmin(grad_prod)
#         gamma = 2.0 / (2.0 + _)
#         scale = (1 - gamma) * scale
#         scale[idx] += gamma
    
#     # Apply the scaling to the gradients
#     for param in model.parameters():
#         if param.grad is not None:
#             param.grad.zero_()
    
#     for i, task in enumerate(gradients.keys()):
#         for param, grad in zip(model.parameters(), gradients[task]):
#             if param.grad is not None:
#                 param.grad += scale[i] * grad
    
#     # Update the model
#     optimizer.step()
    
#     return scale

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


    # def calculate_eo_loss(self, model, data, labels, sensitive_attributes):
    #     outputs = model(data)
    #     # 获取每个参数的梯度
    #     """""
    #     for name, param in model.named_parameters():
    #         print(f"Parameter name: {name}")
    #         print(param)
    #     """""
        
    #     probs = torch.sigmoid(outputs)
    #     ##print('啦啦啦我是敏感属性'+str(sensitive_attributes))
    #     pos_mask = (labels == 1)
    #     group_0_mask = (sensitive_attributes == 0) & pos_mask
    #     group_1_mask = (sensitive_attributes == 1) & pos_mask
    #     ##print('啦啦啦我是概率'+str(probs))
    #     avg_prob_0 = probs[group_0_mask].mean()
    #     avg_prob_1 = probs[group_1_mask].mean()
    #     print('啦啦啦我是0的平均'+ str(avg_prob_0)+'啦啦啦我是1的平均'+str(avg_prob_1) )
    #     return torch.abs(avg_prob_0 - avg_prob_1)
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

    # def generate_fair_gradient(self, model, synthetic_data, learning_rate=0.1, num_epochs=10):
    #     x, y, sensitive_attr = synthetic_data
    #     device = next(model.parameters()).device
    #     x, y, sensitive_attr = x.to(device), y.to(device), sensitive_attr.to(device)

    #     print('Data shapes:')
    #     print('x:', x.shape)
    #     print('y:', y.shape)
    #     print('sensitive_attr:', sensitive_attr.shape)
    #     print('Sample data:')
    #     print('x (first 5):', x[:5])
    #     print('y (first 20):', y[:20])
    #     print('sensitive_attr (first 20):', sensitive_attr[:20])
    #     print('NaN check:')
    #     print('x:', torch.isnan(x).any().item())
    #     print('y:', torch.isnan(y).any().item())
    #     print('sensitive_attr:', torch.isnan(sensitive_attr).any().item())

    #     temp_model = type(model)(
    #         input_dim=self.args.input_dim,
    #         hidden_outdim=self.args.num_hidden,
    #         output_dim=self.args.output_dim
    #     ).to(device)
    #     temp_model.load_state_dict(model.state_dict())

    #     optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
        
    #     prev_losses = None
    #     for epoch in range(num_epochs):
    #         gradients, losses = calculate_gradients(self, temp_model, x, y, sensitive_attr)
    #         scale = mgda_update(gradients, losses, temp_model, optimizer, prev_losses)
            
    #         print(f"Epoch {epoch + 1}")
    #         for task, loss in losses.items():
    #             print(f"  {task} Loss: {loss:.4f}, Scale: {scale[list(gradients.keys()).index(task)]:.4f}")
            
    #         prev_losses = losses

    #     # Gradient clipping
    #     torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)

    #     # Print final gradients
    #     for name, param in temp_model.named_parameters():
    #         if param.grad is not None:
    #             print(f'Gradient - {name}:')
    #             print(f'  Mean: {param.grad.mean().item():.4f}, Std: {param.grad.std().item():.4f}')
    #             print(f'  Min: {param.grad.min().item():.4f}, Max: {param.grad.max().item():.4f}')
    #             print(f'  Contains NaN: {torch.isnan(param.grad).any().item()}')
    #         else:
    #             print(f'Gradient - {name}: None')

    #     # Compute the final gradient
    #     final_gradient = {name: param.grad.clone() for name, param in temp_model.named_parameters() if param.grad is not None}
        
    #     return final_gradient

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
        print('啦啦啦我是x'+str(x)+'啦啦啦我是y'+str(y))
        print('啦啦啦我是x的shape:', x.shape)
        print('啦啦啦我是y的shape:', y.shape)
        print('啦啦啦我是sensitive_attr的shape:', sensitive_attr.shape)
        print('啦啦啦我是x的前5个样本:', x[:5])
        print('啦啦啦我是y的前20个标签:', y[:20])
        print('啦啦啦我是sensitive_attr的前20个值:', sensitive_attr[:20])
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
        # fairness_optimizer = AdaptiveFairnessOptimizer(fairness_metrics=['eo', 'dp'], learning_rate=learning_rate, acc_threshold=0.01)

        # gradients = {}
    
        
        accuracy_loss = self.calculate_accuracy_loss(temp_model,x,y)
        eo_loss = self.calculate_eo_loss(temp_model, x, y, sensitive_attr)
        dp_loss = self.calculate_dp_loss(temp_model, x, y, sensitive_attr)
        con_loss = self.calculate_con_loss(temp_model, x, y)
        ba_loss = self.calculate_ba_loss(temp_model, x, y, sensitive_attr)
        cal_loss = self.calculate_cal_loss(temp_model, x, y, sensitive_attr)

        # print("Optimization weights:", fairness_optimizer.optimization_weights)
        # print(f"Accuracy loss: {accuracy_loss.item():.4f}")
        # print(f"EO loss: {eo_loss.item():.4f}")
        # print(f"DP loss: {dp_loss.item():.4f}")

        # gradients = {
        #     'acc': self._compute_gradient(accuracy_loss, temp_model),
        #     'eo': self._compute_gradient(eo_loss, temp_model),
        #     'dp': self._compute_gradient(dp_loss, temp_model)
        # }

        # current_metrics = {
        #     'acc': 1-self.metrics_history['acc'][-1],
        #     'eo': self.metrics_history['eo'][-1],
        #     'dp': self.metrics_history['dp'][-1]
        # }

        # # Use AdaptiveFairnessOptimizer to get the final gradient
        # fairness_optimizer.print_gradient_info(gradients)


        # learning_rate = fairness_optimizer.get_learning_rate(current_metrics)
        # final_gradient = fairness_optimizer.optimize(gradients, current_metrics)
        combined_loss = 1.5 * eo_loss + (1) * dp_loss  + 0.4*(accuracy_loss)
        # combined_loss = 0.4 * accuracy_loss + 0.2 * eo_loss + 0.3 * dp_loss + 0.05 * cal_loss + 0.05 * con_loss
        # # print('啦啦啦啦我是eo的loss，看看我爆炸没+'f'EO Loss: {combined_loss.item()}')
        # for k in final_gradient:
        #     final_gradient[k] *= learning_rate

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

        # for name, grad in final_gradient.items():
        #     torch.nn.utils.clip_grad_norm_(grad, max_norm=1.0)
        # return {name: grad.clone().detach() for name, grad in final_gradient.items()}
        # final_gradient = {name: param.grad.clone() for name, param in temp_model.named_parameters() if param.grad is not None}


        # return final_gradient

    # def _calculate_and_log_metrics(self, round_idx):
    #     # 使用全局模型在合成数据上计算指标
    #     model = self.model_trainer.model
    #     x, y, sensitive_attr = self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr
        
    #     accuracy_loss = self.calculate_accuracy_loss(model, x, y)
    #     eo_loss = self.calculate_eo_loss(model, x, y, sensitive_attr)
    #     dp_loss = self.calculate_dp_loss(model, x, y, sensitive_attr)
        
    #     self.metrics_history['acc'].append(accuracy_loss.item())
    #     self.metrics_history['eo'].append(eo_loss.item())
    #     self.metrics_history['dp'].append(dp_loss.item())
        
    #     print(f"Round {round_idx}: Accuracy Loss: {accuracy_loss.item():.4f}, EO Loss: {eo_loss.item():.4f}, DP Loss: {dp_loss.item():.4f}")
        
    #     # 更新 AdaptiveFairnessOptimizer 的历史记录
    #     gradients = {
    #         'acc': self._compute_gradient(accuracy_loss, model),
    #         'eo': self._compute_gradient(eo_loss, model),
    #         'dp': self._compute_gradient(dp_loss, model)
    #     }
    #     self.fairness_optimizer.update_angle_history(gradients)

    # def train(self):
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

    #         # 在指定轮次生成合成数据
    #         if round_idx == self.args.synthetic_data_generation_round:
    #             synthesizer = DataSynthesizer(self.args)
    #             self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_dataset)
    #             print('啦啦啦我是假数据'+str(self.synthetic_data))

    #         if round_idx >= self.args.synthetic_data_generation_round:
    #             self._calculate_and_log_metrics(round_idx) 

    #         # 从指定轮次开始生成和应用公平梯度
    #         if round_idx >= self.args.synthetic_data_generation_round:
                
                
    #             device = next(self.model_trainer.model.parameters()).device
    #             self.synthetic_data = self.synthetic_data.to(device)
    #             self.synthetic_labels = self.synthetic_labels.to(device)
    #             self.synthetic_sensitive_attr = self.synthetic_sensitive_attr.to(device)
            
    #             self.fair_gradient = self.generate_fair_gradient(
    #                 self.model_trainer.model,
    #                 (self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr),
    #                 learning_rate=self.args.learning_rate,
    #             )

    #             # 应用公平梯度
    #             fair_gradient_weight = 1
    #             for name, param in self.model_trainer.model.named_parameters():
    #                 if name in self.fair_gradient:
    #                     if param.grad is None:
    #                         param.grad = fair_gradient_weight * self.fair_gradient[name]
    #                     else:
    #                         param.grad += fair_gradient_weight * self.fair_gradient[name]

    #             # 打印梯度统计信息
    #             print("Checking fair gradient after applying:")
    #             for name, param in self.model_trainer.model.named_parameters():
    #                 if param.grad is not None:
    #                     print(f"Gradient stats for {name}:")
    #                     print(f"  Mean: {param.grad.mean().item()}, Std: {param.grad.std().item()}")
    #                     print(f"  Min: {param.grad.min().item()}, Max: {param.grad.max().item()}")
    #                     print(f"  Contains NaN: {torch.isnan(param.grad).any().item()}")

    #         # 聚合梯度
    #         w_global = self._aggregate(w_locals, round_idx)
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

    # def train_one_round(self, round_idx, w_global):
    #     logging.info("################Communication round : {}".format(round_idx))
    #     print('啦啦啦我是生成假梯度的轮次。。。无语了'+str(self.args.synthetic_data_generation_round))
    #     w_locals = []
    #     w_save = []      

    #     client_indexes = self._client_sampling(
    #         round_idx, self.args.client_num_in_total, self.args.client_num_per_round
    #     )
    #     logging.info("client_indexes = " + str(client_indexes))

    #     for idx, client_idx in enumerate(client_indexes):
    #         client = self.client_list[idx]
    #         w = client.train(copy.deepcopy(w_global))
    #         w_locals.append((client.get_sample_number(), copy.deepcopy(w)))
    #     ##print('啦啦啦我是local模型'+str(w_locals))
    #     1.##聚合生成新的全局模型 先生成W_c+1
    #     w_global = self._aggregate(w_locals, round_idx)
    #     self.model_trainer.set_model_params(w_global)

    #     # # 存储全局模型参数
    #     # self.global_model_trajectory.append(copy.deepcopy(w_global))
       
    #     # 在指定轮次生成合成数据和公平梯度
    #     if round_idx == self.args.synthetic_data_generation_round:
    #         synthesizer = DataSynthesizer(self.args)
    #         self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr = synthesizer.synthesize(self.global_dataset)
    #         ##print('啦啦啦我是假数据'+str(self.synthetic_data))
    #     # 从生成公平梯度后的轮次开始使用
    #     if round_idx >= self.args.synthetic_data_generation_round:
    #         # 将公平梯度添加到 w_locals
    #         print('啦啦啦我是假梯度额外的步骤我运行了'+str(self.args.synthetic_data_generation_round))
    #         device = next(self.model_trainer.model.parameters()).device
    #         self.synthetic_data = self.synthetic_data.to(device)
    #         self.synthetic_labels = self.synthetic_labels.to(device)
    #         self.synthetic_sensitive_attr = self.synthetic_sensitive_attr.to(device)
        
    #         self.fair_gradient = self.generate_fair_gradient(
    #             self.model_trainer.model,
    #             (self.synthetic_data, self.synthetic_labels, self.synthetic_sensitive_attr),
    #             learning_rate=self.args.learning_rate,
    #         )

    #         fair_gradient = {k: v.to(device) for k, v in self.fair_gradient.items()}
    #         ##print('啦啦啦我是公平梯度'+str(fair_gradient))
    #         self.model_trainer.set_model_params(fair_gradient)
    
    #     if round_idx % self.args.save_epoches == 0:
    #         torch.save(
    #             self.model_trainer.model.state_dict(),
    #             os.path.join(
    #                 self.args.run_folder,
    #                 "%s_at_%s.pt" % (self.args.save_model_name, round_idx),
    #             ),
    #         )
    #         with open(
    #             "%s/%s_locals_at_%s.pt"
    #             % (self.args.run_folder, self.args.save_model_name, round_idx),
    #             "wb",
    #         ) as f:
    #             pickle.dump(w_save, f, protocol=pickle.HIGHEST_PROTOCOL)

    #     if (
    #         round_idx == self.args.comm_round - 1
    #         or round_idx % self.args.frequency_of_the_test == 0
    #     ):
    #         self._local_test_on_all_clients(round_idx)
    #     ##print('啦啦啦我是全局参数'+str(w_global))
    #     ##全局参数正常，在变化
    #     return w_global

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

        # for idx, client_idx in enumerate(self.args.users):
        #     """
        #     Note: for datasets like "fed_CIFAR100" and "fed_shakespheare",
        #     the training client number is larger than the testing client number
        #     """
        #     if self.test_data_local_dict[client_idx] is None:
        #         continue
        #     client = self.client_list[idx]
        #     train_local_metrics = client.local_test(False)
        #     train_metrics["num_samples"].append(
        #         copy.deepcopy(train_local_metrics["test_total"])
        #     )
        #     train_metrics["num_correct"].append(
        #         copy.deepcopy(train_local_metrics["test_correct"])
        #     )
        #     train_metrics["losses"].append(
        #         copy.deepcopy(train_local_metrics["test_loss"])
        #     )
        #     train_metrics["eo_gap"].append(copy.deepcopy(train_local_metrics["eo_gap"]))
        #     train_metrics["dp_gap"].append(copy.deepcopy(train_local_metrics["dp_gap"]))

        #     test_local_metrics = client.local_test(True)
        #     test_metrics["num_samples"].append(
        #         copy.deepcopy(test_local_metrics["test_total"])
        #     )
        #     test_metrics["num_correct"].append(
        #         copy.deepcopy(test_local_metrics["test_correct"])
        #     )
        #     test_metrics["losses"].append(
        #         copy.deepcopy(test_local_metrics["test_loss"])
        #     )
        #     test_metrics["eo_gap"].append(copy.deepcopy(test_local_metrics["eo_gap"]))
        #     test_metrics["dp_gap"].append(copy.deepcopy(test_local_metrics["dp_gap"]))
        # # test on training dataset
        # train_acc = sum(train_metrics["num_correct"]) / sum(
        #     train_metrics["num_samples"]
        # )
        # train_loss = sum(train_metrics["losses"]) / sum(train_metrics["num_samples"])
        # train_dp_gap = sum(train_metrics["dp_gap"]) / len(self.args.users)
        # train_eo_gap = sum(train_metrics["eo_gap"]) / len(self.args.users)
        # ##print(train_metrics)
        # # test on test dataset
        # test_acc = sum(test_metrics["num_correct"]) / sum(test_metrics["num_samples"])
        # test_loss = sum(test_metrics["losses"]) / sum(test_metrics["num_samples"])
        # test_dp_gap = sum(test_metrics["dp_gap"]) / len(self.args.users)
        # test_eo_gap = sum(test_metrics["eo_gap"]) / len(self.args.users)
        # logging.info(train_metrics["eo_gap"])
        # logging.info(
        #     "Train acc: {} Train Loss: {}, Test acc: {} Test Loss: {}".format(
        #         train_acc, train_loss, test_acc, test_loss
        #     )
        # )
        # logging.info(
        #     "Train dp gap: {} Train eo gap: {}, Test dp gap: {} Test eo gap: {}".format(
        #         train_dp_gap, train_eo_gap, test_dp_gap, test_eo_gap
        #     )
        # )

        # if self.args.enable_wandb:
        #     wandb.log({"Test/Acc": test_acc, "round": round_idx})
        #     wandb.log({"Test/Loss": test_loss, "round": round_idx})
        #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
        #     wandb.log({"Train/Loss": train_loss, "round": round_idx})
        # self.acc_list_all_round.append(test_acc)
        # self.eo_list_all_round.append(test_eo_gap)
        # self.dp_list_all_round.append(test_dp_gap)
        # logging.info( "准确性list: {} eo list: {}, dp list: {}".format(
        #     self.acc_list_all_round, self.eo_list_all_round, self.dp_list_all_round))

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