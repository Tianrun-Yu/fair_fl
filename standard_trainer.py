import os
import torch
from torch import nn
from fedml.core.alg_frame.client_trainer import ClientTrainer
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

"""
Standard  local trainer; the parties minimize the loss
"""
def replace_nan_with_zero(x):
    # Check if there are any NaN values in the tensor
    if torch.isnan(x).any():
        # Replace NaN values with 0
        x[torch.isnan(x)] = 0
    return x

class StandardTrainer(ClientTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        ##print('啦啦啦我是client的训练函数我训练了啊啊啊啊啊')
        ##print('啦啦啦我是client在训练参数'+str(self.get_model_params()))
        epoch_loss = []
        model = self.model

        model.to(device)
        model.train()
        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=args.learning_rate,
                weight_decay=args.weight_decay,
                amsgrad=True,
            )
        epoch_loss = []
        for _ in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels, s) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                ##print('啦啦啦我是client每轮训练数据x'+str(x))
                ##print('啦啦啦我是client每轮训练数据label'+str(labels))
                model.zero_grad()
                x = replace_nan_with_zero(x)
                log_probs = model(x)
                ##print('啦啦啦我是client每轮训练预测值'+str(log_probs)+'label'+str(labels))
                loss = criterion(log_probs, labels)
                ##print('啦啦啦我是clinet每轮训练的loss'+str(batch_loss))
                loss.backward()
                # 假设你有一个 loss 已经定义和计算完毕，并且已经调用了 loss.backward() 来计算梯度

                # 获取每个参数的梯度
                """""
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f'Parameter: {name}, Gradient norm: {param.grad.norm().item()}')
                """""
                # 这段代码将打印出每个参数的梯度范数，以及其他你可能感兴趣的梯度信息。

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))

    def test_on_the_server(self):
        return False

    def test(self, test_data, device, args):
        ##print('我是standard trainer 我说输出了1')
        model = self.model
        model.to(device)
        model.eval()

        metrics = {"test_correct": 0, "test_loss": 0, "test_total": 0}
        criterion = nn.CrossEntropyLoss().to(device)

        target_list = []
        s_list = []
        x_list = []
        pred_list = []

        with torch.no_grad():
            for x, target, s in test_data:
                target_list.extend(target.tolist())
                s_list.extend(s.tolist())
                x_list.extend(x.tolist())
                x = replace_nan_with_zero(x)
                x = x.to(device)
                target = target.to(device)
                s = s.to(device)
                logits = model(x)
                ##print('我是x'+str(x))
                loss = criterion(logits, target)

                _, predicted = torch.max(logits, -1)
                correct = predicted.eq(target).sum()
                pred_list.extend(predicted.detach().cpu().tolist())
                metrics["test_correct"] += correct.item()
                metrics["test_loss"] += loss.item() * target.size(0)
                metrics["test_total"] += target.size(0)
       
        target_list = np.array(target_list)
        s_list = np.array(s_list)
        x_list = np.array(x_list)
        pred_list = np.array(pred_list)
        pred_acc = pred_list == target_list
        ppr_list = []
        tnr_list = []
        tpr_list = []
        converted_s = s_list[:, 1]  # sex, 1 attribute
        
        for s_value in np.unique(converted_s):
            if np.mean(converted_s == s_value) > 0.01:
                indexs0 = np.logical_and(target_list == 0, converted_s == s_value)
                indexs1 = np.logical_and(target_list == 1, converted_s == s_value)
                ppr_list.append(np.mean(pred_list[converted_s == s_value]))
                tnr_list.append(np.mean(pred_acc[indexs0]))
                tpr_list.append(np.mean(pred_acc[indexs1]))
        ##print('啦啦啦我是converted_s'+str(converted_s))
        ##print('啦啦啦我是pred_list'+str(pred_list))
        ##print('啦啦啦我是index0'+str(indexs0))
        eo_gap = max(max(tnr_list) - min(tnr_list), max(tpr_list) - min(tpr_list))
        dp_gap = max(ppr_list) - min(ppr_list)

        metrics["eo_gap"] = eo_gap
        metrics["dp_gap"] = dp_gap
        ##print('啦啦啦我是test正确的个数'+str(metrics["test_correct"]))
        return metrics
