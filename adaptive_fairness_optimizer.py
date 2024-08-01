import numpy as np
import torch

class AdaptiveFairnessOptimizer:
    def __init__(self, fairness_metrics=['eo', 'dp'], learning_rate=0.1, acc_threshold=0.05):
        self.fairness_metrics = fairness_metrics
        self.learning_rate = learning_rate
        self.acc_threshold = acc_threshold
        self.angle_history = {metric: [] for metric in fairness_metrics}
        self.optimization_weights = {metric: 1.0 for metric in fairness_metrics}
        self.metric_history = {metric: [] for metric in fairness_metrics + ['acc']}
        self.gradient_accumulation = {}
        self.accumulation_steps = 0
    def calculate_angle(self, grad1, grad2):
        dot_product = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad1.values(), grad2.values()))
        norm1 = sum(torch.norm(g)**2 for g in grad1.values()).sqrt()
        norm2 = sum(torch.norm(g)**2 for g in grad2.values()).sqrt()
        return torch.acos(dot_product / (norm1 * norm2 + 1e-8)) * 180 / np.pi

    def update_angle_history(self, gradients):
        acc_grad = gradients['acc']
        for metric in self.fairness_metrics:
            angle = self.calculate_angle(acc_grad, gradients[metric])
            self.angle_history[metric].append(angle.item())

    def calculate_optimization_weights(self, current_metrics):
        for metric in self.fairness_metrics:
            avg_angle = np.mean(self.angle_history[metric][-10:]) if self.angle_history[metric] else 90
            ease_of_optimization = 90 - avg_angle  # 角度越小，越容易优化
            metric_value = current_metrics[metric]
            self.optimization_weights[metric] = max(10.0, 1000.0 * metric_value)  # 增加权重

    def project_gradients(self, grad1, grad2):
        dot_product = sum(torch.sum(g1 * g2) for g1, g2 in zip(grad1.values(), grad2.values()))
        if dot_product < 0:
            proj_direction = {k: g - (dot_product / sum(torch.norm(g)**2 for g in grad1.values())) * grad1[k]
                              for k, g in grad2.items()}
            return proj_direction
        return grad2

    def optimize(self, gradients, current_metrics):
            self.update_angle_history(gradients)
            self.calculate_optimization_weights(current_metrics)

            final_gradient = {k: torch.zeros_like(v) for k, v in gradients['acc'].items()}
            
            # 初始化 gradient_accumulation
            if not self.gradient_accumulation:
                self.gradient_accumulation = {k: torch.zeros_like(v) for k, v in gradients['acc'].items()}

            # 添加acc梯度，但限制其影响
            acc_norm = sum(torch.norm(g)**2 for g in gradients['acc'].values()).sqrt()
            acc_scale = min(self.acc_threshold / acc_norm, 1.0)
            for k in final_gradient:
                final_gradient[k] += acc_scale * gradients['acc'][k]

            # 使用PCGrad方法添加公平性指标的梯度
            for metric in self.fairness_metrics:
                weight = self.optimization_weights[metric]
                metric_grad = self.project_gradients(gradients['acc'], gradients[metric])
                metric_norm = sum(torch.norm(g)**2 for g in metric_grad.values()).sqrt()
                metric_scale = weight * (1 - acc_scale) * 50  # 增加公平性梯度的影响
                for k in final_gradient:
                    final_gradient[k] += (metric_scale / metric_norm) * metric_grad[k]

            # 累积梯度
            self.accumulation_steps += 1
            for k in final_gradient:
                self.gradient_accumulation[k] += final_gradient[k]

            # 每4步应用一次累积的梯度
            if self.accumulation_steps % 4 == 0:
                for k in final_gradient:
                    final_gradient[k] = self.gradient_accumulation[k] / 4
                    self.gradient_accumulation[k].zero_()
                self.accumulation_steps = 0

                # 应用更激进的梯度裁剪
                grad_norm = sum(torch.norm(g)**2 for g in final_gradient.values()).sqrt()
                if grad_norm > 5.0:
                    for k in final_gradient:
                        final_gradient[k] *= 5.0 / grad_norm

            return final_gradient

    def print_gradient_info(self, gradients):
        for name, grad in gradients.items():
            norm = sum(torch.norm(g)**2 for g in grad.values()).sqrt()
            print(f"Gradient norm for {name}: {norm.item():.4f}")
        for metric1 in gradients:
            for metric2 in gradients:
                if metric1 != metric2:
                    angle = self.calculate_angle(gradients[metric1], gradients[metric2])
                    print(f"Angle between {metric1} and {metric2}: {angle.item():.2f} degrees")
    def get_learning_rate(self, current_metrics):
        eo_dp_sum = current_metrics['eo'] + current_metrics['dp']
        return self.learning_rate * (1 + 10 * eo_dp_sum)

