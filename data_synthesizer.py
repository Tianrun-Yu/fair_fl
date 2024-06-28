import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model import TwoNN

class DataSynthesizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TwoNN(args.input_dim, args.num_hidden, args.output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.inner_lr = args.lr

    def synthesize(self, expert_trajectory, n_iterations=1000):
        # Initialize synthetic data
        synthetic_data = torch.randn(self.args.synthetic_data_size, self.args.input_dim).to(self.device)
        synthetic_labels = torch.randint(0, self.args.output_dim, (self.args.synthetic_data_size,)).to(self.device)
        
        synthetic_data.requires_grad = True
        synthetic_labels.requires_grad = True

        optimizer = optim.Adam([synthetic_data, synthetic_labels], lr=0.01)

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            loss = self.compute_matching_loss(expert_trajectory, synthetic_data, synthetic_labels)
            loss.backward()
            optimizer.step()

            # Project labels back to valid range
            with torch.no_grad():
                synthetic_labels.clamp_(0, self.args.output_dim - 1)

        return synthetic_data.detach(), synthetic_labels.detach().long()

    def compute_matching_loss(self, expert_trajectory, synthetic_data, synthetic_labels):
        synthetic_trajectory = self.get_synthetic_trajectory(expert_trajectory[0], synthetic_data, synthetic_labels)
        loss = sum(self.distance(w_syn, w_exp) for w_syn, w_exp in zip(synthetic_trajectory, expert_trajectory))
        return loss

    def get_synthetic_trajectory(self, init_params, synthetic_data, synthetic_labels):
        synthetic_trajectory = []
        w = init_params.clone().detach()
        for _ in range(len(expert_trajectory) - 1):
            w = self.update(w, synthetic_data, synthetic_labels)
            synthetic_trajectory.append(w)
        return synthetic_trajectory

    def update(self, w, x, y):
        self.net.load_state_dict(w)
        logits = self.net(x)
        loss = self.criterion(logits, y)
        grad = torch.autograd.grad(loss, self.net.parameters(), create_graph=True)
        return OrderedDict({k: w[k] - self.inner_lr * g for k, g in zip(w.keys(), grad)})

    @staticmethod
    def distance(w1, w2):
        return sum((p1 - p2).norm(2) for p1, p2 in zip(w1.values(), w2.values()))
