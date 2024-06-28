import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model import TwoNN

class DataSynthesizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TwoNN(args.model_args.input_dim, args.model_args.num_hidden, args.model_args.output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.inner_lr = args.synthetic_data_args.synthetic_data_lr

    def synthesize(self, global_trajectory, n_iterations=None):
        if n_iterations is None:
            n_iterations = self.args.synthetic_data_args.n_iterations
        
        L = len(global_trajectory)
        s = self.args.synthetic_data_args.inner_steps

        # Initialize synthetic data
        synthetic_data = torch.randn(self.args.synthetic_data_args.synthetic_data_size, self.args.model_args.input_dim).to(self.device)
        synthetic_labels = torch.randint(0, self.args.model_args.output_dim, (self.args.synthetic_data_args.synthetic_data_size,)).to(self.device)
        synthetic_sensitive_attr = torch.randint(0, 2, (self.args.synthetic_data_args.synthetic_data_size,)).to(self.device)
        
        synthetic_data.requires_grad = True
        synthetic_labels.requires_grad = True

        optimizer = optim.Adam([synthetic_data, synthetic_labels], lr=self.args.synthetic_data_args.synthetic_data_lr)

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Sample t ~ U(1, L-s)
            t = torch.randint(1, L-s, (1,)).item()
            wt = global_trajectory[t]
            wt_s = global_trajectory[t+s]
            
            # Get trained parameters
            w_tilde = self.get_trained_params(synthetic_data, synthetic_labels, wt, s)
            
            # Calculate distance and gradients
            loss = self.distance(w_tilde, wt_s)
            loss.backward()
            
            optimizer.step()

            # Project labels back to valid range
            with torch.no_grad():
                synthetic_labels.clamp_(0, self.args.model_args.output_dim - 1)

        return synthetic_data.detach(), synthetic_labels.detach().long(), synthetic_sensitive_attr

    def get_trained_params(self, x, y, w_init, steps):
        w = OrderedDict({k: v.clone().detach().requires_grad_(True) for k, v in w_init.items()})
        optimizer = optim.SGD(w.values(), lr=self.inner_lr)
        
        for _ in range(steps):
            optimizer.zero_grad()
            self.net.load_state_dict(w)
            output = self.net(x)
            loss = self.criterion(output, y)
            loss.backward()
            optimizer.step()
            w = OrderedDict({k: v.clone().detach().requires_grad_(True) for k, v in self.net.state_dict().items()})
        
        return w

    @staticmethod
    def distance(w1, w2):
        return sum((p1 - p2).norm(2) for p1, p2 in zip(w1.values(), w2.values()))
