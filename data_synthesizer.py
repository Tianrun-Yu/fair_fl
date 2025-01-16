import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from model import TwoNN

class DataSynthesizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TwoNN(args.input_dim, args.num_hidden, args.output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.inner_lr = args.learning_rate

    def synthesize(self, global_trajectory, n_iterations=None):
        if n_iterations is None:
            n_iterations = self.args.n_iterations
        
        L = len(global_trajectory)
        s = self.args.local_ep

        # Initialize synthetic data
        synthetic_data = torch.randn(self.args.synthetic_data_size, self.args.input_dim).to(self.device)
        synthetic_labels = F.one_hot(torch.randint(0, self.args.output_dim, (self.args.synthetic_data_size,)), 
                                     num_classes=self.args.output_dim).float().to(self.device)
        synthetic_sensitive_attr = torch.randint(0, 2, (self.args.synthetic_data_size,)).to(self.device)
        
        synthetic_data.requires_grad = True
        synthetic_labels.requires_grad = True

        optimizer = optim.Adam([synthetic_data, synthetic_labels], lr=self.args.learning_rate)

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Sample t ~ U(1, L-s)
            t = torch.randint(1, L-s, (1,)).item()
            wt = self.to_device(global_trajectory[t])
            wt_s = self.to_device(global_trajectory[t+s])
            
            # Get trained parameters
            w_tilde = self.get_trained_params(synthetic_data, synthetic_labels.argmax(dim=1), wt, s)
            
            # Calculate distance and gradients
            loss = self.distance(w_tilde, wt_s)
            loss.backward()
            
            # Implement gradient clipping
            torch.nn.utils.clip_grad_norm_([synthetic_data, synthetic_labels], max_norm=1.0)

            optimizer.step()

            # Check for NaN values
            #if torch.isnan(synthetic_data).any() or torch.isnan(synthetic_labels).any():
                ##print("NaN values detected in synthetic_data or synthetic_labels.")
                # Add additional debugging/logging here

            # Project labels back to valid range
            with torch.no_grad():
                synthetic_labels = F.softmax(synthetic_labels, dim=1)

        # Convert one-hot labels back to class indices
        synthetic_labels = synthetic_labels.argmax(dim=1)
        ##print('啦啦啦我是记录的全局模型'+str(global_trajectory))

        return synthetic_data.detach().cpu(), synthetic_labels.detach().long().cpu(), synthetic_sensitive_attr.cpu()

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

    def to_device(self, w):
        return OrderedDict({k: v.to(self.device) for k, v in w.items()})

# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from collections import OrderedDict
# from model import TwoNN
# import random

# def replace_nan_with_zero(x):
#         if torch.isnan(x).any():
#             x[torch.isnan(x)] = 0
#         return x

# class DataSynthesizer:
#     def __init__(self, args):
#         self.args = args
#         self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.net = TwoNN(args.input_dim, args.num_hidden, args.output_dim).to(self.device)
#         self.criterion = nn.CrossEntropyLoss()
#         self.inner_lr = args.learning_rate

#     def synthesize(self, global_dataset, n_iterations=None):
#         if n_iterations is None:
#             n_iterations = self.args.n_iterations
    
#         # Sample from real dataset
#         sample_size = min(self.args.synthetic_data_size, len(global_dataset))
#         sampled_indices = random.sample(range(len(global_dataset)), sample_size)
    
#         synthetic_data = []
#         synthetic_labels = []
#         synthetic_sensitive_attr = []
    
#         for idx in sampled_indices:
#             x, y, s = global_dataset[idx]
#             synthetic_data.append(x)
#             synthetic_labels.append(y)
#             synthetic_sensitive_attr.append(s)
    
#         synthetic_data = torch.stack(synthetic_data).to(self.device)
#         synthetic_labels = torch.tensor(synthetic_labels).to(self.device)
#         synthetic_sensitive_attr = torch.stack(synthetic_sensitive_attr).to(self.device)

#         # Ensure correct data types
#         synthetic_data = synthetic_data.float()
#         synthetic_labels = synthetic_labels.long()
#         synthetic_sensitive_attr = synthetic_sensitive_attr.long()

#         # Replace NaN with 0 in synthetic_data
#         synthetic_data = replace_nan_with_zero(synthetic_data)

#         # If sensitive_attr is 2D, take the first column
#         if synthetic_sensitive_attr.dim() > 1:
#             synthetic_sensitive_attr = synthetic_sensitive_attr[:, 1]

#         print(f"Sampled data shape: {synthetic_data.shape}")
#         print(f"Sampled labels shape: {synthetic_labels.shape}")
#         print(f"Sampled sensitive attributes shape: {synthetic_sensitive_attr.shape}")
#         print(f"Sampled sensitive attributes unique values: {torch.unique(synthetic_sensitive_attr)}")
    
#         # 添加更多详细的输出
#         print("\nDetailed sample information:")
#         print(f"First 5 samples of data:\n{synthetic_data[:5]}")
#         print(f"\nFirst 20 labels: {synthetic_labels[:20]}")
#         print(f"\nFirst 20 sensitive attributes: {synthetic_sensitive_attr[:20]}")
#         print(f"\nLabel distribution: {torch.bincount(synthetic_labels)}")
#         print(f"\nSensitive attribute distribution: {torch.bincount(synthetic_sensitive_attr)}")
    
#         # Check for remaining NaN values
#         print(f"\nNaN values in synthetic_data: {torch.isnan(synthetic_data).sum()}")
#         print(f"NaN values in synthetic_labels: {torch.isnan(synthetic_labels).sum()}")
#         print(f"NaN values in synthetic_sensitive_attr: {torch.isnan(synthetic_sensitive_attr).sum()}")

#         return synthetic_data.cpu(), synthetic_labels.cpu(), synthetic_sensitive_attr.cpu()

#     def get_trained_params(self, x, y, w_init, steps):
#         w = OrderedDict({k: v.clone().detach().requires_grad_(True) for k, v in w_init.items()})
#         optimizer = optim.SGD(w.values(), lr=self.inner_lr)
        
#         for _ in range(steps):
#             optimizer.zero_grad()
#             self.net.load_state_dict(w)
#             output = self.net(x)
#             loss = self.criterion(output, y)
#             loss.backward()
#             optimizer.step()
#             w = OrderedDict({k: v.clone().detach().requires_grad_(True) for k, v in self.net.state_dict().items()})
        
#         return w

#     @staticmethod
#     def distance(w1, w2):
#         return sum((p1 - p2).norm(2) for p1, p2 in zip(w1.values(), w2.values()))

#     def to_device(self, w):
#         return OrderedDict({k: v.to(self.device) for k, v in w.items()})


"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import OrderedDict
from model import TwoNN

class DataSynthesizer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = TwoNN(args.input_dim, args.num_hidden, args.output_dim).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.inner_lr = args.learning_rate

    def synthesize(self, global_trajectory, n_iterations=None):
        if n_iterations is None:
            n_iterations = self.args.n_iterations
        
        L = len(global_trajectory)
        s = self.args.local_ep

        # Initialize synthetic data
        synthetic_data = torch.randn(self.args.synthetic_data_size, self.args.input_dim).to(self.device)
        synthetic_labels = F.one_hot(torch.randint(0, self.args.output_dim, (self.args.synthetic_data_size,)), 
                                     num_classes=self.args.output_dim).float().to(self.device)
        synthetic_sensitive_attr = torch.randint(0, 2, (self.args.synthetic_data_size,)).to(self.device)
        
        synthetic_data.requires_grad = True
        synthetic_labels.requires_grad = True

        optimizer = optim.Adam([synthetic_data, synthetic_labels], lr=self.args.learning_rate)

        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            # Sample t ~ U(1, L-s)
            t = torch.randint(1, L-s, (1,)).item()
            wt = self.to_device(global_trajectory[t])
            wt_s = self.to_device(global_trajectory[t+s])
            
            # Get trained parameters
            w_tilde = self.get_trained_params(synthetic_data, synthetic_labels.argmax(dim=1), wt, s)
            
            # Calculate distance and gradients
            loss = self.distance(w_tilde, wt_s)
            loss.backward()
            
            optimizer.step()

            # Project labels back to valid range
            with torch.no_grad():
                synthetic_labels = F.softmax(synthetic_labels, dim=1)

        # Convert one-hot labels back to class indices
        synthetic_labels = synthetic_labels.argmax(dim=1)

        return synthetic_data.detach().cpu(), synthetic_labels.detach().long().cpu(), synthetic_sensitive_attr.cpu()

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

    def to_device(self, w):
        return OrderedDict({k: v.to(self.device) for k, v in w.items()}) 
"""
