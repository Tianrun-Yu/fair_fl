

## Introduction

In our paper, we show that participating in federated learning can be detrimental to group fairness. In fact, the bias of a few biased parties against under-represented groups (identified by sensitive attributes such as gender or race) propagates through the network to all parties. On naturally partitioned real-world datasets, we analyze and explain bias propagation in federated learning. Our analysis reveals that biased parties unintentionally yet stealthily encode their bias in a small number of model parameters, and throughout the training, they steadily increase the dependence of the global model on sensitive attributes. What is important to highlight is that the experienced bias in federated learning is higher than what parties would otherwise encounter in centralized training with a model trained on the union of all their data. This indicates that the bias is due to the algorithm. Our work calls for auditing group fairness in federated learning, and designing learning algorithms that are robust to bias propagation.

## Dependencies

Our implementation of federated learning is based on the [FedML library](https://github.com/FedML-AI/FedML), and we use the machine learning tasks provided by [folktables table](https://github.com/socialfoundations/folktables).

We tested our code on `Python 3.8.13` and `cuda 11.4`. The essential environments are listed in the `environment.yml` file. Run the following command to create the conda environment:

```
conda env create -f environment.yml
```

### Usage

#### 1. Training the models for different settings.

To run federated learning on the Income dataset, use the command:

```
conda activate bias_propagation
python3 fairfed/main.py --cf fairfed/config/config_fedavg_income.yaml
```


```
conda activate bias_propagation
python3 fairfed/main.py --cf fairfed/config/config_fedavg_health.yaml

```

```
conda activate bias_propagation
python3 fairfed/main.py --cf fairfed/config/config_fedavg_employee.yaml

```



By default, the script collects the prediction information from 5 runs with random seeds 0 to

#### 3. Generate figures

To generate the figures in the paper, run the `plotting.ipynb` Jupyter notebook.
