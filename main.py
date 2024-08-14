import fedml
import torch
import torch.nn as nn
from data_loader import load_partition_data_census
from fedml.simulation import SimulatorSingleProcess as Simulator
from standard_trainer import StandardTrainer
import pathlib
import os
import time
import copy
import yaml
from model import TwoNN
from data_synthesizer import DataSynthesizer
import argparse

census_input_shape_dict = {"income": 54, "health": 154, "employment": 109}

def load_data(args):
    fedml.logging.info("load_data. dataset_name = %s" % args.dataset)
    if args.cluster_num == 0:
        args.users = [i for i in range(51)]
        (
            client_num,
            _,
            train_data_num,
            test_data_num,
            train_data_global,
            test_data_global,
            val_data_global,
            train_data_local_num_dict,
            test_data_local_num_dict,
            train_data_local_dict,
            test_data_local_dict,
            val_data_local_dict,
            class_num,
            unselected_data_local_dict,
        ) = load_partition_data_census(args.users, args)

    args.client_num_in_total = client_num
    dataset = [
        train_data_num,
        test_data_num,
        train_data_global,
        test_data_global,
        train_data_local_num_dict,
        train_data_local_dict,
        test_data_local_dict,
        val_data_local_dict,
        class_num,
    ]
    return dataset, class_num
def add_custom_arguments(parser):
    parser.add_argument('--aggregation_method', type=str, default='fedavg',
                        choices=['fedavg', 'median', 'trimmed_mean'],
                        help='Aggregation method to use')


def main():
    # 创建自定义参数解析器
    custom_parser = argparse.ArgumentParser(add_help=False)
    add_custom_arguments(custom_parser)
    
    # 先解析自定义参数
    custom_args, _ = custom_parser.parse_known_args()

    # init FedML framework
    args = fedml.init()
    # 将自定义参数添加到 args 对象中
    args.aggregation_method = custom_args.aggregation_method


    args.run_folder = "results/{}/run_{}".format(args.task, args.random_seed)
    os.makedirs(args.data_cache_dir, exist_ok=True)
    pathlib.Path(args.run_folder).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    device = fedml.device.get_device(args)
    dataset, output_dim = load_data(args)
    print("load dataset time {}".format(time.time() - start_time))
    if args.model == "two-layer":
        """
        model = TwoNN(
            input_dim=args.input_dim,
            hidden_outdim=args.num_hidden,  # 注意这里使用 num_hidden
            output_dim=args.output_dim
        )
        """
        model = TwoNN(census_input_shape_dict[args.task], args.num_hidden, output_dim)
    trainer = StandardTrainer(model)
    print("load model time {}".format(time.time() - start_time))

    # 设置合成数据生成的轮次
   ##args.synthetic_data_generation_round = args.comm_round // 2

    simulator = Simulator(args, device, dataset, model, trainer)
    simulator.run()
    simulator.fl_trainer.save()
    print("finishing time {}".format(time.time() - start_time))
    torch.save(
        simulator.fl_trainer.model_trainer.model.state_dict(),
        os.path.join(args.run_folder, "%s.pt" % (args.save_model_name)),
    )

if __name__ == "__main__":
    main()
