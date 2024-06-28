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



def main():
    # init FedML framework
    args = fedml.init()
    args.run_folder = "results/{}/run_{}".format(args.task, args.random_seed)
    os.makedirs(args.data_cache_dir, exist_ok=True)
    pathlib.Path(args.run_folder).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    device = fedml.device.get_device(args)
    dataset, output_dim = load_data(args)
    print("load dataset time {}".format(time.time() - start_time))
    if args.model == "two-layer":
        model = TwoNN(
            input_dim=args.input_dim,
            hidden_outdim=args.num_hidden,  # 注意这里使用 num_hidden
            output_dim=args.output_dim
        )
    trainer = StandardTrainer(model)
    print("load model time {}".format(time.time() - start_time))

    # 设置合成数据生成的轮次
    args.synthetic_data_generation_round = args.comm_round // 2

    simulator = Simulator(args, device, dataset, model, trainer)
    w_global = simulator.fl_trainer.model_trainer.get_model_params()

    # 存储全局模型轨迹
    global_model_trajectory = []

    for round_idx in range(args.comm_round):
        w_global = simulator.fl_trainer.train_one_round(round_idx, w_global)

        # 存储全局模型参数
        global_model_trajectory.append(copy.deepcopy(simulator.fl_trainer.model_trainer.model.state_dict()))

        # 在指定轮次生成合成数据和公平梯度
        if round_idx == args.synthetic_data_generation_round:
            # 生成合成数据
            synthesizer = DataSynthesizer(args)
            synthetic_data, synthetic_labels, synthetic_sensitive_attr = synthesizer.synthesize(
                global_model_trajectory,
                n_iterations=args.n_iterations
            )

            # 生成公平梯度
            fair_gradient = simulator.fl_trainer.generate_fair_gradient(
                simulator.fl_trainer.model_trainer.model,
                (synthetic_data, synthetic_labels, synthetic_sensitive_attr),
                learning_rate=args.learning_rate,
                num_iterations=args.num_iterations
            )

            # 将公平梯度添加到 FedAvgAPI 中
            simulator.fl_trainer.fair_gradient = fair_gradient

        # 从生成公平梯度后的轮次开始使用
        if round_idx > args.synthetic_data_generation_round:
            # 在 FedAvgAPI 中，公平梯度会自动被添加到聚合过程中
            pass

    simulator.fl_trainer.save()
    print("finishing time {}".format(time.time() - start_time))
    torch.save(
        simulator.fl_trainer.model_trainer.model.state_dict(),
        os.path.join(args.run_folder, "%s.pt" % (args.save_model_name)),
    )

if __name__ == "__main__":
    main()
