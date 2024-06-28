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

from model import TwoNN
from image_synthesizer import ImageSynthesizer

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

def update_with_synthetic_data(model, Dsyn, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(args.local_ep):
        for batch_idx, (data, labels) in enumerate(Dsyn):
            data, labels = data.to(args.device), labels.to(args.device)
            optimizer.zero_grad()
            log_probs = model(data)
            loss = criterion(log_probs, labels)
            loss.backward()
            optimizer.step()

def main():
    # init FedML framework
    args = fedml.init()
    args.run_folder = "results/{}/run_{}".format(args.task, args.random_seed)
    pathlib.Path(args.run_folder).mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    device = fedml.device.get_device(args)
    dataset, output_dim = load_data(args)
    print("load dataset time {}".format(time.time() - start_time))
    if args.model == "two-layer":
        model = TwoNN(census_input_shape_dict[args.task], args.num_hidden, output_dim)
    trainer = StandardTrainer(model)
    print("load model time {}".format(time.time() - start_time))

    simulator = Simulator(args, device, dataset, model, trainer)

    # 添加全局模型轨迹的存储
    global_model_trajectory = []
    Dsyn = None

    for round_idx in range(args.comm_round):
        simulator.run_one_round()

        # 存储全局模型参数
        global_model_trajectory.append(copy.deepcopy(simulator.fl_trainer.model_trainer.model.state_dict()))

        # 在一半的轮次时生成伪数据
        if round_idx == args.comm_round // 2:
            synthesizer = ImageSynthesizer(args)
            Dsyn = synthesizer.synthesize(global_model_trajectory)

        # 从生成伪数据后的下一轮开始使用
        if round_idx > args.comm_round // 2 and Dsyn is not None:
            # 使用伪数据更新模型
            update_with_synthetic_data(simulator.fl_trainer.model_trainer.model, Dsyn, args)

    simulator.fl_trainer.save()
    print("finishing time {}".format(time.time() - start_time))
    torch.save(
        simulator.fl_trainer.model_trainer.model.state_dict(),
        os.path.join(args.run_folder, "%s.pt" % (args.save_model_name)),
    )

if __name__ == "__main__":
    main()
