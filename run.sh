#!/bin/bash

# 检查是否提供了足够的参数
if [ $# -ne 2 ]; then
    echo "Usage: $0 <f|c|s> <income|health|employment>"
    echo "  First parameter:"
    echo "    f: Run FedAvg config"
    echo "    c: Run Centralized config"
    echo "    s: Run Standalone config"
    echo "  Second parameter:"
    echo "    income: Use Income dataset"
    echo "    health: Use Health dataset"
    echo "    employment: Use Employment dataset"
    exit 1
fi

# 设置模式和数据集
mode=$1
dataset=$2

# 验证数据集参数
if [[ ! "$dataset" =~ ^(income|health|employment)$ ]]; then
    echo "Invalid dataset. Please use 'income', 'health', or 'employment'."
    exit 1
fi

# 根据输入参数运行相应的命令
case $mode in
    f)
        echo "Running FedAvg $dataset config..."
        python main.py --cf config/config_fedavg_${dataset}.yaml
        ;;
    c)
        echo "Running Centralized $dataset config..."
        python main.py --cf config/config_centralized_${dataset}.yaml
        ;;
    s)
        echo "Running Standalone $dataset config..."
        python main.py --cf config/config_standalone_${dataset}.yaml
        ;;
    *)
        echo "Error: Invalid mode"
        echo "Usage: $0 <f|c|s> <income|health|employment>"
        echo "  First parameter:"
        echo "    f: Run FedAvg config"
        echo "    c: Run Centralized config"
        echo "    s: Run Standalone config"
        echo "  Second parameter:"
        echo "    income: Use Income dataset"
        echo "    health: Use Health dataset"
        echo "    employment: Use Employment dataset"
        exit 1
        ;;
esac