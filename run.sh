#!/bin/bash

# 检查是否提供了参数
if [ $# -eq 0 ]; then
    echo "Using: $0 <f|c|s>"
    echo "  f: Run FedAvg Income config"
    echo "  c: Run Centralized Income config"
    echo "  s: Run Standalone Income config"
    exit 1
fi

# 根据输入参数运行相应的命令
case $1 in
    f)
        echo "Run FedAvg Income config..."
        python main.py --cf config/config_fedavg_income.yaml
        ;;
    c)
        echo "Run Centralized Income config..."
        python main.py --cf config/config_centralized_income.yaml
        ;;
    s)
        echo "Run Standalone Income config..."
        python main.py --cf config/config_standalone_income.yaml
        ;;
    *)
        echo "Error"
        echo "Using: $0 <f|c|s>"
        echo "  f: Run FedAvg Income config"
        echo "  c: Run Centralized Income config"
        echo "  s: Run Standalone Income config"
        exit 1
        ;;
esac