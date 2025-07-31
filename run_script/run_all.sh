#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate yolo13

# 进入项目目录
cd /workspace/yolov13

echo "=== YOLOv13 训练和验证脚本 ==="
echo "1. 训练模型"
echo "2. 验证模型"
echo "3. 预测图像"
echo "4. 导出模型"
echo "5. 运行所有步骤"
echo "请选择要执行的操作 (1-5):"

read choice

case $choice in
    1)
        echo "开始训练..."
        python train.py
        ;;
    2)
        echo "开始验证..."
        python val.py
        ;;
    3)
        echo "开始预测..."
        python predict.py
        ;;
    4)
        echo "开始导出..."
        python export.py
        ;;
    5)
        echo "运行所有步骤..."
        echo "1. 训练模型..."
        python train.py
        echo "2. 验证模型..."
        python val.py
        echo "3. 导出模型..."
        python export.py
        ;;
    *)
        echo "无效选择，退出"
        exit 1
        ;;
esac

echo "操作完成！" 