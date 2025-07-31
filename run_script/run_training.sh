#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate yolo13

# 进入项目目录（从 run_script 目录回到项目根目录）
cd /workspace/yolov13

# 运行训练
echo "开始训练 YOLOv13..."
python train.py

echo "训练完成！" 