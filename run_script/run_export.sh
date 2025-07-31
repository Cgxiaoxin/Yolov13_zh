#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate yolo13

# 进入项目目录（从 run_script 目录回到项目根目录）
cd /workspace/yolov13

# 运行模型导出
echo "开始导出模型..."
python export.py

echo "导出完成！" 