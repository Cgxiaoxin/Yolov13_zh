#!/bin/bash

# 激活conda环境
source ~/.bashrc
conda activate yolo13

# 进入项目目录
cd /workspace/yolov13

# 运行验证
echo "开始验证 YOLOv13..."
python val.py

echo "验证完成！" 