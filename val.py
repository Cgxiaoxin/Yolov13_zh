from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('model/yolov13s.pt')  # 使用你的预训练权重

# 在验证集上评估模型
results = model.val(
    data='datasets/coco/coco.yaml',  # 修正数据集配置文件路径
    imgsz=640,
    batch=16,
    device="0",  # 使用GPU 0，或者改为"cpu"使用CPU
    save_json=True,  # 保存JSON格式的结果
    save_txt=True,   # 保存TXT格式的结果
    project='runs/val',  # 保存验证结果的目录
    name='yolov13_validation',  # 验证实验名称
)

print("验证完成！")
print(f"验证结果保存在: runs/val/yolov13_validation/")