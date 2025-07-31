from ultralytics import YOLO

# 加载预训练模型
model = YOLO('model/yolov13n.pt')  # 使用你的预训练权重

# 训练模型
results = model.train(
    data='datasets/coco/coco.yaml',  # 修正数据集配置文件路径
    epochs=600, 
    batch=256, 
    imgsz=640,
    scale=0.5,  # S:0.9; L:0.9; X:0.9
    mosaic=1.0,
    mixup=0.0,  # S:0.05; L:0.15; X:0.2
    copy_paste=0.1,  # S:0.15; L:0.5; X:0.6
    device="0",
    project='runs/train',  # 保存训练结果的目录
    name='yolov13_coco',  # 实验名称
    save=True,  # 保存模型
    save_period=50,  # 每50个epoch保存一次
)

# 在验证集上评估模型性能
metrics = model.val(data='datasets/coco/coco.yaml')

# 对图像进行目标检测（示例）
# results = model("path/to/your/image.jpg")
# results[0].show()

print("训练完成！")
print(f"模型保存在: {model.ckpt_path}")
