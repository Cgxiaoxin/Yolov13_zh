from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('model/yolov13n.pt')  # 使用你的预训练权重，可以改为 yolov13s.pt 或 yolov13x.pt

# 对单张图像进行预测
results = model.predict(
    source='path/to/your/image.jpg',  # 替换为你的图像路径
    imgsz=640,
    conf=0.25,  # 置信度阈值
    iou=0.45,   # NMS IoU阈值
    device="0",  # 使用GPU 0，或者改为"cpu"使用CPU
    save=True,   # 保存预测结果
    project='runs/predict',  # 保存预测结果的目录
    name='yolov13_prediction',  # 预测实验名称
)

# 显示预测结果
for result in results:
    result.show()  # 显示图像
    result.save()  # 保存结果图像

print("预测完成！")
print(f"预测结果保存在: runs/predict/yolov13_prediction/")

# 批量预测示例
# results = model.predict(
#     source='path/to/your/images/folder/',  # 图像文件夹路径
#     imgsz=640,
#     conf=0.25,
#     save=True,
#     project='runs/predict',
#     name='batch_prediction',
# )