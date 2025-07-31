''' 将 YOLOv13 模型导出为 ONNX 或 TensorRT 格式。'''
from ultralytics import YOLO

# 加载训练好的模型
model = YOLO('model/yolov13n.pt')  # 使用你的预训练权重，可以改为 yolov13s.pt 或 yolov13x.pt

# 导出为 ONNX 格式
print("导出为 ONNX 格式...")
model.export(format="onnx", half=True, imgsz=640)

# 导出为 TensorRT 格式（需要 CUDA 支持）
print("导出为 TensorRT 格式...")
model.export(format="engine", half=True, imgsz=640)

# 导出为其他格式
# model.export(format="torchscript")  # PyTorch TorchScript
# model.export(format="coreml")       # CoreML
# model.export(format="tflite")       # TensorFlow Lite

print("导出完成！")
print("导出的文件保存在模型文件同目录下")