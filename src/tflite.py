from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("best.pt file path")

# Export the model to TFLite format
model.export(format="tflite", imgsz=320, nms=True) 