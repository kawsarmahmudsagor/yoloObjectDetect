from ultralytics import YOLO

# Load the YOLO11 model
model = YOLO("/home/sagor/Projects/YOLO/runs/pose/train/weights/best.pt")

# Export the model to TFLite format
model.export(format="tflite", imgsz=320, nms=True)  # creates 'yolo11n_float32.tflite'



# Run inference
# results = tflite_model("https://ultralytics.com/images/bus.jpg")