from ultralytics import YOLO

# Load a model
# model = YOLO("yolo11n-pose.yaml")  # build a new model from YAML
model = YOLO("yolo11n-pose.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="/home/sagor/Projects/YOLO/src/Foot-Landmark-Detection-1/data.yaml", epochs=100, imgsz=320)