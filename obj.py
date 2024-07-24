import os
from ultralytics import YOLO

def train_yolo(model_arch, data_path, epochs=25, img_size=640, batch_size=8):
    """Train the YOLO model."""
    model = YOLO(model_arch)
    model.train(data=data_path, epochs=epochs, imgsz=img_size, batch=batch_size, plots=True)

def validate_yolo(model_path, data_path):
    """Validate the YOLO model."""
    model = YOLO(model_path)
    model.val(data=data_path)

if __name__ == "__main__":
    # Define paths
    model_arch = 'yolov8s.yaml'  # Path to the YOLO model architecture file
    data_path = '/workspaces/Drone_project---Copy/mango_project/data.yaml'  # Path to your data.yaml file

    # Train the YOLO model
    train_yolo(model_arch, data_path, epochs=5, img_size=640)

    # Validate the YOLO model
    validate_yolo('/workspaces/Drone_project---Copy/runs/detect/train2/weights/best.pt', data_path)