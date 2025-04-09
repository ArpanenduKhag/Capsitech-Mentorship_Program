from ultralytics import YOLO
import cv2


def detect_objects(image_path):
    model = YOLO("yolov8n.pt")  # Use a smaller model for faster inference
    results = model(image_path)
    result_image = results[0].plot()  # Annotated image
    return result_image
