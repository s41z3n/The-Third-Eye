from ultralytics import YOLO
import cv2
import pyttsx3


model = YOLO("yolo11n-seg.pt")

results = model.predict(source=0,show=True)

print(results)