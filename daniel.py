import cv2
import pyttsx3
from ultralytics import YOLO
import time
import detection

engine = pyttsx3.init()

while True:
    speak = detection.text()
    
    if speak:  # Only speak if objects detected
        print(speak)
        engine.say("caution")
        engine.runAndWait()
            
    
    
    