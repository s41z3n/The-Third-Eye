import cv2
import pyttsx3
from ultralytics import YOLO
import time
import detection

engine = pyttsx3.init()
previous_status = "SAFE"
while True:
    current_status = detection.current_status
    
    if current_status == "STOP!" and previous_status == "SAFE":
        print("Obstacle detected - Speaking warning")
        engine.say("caution")
        engine.runAndWait()
    
   
    previous_status = current_status
    
    # Small delay to avoid overwhelming the system
    time.sleep(0.1)
    
    # Check if main program stopped
    if not detection.program_running:
        break
            
    
    
    