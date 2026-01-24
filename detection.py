import cv2
import time
import threading
import numpy as np
from PIL import Image
from ultralytics import YOLO
from transformers import pipeline
import pyttsx3
engine = pyttsx3.init()
previous_status = "SAFE"

# --- 1. UTILITY FUNCTIONS ---
def getCoords(image):
    h, w = image.shape[:2]
    
    target_w_ratio = 0.15
    target_h_ratio = 0.55
    
    box_width = int(w * target_w_ratio)
    box_height = int(h * target_h_ratio)
    
    start_x = (w - box_width) // 2
    start_y = (h - box_height) // 2

    end_x = start_x + box_width
    end_y = start_y + box_height
    
    return start_x, start_y, end_x, end_y

# --- 2. SETUP ---
pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device="cpu", use_fast=True)
model = YOLO("yolo11n.pt")

CHECK_INTERVAL = 0.25
DEPTH_THRESHOLD = 170

cap = cv2.VideoCapture(0)

# State Variables
current_frame = None       
latest_heatmap = None      
program_running = True     
current_status = "SAFE"
box_color = (0, 255, 0) 
previous_status = "SAFE"

# --- 3. THE BRAIN (THREAD) ---
def depth_thread():
    global current_frame, latest_heatmap, program_running, current_status, box_color
    
    while program_running:
        if current_frame is None:
            time.sleep(0.1)
            continue
        
        input_image = current_frame.copy()
        pil_image = Image.fromarray(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB))
        result = pipe(pil_image)
        depth_data = np.array(result["depth"])
        
        sx, sy, ex, ey = getCoords(depth_data)
        danger_zone = depth_data[sy:ey, sx:ex]

        close_pixels = np.sum(danger_zone > DEPTH_THRESHOLD)
        percent_blocked = close_pixels / danger_zone.size

        if percent_blocked > 0.40:
            current_status = "STOP!"
            box_color = (0, 0, 255)
        
        else:
            current_status = "SAFE"
            box_color = (0, 255, 0)
        
        depth_display = cv2.resize(depth_data, (input_image.shape[1], input_image.shape[0]))
        latest_heatmap = cv2.applyColorMap(depth_display, cv2.COLORMAP_INFERNO)
        
        time.sleep(CHECK_INTERVAL)

thread = threading.Thread(target=depth_thread, daemon=True)
thread.start()

# --- 4. THE EYES (MAIN LOOP) ---
print("System Ready.")
if __name__ == "__main__":
    print("System Ready.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        current_frame = frame
        if current_status == "STOP!" and previous_status == "SAFE":
            print("Speaking: caution")
            engine.say("caution")
            engine.runAndWait()

        previous_status = current_status
        results = model.predict(source=frame, verbose=False, conf=0.5)
        frame = results[0].plot()

        # DRAW ON MAIN FRAME
        sx, sy, ex, ey = getCoords(frame)
        cv2.rectangle(frame, (sx, sy), (ex, ey), box_color, 2)
        cv2.putText(frame, current_status, (sx, sy-10), cv2.FONT_HERSHEY_SIMPLEX, 1, box_color, 2)
        
        # DRAW ON THERMAL OVERLAY
        if latest_heatmap is not None:
            
            h, w, _ = frame.shape
            
            scale = 0.35
            aspect_ratio = 4/3
            
            hm_width = int(w * scale)
            hm_height = int(hm_width / aspect_ratio)
            
            small_thermal = cv2.resize(latest_heatmap, (hm_width, hm_height))
            
            frame[h-hm_height:h, 0:hm_width] = small_thermal
            cv2.rectangle(frame, (0, h-hm_height), (hm_width, h), (255, 255, 255), 2)
            
        cv2.imshow("The Third Eye", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            program_running = False
            break

