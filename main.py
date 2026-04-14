import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pyautogui
import tkinter as tk
from threading import Thread
import math
import time
import sys
import os

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# PyAutoGUI settings
pyautogui.FAILSAFE = False

# Initialize MediaPipe HandLandmarker Tasks API with High Precision Settings
base_options = python.BaseOptions(model_asset_path=resource_path('hand_landmarker.task'))
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.8, # Improved Accuracy
    min_hand_presence_confidence=0.8,  # Improved Accuracy
    running_mode=vision.RunningMode.IMAGE
)
detector = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]

def draw_landmarks_manual(img, landmarks):
    h, w, c = img.shape
    pixel_landmarks = []
    
    for landmark in landmarks:
        px, py = int(landmark.x * w), int(landmark.y * h)
        pixel_landmarks.append((px, py))
        cv2.circle(img, (px, py), 5, (0, 0, 255), -1) 
        
    for connection in HAND_CONNECTIONS:
        start_idx, end_idx = connection
        cv2.line(img, pixel_landmarks[start_idx], pixel_landmarks[end_idx], (0, 255, 0), 2)

screen_width, screen_height = pyautogui.size()
running = False

last_click_time = 0
last_media_time = 0
last_vol_time = 0
prev_scroll_y = 0
is_dragging = False

def get_fingers_up(landmarks):
    fingers = []
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    
    for tip, pip in zip(tips, pips):
        if landmarks[tip].y < landmarks[pip].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def gesture_loop():
    global running, last_click_time, last_media_time, last_vol_time, prev_scroll_y, is_dragging
    cap = cv2.VideoCapture(0)
    
    # Force higher resolution for better tracking precision
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Use exponential smoothing (alpha) for stable, responsive motion
    smooth_alpha = 0.32
    prev_x, prev_y = 0.0, 0.0
    last_move_time = 0
    move_interval = 0.01  # seconds: throttle mouse movement calls
    margin = 80 # Inner interaction boundary (makes it easy to reach screen edges without extending arm completely)
    
    while running:
        success, img = cap.read()
        if not success:
            continue
            
        img = cv2.flip(img, 1)
        h, w, c = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Draw interaction Box
        cv2.rectangle(img, (margin, margin), (w - margin, h - margin), (255, 0, 255), 2)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        detection_result = detector.detect(mp_image)
        
        # Reset continuous variables if no hands
        if not detection_result.hand_landmarks:
            prev_scroll_y = 0
            if is_dragging:
                pyautogui.mouseUp()
                is_dragging = False
            # reduce CPU when idle
            time.sleep(0.01)
        
        if detection_result.hand_landmarks:
            for hand_landmarks_raw in detection_result.hand_landmarks:
                draw_landmarks_manual(img, hand_landmarks_raw)
                
                landmarks = hand_landmarks_raw
                fingers_up = get_fingers_up(landmarks)
                
                thumb_tip = landmarks[4]
                thumb_mcp = landmarks[2]
                current_time = time.time()
                
                # Release Drag if fingers differ from drag posture
                if fingers_up != [1, 1, 0, 0] and fingers_up != [1, 0, 0, 0] and is_dragging:
                    pyautogui.mouseUp()
                    is_dragging = False

                # 1. 4 Fingers: Play/Pause or Next Track
                if sum(fingers_up) == 4:
                    if current_time - last_media_time > 1.5:
                        pyautogui.press('playpause')
                        last_media_time = current_time
                        cv2.putText(img, "Play/Pause", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        
                # 2. 0 Fingers (Fist or Thumb isolated)
                elif sum(fingers_up) == 0:
                    if thumb_tip.y < thumb_mcp.y - 0.05: # Thumb up
                        if current_time - last_vol_time > 0.1:
                            pyautogui.press('volumeup')
                            last_vol_time = current_time
                            cv2.putText(img, "Volume +", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                    elif thumb_tip.y > thumb_mcp.y + 0.05: # Thumb down
                        if current_time - last_vol_time > 0.1:
                            pyautogui.press('volumedown')
                            last_vol_time = current_time
                            cv2.putText(img, "Volume -", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                  #  else: # Solid Fist (Desktop)
                  #      if current_time - last_click_time > 2.0:
                   #         pyautogui.hotkey('win', 'd')
                    #        last_click_time = current_time
                    #       cv2.putText(img, "Show Desktop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            
                # 3. 3 Fingers (Index, Middle, Ring): Right Click OR Double Click based on distance
                elif fingers_up == [1, 1, 1, 0]:
                    # Let's say spreading 3 fingers = Double Click; joining them = Right Click
                    dist_ir = math.hypot(landmarks[8].x - landmarks[16].x, landmarks[8].y - landmarks[16].y)
                    if current_time - last_click_time > 1.0:
                        if dist_ir < 0.1: # Joined
                            pyautogui.rightClick()
                            cv2.putText(img, "Right Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        else: # Spread out
                            pyautogui.doubleClick()
                            cv2.putText(img, "Double Click", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        last_click_time = current_time
                        
                # 4. 2 Fingers (Index & Middle): Scroll or Click/Drag
                elif fingers_up == [1, 1, 0, 0]:
                    index_tip = landmarks[8]
                    middle_tip = landmarks[12]
                    dist = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y)
                    
                    if dist < 0.05: # Pinch -> Drag
                        if not is_dragging:
                            pyautogui.mouseDown()
                            is_dragging = True
                        
                        # High accuracy interpolation based mapping (use full frame for mapping)
                        x_mapped = np.interp(index_tip.x * w, (0, w), (0, screen_width))
                        y_mapped = np.interp(index_tip.y * h, (0, h), (0, screen_height))
                        # clamp coordinates
                        x_mapped = max(0, min(screen_width - 1, x_mapped))
                        y_mapped = max(0, min(screen_height - 1, y_mapped))
                        # exponential smoothing
                        curr_x = prev_x + smooth_alpha * (x_mapped - prev_x)
                        curr_y = prev_y + smooth_alpha * (y_mapped - prev_y)
                        # throttle mouse moves
                        if time.time() - last_move_time >= move_interval:
                            pyautogui.moveTo(curr_x, curr_y)
                            last_move_time = time.time()
                        prev_x, prev_y = curr_x, curr_y
                        
                        cv2.putText(img, "Holding -> Dragging", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    else: # Open V shape -> Scroll
                        if is_dragging:
                            pyautogui.mouseUp()
                            is_dragging = False
                            last_click_time = current_time # Simple click fallback when released quickly
                            
                        if prev_scroll_y == 0:
                            prev_scroll_y = index_tip.y
                        
                        y_diff = index_tip.y - prev_scroll_y
                        if y_diff > 0.012: # Moved down -> Scroll Down
                            pyautogui.scroll(-120)
                            prev_scroll_y = index_tip.y
                            cv2.putText(img, "Scroll Down", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        elif y_diff < -0.012: # Moved up -> Scroll Up
                            pyautogui.scroll(120)
                            prev_scroll_y = index_tip.y
                            cv2.putText(img, "Scroll Up", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                            
                # 5. Index and Pinky (Spider-Man web sign): Task View
                elif fingers_up == [1, 0, 0, 1] and thumb_tip.y > thumb_mcp.y: # Thumb folded
                    if current_time - last_click_time > 1.5:
                        pyautogui.hotkey('win', 'tab')
                        last_click_time = current_time
                        cv2.putText(img, "Task View", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)
                        
                # 6. Pinky only / Shaka (Thumb+Pinky)
                elif fingers_up == [0, 0, 0, 1]:
                    if thumb_tip.x < thumb_mcp.x - 0.05 or thumb_tip.x > thumb_mcp.x + 0.05: # Thumb is out (Shaka)
                        if current_time - last_media_time > 1.5:
                            pyautogui.press('nexttrack')
                            last_media_time = current_time
                            cv2.putText(img, "Next Track", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                    else: # Pinky only (Screenshot)
                        if current_time - last_click_time > 2.0:
                            pyautogui.hotkey('win', 'prtsc')
                            last_click_time = current_time
                            cv2.putText(img, "Screenshot", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
                        
                # 7. Only Index UP: Move Mouse Precision Mode
                elif fingers_up == [1, 0, 0, 0]:
                    index_tip = landmarks[8]
                    
                    # High precision mapping using full webcam frame for mapping
                    x_mapped = np.interp(index_tip.x * w, (0, w), (0, screen_width))
                    y_mapped = np.interp(index_tip.y * h, (0, h), (0, screen_height))
                    
                    curr_x = prev_x + smooth_alpha * (x_mapped - prev_x)
                    curr_y = prev_y + smooth_alpha * (y_mapped - prev_y)
                    if time.time() - last_move_time >= move_interval:
                        pyautogui.moveTo(curr_x, curr_y)
                        last_move_time = time.time()
                    prev_x, prev_y = curr_x, curr_y
                    
                    cv2.putText(img, "Precision Move", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 2)

        cv2.imshow("Webcam Feed - Hand Gesture App", img)
        if cv2.waitKey(1) & 0xFF == 27:
            break
            
    cap.release()
    cv2.destroyAllWindows()
    running = False

def start_app():
    global running
    if not running:
        running = True
        t = Thread(target=gesture_loop)
        t.start()
        status_label.config(text="Status: Active", fg="green")

def stop_app():
    global running
    running = False
    status_label.config(text="Status: Inactive", fg="red")

root = tk.Tk()
root.title("Simple Hand Gesture Control")
root.geometry("480x750")
root.resizable(False, False)

tk.Label(root, text="Hand Gesture Controller", font=("Segoe UI", 18, "bold"), fg="#1E88E5").pack(pady=10)

start_btn = tk.Button(root, text="Start Webcam", command=start_app, bg="#4CAF50", fg="white", font=("Segoe UI", 12, "bold"), width=30)
start_btn.pack(pady=5)
stop_btn = tk.Button(root, text="Stop Webcam", command=stop_app, bg="#F44336", fg="white", font=("Segoe UI", 12, "bold"), width=30)
stop_btn.pack(pady=5)
status_label = tk.Label(root, text="Status: Inactive", fg="red", font=("Segoe UI", 12, "italic"))
status_label.pack(pady=5)

# Gesture Manual Frame
frame = tk.LabelFrame(root, text="📱 Easy Gesture Guide", font=("Segoe UI", 12, "bold"), padx=10, pady=10)
frame.pack(fill="both", expand="yes", padx=20, pady=10)

gestures = [
    ("☝️ One Index Finger Up", "Move mouse pointer freely"),
    ("✌️ Two Fingers Closed", "Click and drag objects"),
    ("✌️ Two Fingers Open (V)", "Scroll up or down"),
    ("🖖 Three Fingers Together", "Right click"),
    ("🖐️ Three Fingers Spread", "Double click"),
    ("🤘 Index + Pinky Only", "Open Task View"),
    ("🖐️ All Four Fingers Up", "Play or pause media"),
    ("🤙 Thumb + Pinky Out", "Play next track"),
    ("🤏 Pinky Finger Alone", "Take screenshot"),
    ("👍 Thumb Pointing Up", "Volume increase"),
    ("👎 Thumb Pointing Down", "Volume decrease"),
    ("✊ Make a Fist", "Show desktop")
]
for g, desc in gestures:
    lbl = tk.Label(frame, text=f"{g} → {desc}", font=("Segoe UI", 10), anchor="w", justify="left", fg="#1E88E5")
    lbl.pack(fill="x", pady=3)
tk.Label(root, text="Keep inside the purple box for best accuracy", font=("Segoe UI", 9, "italic"), fg="#FF6F00").pack(pady=5)

root.protocol("WM_DELETE_WINDOW", lambda: (stop_app(), root.destroy()))
root.mainloop()