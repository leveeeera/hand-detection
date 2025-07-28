import cv2
import mediapipe as mp
import numpy as np
import csv
import json

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Load configuration for camera URL
with open("config.json") as f:
    config = json.load(f)

url = config['camera_url']  # Use the camera URL from config 
cap = cv2.VideoCapture(url)
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("Press 'q' to quit or any key to record a gesture.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    landmarks = None
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
    cv2.imshow("Collecting Data", frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        print("Exiting...")
        break
    elif landmarks is not None and key != -1:
        label = chr(key)
        with open("data.csv", mode="a", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(landmarks + [label])
        print(f"Saved: {label}")
cap.release()
cv2.destroyAllWindows()