import cv2
import mediapipe as mp
import numpy as np
import pickle
import json

# uploaded model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# uploaded labels
with open("config.json") as f:
    config = json.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

url = config['camera_url']  # Use the camera URL from config
cap = cv2.VideoCapture(url)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            prediction = model.predict([landmarks])
            cv2.putText(frame, f"Prediction: {prediction[0]}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("Real-time Prediction", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()