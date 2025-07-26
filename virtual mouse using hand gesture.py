import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math

# Initialize webcam
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# MediaPipe Hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Smoothing parameters
prev_x, prev_y = 0, 0
smoothening = 5

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror the image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            lm_list = []
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            # Index finger tip = id 8, Thumb tip = id 4
            x1, y1 = lm_list[8][1:]  # Index tip
            x2, y2 = lm_list[4][1:]  # Thumb tip

            # Draw circles
            cv2.circle(img, (x1, y1), 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 8, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 255, 0), 2)

            # Convert to screen coordinates
            screen_x = np.interp(x1, (100, w - 100), (0, screen_width))
            screen_y = np.interp(y1, (100, h - 100), (0, screen_height))

            # Smooth cursor movement
            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - pre_)
