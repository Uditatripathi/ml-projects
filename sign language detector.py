import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from gtts import gTTS
import os
from io import BytesIO
from tempfile import NamedTemporaryFile
import base64

# Load pre-trained model (online)
@st.cache_resource
def load_sign_model():
    model = load_model('https://huggingface.co/spaces/udita-tripathi/asl-sign-model/resolve/main/asl_model.h5')
    return model

model = load_sign_model()
labels = [chr(i) for i in range(65, 91)]

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Streamlit UI
st.title("🖐️ Sign Language to Speech Translator")
st.markdown("Make hand gestures (A–Z) in front of your webcam. Click 'Speak' to hear the word.")

run = st.checkbox("Start Webcam")
speak_button = st.button("🔊 Speak Word")
clear_button = st.button("🗑️ Clear")

# Result word
if 'sentence' not in st.session_state:
    st.session_state['sentence'] = ""

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                x_min, y_min = w, h
                x_max, y_max = 0, 0
                for lm in hand_landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_min, y_min = min(x, y, x_min), min(y, y_min)
                    x_max, y_max = max(x, x_max), max(y, y_max)

                x_min = max(x_min - 20, 0)
                y_min = max(y_min - 20, 0)
                x_max = min(x_max + 20, w)
                y_max = min(y_max + 20, h)

                hand_img = frame[y_min:y_max, x_min:x_max]
                if hand_img.size == 0:
                    continue
                try:
                    hand_img = cv2.cvtColor(hand_img, cv2.COLOR_BGR2GRAY)
                    hand_img = cv2.resize(hand_img, (64, 64))
                    hand_img = hand_img / 255.0
                    hand_img = np.expand_dims(hand_img, axis=[0, -1])
                    pred = model.predict(hand_img)
                    pred_letter = labels[np.argmax(pred)]
                    st.session_state['sentence'] += pred_letter
                    cv2.putText(frame, f'{pred_letter}', (x_min, y_min - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                except:
                    pass

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Word: {st.session_state['sentence']}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        stframe.image(frame, channels="BGR")

        if not run:
            break

    cap.release()

# Speak button logic
if speak_button and st.session_state['sentence']:
    tts = gTTS(st.session_state['sentence'], lang='en')
    with NamedTemporaryFile(delete=True) as fp:
        tts.save(f"{fp.name}.mp3")
        audio_file = open(f"{fp.name}.mp3", 'rb')
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")

# Clear sentence
if clear_button:
    st.session_state['sentence'] = ""
