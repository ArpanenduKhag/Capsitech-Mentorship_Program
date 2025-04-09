import streamlit as st
import cv2
import numpy as np
from gesture import detect_gesture

st.title("🖐 Gesture Recognition with MediaPipe")

run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

camera = cv2.VideoCapture(0)

while run:
    ret, frame = camera.read()
    if not ret:
        st.error("Camera error")
        break

    frame = cv2.flip(frame, 1)
    frame, gesture = detect_gesture(frame)

    st.markdown(f"### Detected Gesture: {gesture}")
    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

camera.release()
