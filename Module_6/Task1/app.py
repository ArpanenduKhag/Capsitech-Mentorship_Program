import streamlit as st
from detect import detect_objects
from PIL import Image
import numpy as np
import os

st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("🔍 Object Detection with YOLOv8")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Save temporarily
    img_path = "temp_input.jpg"
    img.save(img_path)

    with st.spinner("Running object detection..."):
        result_img = detect_objects(img_path)

    st.subheader("📸 Detection Result")
    st.image(result_img, use_column_width=True)

    # Clean up
    os.remove(img_path)
