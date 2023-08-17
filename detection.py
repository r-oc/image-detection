from ultralytics import YOLO
import streamlit as st
import PIL
from PIL import Image

st.title("Welcome to r-oc's object detection!")

picture_taken = st.camera_input("Take a photo of something, and see what it detects. Trained using YOLOv8 NANO model.")
if picture_taken:
    uploaded_image = PIL.Image.open(picture_taken)

    with st.spinner("Detecting objects..."):
        model = YOLO("yolov8n.pt")
        res = model.predict(uploaded_image)

        boxes = res[0].boxes
        res_plotted = res[0].plot()[:, :, ::-1]
        st.image(res_plotted,
                 caption='Detected Image',
                 use_column_width=True
                 )


