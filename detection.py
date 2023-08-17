from ultralytics import YOLO
import streamlit as st

picture = st.camera_input("Take a pic")

model = YOLO("yolov8n.pt")

results = model(source=picture, show=True)

if results:
    st.image(results)
