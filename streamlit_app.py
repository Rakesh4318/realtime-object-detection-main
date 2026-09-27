from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from src.detector import DetectionConfig, ObjectDetector

st.set_page_config(page_title="VisionPulse — Object Detection", page_icon="👁️", layout="wide")

st.title("VisionPulse")
st.caption("YOLO + OpenCV real-time object detection demo")

with st.sidebar:
    st.header("Detection settings")
    model_name = st.selectbox("YOLO model", ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt"], index=0)
    confidence = st.slider("Confidence", 0.10, 0.90, 0.35, 0.05)
    iou = st.slider("IoU", 0.10, 0.90, 0.45, 0.05)
    mode = st.radio("Input", ["Image", "Video"])

@st.cache_resource(show_spinner=True)
def load_detector(model: str, conf: float, iou_value: float):
    return ObjectDetector(DetectionConfig(model_path=model, confidence=conf, iou=iou_value))

detector = load_detector(model_name, confidence, iou)

if mode == "Image":
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        frame = cv2.cvtColor(__import__("numpy").array(image), cv2.COLOR_RGB2BGR)
        annotated, result = detector.annotate(frame)
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

        left, right = st.columns([2, 1])
        with left:
            st.image(annotated_rgb, caption="Detected objects", use_container_width=True)
        with right:
            st.subheader("Object counts")
            counts = detector.count_objects(result)
            if counts:
                st.dataframe({"Class": list(counts.keys()), "Count": list(counts.values())}, hide_index=True)
            else:
                st.info("No objects detected above the selected threshold.")

else:
    uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "avi", "mkv"])
    if uploaded:
        temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        temp_input.write(uploaded.read())
        temp_input.close()

        cap = cv2.VideoCapture(temp_input.name)
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        temp_output.close()
        writer = cv2.VideoWriter(temp_output.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

        progress = st.progress(0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_no = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            annotated, _ = detector.annotate(frame)
            writer.write(annotated)
            frame_no += 1
            progress.progress(min(frame_no / total, 1.0))

        cap.release()
        writer.release()

        st.success("Video processed successfully.")
        with open(temp_output.name, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
        st.download_button("Download detected video", video_bytes, file_name="detected_video.mp4", mime="video/mp4")
