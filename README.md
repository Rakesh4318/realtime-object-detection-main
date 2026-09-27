#  Real-Time Object Detection



## Features

- Real-time webcam object detection
- Image inference
- Video-file inference
- YOLO confidence and IoU controls
- Bounding boxes and class labels
- Live FPS display
- Per-frame object counts
- Optional processed-video recording
- Streamlit web demo for image/video uploads
- Automatic model download on first run

## Tech Stack

- Python 3.10+
- YOLO (Ultralytics)
- OpenCV
- NumPy
- Streamlit

## Installation

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

macOS/Linux:

```bash
source .venv/bin/activate
```

Then:

```bash
pip install -r requirements.txt
```

## Run Webcam Detection

```bash
python app.py --source 0
```

Press **Q** or **Esc** to close the camera window.

To save the detected webcam feed:

```bash
python app.py --source 0 --save
```

## Run on an Image

```bash
python app.py --source path/to/image.jpg
```

The processed image is saved in `outputs/`.

## Run on a Video

```bash
python app.py --source path/to/video.mp4 --save
```

## Change Detection Threshold

```bash
python app.py --source 0 --conf 0.50 --iou 0.45
```

## Use GPU

If PyTorch detects your CUDA setup, use:

```bash
python app.py --source 0 --device 0
```

For CPU:

```bash
python app.py --source 0 --device cpu
```

## Run the Streamlit App

```bash
streamlit run streamlit_app.py
```

Open the local URL shown by Streamlit. Upload an image or video and the app will run object detection in the browser UI.

## Model

The default model is `yolo11n.pt`, a lightweight YOLO11 model suitable for real-time demos. Ultralytics downloads the weights automatically the first time the project runs.

You can use a larger model for better accuracy:

```bash
python app.py --source 0 --model yolo11s.pt
```

