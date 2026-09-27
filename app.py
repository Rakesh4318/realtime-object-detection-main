from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Union

import cv2

from src.detector import DetectionConfig, ObjectDetector
from src.utils import ensure_dir, open_video_writer, timestamp_name


def parse_source(value: str) -> Union[int, str]:
    return int(value) if value.isdigit() else value


def draw_hud(frame, fps: float, counts: dict[str, int]):
    cv2.rectangle(frame, (8, 8), (390, 72), (0, 0, 0), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    summary = ", ".join(f"{k}:{v}" for k, v in list(counts.items())[:5]) or "No objects"
    cv2.putText(frame, summary, (18, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return frame


def run_image(detector: ObjectDetector, image_path: str, output_dir: Path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, result = detector.annotate(image)
    counts = detector.count_objects(result)
    annotated = draw_hud(annotated, 0.0, counts)

    output_path = output_dir / timestamp_name("detected_image", "jpg")
    cv2.imwrite(str(output_path), annotated)
    print(f"Saved result to: {output_path}")
    print(f"Detected: {counts}")

    cv2.imshow("YOLO Object Detection", annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def run_stream(detector: ObjectDetector, source: Union[int, str], output_dir: Path, save: bool):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    writer = None
    previous = time.perf_counter()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.perf_counter()
            fps = 1.0 / max(now - previous, 1e-6)
            previous = now

            annotated, result = detector.annotate(frame)
            counts = detector.count_objects(result)
            annotated = draw_hud(annotated, fps, counts)

            if save and writer is None:
                h, w = annotated.shape[:2]
                source_fps = cap.get(cv2.CAP_PROP_FPS)
                if source_fps <= 1 or source_fps > 240:
                    source_fps = 30.0
                output_path = output_dir / timestamp_name("detected_video", "mp4")
                writer = open_video_writer(output_path, source_fps, w, h)
                print(f"Saving video to: {output_path}")

            if writer is not None:
                writer.write(annotated)

            cv2.imshow("Real-Time YOLO Object Detection", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-Time Object Detection with YOLO + OpenCV")
    parser.add_argument("--source", default="0", help="0 for webcam, video path, stream URL, or image path")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLO model, e.g. yolo11n.pt")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="IoU threshold")
    parser.add_argument("--device", default=None, help="cpu, 0, 1, etc.")
    parser.add_argument("--save", action="store_true", help="Save processed video stream")
    parser.add_argument("--output", default="outputs", help="Output directory")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)
    detector = ObjectDetector(DetectionConfig(args.model, args.conf, args.iou, args.device))

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    source_path = Path(args.source)
    if source_path.exists() and source_path.suffix.lower() in image_extensions:
        run_image(detector, args.source, output_dir)
    else:
        run_stream(detector, parse_source(args.source), output_dir, args.save)


if __name__ == "__main__":
    main()
