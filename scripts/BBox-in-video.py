# Prefer: python cli.py video -m models/yolo26n.pt -i YOUR.mp4 -o outputs/annotated.mp4
# This script uses the same defaults (model + output folder).

import subprocess
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent  # Bounding-Box (this file lives in scripts/)
sys.path.insert(0, str(ROOT / "src"))

from football_detection.core import annotate_image, load_model, predict_image  # noqa: E402

MODEL_PATH = ROOT / "models" / "yolo26n.pt"
OUTPUT_PATH = ROOT / "outputs" / "annotated.avi"
# Set M3U8_URL to stream, or leave empty and use INPUT_VIDEO file.
M3U8_URL = ""
INPUT_VIDEO = ROOT / "samples" / "Original.mp4"  # used when M3U8_URL is empty

SKIP_N_FRAMES = 1
CLASS_NAMES = ["player", "ball", "logo"]


def main() -> None:
    if not MODEL_PATH.is_file():
        raise SystemExit(f"Missing weights: {MODEL_PATH}")

    model = load_model(MODEL_PATH)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if M3U8_URL:
        temp_video = ROOT / "temp_video.mp4"
        subprocess.run(["ffmpeg", "-y", "-i", M3U8_URL, "-c", "copy", str(temp_video)], check=True)
        cap_path = str(temp_video)
    else:
        cap_path = str(INPUT_VIDEO)
        if not Path(cap_path).is_file():
            raise SystemExit(
                f"No input video at {INPUT_VIDEO}. Set INPUT_VIDEO or M3U8_URL in scripts/BBox-in-video.py, "
                "or use: python cli.py video -m models/yolo26n.pt -i clip.mp4 -o outputs/annotated.mp4"
            )
        temp_video = None

    video = cv2.VideoCapture(cap_path)
    if not video.isOpened():
        raise SystemExit(f"Could not open video: {cap_path}")

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    out = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"XVID"),
        fps,
        (frame_w, frame_h),
    )

    frame_count = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if SKIP_N_FRAMES > 1 and frame_count % SKIP_N_FRAMES != 0:
                out.write(frame)
                continue
            _, detections = predict_image(
                model,
                frame,
                conf=0.5,
                device=None,
                verbose=False,
                class_names_override=CLASS_NAMES,
            )
            annotated = annotate_image(frame, detections, conf_threshold=0.5)
            out.write(annotated)
    finally:
        video.release()
        out.release()
        if temp_video and Path(temp_video).exists():
            Path(temp_video).unlink()

    print(f"Wrote {OUTPUT_PATH} ({frame_count} frames read)")


if __name__ == "__main__":
    main()
