from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Tuple

import cv2
import gradio as gr
import numpy as np

from football_detection.core import (
    annotate_image,
    load_model,
    parse_class_names_arg,
    predict_image,
)
from football_detection.paths import resolve_weights_path

_model = None
_model_path: str | None = None


def _get_model():
    global _model, _model_path
    try:
        path = resolve_weights_path()
    except RuntimeError as e:
        raise gr.Error(str(e)) from e
    if _model is None or _model_path != path:
        _model = load_model(path)
        _model_path = path
    return _model


def _run(
    image: np.ndarray | None,
    conf: float,
    class_names_csv: str,
) -> Tuple[Any, str]:
    if image is None:
        return None, "Upload an image."
    # Gradio may pass RGB; OpenCV / Ultralytics expect BGR for predict on ndarray
    if image.ndim == 3 and image.shape[2] == 3:
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image

    names = parse_class_names_arg(class_names_csv)
    model = _get_model()
    _, detections = predict_image(
        model,
        image_bgr,
        conf=conf,
        device=None,
        verbose=False,
        class_names_override=names,
    )
    vis_bgr = annotate_image(image_bgr, detections, conf_threshold=conf)
    vis_rgb = cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB)
    summary = f"{len(detections)} detection(s)\n" + "\n".join(
        f"- {d['class_name']}: {d['confidence']:.3f}" for d in detections
    )
    return vis_rgb, summary


def _video_input_path(video_file: Any) -> str | None:
    """Normalize Gradio Video value to a filesystem path."""
    if video_file is None:
        return None
    if isinstance(video_file, str) and video_file.strip():
        return video_file
    if isinstance(video_file, dict):
        return video_file.get("name") or video_file.get("path")
    return None


def _run_video(
    video_file: Any,
    conf: float,
    class_names_csv: str,
    skip_frames: int,
    progress: gr.Progress = gr.Progress(),
) -> Tuple[str | None, str]:
    path = _video_input_path(video_file)
    if not path or not Path(path).is_file():
        return None, "Upload a video file."

    skip_n = max(1, int(skip_frames))
    names = parse_class_names_arg(class_names_csv)
    model = _get_model()

    video = cv2.VideoCapture(path)
    if not video.isOpened():
        return None, f"Could not open video: {path}"

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    total_guess = int(video.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    if total_guess <= 0:
        total_guess = 1

    fd, out_tmp = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    out_path = Path(out_tmp)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    frame_count = 0
    try:
        progress(0.0, desc="Annotating video…")
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if skip_n > 1 and frame_count % skip_n != 0:
                writer.write(frame)
                continue
            _, detections = predict_image(
                model,
                frame,
                conf=conf,
                device=None,
                verbose=False,
                class_names_override=names,
            )
            annotated = annotate_image(frame, detections, conf_threshold=conf)
            writer.write(annotated)
            progress(min(1.0, frame_count / total_guess), desc="Annotating video…")
    finally:
        video.release()
        writer.release()

    msg = f"Done: {frame_count} frame(s) read → {out_path.name}\n(same pipeline as scripts/BBox-in-video.py / cli.py video)"
    return str(out_path), msg


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="YOLO detection") as demo:
        gr.Markdown(
            "# Bounding-box detection\n"
            "Uses `models/yolo26n.pt` if present, or set `YOLO_MODEL`."
        )
        with gr.Tabs():
            with gr.Tab("Image"):
                with gr.Row():
                    inp = gr.Image(type="numpy", label="Input")
                    out = gr.Image(type="numpy", label="Annotated")
                conf = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Confidence")
                class_names = gr.Textbox(
                    label="Class names (optional, comma-separated, index order)",
                    placeholder="e.g. ball,player,logo",
                )
                btn = gr.Button("Run")
                log = gr.Textbox(label="Detections", lines=8)
                btn.click(_run, inputs=[inp, conf, class_names], outputs=[out, log])
            with gr.Tab("Video"):
                gr.Markdown(
                    "Upload a video to run the same per-frame detection as "
                    "`scripts/BBox-in-video.py` (bounding boxes written to an output video)."
                )
                vid_in = gr.Video(label="Input video", sources=["upload"])
                vid_out = gr.Video(label="Annotated video")
                conf_v = gr.Slider(0.05, 1.0, value=0.5, step=0.05, label="Confidence")
                class_names_v = gr.Textbox(
                    label="Class names (optional, comma-separated, index order)",
                    placeholder="e.g. player,ball,logo",
                    value="player,ball,logo",
                )
                skip_v = gr.Slider(
                    1,
                    10,
                    value=1,
                    step=1,
                    label="Process every Nth frame (1 = all frames; others copied unchanged)",
                )
                btn_v = gr.Button("Run")
                log_v = gr.Textbox(label="Status", lines=4)
                btn_v.click(
                    _run_video,
                    inputs=[vid_in, conf_v, class_names_v, skip_v],
                    outputs=[vid_out, log_v],
                )
    return demo


def main() -> None:
    demo = build_demo()
    # 127.0.0.1 opens in the browser; 0.0.0.0 is bind-only and often fails as a URL (e.g. ERR_ADDRESS_INVALID).
    host = os.environ.get("GRADIO_HOST", "127.0.0.1")
    demo.launch(server_name=host, server_port=int(os.environ.get("GRADIO_PORT", "7860")))


if __name__ == "__main__":
    main()
