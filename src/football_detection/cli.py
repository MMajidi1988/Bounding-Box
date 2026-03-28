from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2

from football_detection.core import (
    annotate_image,
    load_model,
    parse_class_names_arg,
    predict_image,
)
from football_detection.paths import DEFAULT_WEIGHTS_PATH, resolve_weights_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="football-detection",
        description="Run YOLO detection on video (local file or m3u8 stream).",
    )
    sub = p.add_subparsers(dest="command", required=True)

    pv = sub.add_parser("video", help="Detect on a video file or m3u8 stream")
    pv.add_argument(
        "-m",
        "--model",
        default=None,
        help=f"Path to YOLO weights (.pt). Default: {DEFAULT_WEIGHTS_PATH} if that file exists.",
    )
    pv.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    pv.add_argument(
        "--device",
        default=None,
        help="Torch device, e.g. cpu, 0, cuda:0 (default: Ultralytics auto)",
    )
    pv.add_argument(
        "--class-names",
        default=None,
        help="Comma-separated class names (index order), optional override for labels",
    )
    g = pv.add_mutually_exclusive_group(required=True)
    g.add_argument("-i", "--input", help="Input video file path")
    g.add_argument("--m3u8-url", dest="m3u8_url", help="HTTP URL to m3u8 (requires ffmpeg on PATH)")
    pv.add_argument("-o", "--output", required=True, help="Output video path (.mp4 or .avi)")
    pv.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames)",
    )
    pv.add_argument(
        "--codec",
        choices=("mp4v", "XVID"),
        default="mp4v",
        help="Fourcc for output (default mp4v)",
    )

    return p


def _model_path_from_args(model_arg: str | None) -> str:
    if model_arg:
        return model_arg
    return resolve_weights_path()


def cmd_video(args: argparse.Namespace) -> int:
    names = parse_class_names_arg(args.class_names)
    try:
        weights = _model_path_from_args(args.model)
    except RuntimeError as e:
        print(str(e), file=sys.stderr)
        return 1
    model = load_model(weights)

    temp_mp4: Path | None = None
    if args.m3u8_url:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp.close()
        temp_mp4 = Path(tmp.name)
        cmd = ["ffmpeg", "-y", "-i", args.m3u8_url, "-c", "copy", str(temp_mp4)]
        print("Running ffmpeg to fetch stream...")
        subprocess.run(cmd, check=True)
        cap_path = str(temp_mp4)
    else:
        cap_path = args.input

    video = cv2.VideoCapture(cap_path)
    if not video.isOpened():
        print(f"Could not open video: {cap_path}", file=sys.stderr)
        if temp_mp4 and temp_mp4.exists():
            temp_mp4.unlink()
        return 1

    frame_w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS) or 25.0
    fourcc = cv2.VideoWriter_fourcc(*args.codec)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (frame_w, frame_h))

    frame_count = 0
    try:
        while True:
            ret, frame = video.read()
            if not ret:
                break
            frame_count += 1
            if args.skip_frames > 1 and frame_count % args.skip_frames != 0:
                writer.write(frame)
                continue
            _, detections = predict_image(
                model,
                frame,
                conf=args.conf,
                device=args.device,
                verbose=False,
                class_names_override=names,
            )
            annotated = annotate_image(frame, detections, conf_threshold=args.conf)
            writer.write(annotated)
    finally:
        video.release()
        writer.release()
        if temp_mp4 and temp_mp4.exists():
            temp_mp4.unlink()

    print(f"Wrote {out_path} ({frame_count} frames read)")
    return 0


def main(argv: List[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    args = _build_parser().parse_args(argv)
    if args.command == "video":
        return cmd_video(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
