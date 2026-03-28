"""Gradio UI: uses `models/yolo26n.pt` if present, else set YOLO_MODEL."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from football_detection.gradio_ui import main

if __name__ == "__main__":
    main()
