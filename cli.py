"""Entry point: run from repo root with `python cli.py ...`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from football_detection.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
