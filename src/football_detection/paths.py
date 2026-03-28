"""Repository paths (Bounding-Box/ is two levels above this package)."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS_PATH = REPO_ROOT / "models" / "yolo26n.pt"


def resolve_weights_path() -> str:
    """Prefer YOLO_MODEL env, else models/yolo26n.pt if present."""
    import os

    env = os.environ.get("YOLO_MODEL", "").strip()
    if env:
        return env
    if DEFAULT_WEIGHTS_PATH.is_file():
        return str(DEFAULT_WEIGHTS_PATH.resolve())
    raise RuntimeError(
        f"Set environment variable YOLO_MODEL or add weights at {DEFAULT_WEIGHTS_PATH}"
    )
