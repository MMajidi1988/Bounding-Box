from __future__ import annotations

from pathlib import Path
from typing import Any, List, Sequence

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

Detection = dict[str, Any]


def load_model(weights: str | Path) -> YOLO:
    return YOLO(str(weights))


def _resolve_class_name(
    cls_id: int,
    names: dict[int, str] | Any,
    class_names_override: Sequence[str] | None,
) -> str:
    if class_names_override is not None and 0 <= cls_id < len(class_names_override):
        return class_names_override[cls_id]
    if isinstance(names, dict):
        return names.get(cls_id, str(cls_id))
    return str(cls_id)


def result_to_detections(
    result: Results,
    class_names_override: Sequence[str] | None = None,
) -> List[Detection]:
    out: List[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    names = result.names
    boxes = result.boxes
    for i in range(len(boxes)):
        xyxy = boxes.xyxy[i].cpu().tolist()
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        name = _resolve_class_name(cls_id, names, class_names_override)
        out.append(
            {
                "x1": float(xyxy[0]),
                "y1": float(xyxy[1]),
                "x2": float(xyxy[2]),
                "y2": float(xyxy[3]),
                "confidence": conf,
                "class_id": cls_id,
                "class_name": name,
            }
        )
    return out


def predict_image(
    model: YOLO,
    image_bgr: np.ndarray,
    *,
    conf: float,
    device: str | None,
    verbose: bool = False,
    class_names_override: Sequence[str] | None = None,
) -> tuple[Results, List[Detection]]:
    kwargs: dict[str, Any] = {"conf": conf, "verbose": verbose}
    if device:
        kwargs["device"] = device
    results = model.predict(source=image_bgr, **kwargs)
    r = results[0]
    return r, result_to_detections(r, class_names_override)


def predict_path(
    model: YOLO,
    path: str | Path,
    *,
    conf: float,
    device: str | None,
    verbose: bool = False,
    class_names_override: Sequence[str] | None = None,
) -> tuple[Results, List[Detection]]:
    kwargs: dict[str, Any] = {"conf": conf, "verbose": verbose}
    if device:
        kwargs["device"] = device
    results = model.predict(source=str(path), **kwargs)
    r = results[0]
    return r, result_to_detections(r, class_names_override)


def annotate_image(
    image_bgr: np.ndarray,
    detections: List[Detection],
    *,
    conf_threshold: float,
) -> np.ndarray:
    img = image_bgr.copy()
    for d in detections:
        if d["confidence"] < conf_threshold:
            continue
        x1, y1, x2, y2 = map(int, [d["x1"], d["y1"], d["x2"], d["y2"]])
        label = f'{d["class_name"]}: {d["confidence"]:.2f}'
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y_text = max(y1, label_size[1])
        cv2.rectangle(
            img,
            (x1, y_text - label_size[1]),
            (x1 + label_size[0], y_text + base_line),
            (255, 255, 255),
            cv2.FILLED,
        )
        cv2.putText(img, label, (x1, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    return img


def read_image_bgr(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return img


def write_image_bgr(path: str | Path, image_bgr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image_bgr)
    if not ok:
        raise OSError(f"Failed to write image: {path}")


def parse_class_names_arg(value: str | None) -> list[str] | None:
    if not value or not value.strip():
        return None
    return [s.strip() for s in value.split(",") if s.strip()]
