from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, List

from football_detection.core import Detection


def detection_rows_for_image(
    source: str,
    detections: Iterable[Detection],
) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    for d in detections:
        rows.append(
            {
                "source": source,
                "x1": d["x1"],
                "y1": d["y1"],
                "x2": d["x2"],
                "y2": d["y2"],
                "confidence": d["confidence"],
                "class_id": d["class_id"],
                "class_name": d["class_name"],
            }
        )
    return rows


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def write_csv(path: str | Path, rows: List[dict[str, Any]], fieldnames: List[str] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = [
                "source",
                "x1",
                "y1",
                "x2",
                "y2",
                "confidence",
                "class_id",
                "class_name",
            ]
        with path.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
