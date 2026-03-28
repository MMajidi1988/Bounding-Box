from __future__ import annotations

import base64
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Query, UploadFile

from football_detection.core import (
    annotate_image,
    load_model,
    parse_class_names_arg,
    predict_image,
)
from football_detection.paths import resolve_weights_path

_model = None
_model_path: str | None = None


def get_model():
    global _model, _model_path
    path = resolve_weights_path()
    if _model is None or _model_path != path:
        _model = load_model(path)
        _model_path = path
    return _model


def create_app() -> FastAPI:
    app = FastAPI(title="YOLO detection", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/detect")
    async def detect(
        file: UploadFile = File(...),
        conf: float = Query(0.5, ge=0.0, le=1.0),
        device: str | None = Query(None),
        class_names: str | None = Query(None, description="Comma-separated class name overrides"),
        annotate: bool = Query(True, description="Return base64 JPEG with boxes drawn"),
    ) -> dict[str, Any]:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")
        arr = np.frombuffer(data, dtype=np.uint8)
        image_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        names = parse_class_names_arg(class_names)
        try:
            model = get_model()
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e)) from e

        _, detections = predict_image(
            model,
            image_bgr,
            conf=conf,
            device=device,
            verbose=False,
            class_names_override=names,
        )
        out: dict[str, Any] = {"detections": detections, "count": len(detections)}
        if annotate:
            vis = annotate_image(image_bgr, detections, conf_threshold=conf)
            ok, buf = cv2.imencode(".jpg", vis)
            if not ok:
                raise HTTPException(status_code=500, detail="Failed to encode output image")
            out["image_base64"] = base64.b64encode(buf.tobytes()).decode("ascii")
        return out

    return app
