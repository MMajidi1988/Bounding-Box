import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from football_detection.api import create_app


def test_health():
    client = TestClient(create_app())
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_detect_without_weights_returns_503(monkeypatch, tmp_path):
    import football_detection.paths as paths_mod

    monkeypatch.delenv("YOLO_MODEL", raising=False)
    monkeypatch.setattr(paths_mod, "DEFAULT_WEIGHTS_PATH", tmp_path / "missing.pt")
    client = TestClient(create_app())
    img = np.zeros((4, 4, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    r = client.post("/detect", files={"file": ("x.jpg", buf.tobytes(), "image/jpeg")})
    assert r.status_code == 503
