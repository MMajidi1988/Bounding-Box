from football_detection.export import detection_rows_for_image, write_csv, write_json
from football_detection.core import Detection


def test_detection_rows_for_image():
    dets: list[Detection] = [
        {
            "x1": 1.0,
            "y1": 2.0,
            "x2": 3.0,
            "y2": 4.0,
            "confidence": 0.9,
            "class_id": 0,
            "class_name": "ball",
        }
    ]
    rows = detection_rows_for_image("a.jpg", dets)
    assert len(rows) == 1
    assert rows[0]["source"] == "a.jpg"
    assert rows[0]["class_name"] == "ball"


def test_write_json_csv(tmp_path):
    write_json(tmp_path / "a.json", [{"x": 1}])
    assert (tmp_path / "a.json").read_text(encoding="utf-8").strip().startswith("[")

    write_csv(tmp_path / "b.csv", [])
    assert "source" in (tmp_path / "b.csv").read_text(encoding="utf-8")
