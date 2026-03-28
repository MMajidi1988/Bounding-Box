from football_detection.core import parse_class_names_arg


def test_parse_class_names_arg():
    assert parse_class_names_arg(None) is None
    assert parse_class_names_arg("") is None
    assert parse_class_names_arg("a,b, c") == ["a", "b", "c"]
