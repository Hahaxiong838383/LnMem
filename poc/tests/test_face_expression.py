"""AU expression feature tests from FaceMesh landmarks."""

from __future__ import annotations

from types import SimpleNamespace

from poc.src.capture.face_signals import _extract_expression_features


def _make_landmarks() -> list[SimpleNamespace]:
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    points[10] = SimpleNamespace(x=0.5, y=0.2, z=0.0)
    points[152] = SimpleNamespace(x=0.5, y=0.8, z=0.0)
    points[234] = SimpleNamespace(x=0.2, y=0.5, z=0.0)
    points[454] = SimpleNamespace(x=0.8, y=0.5, z=0.0)
    points[13] = SimpleNamespace(x=0.5, y=0.6, z=0.0)
    points[14] = SimpleNamespace(x=0.5, y=0.602, z=0.0)
    points[61] = SimpleNamespace(x=0.42, y=0.6, z=0.0)
    points[291] = SimpleNamespace(x=0.58, y=0.6, z=0.0)
    points[55] = SimpleNamespace(x=0.45, y=0.4, z=0.0)
    points[285] = SimpleNamespace(x=0.55, y=0.4, z=0.0)
    points[112] = SimpleNamespace(x=0.43, y=0.48, z=0.0)
    points[341] = SimpleNamespace(x=0.57, y=0.48, z=0.0)
    points[468] = SimpleNamespace(x=0.43, y=0.46, z=0.0)
    points[473] = SimpleNamespace(x=0.57, y=0.46, z=0.0)
    return points


def test_smile_face_smile_intensity_high() -> None:
    points = _make_landmarks()
    points[61].y = 0.56
    points[291].y = 0.56
    out = _extract_expression_features(points, len(points))
    assert float(out["smile_intensity"]) > 0.3


def test_frown_face_brow_furrow_high() -> None:
    points = _make_landmarks()
    points[55].x = 0.49
    points[285].x = 0.51
    out = _extract_expression_features(points, len(points))
    assert float(out["brow_furrow"]) > 0.3


def test_lip_corner_depress_high() -> None:
    points = _make_landmarks()
    points[61].y = 0.64
    points[291].y = 0.64
    out = _extract_expression_features(points, len(points))
    assert float(out["lip_corner_depress"]) > 0.3


def test_mouth_open_jaw_drop_and_ratio_high() -> None:
    points = _make_landmarks()
    points[14].y = 0.64
    out = _extract_expression_features(points, len(points))
    assert float(out["jaw_drop"]) > 0.3
    assert float(out["mouth_open_ratio"]) > 0.3


def test_neutral_face_all_expression_features_low() -> None:
    points = _make_landmarks()
    out = _extract_expression_features(points, len(points))
    assert float(out["smile_intensity"]) < 0.3
    assert float(out["brow_furrow"]) < 0.3
    assert float(out["cheek_raise"]) < 0.3
    assert float(out["lip_corner_depress"]) < 0.3
    assert float(out["jaw_drop"]) < 0.3
    assert float(out["mouth_open_ratio"]) < 0.3


def test_insufficient_landmarks_returns_all_zero() -> None:
    points = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(100)]
    out = _extract_expression_features(points, len(points))
    assert out == {
        "smile_intensity": 0.0,
        "brow_furrow": 0.0,
        "cheek_raise": 0.0,
        "lip_corner_depress": 0.0,
        "jaw_drop": 0.0,
        "mouth_open_ratio": 0.0,
    }
