"""Vision AU calibration tests. / 视觉 AU 校准测试。"""

from __future__ import annotations

from poc.src.vision.feature_calibrator import calibrate_to_lnn_expression


def test_neutral_face_all_expression_features_low() -> None:
    au = {"AU04": 0.08, "AU06": 0.08, "AU12": 0.08, "AU15": 0.06, "AU25": 0.05, "AU26": 0.07}
    out = calibrate_to_lnn_expression(au)
    assert out["smile_intensity"] < 0.3
    assert out["brow_furrow"] < 0.3
    assert out["cheek_raise"] < 0.3
    assert out["lip_corner_depress"] < 0.3
    assert out["jaw_drop"] < 0.3
    assert out["mouth_open_ratio"] < 0.3


def test_smile_face_smile_intensity_high() -> None:
    out = calibrate_to_lnn_expression({"AU06": 0.71, "AU12": 0.76})
    assert out["smile_intensity"] > 0.7


def test_frown_face_brow_furrow_high() -> None:
    out = calibrate_to_lnn_expression({"AU04": 0.50})
    assert out["brow_furrow"] > 0.95


def test_light_frown_brow_furrow_stretched() -> None:
    out = calibrate_to_lnn_expression({"AU04": 0.20})
    assert out["brow_furrow"] > 0.7


def test_missing_aus_use_defaults_without_exception() -> None:
    out = calibrate_to_lnn_expression({})
    assert out == {
        "smile_intensity": 0.0,
        "brow_furrow": 0.0,
        "cheek_raise": 0.0,
        "lip_corner_depress": 0.0,
        "jaw_drop": 0.0,
        "mouth_open_ratio": 0.0,
    }


def test_all_outputs_are_in_range_0_1() -> None:
    out = calibrate_to_lnn_expression(
        {"AU04": 2.0, "AU06": 1.6, "AU12": 1.5, "AU15": -0.4, "AU25": 8.0, "AU26": 9.0}
    )
    for value in out.values():
        assert 0.0 <= value <= 1.0
