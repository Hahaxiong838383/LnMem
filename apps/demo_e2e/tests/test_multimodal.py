"""Phase α 多模态 snapshot 单测（Mock 模式覆盖）。

不测真 py-feat（依赖 ~150MB 模型 + CPU 慢），留 Phase β 集成测。
"""
from __future__ import annotations

import hashlib

import pytest

from apps.demo_e2e.multimodal import (
    AU_NAMES_20,
    STATE_LABELS_9,
    STATE_LABELS_ZH,
    SnapshotResult,
    _deterministic_va_from_bytes,
    map_va_to_state_label,
    snapshot_to_state,
)


# ──────────────── Russell 几何映射测试 ────────────────

def test_map_va_neutral_center():
    """V-A 落在圆心 r<0.35 → neutral。"""
    assert map_va_to_state_label(0.0, 0.0) == "neutral"
    assert map_va_to_state_label(0.1, 0.1) == "neutral"  # r ≈ 0.14
    assert map_va_to_state_label(0.2, 0.2) == "neutral"  # r ≈ 0.28


def test_map_va_happy_quadrant_high_v_mid_a():
    """高 V 中 A → happy（sector 1, 0°-45°）。"""
    # V=0.6, A=0.2 → r ≈ 0.63, θ ≈ 18° → happy
    assert map_va_to_state_label(0.6, 0.2) == "happy"


def test_map_va_sad_quadrant_low_v_mid_a():
    """低 V 中 A → sad（sector 5, 180°-225°）。"""
    # V=-0.6, A=-0.2 → θ ≈ 198° → sad
    assert map_va_to_state_label(-0.6, -0.2) == "sad"


def test_map_va_excited_high_v_high_a():
    """中高 V 高 A → excited（sector 2, 45°-90°）。"""
    # V=0.4, A=0.6 → θ ≈ 56° → excited
    assert map_va_to_state_label(0.4, 0.6) == "excited"


def test_map_va_tense_low_v_high_a():
    """低 V 高 A → tense（sector 4, 135°-180°）。"""
    # V=-0.5, A=0.4 → θ ≈ 141° → tense
    assert map_va_to_state_label(-0.5, 0.4) == "tense"


# ──────────────── 确定性 hash → V-A 测试 ────────────────

def test_deterministic_va_same_bytes_same_va():
    """相同输入 bytes 必须出相同 V-A（demo 可复现）。"""
    img = b"fake_image_bytes_chuange_2026"
    v1, a1 = _deterministic_va_from_bytes(img)
    v2, a2 = _deterministic_va_from_bytes(img)
    assert v1 == v2
    assert a1 == a2


def test_deterministic_va_in_range():
    """V/A 必须在 [-0.8, 0.8] 范围内。"""
    for seed in [b"a", b"hello world", b"\x00\xff" * 100, b"chuange"]:
        v, a = _deterministic_va_from_bytes(seed)
        assert -0.8 <= v <= 0.8, f"V out of range: {v} for {seed!r}"
        assert -0.8 <= a <= 0.8, f"A out of range: {a} for {seed!r}"


def test_deterministic_va_empty_returns_zero():
    """空 bytes → (0, 0)。"""
    assert _deterministic_va_from_bytes(b"") == (0.0, 0.0)


# ──────────────── 端到端 snapshot 测试 ────────────────

def test_snapshot_empty_bytes_returns_error():
    """空 bytes → error 字段非 None，state=neutral 兜底。"""
    result = snapshot_to_state(None)
    assert result["face_detected"] is False
    assert result["error"] == "empty image bytes"
    assert result["state_label"] == "neutral"
    assert result["confidence"] == 0.0

    result2 = snapshot_to_state(b"")
    assert result2["face_detected"] is False
    assert result2["error"] is not None


def test_snapshot_mock_mode_consistent():
    """Mock 模式：相同输入 → 相同 SnapshotResult（demo 可复现 + 测试稳定）。"""
    img = b"chuange_test_image_2026_05_21"
    r1 = snapshot_to_state(img, mode="mock")
    r2 = snapshot_to_state(img, mode="mock")
    assert r1 == r2


def test_snapshot_mock_returns_full_schema():
    """Mock 模式返回所有 SnapshotResult 字段。"""
    result = snapshot_to_state(b"some_image_data", mode="mock")
    # TypedDict 必填字段
    assert set(result.keys()) == {
        "au_vector", "face_detected", "v_a", "state_label",
        "state_label_zh", "confidence", "source", "error",
    }
    assert result["face_detected"] is True
    assert result["error"] is None
    assert result["source"] == "mock"
    assert result["state_label"] in STATE_LABELS_9
    assert result["state_label_zh"] in STATE_LABELS_ZH.values()
    assert 0.0 <= result["confidence"] <= 1.0


def test_snapshot_au_vector_20_dims():
    """AU vector 必须含全部 20 个 AU 维度。"""
    result = snapshot_to_state(b"test", mode="mock")
    assert set(result["au_vector"].keys()) == set(AU_NAMES_20)
    for name, val in result["au_vector"].items():
        # AU 范围 [0, 1]（py-feat XGB 概率 + mock 反推都对齐到这个范围，2026-05-21 修复）
        assert 0.0 <= val <= 1.0, f"{name} out of range: {val}"


def test_snapshot_va_in_unit_circle():
    """V-A 必须在单位圆内 [-1, 1]。"""
    for seed in [b"img1", b"img2", b"\xff" * 1000, b"\x00" * 50]:
        result = snapshot_to_state(seed, mode="mock")
        v, a = result["v_a"]
        assert -1.0 <= v <= 1.0
        assert -1.0 <= a <= 1.0


def test_snapshot_pyfeat_mode_falls_back_to_mock():
    """py-feat 不可用时，pyfeat 模式自动降级 mock（不抛异常）。

    本地未必装 py-feat（依赖重），降级路径必须可用。
    """
    img = b"test_pyfeat_fallback"
    result = snapshot_to_state(img, mode="pyfeat")
    # 不管 py-feat 是否装上，都返回合法结果
    assert result["state_label"] in STATE_LABELS_9
    assert result["error"] is None
    # 若降级，source 字段标 fallback
    if result["source"].startswith("mock"):
        assert "fallback" in result["source"]


def test_state_labels_count():
    """9-state 标签必须正好 9 个（1 neutral + 8 sectors）。"""
    assert len(STATE_LABELS_9) == 9
    assert len(STATE_LABELS_ZH) == 9
    assert STATE_LABELS_9[0] == "neutral"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
