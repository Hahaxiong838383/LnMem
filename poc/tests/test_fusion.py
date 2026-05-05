"""Day 2 v0.9 — MultimodalFusion 单元测试（纯 mock，不依赖外部 lib）。"""
from __future__ import annotations

from poc.src.fusion import (
    DEFAULT_CONFIG,
    FEATURE_DIM,
    FusionConfig,
    FusionFrame,
    MultimodalFusion,
)


def _full_frame(idx: int = 0) -> dict:
    """构造一个所有子模态齐备的输入帧。"""
    return {
        "ts": "2026-05-05T12:00:00",
        "frame_idx": idx,
        "face": {
            "eye_closure": 0.2,
            "head_pitch_deg": 5.0,
            "head_yaw_deg": 8.0,
            "gaze_stability": 0.85,
            "landmarks_n": 478,
        },
        "audio": {
            "transcript": "今天专注度还不错",
            "lang": "zh",
            "duration_s": 1.5,
        },
        "context": {
            "active_app": "Code",
            "app_switch_count_5min": 2,
        },
    }


def test_fusion_full_frame_vector_dim_and_range() -> None:
    """完整帧融合 → 向量 12 维 + 全部 [0,1]。"""
    fusion = MultimodalFusion()
    out = fusion.fuse(_full_frame(idx=42))
    assert isinstance(out, FusionFrame)
    assert len(out.vector) == FEATURE_DIM == 12
    assert all(0.0 <= v <= 1.0 for v in out.vector)
    assert out.missing == ()
    assert out.frame_idx == 42
    assert out.meta["lang"] == "zh"


def test_fusion_missing_face_falls_back() -> None:
    """face 缺失 → vector 仍 12 维 + missing 含 'face' + 默认值填补。"""
    fusion = MultimodalFusion()
    frame = _full_frame()
    frame["face"] = None
    out = fusion.fuse(frame)
    assert len(out.vector) == FEATURE_DIM
    assert "face" in out.missing
    # 视觉前 8 维（face 4 + derived 4）应全 = missing_default
    expected = DEFAULT_CONFIG.missing_default
    assert all(v == expected for v in out.vector[:8])


def test_fusion_missing_audio_and_context() -> None:
    """audio + context 同时缺失 → missing 双标 + 后 4 维默认值。"""
    fusion = MultimodalFusion()
    frame = _full_frame()
    frame.pop("audio")
    frame.pop("context")
    out = fusion.fuse(frame)
    assert "audio" in out.missing
    assert "context" in out.missing
    expected = DEFAULT_CONFIG.missing_default
    # 索引 8-9 = audio，10-11 = context
    assert all(v == expected for v in out.vector[8:12])


def test_fusion_empty_frame_all_missing() -> None:
    """空帧 → missing 三模态全标。"""
    fusion = MultimodalFusion()
    out = fusion.fuse({"ts": "2026-05-05T12:00:00", "frame_idx": 0})
    assert set(out.missing) == {"face", "audio", "context"}
    assert len(out.vector) == FEATURE_DIM


def test_fusion_extreme_head_pose_clamped_to_one() -> None:
    """极端头部姿态（> max_deg）→ 归一化到 1.0，不溢出。"""
    fusion = MultimodalFusion()
    frame = _full_frame()
    frame["face"]["head_pitch_deg"] = 90.0  # > 45 (max)
    frame["face"]["head_yaw_deg"] = -120.0  # |abs| > 60 (max)
    out = fusion.fuse(frame)
    # face_features 的索引 1=pitch_norm, 2=yaw_norm
    assert out.vector[1] == 1.0
    assert out.vector[2] == 1.0
    assert all(0.0 <= v <= 1.0 for v in out.vector)


def test_fusion_blink_burst_threshold() -> None:
    """eye_closure > 阈值 → blink_burst (索引 7) = 1.0；低于则 = 0.0。"""
    fusion = MultimodalFusion(FusionConfig(eye_closure_fatigue_threshold=0.30))
    high = _full_frame()
    high["face"]["eye_closure"] = 0.50
    low = _full_frame()
    low["face"]["eye_closure"] = 0.10
    assert fusion.fuse(high).vector[7] == 1.0
    assert fusion.fuse(low).vector[7] == 0.0


def test_fuse_batch_returns_list_of_frames() -> None:
    """batch fusion → 列表长度 = 输入 + 类型一致。"""
    fusion = MultimodalFusion()
    frames = [_full_frame(idx=i) for i in range(5)]
    out = fusion.fuse_batch(frames)
    assert len(out) == 5
    assert all(isinstance(f, FusionFrame) for f in out)
    assert [f.frame_idx for f in out] == [0, 1, 2, 3, 4]


def test_fusion_frame_is_immutable() -> None:
    """FusionFrame 是 frozen dataclass — 强制 immutability（coding-style.md）。"""
    fusion = MultimodalFusion()
    out = fusion.fuse(_full_frame())
    try:
        out.frame_idx = 999  # type: ignore[misc]
    except Exception:
        return  # 预期抛 FrozenInstanceError 或 AttributeError
    raise AssertionError("FusionFrame should be frozen")
