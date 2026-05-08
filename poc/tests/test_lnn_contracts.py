"""Day 5 LNN 契约测试 — 验证 contracts.py 类型/默认值/兼容性。"""
from __future__ import annotations

from pathlib import Path

import pytest

from poc.src.lnn.contracts import (
    EpochMetric,
    LNNCheckpoint,
    LNNDataError,
    LNNModelError,
    LNNTrainConfig,
)


def test_default_config_is_consistent() -> None:
    """默认配置与 fusion FEATURE_DIM=12 + 8 类 State + 30 画像对齐。"""
    cfg = LNNTrainConfig()
    assert cfg.feature_dim == 12
    assert cfg.n_state_classes == 8
    assert cfg.n_personas == 30
    assert cfg.sequence_length >= cfg.sequence_min_frames
    assert 0 < cfg.train_val_split < 1


def test_resolved_device_returns_string() -> None:
    cfg = LNNTrainConfig(device="cpu")
    assert cfg.resolved_device() == "cpu"

    cfg_auto = LNNTrainConfig(device="auto")
    out = cfg_auto.resolved_device()
    assert out in ("mps", "cuda", "cpu")


def test_config_is_frozen() -> None:
    cfg = LNNTrainConfig()
    with pytest.raises((AttributeError, Exception)):
        cfg.feature_dim = 999  # type: ignore[misc]


def test_epoch_metric_defaults() -> None:
    m = EpochMetric(epoch=0, train_loss=0.5, val_loss=0.6, train_state_acc=0.4, val_state_acc=0.45)
    assert m.val_boundary_f1 == 0.0
    assert m.notes == ""


def test_checkpoint_minimal_construction() -> None:
    cfg = LNNTrainConfig()
    ckpt = LNNCheckpoint(
        config=cfg,
        persona_to_idx={"vboss_001": 0, "vboss_002": 1},
        state_model_state_dict={},
    )
    assert ckpt.boundary_model_state_dict is None
    assert ckpt.epoch_metrics == ()
    assert ckpt.final_state_acc == 0.0


def test_signals_dir_default_points_to_llm_v2() -> None:
    cfg = LNNTrainConfig()
    assert cfg.signals_dir == Path("poc/data/synthetic/llm_v2")


def test_exceptions_are_subclasses() -> None:
    """LNNDataError 是 ValueError，LNNModelError 是 RuntimeError——便于 except 通用类型。"""
    assert issubclass(LNNDataError, ValueError)
    assert issubclass(LNNModelError, RuntimeError)
