"""TDD: poc/src/lnn/feedback_to_batch.py — feedback × signals → batch tensor."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from poc.src.fusion import FEATURE_DIM
from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.feedback_to_batch import (
    MATCH_LABEL_PLACEHOLDER,
    STATE_NAME_TO_IDX,
    build_training_batch,
    fill_match_labels_with_argmax,
)
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_contracts import FeedbackDataError


# ---------- Fixtures ----------


def _signal_row(frame_id: int, smile: float = 0.5) -> dict:
    return {
        "ts": f"2026-05-10T04:00:{frame_id:04d}.000Z",
        "device": "air",
        "frame_id": frame_id,
        "raw_video_offset_ms": frame_id * 1000,
        "face_detected": True,
        "raw_au": {},
        "calibrated_au": {f"AU{i:02d}": 0.1 for i in [1, 4, 6, 12, 17, 23, 24, 28, 43]},
        "expression": {
            "smile_intensity": smile,
            "brow_furrow": 0.5,
            "cheek_raise": 0.3,
            "lip_corner_depress": 0.1,
            "jaw_drop": 0.0,
            "mouth_open_ratio": 0.0,
        },
        "head_pose": {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0},
        "session_id": "test",
    }


def _feedback_row(
    frame_id: int,
    label: str,
    *,
    true_state: str | None = None,
    note: str | None = None,
) -> dict:
    return {
        "ts": f"2026-05-10T04:00:{frame_id:04d}Z",
        "frame_id": frame_id,
        "reason": "brow_burst" if label != "skip" else "low_vlm_conf",
        "signal": {},
        "vlm_label": None,
        "user_response": {"label": label, "true_state": true_state, "note": note},
        "latency_s": 5.0,
    }


@pytest.fixture
def setup_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """build feedback + signals dirs。"""
    fb = tmp_path / "feedback"
    sig = tmp_path / "data"
    fb.mkdir()
    sig.mkdir()

    # 100 帧 signals（足够 window=60）
    rows = [_signal_row(frame_id=i, smile=0.1 + i * 0.005) for i in range(100)]
    (sig / "signals_2026-05-10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n",
        encoding="utf-8",
    )

    # feedback 4 条：mismatch w/ true_state, match, skip, mismatch w/o true_state
    fb_rows = [
        _feedback_row(frame_id=70, label="mismatch", true_state="distracted"),
        _feedback_row(frame_id=80, label="match"),
        _feedback_row(frame_id=85, label="skip"),
        _feedback_row(frame_id=90, label="mismatch", true_state=None),  # 缺 true_state
    ]
    (fb / "queries_2026-05-10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in fb_rows) + "\n",
        encoding="utf-8",
    )
    return fb, sig


@pytest.fixture
def small_config() -> LNNTrainConfig:
    return LNNTrainConfig(
        feature_dim=FEATURE_DIM,  # 18
        n_state_classes=8,
        n_personas=31,
        persona_embed_dim=8,
        hidden_dim=8,
        sequence_length=60,
        sequence_stride=30,
        sequence_min_frames=30,
        batch_size=2,
        epochs=1,
        device="cpu",
    )


# ---------- 行为测试 ----------


class TestStateNameMapping:
    def test_state_name_to_idx_complete(self) -> None:
        assert STATE_NAME_TO_IDX["low_energy"] == 0
        assert STATE_NAME_TO_IDX["distracted"] == 2
        assert STATE_NAME_TO_IDX["transitioning"] == 7
        assert len(STATE_NAME_TO_IDX) == len(STATE_LABELS)


class TestBuildBatch:
    def test_only_match_and_mismatch_with_valid_true_state(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        """skip / input / mismatch w/o true_state 应被过滤掉。"""
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # 抑制 mismatch w/o ts warning
            batch = build_training_batch(
                fb, sig, persona_idx=30, window_size=60
            )
        # frame 70 mismatch+distracted ✅ + frame 80 match ✅
        # frame 85 skip ❌（filter_for_training）
        # frame 90 mismatch w/o true_state ❌（_label_for_event 返 None）
        assert batch.x.shape == (2, 60, 18)
        assert batch.persona_idx.tolist() == [30, 30]

    def test_label_assignment(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(fb, sig, persona_idx=30, window_size=60)
        # 顺序：mismatch(70, distracted=2) → match(80, placeholder=-1)
        assert batch.labels[0].item() == STATE_NAME_TO_IDX["distracted"]
        assert batch.labels[1].item() == MATCH_LABEL_PLACEHOLDER
        assert batch.label_sources == ("mismatch_true_state", "match_argmax_pending")

    def test_rewards_assigned(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(fb, sig, persona_idx=30, window_size=60)
        # mismatch reward=-1 / match reward=+1
        assert batch.rewards[0].item() == pytest.approx(-1.0)
        assert batch.rewards[1].item() == pytest.approx(1.0)

    def test_skipped_count(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(fb, sig, persona_idx=30, window_size=60)
        # 1 skipped: mismatch w/o true_state（frame 90）
        assert batch.skipped_count == 1

    def test_window_too_short_skipped(
        self, tmp_path: Path
    ) -> None:
        """frame_id 太靠前（< window_size）的 event 应被跳过。"""
        fb = tmp_path / "feedback"
        sig = tmp_path / "data"
        fb.mkdir()
        sig.mkdir()
        # 仅 30 帧 signals，window=60 不够
        rows = [_signal_row(frame_id=i) for i in range(30)]
        (sig / "signals_2026-05-10.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8"
        )
        # feedback 在 frame 20（早于 window=60 边界）
        (fb / "queries_2026-05-10.jsonl").write_text(
            json.dumps(_feedback_row(frame_id=20, label="match")) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(FeedbackDataError, match="no usable training samples"):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                build_training_batch(fb, sig, persona_idx=30, window_size=60)

    def test_frame_ids_recorded(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(fb, sig, persona_idx=30, window_size=60)
        assert batch.frame_ids == (70, 80)

    def test_include_skip_input_false_by_default(
        self, setup_dirs: tuple[Path, Path]
    ) -> None:
        """默认不含 skip / input（filter_for_training 已过滤）。"""
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(
                fb, sig, persona_idx=30, window_size=60,
                include_skip_input=False,
            )
        # 仅 match + mismatch 进入流程（skip 在 filter_for_training 阶段已过滤）
        assert batch.x.shape[0] == 2

    def test_no_signals_raises(self, tmp_path: Path) -> None:
        fb = tmp_path / "feedback"
        sig = tmp_path / "data"
        fb.mkdir()
        sig.mkdir()
        (fb / "queries_2026-05-10.jsonl").write_text(
            json.dumps(_feedback_row(frame_id=70, label="match")) + "\n",
            encoding="utf-8",
        )
        # 空 signals 文件
        (sig / "signals_2026-05-10.jsonl").write_text("", encoding="utf-8")
        with pytest.raises(FeedbackDataError):
            build_training_batch(fb, sig, persona_idx=30, window_size=60)


# ---------- fill_match_labels_with_argmax ----------


class TestFillArgmax:
    def test_fills_match_placeholder_only(
        self, setup_dirs: tuple[Path, Path], small_config: LNNTrainConfig
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(
                fb, sig, persona_idx=30, window_size=small_config.sequence_length
            )
        # mismatch label=2 不应被改；match label=-1 应被填
        torch.manual_seed(0)
        factory = DefaultLNNFactory()
        model = factory.create_state_model(small_config)
        new_batch = fill_match_labels_with_argmax(batch, model)

        # mismatch row labels[0] 仍是 distracted=2
        assert new_batch.labels[0].item() == STATE_NAME_TO_IDX["distracted"]
        # match row labels[1] 应被填为 [0, n_state_classes)
        assert 0 <= new_batch.labels[1].item() < small_config.n_state_classes

    def test_immutable_returns_new_object(
        self, setup_dirs: tuple[Path, Path], small_config: LNNTrainConfig
    ) -> None:
        fb, sig = setup_dirs
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            batch = build_training_batch(
                fb, sig, persona_idx=30, window_size=small_config.sequence_length
            )
        torch.manual_seed(0)
        factory = DefaultLNNFactory()
        model = factory.create_state_model(small_config)
        new_batch = fill_match_labels_with_argmax(batch, model)
        # original batch.labels 仍含 -1（不被 in-place 改）
        assert (batch.labels == MATCH_LABEL_PLACEHOLDER).any().item()
        # new batch labels 全 ≥ 0
        assert (new_batch.labels >= 0).all().item()
        # x / persona_idx / rewards 共享（同 tensor 引用 OK）
        assert new_batch.x is batch.x
