"""TDD: poc/src/lnn/signals_loader.py — daemon V0.2 signals jsonl → 18 维 fusion vector."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest
import torch

from poc.src.fusion import FEATURE_DIM
from poc.src.lnn.online_contracts import FeedbackDataError
from poc.src.lnn.signals_loader import (
    SignalsFrame,
    build_frame_id_index,
    build_session_frame_index,
    load_signals,
    parse_signals_row,
    signals_to_tensor,
)


# ---------- Fixtures ----------


def _make_signals_row(
    frame_id: int = 0,
    *,
    smile: float = 0.5,
    brow: float = 0.5,
    pitch: float = 0.0,
    yaw: float = 0.0,
    au43: float = 0.3,
    face_detected: bool = True,
) -> dict:
    """构造一条 daemon V0.2 格式的 signals jsonl row。"""
    return {
        "ts": f"2026-05-10T04:00:{frame_id:02d}.000Z",
        "device": "air",
        "frame_id": frame_id,
        "raw_video_offset_ms": frame_id * 1000,
        "face_detected": face_detected,
        "raw_au": {f"AU{i:02d}": 0.1 for i in range(1, 28)},
        "calibrated_au": {
            "AU01": 0.1, "AU02": 0.1, "AU04": 0.2, "AU05": 0.1,
            "AU06": 0.3, "AU07": 0.1, "AU09": 0.0, "AU10": 0.0,
            "AU11": 0.0, "AU12": 0.4, "AU14": 0.1, "AU15": 0.1,
            "AU17": 0.1, "AU20": 0.0, "AU23": 0.1, "AU24": 0.1,
            "AU25": 0.1, "AU26": 0.1, "AU28": 0.1, "AU43": au43,
        },
        "expression": {
            "smile_intensity": smile,
            "brow_furrow": brow,
            "cheek_raise": 0.3,
            "lip_corner_depress": 0.1,
            "jaw_drop": 0.0,
            "mouth_open_ratio": 0.0,
        },
        "head_pose": {
            "pitch_deg": pitch,
            "roll_deg": 0.0,
            "yaw_deg": yaw,
        },
        "session_id": "test-session",
    }


@pytest.fixture
def signals_dir(tmp_path: Path) -> Path:
    d = tmp_path / "signals"
    d.mkdir()
    rows_0510 = [_make_signals_row(frame_id=i, smile=0.1 + i * 0.05) for i in range(5)]
    (d / "signals_2026-05-10.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows_0510) + "\n",
        encoding="utf-8",
    )
    return d


# ---------- parse_signals_row ----------


class TestParseRow:
    def test_18_dim_vector(self) -> None:
        row = _make_signals_row()
        sf = parse_signals_row(row)
        assert sf is not None
        assert len(sf.vector) == FEATURE_DIM

    def test_smile_field_propagates(self) -> None:
        """expression.smile_intensity 应直接映射到 fusion vector 的 expression 段。"""
        sf = parse_signals_row(_make_signals_row(smile=0.9, brow=0.1))
        # face_feats 4 + derived 4 = 8 之后是 expression 6 维
        # idx 8: smile, idx 9: brow
        assert sf is not None
        assert sf.vector[8] == pytest.approx(0.9)
        assert sf.vector[9] == pytest.approx(0.1)

    def test_face_detected_preserved(self) -> None:
        sf_true = parse_signals_row(_make_signals_row(face_detected=True))
        sf_false = parse_signals_row(_make_signals_row(face_detected=False))
        assert sf_true.face_detected
        assert not sf_false.face_detected

    def test_non_dict_row_skipped(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            assert parse_signals_row([1, 2, 3]) is None
            assert parse_signals_row("string") is None
            assert parse_signals_row(42) is None
        assert any("non-dict" in str(x.message).lower() for x in w)

    def test_eye_closure_from_AU43(self) -> None:
        sf = parse_signals_row(_make_signals_row(au43=0.85))
        # eye_closure 是 face_feats 第 0 维
        assert sf.vector[0] == pytest.approx(0.85)


# ---------- load_signals ----------


class TestLoadSignals:
    def test_loads_all_rows(self, signals_dir: Path) -> None:
        frames = load_signals(signals_dir)
        assert len(frames) == 5
        assert all(isinstance(f, SignalsFrame) for f in frames)

    def test_returns_frozen_tuple(self, signals_dir: Path) -> None:
        frames = load_signals(signals_dir)
        assert isinstance(frames, tuple)

    def test_frame_id_preserved(self, signals_dir: Path) -> None:
        frames = load_signals(signals_dir)
        ids = [f.frame_id for f in frames]
        assert ids == [0, 1, 2, 3, 4]

    def test_missing_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FeedbackDataError):
            load_signals(tmp_path / "nonexistent")

    def test_skip_invalid_json(self, tmp_path: Path) -> None:
        d = tmp_path / "signals"
        d.mkdir()
        valid = json.dumps(_make_signals_row(frame_id=0))
        (d / "signals_2026-05-10.jsonl").write_text(
            valid + "\n{not-json}\n" + json.dumps(_make_signals_row(frame_id=2)) + "\n",
            encoding="utf-8",
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            frames = load_signals(d)
        assert len(frames) == 2

    def test_date_prefix_filter(self, tmp_path: Path) -> None:
        d = tmp_path / "signals"
        d.mkdir()
        (d / "signals_2026-05-09.jsonl").write_text(
            json.dumps(_make_signals_row(frame_id=99)) + "\n", encoding="utf-8"
        )
        (d / "signals_2026-05-10.jsonl").write_text(
            json.dumps(_make_signals_row(frame_id=0)) + "\n", encoding="utf-8"
        )
        frames = load_signals(d, date_prefix="2026-05-10")
        assert len(frames) == 1
        assert frames[0].frame_id == 0


# ---------- index + tensor ----------


class TestIndexAndTensor:
    def test_build_index(self, signals_dir: Path) -> None:
        frames = load_signals(signals_dir)
        idx = build_frame_id_index(frames)
        assert idx[0] == 0
        assert idx[4] == 4
        assert len(idx) == 5

    def test_index_takes_last_when_duplicate(self) -> None:
        """跨天 frame_id 重复时，应取后到的（更近的）index。"""
        # 手动构造重复 frame_id
        frames = (
            SignalsFrame(ts="t1", frame_id=10, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}),
            SignalsFrame(ts="t2", frame_id=20, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}),
            SignalsFrame(ts="t3", frame_id=10, vector=tuple([1.0] * 18),
                         face_detected=True, expression={}),
        )
        idx = build_frame_id_index(frames)
        assert idx[10] == 2  # 后到覆盖前
        assert idx[20] == 1

    def test_signals_to_tensor_shape(self, signals_dir: Path) -> None:
        frames = load_signals(signals_dir)
        t = signals_to_tensor(frames)
        assert t.shape == (5, FEATURE_DIM)
        assert t.dtype == torch.float32

    def test_tensor_empty_raises(self) -> None:
        with pytest.raises(FeedbackDataError):
            signals_to_tensor(())


class TestF1_01_SessionFrameIndex:
    """F1-01 (codex): 复合键 (session_id, frame_id) 防跨天 silent overwrite。"""

    def test_session_frame_index_basic(self) -> None:
        frames = (
            SignalsFrame(ts="t1", frame_id=10, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}, session_id="A"),
            SignalsFrame(ts="t2", frame_id=20, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}, session_id="A"),
        )
        idx = build_session_frame_index(frames)
        assert idx[("A", 10)] == 0
        assert idx[("A", 20)] == 1

    def test_cross_session_same_frame_id_no_collision(self) -> None:
        """两个 session 同 frame_id 不再 silent overwrite。"""
        frames = (
            SignalsFrame(ts="t1", frame_id=10, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}, session_id="A"),
            SignalsFrame(ts="t2", frame_id=10, vector=tuple([1.0] * 18),
                         face_detected=True, expression={}, session_id="B"),
        )
        idx = build_session_frame_index(frames)
        assert idx[("A", 10)] == 0
        assert idx[("B", 10)] == 1
        assert len(idx) == 2  # 两个独立 entry

    def test_legacy_build_frame_id_index_warns_on_duplicates(self) -> None:
        """老 build_frame_id_index 跨天重复发 warning（F1-01 风险标识）。"""
        frames = (
            SignalsFrame(ts="t1", frame_id=10, vector=tuple([0.0] * 18),
                         face_detected=True, expression={}),
            SignalsFrame(ts="t2", frame_id=10, vector=tuple([1.0] * 18),
                         face_detected=True, expression={}),
        )
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            idx = build_frame_id_index(frames)
        assert idx[10] == 1  # 后到覆盖前（行为保留）
        assert any("F1-01" in str(x.message) for x in w)

    def test_session_id_attribute_default_none(self) -> None:
        """SignalsFrame.session_id 默认 None（向后兼容老 V0.1 数据）。"""
        sf = SignalsFrame(
            ts="t", frame_id=0, vector=tuple([0.0] * 18),
            face_detected=True, expression={},
        )
        assert sf.session_id is None


class TestF2_04_MissingFieldWarning:
    """F2-04 (codex): signals 字段缺失发 warning（不阻断 fail-soft）。"""

    def test_missing_expression_warns(self) -> None:
        row = _make_signals_row()
        del row["expression"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sf = parse_signals_row(row)
        assert sf is not None  # 仍 fail-soft 解析
        assert any("'expression' field missing" in str(x.message) for x in w)

    def test_missing_head_pose_warns(self) -> None:
        row = _make_signals_row()
        del row["head_pose"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sf = parse_signals_row(row)
        assert sf is not None
        assert any("'head_pose' field missing" in str(x.message) for x in w)

    def test_missing_calibrated_au_warns(self) -> None:
        row = _make_signals_row()
        del row["calibrated_au"]
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            sf = parse_signals_row(row)
        assert sf is not None
        assert any("'calibrated_au' field missing" in str(x.message) for x in w)

    def test_session_id_extracted_from_jsonl(self) -> None:
        row = _make_signals_row()
        row["session_id"] = "test-session-uuid-1234"
        sf = parse_signals_row(row)
        assert sf.session_id == "test-session-uuid-1234"
