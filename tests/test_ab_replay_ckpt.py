"""Tests for ab_replay_ckpt_v1_v2 helpers with mocked LNN forward."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "ab_replay_ckpt_v1_v2.py"
_SPEC = importlib.util.spec_from_file_location("ab_replay_ckpt_v1_v2", _SCRIPT)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = mod
_SPEC.loader.exec_module(mod)


class _FakeRunner:
    def __init__(self, prediction):
        self.prediction = prediction
        self.calls: list[tuple[tuple[int, ...], int]] = []

    def predict(self, window: torch.Tensor, persona_idx: int):
        self.calls.append((tuple(window.shape), persona_idx))
        return self.prediction


def _make_frame(ts: str, frame_id: int) -> mod.SignalsFrame:
    return mod.SignalsFrame(
        ts=ts,
        frame_id=frame_id,
        vector=tuple(float(frame_id % 5) for _ in range(18)),
        face_detected=True,
        expression={},
        session_id="sess-1",
    )


def test_load_evaluable_events_filters_brow_burst_ground_truth(tmp_path: Path) -> None:
    """Only brow_burst + match/mismatch + non-null true_state should survive."""
    path = tmp_path / "queries.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"ts":"2026-05-19T00:00:00Z","frame_id":10,"reason":"brow_burst","user_response":{"label":"mismatch","true_state":"focused"},"session_id":"s"}',
                '{"ts":"2026-05-19T00:00:01Z","frame_id":11,"reason":"brow_burst","user_response":{"label":"skip","true_state":null},"session_id":"s"}',
                '{"ts":"2026-05-19T00:00:02Z","frame_id":12,"reason":"pose_held","user_response":{"label":"mismatch","true_state":"distracted"},"session_id":"s"}',
            ]
        ),
        encoding="utf-8",
    )
    events = mod.load_evaluable_events([path], default_persona_idx=30)
    assert len(events) == 1
    assert events[0].true_state == "focused"
    assert events[0].persona_idx == 30


def test_replay_events_uses_last_60_frames_and_default_persona() -> None:
    """Replay should feed a 60-frame window into each mocked runner."""
    event = mod.ReplayEvent(
        ts="2026-05-19T00:01:00Z",
        frame_id=159,
        session_id="sess-1",
        persona_idx=30,
        response_label="mismatch",
        true_state="focused",
        signal_path="raw",
        vlm_label="none",
        source_file="queries.jsonl",
    )
    sessions = {"sess-1": [_make_frame(f"2026-05-19T00:00:{i:02d}Z", 100 + i) for i in range(60)]}
    pred_v1 = mod.Prediction("focused", 0.8, 1, 1)
    pred_v2 = mod.Prediction("focused", 0.7, 2, 2)
    runner_v1 = _FakeRunner(pred_v1)
    runner_v2 = _FakeRunner(pred_v2)
    rows, skipped = mod.replay_events(
        [event],
        sessions,
        {"base_v1": runner_v1, "base_v2": runner_v2},
        window_size=60,
    )
    assert not skipped
    assert len(rows) == 1
    assert rows[0].predictions["base_v1"].label == "focused"
    assert rows[0].predictions["base_v2"].label == "focused"
    assert runner_v1.calls == [((1, 60, 18), 30)]
    assert runner_v2.calls == [((1, 60, 18), 30)]


def test_base_v2_runner_drops_neutral_top1_to_second_best(tmp_path: Path) -> None:
    """When class 0 is neutral top-1, mapping should use the next-highest class."""
    runner = mod.BaseV2Runner.__new__(mod.BaseV2Runner)
    runner.device = "cpu"
    runner.persona_idx = 0
    runner.label_order = tuple(mod.STATE_LABELS)

    class _FakeModel:
        def __call__(self, window, pid):
            del window, pid
            logits = torch.tensor([[[5.0, 1.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]])
            return logits

    runner.model = _FakeModel()
    runner.n_personas = 31
    prediction = mod.BaseV2Runner.predict(runner, torch.zeros((1, 60, 18)), persona_idx=30)
    assert prediction.top_index == 0
    assert prediction.selected_index == 2
    assert prediction.label == mod.STATE_LABELS[1]
    assert prediction.raw_top1_label == "neutral_top1"
