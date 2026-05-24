"""Tests for /intent_feedback handler semantics."""

from __future__ import annotations

import io
import json
from collections import deque
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.intent_feedback_server import (  # noqa: E402
    IntentFeedbackHandler,
    IntentFeedbackStore,
    SessionTsRateLimiter,
)
from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer  # noqa: E402


class _WindowProvider:
    def __init__(self, window: list[dict[str, object]] | None = None) -> None:
        self._window = window or []

    def get_recent_window(self, seconds: int = 10) -> list[dict[str, object]]:
        _ = seconds
        return list(self._window)


class _DummyPersonalization:
    def get_tau_bias(self, persona_idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros((persona_idx.shape[0], 64), dtype=torch.float32)

    def get_film_params(self, persona_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((persona_idx.shape[0], 64), dtype=torch.float32)
        return zeros, zeros


class _DummyBackbone:
    def rnn_cell(
        self, features: torch.Tensor, hidden: torch.Tensor, timespans: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        _ = features, timespans
        return hidden + 0.1, None


class _DummyStateModel:
    def __init__(self) -> None:
        self.hidden_dim = 64
        self.input_projection = torch.nn.Identity()
        self.personalization = _DummyPersonalization()
        self.backbone = _DummyBackbone()

    def _timespans_from_tau(self, tau_bias: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tau_bias)

    def _apply_film(self, hidden: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        _ = gamma, beta
        return hidden

    def __call__(self, x: torch.Tensor, persona_idx: torch.Tensor) -> torch.Tensor:
        _ = persona_idx
        logits = torch.zeros((x.shape[0], x.shape[1], 8), dtype=torch.float32)
        logits[:, :, 1] = 2.0
        return logits


def _signal_row(frame_id: int) -> dict[str, object]:
    aus = {f"AU{i:02d}": 0.2 for i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]}
    return {
        "ts": f"2026-06-01T00:00:{frame_id:02d}Z",
        "frame_id": frame_id,
        "face_detected": True,
        "raw_au": dict(aus),
        "calibrated_au": dict(aus),
        "expression": {
            "brow_furrow": 0.2,
            "smile_intensity": 0.1,
            "eye_closure": 0.1,
            "cheek_raise": 0.1,
            "lip_corner_depress": 0.0,
            "jaw_drop": 0.0,
            "mouth_open_ratio": 0.0,
        },
        "head_pose": {"pitch_deg": 1.0, "yaw_deg": 0.0, "roll_deg": 0.0},
    }


def _run_post(
    payload: dict[str, object],
    store: IntentFeedbackStore,
    limiter: SessionTsRateLimiter,
    path: str = "/intent_feedback",
    inferencer: object | None = None,
) -> tuple[int, dict[str, object]]:
    raw = json.dumps(payload).encode("utf-8")
    handler = IntentFeedbackHandler.__new__(IntentFeedbackHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler.rate_limiter = limiter
    handler.store = store
    handler.inferencer = inferencer
    captured: dict[str, object] = {}

    def _reply(status: int, body: dict[str, object]) -> None:
        captured["status"] = status
        captured["body"] = body

    handler._reply = _reply  # type: ignore[method-assign]
    handler.do_POST()
    return int(captured["status"]), captured["body"]  # type: ignore[return-value]


def test_intent_feedback_accept_and_store(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    status, body = _run_post(
        {
            "ts_ns": 1_717_000_000_000_000_000,
            "predicted": "positive",
            "corrected": "negative",
            "ui_session": "session-a",
            "note": "fix label",
        },
        store=store,
        limiter=limiter,
    )

    assert status == 200
    assert body["accepted"] is True
    assert "stored_at" in body

    files = list((tmp_path / "intent_feedback").glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ts_ns"] == 1_717_000_000_000_000_000
    assert record["predicted"] == "positive"
    assert record["corrected"] == "negative"


def test_intent_feedback_rate_limit_same_session_and_ts_window(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    first_ts = 1_717_000_000_000_000_000

    status1, body1 = _run_post(
        {
            "ts_ns": first_ts,
            "predicted": "none",
            "corrected": "positive",
            "ui_session": "session-a",
        },
        store=store,
        limiter=limiter,
    )
    status2, body2 = _run_post(
        {
            "ts_ns": first_ts + 500_000_000,
            "predicted": "none",
            "corrected": "negative",
            "ui_session": "session-a",
        },
        store=store,
        limiter=limiter,
    )

    assert status1 == 200 and body1["accepted"] is True
    assert status2 == 429
    assert body2["accepted"] is False
    assert body2["error"] == "rate_limited"


def test_intent_feedback_endpoint_window_attached(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    inferencer = RealtimeLNNInferencer(ckpt_path=None, window=2)
    inferencer._enabled = True
    inferencer._state_model = _DummyStateModel()
    inferencer._history = deque(maxlen=2)
    inferencer._window = 2
    for idx in range(1, 11):
        inferencer.infer(_signal_row(frame_id=idx))

    status, _body = _run_post(
        {
            "ts_ns": 1_717_000_000_000_000_000,
            "predicted": "positive",
            "corrected": "negative",
            "ui_session": "session-window",
        },
        store=store,
        limiter=limiter,
        inferencer=inferencer,  # type: ignore[arg-type]
    )

    assert status == 200
    files = list((tmp_path / "intent_feedback").glob("*.jsonl"))
    record = json.loads(files[0].read_text(encoding="utf-8").strip())
    assert "hidden_state_window" in record
    assert "au_window" in record
    assert len(record["hidden_state_window"]) >= 1
    assert len(record["hidden_state_window"][-1]["h"]) == 64
    assert len(record["au_window"][-1]["au"]) == 18


def test_admin_trigger_training_endpoint_returns_marker(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    status, body = _run_post(
        payload={},
        store=store,
        limiter=limiter,
        path="/admin/trigger_training",
    )

    assert status == 200
    assert body["triggered"] is True
    assert isinstance(body["marker_at"], str)
