"""Tests for LTC ODE future prediction endpoint and inferencer."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.daemon_main import DaemonApp  # noqa: E402
from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer  # noqa: E402


class _DummyPersonalization:
    def get_tau_bias(self, persona_idx: torch.Tensor) -> torch.Tensor:
        return torch.zeros((persona_idx.shape[0], 64), dtype=torch.float32)

    def get_film_params(self, persona_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        zeros = torch.zeros((persona_idx.shape[0], 64), dtype=torch.float32)
        return zeros, zeros


class _DummyBackbone:
    def rnn_cell(
        self,
        features: torch.Tensor,
        hidden: torch.Tensor,
        timespans: torch.Tensor,
    ) -> tuple[torch.Tensor, None]:
        _ = timespans
        return hidden + 0.05 * torch.tanh(features) + 0.01, None


class _DummyStateModel:
    def __init__(self) -> None:
        self.hidden_dim = 64
        self.input_projection = torch.nn.Linear(18, 64)
        self.personalization = _DummyPersonalization()
        self.backbone = _DummyBackbone()
        self.readout = torch.nn.Linear(64, 8)
        with torch.no_grad():
            self.input_projection.weight.fill_(0.01)
            self.input_projection.bias.zero_()
            self.readout.weight.fill_(0.02)
            self.readout.bias.copy_(torch.tensor([0.1, 0.8, -0.2, 0.0, 0.3, -0.1, 0.2, 0.05]))

    def _timespans_from_tau(self, tau_bias: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(tau_bias)

    def _apply_film(self, hidden: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
        _ = gamma, beta
        return hidden


def _inferencer(intent_head: torch.nn.Module | None = None) -> RealtimeLNNInferencer:
    inferencer = RealtimeLNNInferencer(ckpt_path=None, window=8)
    inferencer._enabled = True
    inferencer._state_model = _DummyStateModel()
    inferencer._intent_head = intent_head
    inferencer._history = deque(
        [[0.1 + frame * 0.01 for _ in range(18)] for frame in range(8)],
        maxlen=8,
    )
    inferencer._ckpt_sha8 = "deadbeef"
    return inferencer


def _intent_head() -> torch.nn.Linear:
    head = torch.nn.Linear(64, 3)
    with torch.no_grad():
        head.weight.fill_(0.01)
        head.bias.copy_(torch.tensor([0.4, -0.2, 0.1]))
    return head


def test_predict_future_returns_five_normalized_frames_with_intent() -> None:
    inferencer = _inferencer(intent_head=_intent_head())

    predictions = inferencer.predict_future(5)

    assert len(predictions) == 5
    for idx, pred in enumerate(predictions):
        assert pred["step"] == idx
        assert "error" not in pred
        assert len(pred["state_probs"]) == 8
        assert sum(pred["state_probs"]) == pytest.approx(1.0, abs=0.01)
        assert pred["top_prob"] >= pred["margin"] >= 0.0
        assert pred["entropy"] >= 0.0
        assert pred["intent_probs"] is not None
        assert len(pred["intent_probs"]) == 3
        assert sum(pred["intent_probs"]) == pytest.approx(1.0, abs=0.01)
        assert pred["intent_pred"] is not None
        assert pred["intent_confidence"] is not None


def test_predict_future_without_intent_head_returns_none_intent_fields() -> None:
    inferencer = _inferencer(intent_head=None)

    predictions = inferencer.predict_future(5)

    assert len(predictions) == 5
    for pred in predictions:
        assert pred["intent_probs"] is None
        assert pred["intent_pred"] is None
        assert pred["intent_confidence"] is None


def test_predict_future_disabled_inferencer_returns_error_dict() -> None:
    inferencer = RealtimeLNNInferencer(ckpt_path=None)

    assert inferencer.predict_future(5) == [{"error": "disabled"}]


def test_daemon_predict_future_disabled_by_env(monkeypatch: pytest.MonkeyPatch) -> None:
    app = DaemonApp.__new__(DaemonApp)
    app._lnn_inferencer = _inferencer(intent_head=_intent_head())
    monkeypatch.setenv("LNMEM_DISABLE_PREDICT_FUTURE", "1")

    code, body = app.handle_predict_future(5)

    assert code == 503
    assert body["error"] == "disabled by env"


def test_daemon_predict_future_response_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    app = DaemonApp.__new__(DaemonApp)
    app._lnn_inferencer = _inferencer(intent_head=_intent_head())
    monkeypatch.delenv("LNMEM_DISABLE_PREDICT_FUTURE", raising=False)

    code, body = app.handle_predict_future(65)

    assert code == 200
    assert body["steps"] == 60
    assert body["ckpt_sha8"] == "deadbeef"
    assert body["method"] == "ltc_ode_persistence"
    assert isinstance(body["ts_ns"], int)
    assert len(body["predictions"]) == 60
