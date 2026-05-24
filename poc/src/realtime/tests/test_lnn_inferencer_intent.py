"""Tests for intent outputs in RealtimeLNNInferencer."""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.intent_head import IntentClassifierHead
from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer


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
    aus = {f"AU{i:02d}": 0.3 for i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]}
    return {
        "ts": "2026-06-01T00:00:00Z",
        "frame_id": frame_id,
        "face_detected": True,
        "raw_au": dict(aus),
        "calibrated_au": dict(aus),
        "expression": {
            "brow_furrow": 0.3,
            "smile_intensity": 0.2,
            "eye_closure": 0.1,
            "cheek_raise": 0.2,
            "lip_corner_depress": 0.1,
            "jaw_drop": 0.0,
            "mouth_open_ratio": 0.0,
        },
        "head_pose": {"pitch_deg": 2.0, "yaw_deg": 1.0, "roll_deg": -1.0},
    }


def test_infer_outputs_intent_fields_when_head_loaded() -> None:
    inferencer = RealtimeLNNInferencer(ckpt_path=None, window=2)
    inferencer._enabled = True
    inferencer._state_model = _DummyStateModel()
    inferencer._history = deque(maxlen=2)
    inferencer._window = 2

    head = IntentClassifierHead(hidden_dim=64)
    with torch.no_grad():
        head.classifier[0].weight.zero_()
        head.classifier[0].bias.zero_()
        head.classifier[3].weight.zero_()
        head.classifier[3].bias.copy_(torch.tensor([3.0, -2.0, 0.0]))
    head.eval()
    inferencer._intent_head = head

    warm = inferencer.infer(_signal_row(frame_id=1))
    assert warm.lnn_status == "warmup"

    out = inferencer.infer(_signal_row(frame_id=2))
    assert out.lnn_status == "ok"
    assert out.intent_probs is not None and len(out.intent_probs) == 3
    assert out.intent_pred == "positive"
    assert out.intent_confidence is not None and out.intent_confidence > 0.8


def test_intent_head_ckpt_loading_works(tmp_path: Path) -> None:
    inferencer = RealtimeLNNInferencer(ckpt_path=None)
    inferencer._state_model = _DummyStateModel()

    head = IntentClassifierHead(hidden_dim=64)
    ckpt = tmp_path / "intent_head.pt"
    torch.save({"state_dict": head.state_dict()}, ckpt)

    inferencer._load_intent_head_ckpt(ckpt)

    assert inferencer._intent_head is not None
    assert inferencer._intent_head_ckpt_sha8 is not None
