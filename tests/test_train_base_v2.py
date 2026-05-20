"""Tests for train_base_v2 helpers; 覆盖 fusion 映射、soft label、balanced sampler。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_base_v2.py"
_SPEC = importlib.util.spec_from_file_location("train_base_v2", _SCRIPT)
assert _SPEC and _SPEC.loader
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_veatic_au_to_fusion_18_shape() -> None:
    """Full AU row should map to 18 dims within [0, 1]."""
    row = {au: 0.4 for au in mod.AU_NAMES}
    row.update(
        {
            "AU43": 0.7,
            "AU12": 0.8,
            "AU04": 0.3,
            "AU06": 0.6,
            "AU15": 0.2,
            "AU26": 0.5,
            "AU25": 0.4,
            "face_detected": True,
            "Pitch": 10.0,
            "Yaw": 15.0,
        }
    )
    vector = mod._veatic_au_to_fusion_18(row)
    assert len(vector) == 18
    assert all(0.0 <= value <= 1.0 for value in vector)


def test_soft_assignment_at_boundary_is_smooth() -> None:
    """Soft labels should change smoothly when V-A drifts across a sector boundary."""
    valence = torch.full((41, 1), 0.5, dtype=torch.float32)
    arousal = torch.linspace(0.45, 0.55, steps=41, dtype=torch.float32).unsqueeze(-1)
    va = torch.cat([valence, arousal], dim=-1)
    labels = mod.build_soft_state_targets(va, torch.tensor(0.35), r_neutral=0.35, theta_offset_deg=0.0)
    diffs = labels[1:] - labels[:-1]
    max_gradient = float(diffs.abs().max().item())
    assert max_gradient < 0.1


def test_class_balanced_sampler_distribution() -> None:
    """Inverse-frequency sampler should upsample rare states to a near-uniform mix."""
    sample_states = [0] * 1000 + [1] * 10
    for state in range(2, 9):
        sample_states.extend([state] * 100)
    sampler = mod.build_class_balanced_sampler(sample_states, num_classes=9, seed=42)
    sampled_indices = list(iter(sampler))
    sampled_states = np.array([sample_states[index] for index in sampled_indices], dtype=np.int64)
    counts = np.bincount(sampled_states, minlength=9)
    proportions = counts / counts.sum()
    assert len(sampled_indices) == len(sample_states)
    assert np.all(proportions >= 0.05)
