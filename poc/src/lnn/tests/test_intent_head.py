"""Tests for intent classifier head."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.intent_head import IntentClassifierHead, cost_sensitive_cross_entropy


def test_forward_shape() -> None:
    """Head returns (B,3) logits for (B,H) hidden states."""
    head = IntentClassifierHead(hidden_dim=64)
    x = torch.randn(5, 64)
    logits = head(x)
    assert logits.shape == (5, 3)


def test_cost_sensitive_loss_matches_weighted_ce() -> None:
    """Loss equals torch weighted CE with none class weight=0.5."""
    logits = torch.tensor([[2.0, -1.0, 0.5], [0.1, 0.2, 0.3]], dtype=torch.float32)
    targets = torch.tensor([0, 2], dtype=torch.long)
    got = cost_sensitive_cross_entropy(logits, targets)
    expected = F.cross_entropy(
        logits,
        targets,
        weight=torch.tensor([1.0, 1.0, 0.5], dtype=torch.float32),
    )
    assert torch.allclose(got, expected)


def test_cost_sensitive_downweights_none() -> None:
    """Batch loss should shrink when high-loss sample belongs to none class."""
    logits = torch.tensor(
        [
            [5.0, -2.0, -2.0],  # target 0 -> very low loss
            [4.0, -1.0, -3.0],  # target 2 -> very high loss
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    weighted = cost_sensitive_cross_entropy(logits, targets)
    unweighted = F.cross_entropy(logits, targets)

    assert weighted.item() < unweighted.item()
