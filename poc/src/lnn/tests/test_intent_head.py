"""Tests for intent classifier head."""

from __future__ import annotations

from pathlib import Path
import sys

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.intent_head import (  # noqa: E402
    IntentClassifierHead,
    cost_sensitive_cross_entropy,
    dynamic_focal_loss,
)


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
            [5.0, -2.0, -2.0],
            [4.0, -1.0, -3.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 2], dtype=torch.long)

    weighted = cost_sensitive_cross_entropy(logits, targets)
    unweighted = F.cross_entropy(logits, targets)

    assert weighted.item() < unweighted.item()


def test_dynamic_focal_loss_handles_imbalance() -> None:
    """Dynamic focal should reduce none-class dominance in highly imbalanced batch."""
    classes = [0] + [2] * 100 + [1]
    targets = torch.tensor(classes, dtype=torch.long)

    logits = torch.full((len(classes), 3), -1.5, dtype=torch.float32)
    logits[:, 2] = 4.5
    logits[0, 0] = -2.0
    logits[-1, 1] = -2.0

    ce_per_sample = F.cross_entropy(logits, targets, reduction="none")
    ce_none_mean = ce_per_sample[targets == 2].mean().item()

    counts = torch.bincount(targets, minlength=3).to(torch.float32)
    inv = counts.sum() / torch.clamp(counts, min=1.0)
    alpha = inv / inv.sum()

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    idx = torch.arange(targets.shape[0])
    pt = probs[idx, targets]
    focal_per_sample = -alpha[targets] * ((1.0 - pt) ** 2.0) * log_probs[idx, targets]
    focal_none_mean = focal_per_sample[targets == 2].mean().item()

    assert dynamic_focal_loss(logits, targets).item() > 0.0
    assert focal_none_mean < ce_none_mean * 0.5
