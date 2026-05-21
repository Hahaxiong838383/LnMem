"""Tests for hidden-state collapse monitor."""
from __future__ import annotations

import torch

from poc.monitoring.collapse_monitor import CollapseMonitor


def test_all_zero_hidden_is_critical() -> None:
    """all-zero hidden states should trigger CRITICAL collapse."""
    monitor = CollapseMonitor(var_threshold=0.01, dist_threshold=0.1)
    hidden = torch.zeros(64, 32)

    metrics = monitor.step(hidden, batch_metadata={})

    assert metrics["collapse_severity"] == "CRITICAL"
    assert float(metrics["var"]) < 0.01


def test_random_gaussian_hidden_is_ok() -> None:
    """random Gaussian hidden states should stay out of collapse region."""
    torch.manual_seed(0)
    monitor = CollapseMonitor(var_threshold=0.01, dist_threshold=0.1)
    hidden = torch.randn(64, 32)

    metrics = monitor.step(hidden, batch_metadata={})

    assert metrics["collapse_severity"] == "OK"
    assert float(metrics["var"]) > 0.01
    assert float(metrics["pairwise_dist"]) > 0.1


def test_two_cluster_hidden_hits_high() -> None:
    """2-cluster embeddings should trigger HIGH via low pairwise distance."""
    torch.manual_seed(1)
    monitor = CollapseMonitor(var_threshold=0.01, dist_threshold=1.0)

    c1 = torch.randn(32, 32) * 0.10 + 0.05
    c2 = torch.randn(32, 32) * 0.10 - 0.05
    hidden = torch.cat([c1, c2], dim=0)

    metrics = monitor.step(hidden, batch_metadata={})

    assert metrics["collapse_severity"] == "HIGH"
    assert 0.01 < float(metrics["var"]) < 0.2
    assert float(metrics["pairwise_dist"]) < 1.0
