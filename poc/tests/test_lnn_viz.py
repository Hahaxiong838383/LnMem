"""Day 5 LNN visualization tests — PNG generation smoke tests."""
from __future__ import annotations

from pathlib import Path

import torch

from poc.src.lnn.contracts import EpochMetric
from poc.src.lnn.viz import MatplotlibViz


def test_loss_curve_creates_png(tmp_path: Path) -> None:
    """plot_loss_curve should create a PNG file."""
    viz = MatplotlibViz()
    metrics = tuple(
        EpochMetric(
            epoch=i,
            train_loss=1.0 / (i + 1),
            val_loss=1.1 / (i + 1),
            train_state_acc=0.3 + 0.05 * i,
            val_state_acc=0.28 + 0.05 * i,
        )
        for i in range(10)
    )
    out = tmp_path / "loss.png"
    viz.plot_loss_curve(metrics, out)

    assert out.exists(), "PNG not created"
    assert out.stat().st_size > 1000, "PNG too small — likely empty"


def test_tsne_creates_png(tmp_path: Path) -> None:
    """plot_tsne_personalization should create a PNG file."""
    viz = MatplotlibViz()
    n_personas = 10
    emb = torch.randn(n_personas, 8)
    p2i = {f"vboss_{i:03d}": i for i in range(n_personas)}

    out = tmp_path / "tsne.png"
    viz.plot_tsne_personalization(emb, p2i, out)

    assert out.exists(), "PNG not created"
    assert out.stat().st_size > 1000, "PNG too small — likely empty"


def test_tsne_with_few_personas(tmp_path: Path) -> None:
    """t-SNE should work with very few personas (perplexity clamped)."""
    viz = MatplotlibViz()
    n_personas = 3
    emb = torch.randn(n_personas, 4)
    p2i = {f"vboss_{i:03d}": i for i in range(n_personas)}

    out = tmp_path / "tsne_small.png"
    viz.plot_tsne_personalization(emb, p2i, out)
    assert out.exists()
