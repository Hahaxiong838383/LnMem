"""Day 5 training visualization: loss curves + persona t-SNE.

Implements ``TrainingViz`` Protocol from ``contracts.py``.

所有方法落盘 PNG 后 ``plt.close()``，零 matplotlib 全局状态污染。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — non-interactive backend before pyplot import
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.manifold import TSNE  # noqa: E402

from poc.src.lnn.contracts import EpochMetric


class MatplotlibViz:
    """Training visualizations — outputs PNG files, zero global state pollution.

    Satisfies ``TrainingViz`` Protocol in ``contracts.py``.
    """

    def plot_loss_curve(
        self,
        epoch_metrics: tuple[EpochMetric, ...],
        out_path: Path,
    ) -> None:
        """Dual Y-axis: left = train/val loss, right = train/val accuracy."""
        epochs = [m.epoch for m in epoch_metrics]
        train_loss = [m.train_loss for m in epoch_metrics]
        val_loss = [m.val_loss for m in epoch_metrics]
        train_acc = [m.train_state_acc for m in epoch_metrics]
        val_acc = [m.val_state_acc for m in epoch_metrics]

        fig, ax1 = plt.subplots(figsize=(10, 6))

        # Left Y-axis: loss (solid lines)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss", color="tab:blue")
        ax1.plot(epochs, train_loss, "b-", linewidth=1.5, label="Train Loss")
        ax1.plot(epochs, val_loss, "r-", linewidth=1.5, label="Val Loss")
        ax1.tick_params(axis="y", labelcolor="tab:blue")
        ax1.legend(loc="upper left")
        ax1.grid(True, alpha=0.3)

        # Right Y-axis: accuracy (dashed lines)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Accuracy", color="tab:green")
        ax2.plot(epochs, train_acc, "b--", linewidth=1.5, label="Train Acc")
        ax2.plot(epochs, val_acc, "r--", linewidth=1.5, label="Val Acc")
        ax2.tick_params(axis="y", labelcolor="tab:green")
        ax2.legend(loc="upper right")

        fig.suptitle("LNN Training: Loss & State Accuracy")
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)

    def plot_tsne_personalization(
        self,
        persona_embedding_weights: torch.Tensor,
        persona_to_idx: dict[str, int],
        out_path: Path,
    ) -> None:
        """t-SNE 2D scatter of persona embeddings with simplified labels."""
        weights = persona_embedding_weights.detach().cpu().numpy()
        n_personas = weights.shape[0]
        perplexity = min(15, n_personas - 1)

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=42,
        )
        coords = tsne.fit_transform(weights)

        # Build label mapping: idx -> short label ("vboss_001" -> "v001")
        idx_to_label: dict[int, str] = {}
        for persona_id, idx in persona_to_idx.items():
            short = persona_id[0] + persona_id.rsplit("_", 1)[-1]
            idx_to_label[idx] = short

        fig, ax = plt.subplots(figsize=(10, 8))
        ax.scatter(coords[:, 0], coords[:, 1], s=60, alpha=0.8, c="steelblue")

        for i in range(n_personas):
            label = idx_to_label.get(i, f"p{i}")
            ax.annotate(
                label,
                (coords[i, 0], coords[i, 1]),
                fontsize=8,
                ha="center",
                va="bottom",
                xytext=(0, 5),
                textcoords="offset points",
            )

        ax.set_title("Persona Embedding t-SNE (2D)")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)


__all__ = ["MatplotlibViz"]
