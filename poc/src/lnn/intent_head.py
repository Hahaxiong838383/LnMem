"""Intent classification head for online LNN hidden states."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

INTENT_LABELS: tuple[str, str, str] = ("positive", "negative", "none")
INTENT_TO_INDEX: dict[str, int] = {label: idx for idx, label in enumerate(INTENT_LABELS)}


class IntentClassifierHead(nn.Module):
    """Map LNN hidden state to 3-class intent logits."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        proj_dim = max(1, hidden_dim // 2)
        self.hidden_dim = hidden_dim
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, len(INTENT_LABELS)),
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        """Return intent logits with shape (B, 3)."""
        if hidden_state.ndim != 2:
            raise ValueError(f"hidden_state must be 2D (B, H), got {tuple(hidden_state.shape)}")
        if hidden_state.shape[1] != self.hidden_dim:
            raise ValueError(
                f"hidden_state dim mismatch: expected {self.hidden_dim}, got {hidden_state.shape[1]}"
            )
        return self.classifier(hidden_state)


def cost_sensitive_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    none_class_weight: float = 0.5,
    pos_neg_weight: float = 1.0,
) -> Tensor:
    """Compute weighted CE where class 'none' is down-weighted."""
    if logits.ndim != 2 or logits.shape[1] != len(INTENT_LABELS):
        raise ValueError(f"logits must be (B, {len(INTENT_LABELS)}), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (B,), got {tuple(targets.shape)}")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("targets batch size must match logits batch size")

    weights = torch.tensor(
        [pos_neg_weight, pos_neg_weight, none_class_weight],
        dtype=logits.dtype,
        device=logits.device,
    )
    return F.cross_entropy(logits, targets.to(device=logits.device, dtype=torch.long), weight=weights)


__all__ = [
    "INTENT_LABELS",
    "INTENT_TO_INDEX",
    "IntentClassifierHead",
    "cost_sensitive_cross_entropy",
]
