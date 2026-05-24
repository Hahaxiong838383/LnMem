"""Intent classification head and losses for online LNN hidden states."""

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


def _validate_logits_targets(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim != 2 or logits.shape[1] != len(INTENT_LABELS):
        raise ValueError(f"logits must be (B, {len(INTENT_LABELS)}), got {tuple(logits.shape)}")
    if targets.ndim != 1:
        raise ValueError(f"targets must be 1D (B,), got {tuple(targets.shape)}")
    if targets.shape[0] != logits.shape[0]:
        raise ValueError("targets batch size must match logits batch size")
    return targets.to(device=logits.device, dtype=torch.long)


def dynamic_focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Tensor | list[float] | tuple[float, ...] | None = None,
) -> Tensor:
    """Compute focal loss with dynamic alpha from batch class distribution."""
    targets_long = _validate_logits_targets(logits, targets)

    if alpha is None:
        counts = torch.bincount(targets_long, minlength=len(INTENT_LABELS)).to(dtype=logits.dtype)
        counts = torch.clamp(counts, min=1.0)
        inv = counts.sum() / counts
        alpha_tensor = inv / inv.sum()
    else:
        alpha_tensor = torch.as_tensor(alpha, dtype=logits.dtype, device=logits.device)
        if alpha_tensor.numel() != len(INTENT_LABELS):
            raise ValueError(f"alpha must have {len(INTENT_LABELS)} elements")
        alpha_tensor = alpha_tensor / torch.clamp(alpha_tensor.sum(), min=1e-12)

    alpha_tensor = alpha_tensor.to(device=logits.device, dtype=logits.dtype)

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    sample_idx = torch.arange(logits.shape[0], device=logits.device)
    pt = probs[sample_idx, targets_long]
    log_pt = log_probs[sample_idx, targets_long]
    alpha_t = alpha_tensor[targets_long]
    focal_factor = torch.pow(1.0 - pt, gamma)
    loss = -alpha_t * focal_factor * log_pt
    return loss.mean()


def cost_sensitive_cross_entropy(
    logits: Tensor,
    targets: Tensor,
    none_class_weight: float = 0.5,
    pos_neg_weight: float = 1.0,
) -> Tensor:
    """Compute weighted CE where class 'none' is down-weighted."""
    targets_long = _validate_logits_targets(logits, targets)
    weights = torch.tensor(
        [pos_neg_weight, pos_neg_weight, none_class_weight],
        dtype=logits.dtype,
        device=logits.device,
    )
    return F.cross_entropy(logits, targets_long, weight=weights)


def default_intent_loss(logits: Tensor, targets: Tensor, gamma: float = 2.0) -> Tensor:
    """Default training loss for intent head (retrofit H2: Dynamic Focal Loss)."""
    return dynamic_focal_loss(logits, targets, gamma=gamma, alpha=None)


__all__ = [
    "INTENT_LABELS",
    "INTENT_TO_INDEX",
    "IntentClassifierHead",
    "dynamic_focal_loss",
    "cost_sensitive_cross_entropy",
    "default_intent_loss",
]
