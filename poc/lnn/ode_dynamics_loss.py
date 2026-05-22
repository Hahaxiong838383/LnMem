"""ODE dynamics losses for LNN training; ADR-023 实施入口."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import InfoNCELoss, InfoNCEConfig


@dataclass(frozen=True)
class OdeDynamicsLossConfig:
    """Config for weighted ODE dynamics losses（ADR-023 组合权重）."""

    w_predict: float = 1.0
    w_recon: float = 1.0
    w_ode: float = 0.1
    w_infonce_aux: float = 0.1
    predict_horizon: int = 5


class Decoder(nn.Module):
    """Decode latent sequence back to input space（64d -> 52d）."""

    def __init__(
        self,
        hidden_dim: int = 64,
        input_dim: int = 52,
        num_layers: int = 1,
    ) -> None:
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(hidden_dim, input_dim)
        elif num_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            raise ValueError("num_layers must be 1 or 2")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Project hidden sequence into reconstruction target space."""
        return self.net(h)


class OdeDynamicsLoss(nn.Module):
    """ADR-023 主损失：L_predict + L_recon + L_ode + L_infonce_aux."""

    def __init__(
        self,
        cfg: OdeDynamicsLossConfig | None = None,
        decoder: Decoder | None = None,
    ) -> None:
        super().__init__()
        self.cfg = cfg or OdeDynamicsLossConfig()
        self.decoder = decoder or Decoder()
        self.infonce_aux = InfoNCELoss(InfoNCEConfig(learnable_sigma=False))

    def forward(
        self,
        x: torch.Tensor,
        h_seq: torch.Tensor,
        backbone_step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        persona_idx: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute all ODE dynamics losses and return per-term dict."""
        if x.dim() != 3 or h_seq.dim() != 3:
            raise ValueError("x and h_seq must be 3D tensors with shape (B, T, D)")
        if x.size(0) != h_seq.size(0) or x.size(1) != h_seq.size(1):
            raise ValueError("x and h_seq must share batch and time dimensions")
        if persona_idx.dim() != 1 or persona_idx.size(0) != x.size(0):
            raise ValueError("persona_idx must be shape (B,)")

        recon = self.decoder(h_seq)
        l_recon = F.mse_loss(recon, x)
        l_predict = self._compute_predict_loss(x, h_seq, backbone_step_fn, persona_idx)
        l_ode = self._compute_ode_consistency_loss(x, h_seq, backbone_step_fn, persona_idx)
        l_infonce_aux = self._compute_infonce_aux(h_seq)

        total = (
            self.cfg.w_predict * l_predict
            + self.cfg.w_recon * l_recon
            + self.cfg.w_ode * l_ode
            + self.cfg.w_infonce_aux * l_infonce_aux
        )
        return {
            "loss": total,
            "L_predict": l_predict.detach(),
            "L_recon": l_recon.detach(),
            "L_ode": l_ode.detach(),
            "L_infonce_aux": l_infonce_aux.detach(),
        }

    def _compute_predict_loss(
        self,
        x: torch.Tensor,
        h_seq: torch.Tensor,
        backbone_step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        persona_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Multi-step rollout MSE from prefix end to future horizon."""
        _, steps, _ = h_seq.shape
        if steps <= 1:
            return h_seq.new_tensor(0.0, requires_grad=True)
        horizon = max(1, min(self.cfg.predict_horizon, steps - 1))
        start = steps - horizon - 1
        h_prev = h_seq[:, start, :]
        preds = []
        for k in range(horizon):
            t = start + 1 + k
            h_prev = backbone_step_fn(x[:, t, :], h_prev, persona_idx)
            preds.append(h_prev)
        pred = torch.stack(preds, dim=1)
        target = h_seq[:, start + 1 : start + 1 + horizon, :]
        return F.mse_loss(pred, target)

    def _compute_ode_consistency_loss(
        self,
        x: torch.Tensor,
        h_seq: torch.Tensor,
        backbone_step_fn: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
        persona_idx: torch.Tensor,
    ) -> torch.Tensor:
        """Discrete Euler self-consistency loss using one-step backbone dynamics."""
        _, steps, _ = h_seq.shape
        if steps <= 1:
            return h_seq.new_tensor(0.0, requires_grad=True)
        per_step_losses = []
        for t in range(steps - 1):
            h_next = backbone_step_fn(x[:, t, :], h_seq[:, t, :], persona_idx)
            predicted_dh = h_next - h_seq[:, t, :]
            expected_dh = h_seq[:, t + 1, :] - h_seq[:, t, :]
            per_step_losses.append(F.mse_loss(predicted_dh, expected_dh))
        return torch.stack(per_step_losses).mean()

    def _compute_infonce_aux(self, h_seq: torch.Tensor) -> torch.Tensor:
        """Auxiliary InfoNCE on adjacent hidden states with cheap in-batch negatives."""
        batch, steps, hidden = h_seq.shape
        if steps <= 1:
            return h_seq.new_tensor(0.0, requires_grad=True)
        anchor = h_seq[:, :-1, :].reshape(-1, hidden)
        positive = h_seq[:, 1:, :].reshape(-1, hidden)
        num_pairs = anchor.size(0)
        if num_pairs < 2:
            return h_seq.new_tensor(0.0, requires_grad=True)
        neg_shift = max(1, num_pairs // 2)
        neg_idx = (torch.arange(num_pairs, device=h_seq.device) + neg_shift) % num_pairs
        negatives = anchor[neg_idx].unsqueeze(1)
        return self.infonce_aux(anchor, positive, negatives)
