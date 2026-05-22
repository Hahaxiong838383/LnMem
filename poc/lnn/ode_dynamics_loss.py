"""ADR-023 v2 latent world-model rollout loss."""
from __future__ import annotations

import dataclasses
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class OdeDynamicsLossConfig:
    """Configuration for v2 rollout loss. / v2 rollout 损失配置。"""

    w_world_model: float = 1.0
    w_recon_prefix: float = 0.5
    prefix_ratio: float = 0.5
    rollout_horizon: int = 15
    detach_prefix_grad: bool = False


class Decoder(nn.Module):
    """Decode hidden state to input space. / hidden 到输入空间解码器。"""

    def __init__(self, hidden_dim: int = 64, input_dim: int = 52, num_layers: int = 1) -> None:
        super().__init__()
        if num_layers == 1:
            self.net: nn.Module = nn.Linear(hidden_dim, input_dim)
        elif num_layers == 2:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )
        else:
            raise ValueError(f"num_layers must be 1 or 2, got {num_layers}")

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Forward decode. / 前向解码。"""
        return self.net(h)


class OdeDynamicsLoss(nn.Module):
    """Latent world-model rollout objective (ADR-023 v2)."""

    def __init__(self, cfg: OdeDynamicsLossConfig | None = None, decoder: Decoder | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OdeDynamicsLossConfig()
        self.decoder = decoder or Decoder()

    def forward(
        self,
        x: torch.Tensor,
        backbone_step_fn: Callable[[torch.Tensor, list[torch.Tensor], torch.Tensor], list[torch.Tensor]],
        persona_idx: torch.Tensor,
        init_hidden_fn: Callable[[int, torch.device], list[torch.Tensor]],
    ) -> dict[str, torch.Tensor | int]:
        """Compute v2 loss. / 计算 v2 loss。"""
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, D), got {tuple(x.shape)}")
        batch_size, total_steps, _ = x.shape
        if total_steps < 2:
            raise ValueError("total_steps must be >= 2 for prefix + rollout")

        prefix_len = int(total_steps * self.cfg.prefix_ratio)
        prefix_len = max(1, min(prefix_len, total_steps - 1))
        rollout_len = min(self.cfg.rollout_horizon, total_steps - prefix_len)
        if rollout_len <= 0:
            raise ValueError("rollout_len must be > 0")

        h_states = init_hidden_fn(batch_size, x.device)
        if not h_states:
            raise ValueError("init_hidden_fn must return non-empty hidden state list")

        h_seq_prefix: list[torch.Tensor] = []
        x_recon_prefix: list[torch.Tensor] = []
        for t in range(prefix_len):
            h_states = backbone_step_fn(x[:, t, :], h_states, persona_idx)
            h_last = h_states[-1]
            h_seq_prefix.append(h_last)
            x_recon_prefix.append(self.decoder(h_last))

        h_seq_prefix_tensor = torch.stack(h_seq_prefix, dim=1)
        x_recon_prefix_tensor = torch.stack(x_recon_prefix, dim=1)
        x_prefix_target = x[:, :prefix_len, :]
        l_recon_prefix = F.mse_loss(x_recon_prefix_tensor, x_prefix_target)

        if self.cfg.detach_prefix_grad:
            h_pred_states = [h.detach() for h in h_states]
        else:
            h_pred_states = h_states

        x_pred_rollout: list[torch.Tensor] = []
        for _ in range(rollout_len):
            # v2 hard rule: rollout input must be decoder(h_pred), never ground-truth x.
            x_pred_t = self.decoder(h_pred_states[-1])
            x_pred_rollout.append(x_pred_t)
            h_pred_states = backbone_step_fn(x_pred_t, h_pred_states, persona_idx)

        x_pred_rollout_tensor = torch.stack(x_pred_rollout, dim=1)
        x_rollout_target = x[:, prefix_len : prefix_len + rollout_len, :]
        l_world_model = F.mse_loss(x_pred_rollout_tensor, x_rollout_target)
        loss = self.cfg.w_world_model * l_world_model + self.cfg.w_recon_prefix * l_recon_prefix

        return {
            "loss": loss,
            "L_world_model": l_world_model,
            "L_recon_prefix": l_recon_prefix,
            "prefix_len": prefix_len,
            "rollout_len": rollout_len,
            "x_pred_rollout": x_pred_rollout_tensor,
            "h_seq_prefix": h_seq_prefix_tensor,
        }
