"""ADR-023 v2 latent world-model rollout loss."""
from __future__ import annotations

import dataclasses
import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


def clamp_tau(cell: nn.Module, upper_bound: float) -> int:
    """Clamp log-space tau_base upper bound and return clamped parameter count."""
    if upper_bound <= 0.0:
        raise ValueError(f"upper_bound must be > 0, got {upper_bound}")
    tau_base = getattr(cell, "tau_base", None)
    if tau_base is None:
        return 0
    max_value = math.log(float(upper_bound))
    with torch.no_grad():
        before = tau_base.data.clone()
        tau_base.data.clamp_(max=max_value)
        return int((before > max_value).sum().item())


@dataclasses.dataclass(frozen=True)
class OdeDynamicsLossConfig:
    """Configuration for v2 rollout loss. / v2 rollout 损失配置。

    v2.1 整改（cc 第 11 轮 + gemini 第 10 轮 F-3 应验后）：
    - 加 w_motion 直击 trivial 均值预测路径
    - 默认 detach_prefix_grad=True（让 rollout 学独立动力学，不污染 prefix encoder）
    - rollout_horizon 15 → 24（prefix=6, rollout=24, seq_len=30，让短期预测不够）
    - prefix_ratio 0.5 → 0.2
    """

    w_world_model: float = 1.0
    w_recon_prefix: float = 0.5
    # M2 撤回（gemini 第 11 轮指出: diff(actual)≈0 时会驱使 ODE 输出绝对平直，加剧 shortcut）
    w_motion: float = 0.0
    prefix_ratio: float = 0.2   # M4: prefix=6 短，rollout=24 长（动力学主导）
    rollout_horizon: int = 24   # M4: K=24 让短期均值预测不够
    # M5 撤回（gemini 第 11 轮指出: detach 断绝 encoder 梯度，毁灭性毒药）
    detach_prefix_grad: bool = False
    # 第 11 轮 gemini 钦定核心解：时间重采样 stride=3（6Hz→2Hz）拉大相邻帧差异破除准静态死局
    # DEPRECATED: v2.2 起建议用 dt_target + fps 推导 stride；保留仅用于 backward compat。
    input_stride: int = 3
    # v2.2 dt-aware rollout（TODO-1）
    dt_target: float = 0.1
    fps_default: float = 30.0
    use_dt_aware: bool = True


class Decoder(nn.Module):
    """Decode hidden state to input space. / hidden 到输入空间解码器。

    第 11 轮 gemini 钦定：必须有低秩瓶颈（bottleneck=16）截断 hidden→input 无损直通。
    防 cc M3 weight_decay 治标不治本（h≈Wx encoder shortcut 仍在）。
    """

    def __init__(self, hidden_dim: int = 64, input_dim: int = 52, num_layers: int = 1,
                 bottleneck_dim: int = 16) -> None:
        super().__init__()
        if num_layers == 1:
            self.net: nn.Module = nn.Linear(hidden_dim, input_dim)
        elif num_layers == 2:
            # 第 11 轮 gemini 钦定瓶颈：hidden(64) → 16 → 52
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, bottleneck_dim),
                nn.ReLU(),
                nn.Linear(bottleneck_dim, input_dim),
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
        backbone_step_fn: Callable[..., list[torch.Tensor]],
        persona_idx: torch.Tensor,
        init_hidden_fn: Callable[[int, torch.device], list[torch.Tensor]],
        fps: float | None = None,
    ) -> dict[str, torch.Tensor | int | float]:
        """Compute v2 loss. / 计算 v2 loss。"""
        if x.dim() != 3:
            raise ValueError(f"x must be (B, T, D), got {tuple(x.shape)}")
        actual_dt: float | None = None
        if self.cfg.use_dt_aware:
            fps_value = float(self.cfg.fps_default if fps is None else fps)
            if fps_value <= 0.0:
                raise ValueError(f"fps must be > 0, got {fps_value}")
            # F2: 采样定理约束，实际步长不得小于 1/fps。
            raw_dt = float(self.cfg.dt_target)
            actual_dt = max(raw_dt, 1.0 / fps_value)
            stride = max(1, round(fps_value * actual_dt))
            actual_dt = float(stride / fps_value)
            if actual_dt > self.cfg.dt_target * 1.5:
                # cc 修补：节流 print（once-per-fps_value），1.6M 行 stdout I/O 拖死训练
                _w_set = getattr(OdeDynamicsLoss, "_warned_fps", None)
                if _w_set is None:
                    _w_set = set()
                    setattr(OdeDynamicsLoss, "_warned_fps", _w_set)
                if fps_value not in _w_set:
                    _w_set.add(fps_value)
                    print(
                        "[dt-aware] WARNING (once-per-fps): "
                        f"actual_dt={actual_dt:.3f} >> dt_target={self.cfg.dt_target:.3f}, "
                        f"fps={fps_value} 采样率不足",
                        flush=True,
                    )
            if stride > 1:
                x = x[:, ::stride, :]
        else:
            # backward compat: 旧行为完全保留，仍用 input_stride，且不向 backbone 传 dt。
            if self.cfg.input_stride > 1:
                x = x[:, :: self.cfg.input_stride, :]
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
            if self.cfg.use_dt_aware:
                h_states = backbone_step_fn(x[:, t, :], h_states, persona_idx, dt=float(actual_dt))
            else:
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
            if self.cfg.use_dt_aware:
                h_pred_states = backbone_step_fn(x_pred_t, h_pred_states, persona_idx, dt=float(actual_dt))
            else:
                h_pred_states = backbone_step_fn(x_pred_t, h_pred_states, persona_idx)

        x_pred_rollout_tensor = torch.stack(x_pred_rollout, dim=1)
        x_rollout_target = x[:, prefix_len : prefix_len + rollout_len, :]
        l_world_model = F.mse_loss(x_pred_rollout_tensor, x_rollout_target)

        # M2 motion loss: 强制预测变化（diff）而非均值（gemini 第 10 轮 F-3 应验后整改）
        # 关键：sin trivial 均值预测让 diff(x_pred) ≈ 0，但 diff(x_actual) != 0 → 大惩罚
        if rollout_len >= 2:
            diff_pred = x_pred_rollout_tensor[:, 1:, :] - x_pred_rollout_tensor[:, :-1, :]
            diff_target = x_rollout_target[:, 1:, :] - x_rollout_target[:, :-1, :]
            l_motion = F.mse_loss(diff_pred, diff_target)
        else:
            l_motion = torch.tensor(0.0, device=x.device, requires_grad=True)

        loss = (
            self.cfg.w_world_model * l_world_model
            + self.cfg.w_recon_prefix * l_recon_prefix
            + self.cfg.w_motion * l_motion
        )

        out: dict[str, torch.Tensor | int | float] = {
            "loss": loss,
            "L_world_model": l_world_model,
            "L_recon_prefix": l_recon_prefix,
            "L_motion": l_motion,
            "prefix_len": prefix_len,
            "rollout_len": rollout_len,
            "x_pred_rollout": x_pred_rollout_tensor,
            "h_seq_prefix": h_seq_prefix_tensor,
        }
        if self.cfg.use_dt_aware and actual_dt is not None:
            out["actual_dt"] = float(actual_dt)
            out["total_physical_horizon_s"] = float(rollout_len * actual_dt)
        return out
