"""CustomCfCCell · γ-2 Path C 手写单步隐式欧拉求解器

源：spec §2.1 ODE 公式 + ADR-021 §2.3 伪代码（cc 实测 ncps Path A/B 都 WEAK 后启用）

数学锚（spec §2.1）：
    τ_i = τ_base_i * exp(persona_bias_i)        # per-neuron + per-persona τ
    τ * dh/dt = -h + f(x_t, h_prev)             # LTC/CfC ODE
    h_new = (τ * h_prev + dt * f) / (τ + dt)    # 单步隐式欧拉解析解

数值正确性（隐式欧拉的物理直觉）：
    - dt → 0    → h_new ≈ h_prev    （积分步长极小，hidden 几乎不变）
    - dt → ∞    → h_new ≈ f(x, h_prev) （完全更新到 f 的目标值）
    - τ ↑       → 反应迟缓（同样 dt 下 h_new 更接近 h_prev）
    - τ ↓       → 反应敏锐

verdict 门槛（spec §11.1 #12）：persona separation ratio > 1.5（严于 spike 阈值 1.1）
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class CustomCfCCellConfig:
    """spec §2.2 表格参数对齐."""
    input_dim: int
    hidden_dim: int
    num_personas: int = 64        # spec §2.1 persona embedding table 上限
    tau_init: float = 1.0         # base τ 初始值
    f_net_hidden: int = 128       # f_net 中间层（接近 ncps backbone_units=128）
    use_film: bool = False        # FiLM 个性化（留 γ-2 后续 task，本 cell 暂不实施）


class CustomCfCCell(nn.Module):
    """手写 CfC 单步 Cell · per-neuron + per-persona τ-bias 真进 ODE.

    Forward 接口（per-frame 调用）：
        h_new = cell(x_t, h_prev, persona_idx, dt)

    参数预估（input=52, hidden=64, num_personas=64）：
        - tau_base: (64,) = 64
        - persona_tau_bias: (64, 64) = 4096
        - f_net.0: (52+64, 128) + bias = 14976
        - f_net.2: (128, 64) + bias = 8256
        - 总 ≈ 27.4K per cell（2 层 stack 估 ≈ 55K，比 ncps 49K 略大但可控）
    """

    def __init__(self, cfg: CustomCfCCellConfig):
        super().__init__()
        self.cfg = cfg

        # τ_base: per-neuron 基础时间常数，可学习
        self.tau_base = nn.Parameter(torch.ones(cfg.hidden_dim) * cfg.tau_init)

        # per-persona τ-bias: persona_idx → hidden_dim 偏置（exp 进 τ）
        self.persona_tau_bias = nn.Embedding(cfg.num_personas, cfg.hidden_dim)
        nn.init.zeros_(self.persona_tau_bias.weight)  # 默认无偏置 = base τ

        # f_net: f(x, h) → 目标 hidden（隐式欧拉的 driving force）
        self.f_net = nn.Sequential(
            nn.Linear(cfg.input_dim + cfg.hidden_dim, cfg.f_net_hidden),
            nn.GELU(),
            nn.Linear(cfg.f_net_hidden, cfg.hidden_dim),
        )

    def compute_tau(self, persona_idx: torch.Tensor) -> torch.Tensor:
        """计算 per-batch + per-neuron τ.

        Returns: tau shape (batch, hidden_dim), values > 0 (因为 exp).
        """
        bias = self.persona_tau_bias(persona_idx)  # (batch, hidden_dim)
        # τ_i = τ_base_i * exp(persona_bias_i)
        # clip bias 避免 exp 数值爆炸（最大放大 / 缩小 e^3 ≈ 20x，spec §2.1 物理合理范围）
        bias_clipped = torch.clamp(bias, min=-3.0, max=3.0)
        tau = self.tau_base.unsqueeze(0) * torch.exp(bias_clipped)
        # 数值稳定：τ ≥ 1e-3 防 0 除
        return torch.clamp(tau, min=1e-3)

    def forward(
        self,
        x_t: torch.Tensor,
        h_prev: torch.Tensor,
        persona_idx: torch.Tensor,
        dt: float = 1.0,
    ) -> torch.Tensor:
        """单步隐式欧拉 forward.

        Args:
            x_t: (batch, input_dim) 当前帧输入
            h_prev: (batch, hidden_dim) 上一步 hidden
            persona_idx: (batch,) long tensor
            dt: 时间步长（默认 1.0 = 单帧）

        Returns:
            h_new: (batch, hidden_dim)
        """
        tau = self.compute_tau(persona_idx)  # (batch, hidden_dim)
        # f_net 输入：concat(x_t, h_prev)
        f = self.f_net(torch.cat([x_t, h_prev], dim=-1))  # (batch, hidden_dim)
        # 单步隐式欧拉解析解
        h_new = (tau * h_prev + dt * f) / (tau + dt)
        return h_new

    def init_hidden(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """返回初始 zero hidden state."""
        return torch.zeros(batch_size, self.cfg.hidden_dim,
                           device=device or self.tau_base.device)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
