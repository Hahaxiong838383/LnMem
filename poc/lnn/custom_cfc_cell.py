"""CustomCfCCell · γ-2 Path C 手写**半隐式欧拉**离散化求解器

源：spec §2.1 ODE 公式 + ADR-021 §2.3 伪代码（cc 实测 ncps Path A/B 都 WEAK 后启用）

**数学定义修正（2026-05-22 00:50 gemini F-1.1 评审）**：
本实现是 **半隐式欧拉（Semi-Implicit Euler）/ 一阶指数积分器近似**，
**不是** 真隐式欧拉解析解（真隐式欧拉公式右端 f 作用于 h_new，需 Newton-Raphson 迭代）。
半隐式欧拉的右端 f 作用于 h_prev 而非 h_new，避免迭代但精度降一阶。

数学锚（spec §2.1）：
    τ_i = τ_base_i * exp(α·tanh(persona_bias_i/α)) with α=3   # gemini F-1.2: 软有界防梯度死区
    τ * dh/dt = -h + f(x_t, h_prev)                            # LTC/CfC ODE
    h_new = (τ * h_prev + dt * f(x_t, h_prev)) / (τ + dt)      # 半隐式欧拉离散化

数值正确性（半隐式欧拉的物理直觉）：
    - dt → 0    → h_new ≈ h_prev    （积分步长极小，hidden 几乎不变）
    - dt → ∞    → h_new ≈ f(x, h_prev) （完全更新到 f 的目标值）
    - τ ↑       → 反应迟缓（同样 dt 下 h_new 更接近 h_prev）
    - τ ↓       → 反应敏锐

**已知风险（gemini F-4.1，γ-2 训练时必缓解）**：
f_net 参数量（~27K）远 > persona_tau_bias（~4K），优化器倾向走 f_net shortcut，
τ-bias 梯度被边缘化 → LNN 退化为普通 RNN（伪 LNN-FIRST）。
γ-2 训练缓解策略:
  - f_net 加 L2 正则（weight_decay > 1e-3）
  - persona_tau_bias 用独立更大学习率（LR scaling 5-10x）
  - 训练初期 freeze f_net 1k step，强迫先学 τ-bias

verdict 门槛（spec §11.1 #12 修正后）：persona separation ratio > 1.5
**测试必加 input noise**（gemini F-2.1: 无噪声 stress test 会让 sep ratio 撞 1e-6 分母安全垫，
产生 363995x 数值幻觉，不是真物理意义）
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class CustomCfCCellConfig:
    """spec §2.2 表格参数对齐.

    gemini F-3.1 修正（init gain 调整）：
        tau_init 默认 0.5（不是 1.0），让初始 var > 0.005 不依赖测试阈值放水。
    """
    input_dim: int
    hidden_dim: int
    num_personas: int = 64        # spec §2.1 persona embedding table 上限
    tau_init: float = 0.5         # base τ 初始值（gemini F-3.1：1.0→0.5 让 init var > 0.005）
    f_net_hidden: int = 128       # f_net 中间层（接近 ncps backbone_units=128）
    use_film: bool = False        # FiLM 个性化（留 γ-2 后续 task，本 cell 暂不实施）
    f_net_init_gain: float = 2.5  # f_net 输出层 gain 增大（防 init var 过小，gemini F-3.1 调）


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

        # f_net: f(x, h) → 目标 hidden（半隐式欧拉的 driving force）
        # gemini F-3.1: 输出层 gain 增大避免 init var 过小
        self.f_net = nn.Sequential(
            nn.Linear(cfg.input_dim + cfg.hidden_dim, cfg.f_net_hidden),
            nn.GELU(),
            nn.Linear(cfg.f_net_hidden, cfg.hidden_dim),
        )
        # 输出层 Kaiming gain × f_net_init_gain，让 init var 合理
        with torch.no_grad():
            self.f_net[-1].weight.mul_(cfg.f_net_init_gain)

    def compute_tau(self, persona_idx: torch.Tensor) -> torch.Tensor:
        """计算 per-batch + per-neuron τ（gemini F-1.2 修正：硬 clamp → scaled tanh 防梯度死区）.

        τ_i = τ_base_i * exp( α * tanh(bias_i / α) )  with α=3

        全域可微（梯度永不为 0），软有界 (-α, α) ≈ (-3, 3) 等价物理范围.

        Returns: tau shape (batch, hidden_dim), values > 0 (因为 exp).
        """
        bias = self.persona_tau_bias(persona_idx)  # (batch, hidden_dim)
        # scaled tanh: 软有界，全域梯度不为 0
        alpha = 3.0
        bias_soft = alpha * torch.tanh(bias / alpha)
        tau = self.tau_base.unsqueeze(0) * torch.exp(bias_soft)
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
