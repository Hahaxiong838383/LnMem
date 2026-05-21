"""Phase γ Fusion Layer + 5 Readout Heads（spec §2.1 + §2.3）

架构总览（spec §2.1）：
    [Input 52d] → Fusion(52d → fusion_dim) → CfC backbone → hidden 64d → 5 Readout Heads

5 Readout Heads（spec §2.3 + §3.4 Loss 4 弱监督）：
    Head 1 state_8       : Linear 64 → 8 softmax    EMOTIC + mst 8-state
    Head 2 va_cont       : Linear 64 → 2 tanh       VEATIC + EMOTIC V-A 连续
    Head 3 intent        : Linear 64 → 5 softmax    parser 5 intent_type
    Head 4 conf          : Linear 64 → 1 sigmoid    模型置信度
    Head 5 hidden_repr   : Identity 64 → 64         直接进 graphiti KG

**重要纪律（spec §3.4）**：Loss 4 Weakly-supervised Readout 训练时 backbone 必须 freeze
（防止 readout grad 反流污染表征）。本模块提供 freeze_backbone / unfreeze_backbone 接口。
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Fusion Layer (spec §2.1)
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class FusionConfig:
    input_dim: int = 52          # 24 AU + 20 槽 + 5 intent + 2 时间 + 6 音频占位
    fusion_dim: int = 64         # spec §2.1 「α 试跑 32 / β 全量 64」选 64
    dropout: float = 0.1


class FusionLayer(nn.Module):
    """Input Projection + LayerNorm + GELU（spec §2.1）.

    52d → 64d，作为 CfC backbone 输入。
    """

    def __init__(self, cfg: FusionConfig | None = None):
        super().__init__()
        self.cfg = cfg or FusionConfig()
        self.proj = nn.Linear(self.cfg.input_dim, self.cfg.fusion_dim)
        self.norm = nn.LayerNorm(self.cfg.fusion_dim)
        self.dropout = nn.Dropout(self.cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., input_dim) → (..., fusion_dim)"""
        return self.dropout(F.gelu(self.norm(self.proj(x))))


# -----------------------------------------------------------------------------
# 5 Readout Heads
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ReadoutHeadsConfig:
    hidden_dim: int = 64
    num_states: int = 8          # state_8
    va_dim: int = 2              # valence + arousal
    num_intents: int = 5         # spec ADR-018 IntentType 5 类


class ReadoutHeads(nn.Module):
    """5 Readout Heads，spec §2.3 + §3.4.

    每个 head 单独 forward 接口，便于 Loss 4 弱监督灵活 mask 缺失标签的样本。
    """

    def __init__(self, cfg: ReadoutHeadsConfig | None = None):
        super().__init__()
        self.cfg = cfg or ReadoutHeadsConfig()
        d = self.cfg.hidden_dim

        # Head 1: state_8 (logits, CE loss 用)
        self.head_state = nn.Linear(d, self.cfg.num_states)
        # Head 2: va_cont (tanh → [-1, 1] V-A 连续值)
        self.head_va = nn.Linear(d, self.cfg.va_dim)
        # Head 3: intent (logits)
        self.head_intent = nn.Linear(d, self.cfg.num_intents)
        # Head 4: confidence (sigmoid → [0, 1])
        self.head_conf = nn.Linear(d, 1)
        # Head 5: hidden_repr 直出，无参数

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        """h: (..., hidden_dim) → dict of 5 outputs"""
        return {
            "state_logits": self.head_state(h),                # (..., 8)
            "va_cont": torch.tanh(self.head_va(h)),            # (..., 2)
            "intent_logits": self.head_intent(h),              # (..., 5)
            "confidence": torch.sigmoid(self.head_conf(h)).squeeze(-1),  # (...,)
            "hidden_repr": h,                                  # (..., 64) 直出 KG
        }

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# -----------------------------------------------------------------------------
# Freeze Helpers (spec §3.4 Loss 4 弱监督铁律)
# -----------------------------------------------------------------------------
# gemini F-4 CRITICAL 修正（2026-05-22 01:50）：
# 仅 freeze backbone 不够 — Autograd 会穿透 backbone 通过 unfrozen FusionLayer 反流，
# Readout Heads 的弱监督信号将污染 fusion projection 矩阵，间接污染表征。
# 必须把整个表征生成流水线（Fusion + Backbone）作为整体 freeze.
def freeze_module(module: nn.Module) -> None:
    """单模块 freeze（兼容旧 API，但 Loss 4 训练禁止单独用，请用 freeze_representation_pipeline）"""
    for p in module.parameters():
        p.requires_grad = False
    module.eval()


def unfreeze_module(module: nn.Module) -> None:
    for p in module.parameters():
        p.requires_grad = True
    module.train()


def freeze_representation_pipeline(*modules: nn.Module) -> None:
    """**gemini F-4 CRITICAL**：Loss 4 弱监督训练必用此函数同时 freeze（Fusion + Backbone）.

    传入参数应包含所有表征生成路径模块（一般是 fusion + multilayer_cfc），
    确保梯度不能通过任何 unfrozen 中间模块反流污染.

    Usage:
        freeze_representation_pipeline(fusion, backbone)
        # train readout heads ...
    """
    for m in modules:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()


def unfreeze_representation_pipeline(*modules: nn.Module) -> None:
    """对应解冻，spec §3.4 Step 3.bis 联合微调阶段统一启用"""
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True
        m.train()
