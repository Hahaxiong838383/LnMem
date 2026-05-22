"""PC-3 VICReg Std+Cov auto-trigger（cc · 2026-05-22 v2 整改后）

契约：`docs/plans/2026-05-22-p0-pre-conditions.md` §PC-3（v2）
gemini 第 7 轮前置评审 REVISE 整改：
- F-1 CRITICAL：cov 孤立使用 → 特征自我静音作弊 → **必须 std + cov 联合**
- F-2 HIGH：latch 太硬 → **软门控 sigmoid(4·(threshold-effective_dim))**

设计原则
--------
- **std + cov 联合**：cov 解耦 + std 防方差归零，杜绝 "维度静音" 作弊路径
- **软门控**：λ_cov = sigmoid(steepness · (threshold - effective_dim))
  - effective_dim ≪ threshold → λ ≈ 1（强约束）
  - effective_dim = threshold → λ = 0.5（临界）
  - effective_dim ≫ threshold → λ ≈ 0（effectively off）
- **streak 仅用于诊断**：保留 streak/active 字段供日志观察，不再控制开关
"""
from __future__ import annotations

import dataclasses
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class VICRegCovTriggerConfig:
    """gemini 第 7 轮前置评审钦定参数（v2 整改）.

    - threshold: effective_dim 临界值（< 则 λ > 0.5）
    - patience: streak 阈值（仅日志/诊断用，不再用于二元开关）
    - gate_steepness: sigmoid 斜率（gemini 钦定 4.0）
    - std_weight / cov_weight: std + cov 联合权重，1:1（gemini F-1 钦定）
    - std_target: 目标 std（沿用 VICRegLoss 1.0）
    - std_eps: std 计算稳定项
    """

    enabled: bool = True
    threshold: float = 2.5
    patience: int = 5
    gate_steepness: float = 4.0
    std_weight: float = 1.0
    cov_weight: float = 1.0
    std_target: float = 1.0
    std_eps: float = 1e-4


class VICRegCovTrigger:
    """effective_dim 监控 → VICReg Std+Cov auto-trigger（软门控）.

    使用方式
    --------
    >>> trigger = VICRegCovTrigger(VICRegCovTriggerConfig())
    >>> # 每 epoch 末尾跑完 evaluate 之后
    >>> trigger.update(effective_dim=2.1, epoch=epoch_idx)
    >>> lam = trigger.current_lambda  # 软门控权重 ∈ [0, 1]
    >>> # 训练 step 内（lam > 1e-3 才计算，省 compute）
    >>> if lam > 1e-3:
    ...     aux = trigger.compute_loss(z_anchor, z_positive)
    ...     loss = loss + lam * aux

    数学
    ----
    λ_cov = sigmoid(gate_steepness · (threshold - effective_dim))
    aux_loss = std_weight · std_loss + cov_weight · cov_loss
        std_loss = Σ_d max(0, std_target - std(z[:, d]))   防方差归零
        cov_loss = (1/D) · ||off_diag(Cov(Z))||²_F          解耦
    """

    def __init__(self, cfg: Optional[VICRegCovTriggerConfig] = None) -> None:
        self.cfg = cfg or VICRegCovTriggerConfig()
        self._streak: int = 0
        self._active: bool = False
        self._activated_at_epoch: Optional[int] = None
        self._current_effective_dim: float = float("inf")
        self._epoch: int = -1

    # ---- 状态查询 ----
    @property
    def is_active(self) -> bool:
        """streak >= patience（诊断用，软门控下不再用于开关）."""
        return self._active

    @property
    def streak(self) -> int:
        return self._streak

    @property
    def activated_at_epoch(self) -> Optional[int]:
        return self._activated_at_epoch

    @property
    def current_lambda(self) -> float:
        """当前软门控权重 λ_cov ∈ [0, 1].

        disabled 或 effective_dim 未观测时返回 0.
        """
        if not self.cfg.enabled:
            return 0.0
        if math.isinf(self._current_effective_dim):
            return 0.0
        x = self.cfg.gate_steepness * (self.cfg.threshold - self._current_effective_dim)
        # 数值稳定 sigmoid（防 exp 溢出）
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    # ---- 状态推进 ----
    def update(self, effective_dim: float, epoch: int) -> bool:
        """根据本 epoch 的 effective_dim 推进 streak / 当前 λ.

        Returns
        -------
        newly_activated : bool
            streak 首次 >= patience 时为 True；其他为 False.
            （软门控下 newly_activated 仅用于日志通知，不影响 λ 计算）
        """
        if not self.cfg.enabled:
            return False
        self._current_effective_dim = float(effective_dim)
        self._epoch = epoch

        if effective_dim < self.cfg.threshold:
            self._streak += 1
            if self._streak >= self.cfg.patience and not self._active:
                self._active = True
                self._activated_at_epoch = epoch
                return True
        else:
            # streak 重置（软门控下仍保留，便于诊断）
            self._streak = 0
        return False

    # ---- 损失计算 ----
    def compute_std_loss(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """std loss = Σ_d max(0, std_target - std(z[:, d]))（防方差归零作弊）.

        与 `poc/lnn/losses.py::VICRegLoss` 内部 std 计算严格一致.
        """
        N, _ = z_a.shape
        if N <= 1:
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        std_a = torch.sqrt(z_a.var(dim=0) + self.cfg.std_eps)
        std_p = torch.sqrt(z_p.var(dim=0) + self.cfg.std_eps)
        std_loss = torch.mean(F.relu(self.cfg.std_target - std_a)) + torch.mean(
            F.relu(self.cfg.std_target - std_p)
        )
        return std_loss

    def compute_cov_loss(self, z_a: torch.Tensor, z_p: torch.Tensor) -> torch.Tensor:
        """cov loss = (1/D) · ||off_diag(Cov(Z))||²_F.

        与 `poc/lnn/losses.py::VICRegLoss` 内部 cov 计算严格一致.
        """
        N, D = z_a.shape
        if N <= 1:
            return torch.tensor(0.0, device=z_a.device, requires_grad=True)
        z_a_c = z_a - z_a.mean(dim=0, keepdim=True)
        z_p_c = z_p - z_p.mean(dim=0, keepdim=True)
        denom = max(N - 1, 1)
        cov_a = (z_a_c.T @ z_a_c) / denom
        cov_p = (z_p_c.T @ z_p_c) / denom
        cov_loss = (
            (cov_a - torch.diag(torch.diag(cov_a))).pow(2).sum() / D
            + (cov_p - torch.diag(torch.diag(cov_p))).pow(2).sum() / D
        )
        return cov_loss

    def compute_loss(self, z_a: torch.Tensor, z_p: torch.Tensor) -> dict[str, torch.Tensor]:
        """std + cov 联合损失（gemini F-1 钦定 1:1 权重）.

        Returns
        -------
        {"loss", "std_loss", "cov_loss"} 全 differentiable.
        """
        std_loss = self.compute_std_loss(z_a, z_p)
        cov_loss = self.compute_cov_loss(z_a, z_p)
        total = self.cfg.std_weight * std_loss + self.cfg.cov_weight * cov_loss
        return {
            "loss": total,
            "std_loss": std_loss.detach(),
            "cov_loss": cov_loss.detach(),
        }

    # ---- 日志导出 ----
    def to_log(self) -> dict:
        """epoch log 字典字段（供 trainer 合并到 epoch_logs）."""
        return {
            "vicreg_cov_active": self._active,
            "vicreg_cov_streak": self._streak,
            "vicreg_cov_activated_at_epoch": self._activated_at_epoch,
            "vicreg_lambda": self.current_lambda,
            "vicreg_effective_dim_observed": self._current_effective_dim
            if not math.isinf(self._current_effective_dim)
            else None,
        }
