"""PC-3 v3 解耦修法单测（cc · 2026-05-22 gemini 第 8b + codex 第 8b 协同钦定后）

核心契约：
- std_loss 必须**常开**（不受 λ 控制），防方差归零作弊
- cov_loss 必须受 λ 软门控（仅在 effective_dim 低时才启用）

前因：v2 把 std+cov 整体乘 λ，λ<0.01 时 std 也被压成 0 → 表征冷凝
"""
from __future__ import annotations

import math

import torch

from poc.training.vicreg_trigger import (
    VICRegCovTrigger,
    VICRegCovTriggerConfig,
)


class TestStdAlwaysActiveDecoupled:
    """validates std loss is always active regardless of λ value."""

    def test_std_loss_independent_of_lambda(self) -> None:
        """std_loss 计算不依赖于 λ — 即使 λ=0，std_loss 仍然非零."""
        # 用 effective_dim=10 模拟 λ→0
        trigger = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        trigger.update(effective_dim=10.0, epoch=0)
        assert trigger.current_lambda < 1e-7, f"λ at d=10 should be ~0, got {trigger.current_lambda}"

        # 制造方差归零作弊（5/16 维 silenced）
        torch.manual_seed(42)
        z_a = torch.randn(64, 16)
        z_a[:, :5] = 0.0
        z_p = torch.randn(64, 16)
        z_p[:, :5] = 0.0

        std_loss = trigger.compute_std_loss(z_a, z_p)
        assert std_loss.item() > 0.1, (
            f"std loss must catch silenced dims regardless of λ, got {std_loss.item()}"
        )

    def test_cov_loss_gated_by_lambda(self) -> None:
        """cov_loss 应受 λ 软门控 — 高 effective_dim 时 λ≈0 表示 cov 应被忽略."""
        trigger = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))

        # case 1: d 低 → λ 高
        trigger.update(effective_dim=1.0, epoch=0)
        assert trigger.current_lambda > 0.99

        # case 2: d 高 → λ ≈ 0（cov 关闭）
        trigger2 = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        trigger2.update(effective_dim=8.0, epoch=0)
        assert trigger2.current_lambda < 1e-5


class TestDecoupledLossCalculation:
    """模拟 trainer.py PC-3 v3 解耦修法的等效计算，验逻辑正确."""

    @staticmethod
    def _trainer_v3_aux_loss(trigger: VICRegCovTrigger, z_a: torch.Tensor, z_p: torch.Tensor) -> dict:
        """模拟 trainer step loop 的辅助 loss 计算（v3 解耦修法）."""
        cfg = trigger.cfg
        out = {"std_term": 0.0, "cov_term": 0.0, "lam": 0.0}
        if not cfg.enabled:
            return out
        # std 常开
        std_loss = trigger.compute_std_loss(z_a, z_p)
        out["std_term"] = (cfg.std_weight * std_loss).item()
        # cov 受 λ 软门控
        lam = trigger.current_lambda
        out["lam"] = lam
        if lam > 1e-3:
            cov_loss = trigger.compute_cov_loss(z_a, z_p)
            out["cov_term"] = (lam * cfg.cov_weight * cov_loss).item()
        return out

    def test_high_effective_dim_only_std_active(self) -> None:
        """effective_dim=10 → λ ≈ 0 → 只有 std 项贡献"""
        torch.manual_seed(0)
        z_a = torch.randn(32, 16)
        z_p = torch.randn(32, 16)

        trigger = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        trigger.update(effective_dim=10.0, epoch=0)
        out = self._trainer_v3_aux_loss(trigger, z_a, z_p)

        assert out["lam"] < 1e-7
        assert out["cov_term"] == 0.0, f"cov should be 0 at λ≈0, got {out['cov_term']}"
        # std 应该有值（即使 z 是 unit-std 也可能很小，但不会被门控压成 0）
        # 这里关键不是 std=0，而是 cov 被门控、std 没有

    def test_low_effective_dim_both_active(self) -> None:
        """effective_dim=1.5 → λ ≈ 1 → std + cov 同时贡献"""
        torch.manual_seed(0)
        z_a = torch.randn(64, 16)
        z_a[:, :5] = 0.0  # 制造冷凝场景
        z_p = torch.randn(64, 16)
        z_p[:, :5] = 0.0

        trigger = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        trigger.update(effective_dim=1.5, epoch=0)
        out = self._trainer_v3_aux_loss(trigger, z_a, z_p)

        assert out["lam"] > 0.95
        assert out["std_term"] > 0.1, f"std should catch silenced dims, got {out['std_term']}"
        assert out["cov_term"] > 0.001, f"cov should be active at λ≈1, got {out['cov_term']}"

    def test_v3_vs_v2_self_silencing_path(self) -> None:
        """关键对比：v2 (整体乘 λ) vs v3 (std 常开) 对自我静音的惩罚差距."""
        torch.manual_seed(42)
        # 真实场景：高 effective_dim 但有方差归零（模型在「扩散维度」+「内部静音」混合作弊）
        z_a = torch.randn(64, 16)
        z_a[:, :4] = 0.0
        z_p = torch.randn(64, 16)
        z_p[:, :4] = 0.0

        trigger = VICRegCovTrigger(VICRegCovTriggerConfig(
            threshold=2.5,
            gate_steepness=4.0,
            std_weight=1.0,
            cov_weight=1.0,
        ))
        trigger.update(effective_dim=8.0, epoch=0)  # 高维但有死维
        lam = trigger.current_lambda
        assert lam < 1e-5, f"λ should be tiny at d=8, got {lam}"

        # v2 路径（整体乘 λ）
        std_loss = trigger.compute_std_loss(z_a, z_p)
        cov_loss = trigger.compute_cov_loss(z_a, z_p)
        v2_total = lam * (std_loss.item() + cov_loss.item())

        # v3 路径（std 常开 + cov 受门控）
        v3_total = std_loss.item() + lam * cov_loss.item()

        ratio = v3_total / max(v2_total, 1e-12)
        assert ratio > 100, (
            f"v3 必须比 v2 在自我静音惩罚上**大数量级**: v3={v3_total:.4f} v2={v2_total:.4e} ratio={ratio:.1f}"
        )
