"""PC-3 VICReg Std+Cov auto-trigger 单测（cc · 2026-05-22 v2 整改后）

契约：`docs/plans/2026-05-22-p0-pre-conditions.md` §PC-3
gemini 第 7 轮整改要点：
- std + cov 联合损失（F-1 CRITICAL）
- 软门控 sigmoid 替代硬 latch（F-2 HIGH）
"""
from __future__ import annotations

import math

import pytest
import torch

from poc.training.vicreg_trigger import (
    VICRegCovTrigger,
    VICRegCovTriggerConfig,
)


# ------------------------------------------------------------------
# 软门控 λ_cov（gemini F-2 钦定）
# ------------------------------------------------------------------
class TestSoftGate:
    def test_lambda_at_threshold_is_half(self) -> None:
        """effective_dim == threshold → λ = 0.5"""
        t = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        t.update(effective_dim=2.5, epoch=0)
        assert abs(t.current_lambda - 0.5) < 1e-6

    def test_lambda_low_dim_close_to_one(self) -> None:
        """effective_dim ≪ threshold → λ ≈ 1"""
        t = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        t.update(effective_dim=1.0, epoch=0)
        assert t.current_lambda > 0.99, f"d=1.0 → λ={t.current_lambda}"

    def test_lambda_high_dim_close_to_zero(self) -> None:
        """effective_dim ≫ threshold → λ ≈ 0（effectively off）"""
        t = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, gate_steepness=4.0))
        t.update(effective_dim=8.0, epoch=0)
        assert t.current_lambda < 1e-7, f"d=8.0 → λ={t.current_lambda}"

    def test_lambda_monotonic_decreasing_with_effective_dim(self) -> None:
        """λ 必须随 effective_dim 单调递减"""
        t = VICRegCovTrigger(VICRegCovTriggerConfig())
        lams = []
        for d in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 8.0]:
            t.update(effective_dim=d, epoch=0)
            lams.append(t.current_lambda)
        for i in range(len(lams) - 1):
            assert lams[i] > lams[i + 1], f"λ 必须递减，但 λ[{i}]={lams[i]} ≤ λ[{i+1}]={lams[i+1]}"

    def test_lambda_zero_when_disabled(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(enabled=False))
        t.update(effective_dim=0.1, epoch=0)
        assert t.current_lambda == 0.0

    def test_lambda_zero_before_first_update(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig())
        assert t.current_lambda == 0.0


# ------------------------------------------------------------------
# Streak / active 字段（诊断用）
# ------------------------------------------------------------------
class TestStreakDiagnostics:
    def test_streak_counts_consecutive_low_dim(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(patience=5))
        for epoch in range(3):
            t.update(effective_dim=2.0, epoch=epoch)
        assert t.streak == 3
        assert t.is_active is False

    def test_active_flag_set_after_patience(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(patience=3))
        for epoch in range(2):
            t.update(effective_dim=2.0, epoch=epoch)
        newly = t.update(effective_dim=2.0, epoch=2)
        assert newly is True
        assert t.is_active is True
        assert t.activated_at_epoch == 2

    def test_high_dim_resets_streak(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(patience=5))
        t.update(effective_dim=2.0, epoch=0)
        t.update(effective_dim=2.0, epoch=1)
        t.update(effective_dim=4.0, epoch=2)  # 高维打断
        assert t.streak == 0

    def test_exactly_threshold_does_not_count_as_low(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(threshold=2.5, patience=2))
        t.update(effective_dim=2.5, epoch=0)
        t.update(effective_dim=2.5, epoch=1)
        assert t.streak == 0
        assert t.is_active is False


# ------------------------------------------------------------------
# Std + Cov 联合损失（F-1 CRITICAL 修复）
# ------------------------------------------------------------------
class TestJointStdCovLoss:
    def test_compute_loss_returns_dict_with_grad(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig())
        z_a = torch.randn(32, 64, requires_grad=True)
        z_p = torch.randn(32, 64, requires_grad=True)
        out = t.compute_loss(z_a, z_p)
        assert {"loss", "std_loss", "cov_loss"}.issubset(out.keys())
        assert out["loss"].requires_grad
        out["loss"].backward()
        assert z_a.grad is not None and z_p.grad is not None

    def test_std_loss_penalizes_low_variance_dim(self) -> None:
        """单维方差归零 → std loss > 0（防 F-1 自我静音作弊）"""
        torch.manual_seed(0)
        N, D = 32, 16
        z_a = torch.randn(N, D)
        z_a[:, 5] = 0.0  # 静音第 5 维
        z_p = torch.randn(N, D)
        z_p[:, 5] = 0.0
        t = VICRegCovTrigger(VICRegCovTriggerConfig())
        loss = t.compute_std_loss(z_a, z_p)
        assert loss.item() > 0.1, (
            f"维度静音应被 std loss 惩罚，但 loss={loss.item():.6f}（应 > 0.1）"
        )

    def test_std_loss_zero_when_all_dims_unit_std(self) -> None:
        """所有 dim std=1 → std loss = 0"""
        N, D = 100, 16
        torch.manual_seed(0)
        z = torch.randn(N, D)
        z = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-6)
        t = VICRegCovTrigger(VICRegCovTriggerConfig(std_target=1.0))
        loss = t.compute_std_loss(z, z)
        assert loss.item() < 0.01, f"unit-std features → std loss ≈ 0, got {loss.item()}"

    def test_joint_loss_against_self_silencing_shortcut(self) -> None:
        """自我静音作弊检测：std 项让"维度静音"路径不再廉价.

        关键 setup：
        - 用大 N（512）减少 sample-cov noise → normal 路径 cov ≈ 0
        - 把 D 中 1/4 维方差归零 → silenced 路径 std loss 不可忽略
        - std_weight=1.0 / cov_weight=1.0（gemini F-1 钦定）
        """
        torch.manual_seed(42)
        N, D = 512, 16
        # 路径 A：自我静音（D/4 维方差归零）
        z_silenced = torch.randn(N, D)
        z_silenced[:, :4] = 0.0
        # 路径 B：正常单位方差（大 N → empirical cov 接近 I）
        z_normal = torch.randn(N, D)
        z_normal = (z_normal - z_normal.mean(0, keepdim=True)) / (z_normal.std(0, keepdim=True) + 1e-6)

        t = VICRegCovTrigger(VICRegCovTriggerConfig(std_weight=1.0, cov_weight=1.0))
        loss_silenced = t.compute_loss(z_silenced, z_silenced)["loss"].item()
        loss_normal = t.compute_loss(z_normal, z_normal)["loss"].item()
        # 自我静音必须比正常更贵（这是 F-1 整改的全部意义）
        assert loss_silenced > loss_normal, (
            f"自我静音应比正常更贵：silenced={loss_silenced:.4f} normal={loss_normal:.4f}"
        )
        # silenced 应至少有 std_loss ≥ 0.5 (4/16 维 * 2(a+p) * 1.0(target))
        std_only = t.compute_std_loss(z_silenced, z_silenced).item()
        assert std_only >= 0.4, f"silenced std loss 应 ≥ 0.4，实测 {std_only}"

    def test_n_le_1_returns_zero(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig())
        z_a = torch.randn(1, 32)
        z_p = torch.randn(1, 32)
        out = t.compute_loss(z_a, z_p)
        assert out["loss"].item() == 0.0


# ------------------------------------------------------------------
# 与 VICRegLoss 内部公式对齐
# ------------------------------------------------------------------
class TestAlignmentWithVICRegLoss:
    def test_cov_loss_matches_vicreg_internal(self) -> None:
        from poc.lnn.losses import VICRegLoss, VICRegConfig

        torch.manual_seed(42)
        z_a = torch.randn(32, 64)
        z_p = torch.randn(32, 64)

        trigger_cov = VICRegCovTrigger(VICRegCovTriggerConfig()).compute_cov_loss(z_a, z_p)
        ref = VICRegLoss(VICRegConfig())(z_a, z_p)["cov_loss"]
        assert abs(trigger_cov.item() - ref.item()) < 1e-5

    def test_std_loss_matches_vicreg_internal(self) -> None:
        from poc.lnn.losses import VICRegLoss, VICRegConfig

        torch.manual_seed(42)
        z_a = torch.randn(32, 64)
        z_p = torch.randn(32, 64)

        trigger_std = VICRegCovTrigger(VICRegCovTriggerConfig()).compute_std_loss(z_a, z_p)
        ref = VICRegLoss(VICRegConfig())(z_a, z_p)["std_loss"]
        assert abs(trigger_std.item() - ref.item()) < 1e-5


# ------------------------------------------------------------------
# to_log 状态导出
# ------------------------------------------------------------------
class TestStateExport:
    def test_log_contains_lambda_and_effective_dim(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(patience=3))
        t.update(effective_dim=2.0, epoch=0)
        log = t.to_log()
        assert "vicreg_lambda" in log
        assert "vicreg_effective_dim_observed" in log
        assert log["vicreg_lambda"] > 0.5
        assert log["vicreg_effective_dim_observed"] == 2.0

    def test_log_active_flips_after_patience(self) -> None:
        t = VICRegCovTrigger(VICRegCovTriggerConfig(patience=2))
        t.update(effective_dim=2.0, epoch=0)
        t.update(effective_dim=2.0, epoch=1)
        log = t.to_log()
        assert log["vicreg_cov_active"] is True
        assert log["vicreg_cov_activated_at_epoch"] == 1
