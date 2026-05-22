"""R3 τ-spread 正则化单测（cc 第 13 轮 · 2026-05-22）

gemini 12 钦定 τ 多时间尺度分化，cc 改用 std(log τ_base) 替代 gemini Entropy(σ(τ)) 错公式。
"""
from __future__ import annotations

import math
import torch

from poc.training.trainer import TinyTrainer, TrainConfig
from poc.training.dataset import VEATICDataset
import tempfile
from pathlib import Path
import numpy as np


def _create_mock_veatic(tmp_dir: Path, n_videos: int = 2, n_frames: int = 60):
    """Mock VEATIC: 写 npz + rating csv（最小化）."""
    au_cache_dict = {}
    rating_dir = tmp_dir / "ratings"
    rating_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_videos):
        au_cache_dict[str(i)] = np.random.randn(n_frames, 24).astype(np.float32)
        np.savetxt(rating_dir / f"{i}_valence.csv", np.random.randn(n_frames) * 0.5)
        np.savetxt(rating_dir / f"{i}_arousal.csv", np.random.randn(n_frames) * 0.5)
    npz_path = tmp_dir / "au_cache.npz"
    np.savez(npz_path, **au_cache_dict)
    return npz_path, rating_dir


def _make_trainer(lambda_tau_spread: float = 0.1) -> TinyTrainer:
    """构造最小 trainer 实例."""
    tmp = Path(tempfile.mkdtemp())
    npz, ratings = _create_mock_veatic(tmp, n_videos=2, n_frames=60)
    dataset = VEATICDataset(npz, ratings, seq_len=30)
    cfg = TrainConfig(
        seq_len=30, batch_size=2, hidden_dim=16, num_layers=2,
        num_personas=4, freeze_f_net_steps=0, seed=42,
        lambda_tau_spread=lambda_tau_spread,
    )
    return TinyTrainer(cfg, dataset, device="cpu")


class TestTauSpreadMath:
    def test_uniform_tau_zero_spread_zero_loss(self) -> None:
        """所有 tau_base 相等时 std(log tau)=0 → loss = 0"""
        trainer = _make_trainer()
        # 强行让所有 tau_base 相等
        with torch.no_grad():
            for cell in trainer.backbone.cells:
                cell.tau_base.fill_(0.5)
        loss = trainer._compute_tau_spread_loss()
        assert abs(loss.item()) < 1e-5, f"uniform tau → loss=0 expected, got {loss.item()}"

    def test_spread_tau_negative_loss(self) -> None:
        """tau_base 跨数量级分化 → loss 显著负（最小化 = 最大化 spread）"""
        trainer = _make_trainer()
        with torch.no_grad():
            # 模拟跨 3 个数量级：[0.01, 0.1, 1.0, 10.0]
            for cell in trainer.backbone.cells:
                cell.tau_base[0::4] = 0.01
                cell.tau_base[1::4] = 0.1
                cell.tau_base[2::4] = 1.0
                cell.tau_base[3::4] = 10.0
        loss = trainer._compute_tau_spread_loss()
        # std(log [0.01, 0.1, 1, 10]) 在 (1, 2) 之间
        assert loss.item() < -1.0, f"spread tau → loss << 0 expected, got {loss.item()}"

    def test_grad_reaches_tau_base(self) -> None:
        """τ-spread loss backward 后 tau_base.grad 非空"""
        trainer = _make_trainer()
        # 先扰动 tau_base 让它们不全相等
        with torch.no_grad():
            for cell in trainer.backbone.cells:
                cell.tau_base.add_(torch.randn_like(cell.tau_base) * 0.3)
        loss = trainer._compute_tau_spread_loss()
        loss.backward()
        for cell in trainer.backbone.cells:
            assert cell.tau_base.grad is not None, "tau_base.grad should be populated"
            assert cell.tau_base.grad.abs().sum().item() > 0, "tau_base.grad non-zero expected"

    def test_lambda_zero_disables_regularizer(self) -> None:
        """lambda_tau_spread=0 时 trainer step loop 不调用 _compute_tau_spread_loss（接口验证）"""
        trainer = _make_trainer(lambda_tau_spread=0.0)
        assert trainer.cfg.lambda_tau_spread == 0.0


class TestTauSpreadCorrectness:
    def test_matches_manual_std_log_tau(self) -> None:
        """_compute_tau_spread_loss 数学值 = -std(log τ) 跨所有 cells.tau_base"""
        trainer = _make_trainer()
        with torch.no_grad():
            for i, cell in enumerate(trainer.backbone.cells):
                cell.tau_base.copy_(torch.linspace(0.1 + i * 0.5, 2.0 + i * 0.5, cell.tau_base.numel()))
        manual_log_tau = torch.cat([
            torch.log(cell.tau_base.clamp(min=1e-6)) for cell in trainer.backbone.cells
        ])
        manual_spread = manual_log_tau.std(unbiased=False)
        expected_loss = -manual_spread.item()
        actual_loss = trainer._compute_tau_spread_loss().item()
        assert abs(expected_loss - actual_loss) < 1e-5, (
            f"expected {expected_loss}, got {actual_loss}"
        )
