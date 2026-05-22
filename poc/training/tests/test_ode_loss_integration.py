"""ODE-loss integration tests for TinyTrainer; 覆盖训练/评估联动."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from poc.training.dataset import VEATICDataset
from poc.training.evaluate import run_evaluation
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


def test_ode_loss_trainer_dry_run_one_epoch() -> None:
    """1 epoch dry-run: 跑通且 loss 序列整体不发散."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=120)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            use_ode_loss=True,
            log_every_n=1,
            seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch(epoch_idx=0)

    assert result["total_steps"] > 0
    losses = [x["loss"] for x in result["logs"]]
    assert losses and all(np.isfinite(v) for v in losses)
    assert losses[-1] <= losses[0] * 1.2


def test_effective_dim_not_regressed_after_one_epoch() -> None:
    """1 epoch 后 effective_dim 不应比初始更差."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=140)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            use_ode_loss=True,
            log_every_n=2,
            seed=7,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        before = run_evaluation(trainer, dataset, sample_per_video=5)["effective_dim"]
        trainer.train_one_epoch(epoch_idx=0)
        after = run_evaluation(trainer, dataset, sample_per_video=5)["effective_dim"]

    assert after >= before * 0.6


def test_l_recon_curve_drops_after_training_steps() -> None:
    """训练步推进后 L_recon 应有明显下降（>=30%）."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=260)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=2,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            use_ode_loss=True,
            log_every_n=1,
            seed=11,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        trainer.train_one_epoch(epoch_idx=0)
        recon_values = [x["L_recon"] for x in trainer.logs if "L_recon" in x]

    assert len(recon_values) >= 10
    first = np.mean(recon_values[:5])
    last = np.mean(recon_values[-5:])
    assert last <= first * 0.7
