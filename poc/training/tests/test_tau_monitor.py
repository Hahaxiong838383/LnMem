"""PC-1: τ monitor tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from poc.training.dataset import VEATICDataset
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


def test_tau_stats_summary_mock() -> None:
    """Mock buffer should produce stable τ monitor summary fields."""
    tau_start = torch.tensor([0.10, 0.12, 0.14, 0.16])
    tau_end = torch.tensor([0.11, 0.13, 0.15, 0.17])
    stats = TinyTrainer._summarize_tau_stats(
        tau_values=tau_end,
        tau_values_start=tau_start,
        tau_grad_buffer=[1e-4, 2e-4, 3e-4],
    )
    assert stats["mean"] > 0.0
    assert stats["max"] >= stats["min"]
    assert stats["grad_norm"] > 0.0
    assert stats["mean_abs_delta"] > 0.0


def test_tau_monitor_real_trainer_one_epoch(capsys) -> None:
    """Real trainer 1 epoch should print τ stats and collect non-zero grad norm."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=80)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            num_negs=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            use_ode_loss=False,
            log_every_n=1,
            seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch(epoch_idx=0)
        out = capsys.readouterr().out

    assert "τ mean=" in out
    assert result["tau_stats"]["grad_norm"] > 0.0
