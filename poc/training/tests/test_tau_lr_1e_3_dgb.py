"""PC-4: tau_lr=1e-3 + DGB control tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

from poc.training.dataset import VEATICDataset
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


def test_tau_lr_1e_3_has_measurable_grad_and_shift() -> None:
    """tau_lr=1e-3 should avoid underflow-like no-op updates."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=100)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            num_negs=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            tau_lr=1e-3,
            dgb_enabled=True,
            dgb_target_ratio=10.0,
            freeze_f_net_steps=0,
            log_every_n=1,
            seed=7,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch(epoch_idx=0)

    tau_stats = result["tau_stats"]
    assert tau_stats["grad_norm"] >= 1e-5
    assert tau_stats["mean_abs_delta"] > 1e-5

    post_ratios = [x["bias_grad_ratio_post_dgb"] for x in result["logs"]]
    assert post_ratios, "expected logged DGB ratios"
    assert max(post_ratios) <= 0.11


def test_dgb_scales_down_when_needed() -> None:
    """At least one step should show post-DGB ratio <= pre-DGB ratio."""
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
            tau_lr=1e-3,
            freeze_f_net_steps=0,
            log_every_n=1,
            seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch(epoch_idx=0)

    assert any(
        x["bias_grad_ratio_post_dgb"] <= x["bias_grad_ratio_pre_dgb"]
        for x in result["logs"]
    )
