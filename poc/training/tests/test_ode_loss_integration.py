"""Integration tests for ADR-023 v2 ODE loss training path."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from poc.training.dataset import VEATICDataset, split_to_windows
from poc.training.evaluate import run_evaluation
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


def test_trainer_ode_loss_dry_run_one_epoch_world_model_trend():
    """1 epoch dry-run should emit non-zero world-model loss and basic downward trend."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=180)
        dataset = VEATICDataset(npz, ratings, seq_len=30)
        cfg = TrainConfig(
            seq_len=30,
            batch_size=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            log_every_n=1,
            seed=42,
            use_ode_loss=True,
            use_vicreg=False,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch(epoch_idx=0)

    losses = [x["L_world_model"] for x in result["logs"] if "L_world_model" in x]
    assert result["total_steps"] > 0
    assert len(losses) > 3
    assert losses[0] > 0.0
    assert losses[-1] > 0.0
    assert losses[-1] <= losses[0] * 1.2


def test_effective_dim_from_prefix_hseq_is_not_degenerate():
    """Evaluation should report non-degenerate effective_dim on v2 path."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=160)
        dataset = VEATICDataset(npz, ratings, seq_len=30)
        cfg = TrainConfig(
            seq_len=30,
            batch_size=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            log_every_n=2,
            seed=11,
            use_ode_loss=True,
            use_vicreg=False,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        trainer.train_one_epoch(epoch_idx=0)
        eval_result = run_evaluation(trainer, dataset, sample_per_video=4)

    assert "effective_dim" in eval_result
    assert eval_result["effective_dim"] >= 1.0
    assert "L_world_model_train" in eval_result
    assert "L_world_model_unseen" in eval_result
    assert "L_world_model_shuffled" in eval_result


def test_prefix_rollout_split_boundary_n6_k24_t30():
    """Prefix/rollout split should follow N=6, K=24 when T=30 (M4 prefix_ratio 0.5→0.2, input_stride=1 显式)."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=3, n_frames=100)
        dataset = VEATICDataset(npz, ratings, seq_len=30)
        cfg = TrainConfig(
            seq_len=30,
            batch_size=2,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            log_every_n=1,
            seed=7,
            use_ode_loss=True,
            use_vicreg=False,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        # 第 11 轮整改：显式禁用 input_stride（默认 stride=3 会让 30→10 帧，prefix=2/rollout=8）
        import dataclasses as _dc
        from poc.lnn.ode_dynamics_loss import OdeDynamicsLoss as _ODL
        new_cfg = _dc.replace(trainer.ode_loss.cfg, input_stride=1, use_dt_aware=False)
        trainer.ode_loss = _ODL(cfg=new_cfg, decoder=trainer.decoder)
        windows = split_to_windows(dataset[0], seq_len=30, stride=10)
        x = torch.stack([windows[0]["event_tensor"], windows[1]["event_tensor"]]).to(trainer.device)
        persona_idx = torch.zeros(x.size(0), dtype=torch.long, device=trainer.device)
        out = trainer.ode_loss(
            x=x,
            backbone_step_fn=trainer.backbone_step,
            persona_idx=persona_idx,
            init_hidden_fn=lambda b, dev: [cell.init_hidden(b, dev) for cell in trainer.backbone.cells],
        )

    assert out["prefix_len"] == 6
    assert out["rollout_len"] == 24
