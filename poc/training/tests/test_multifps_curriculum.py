"""Tests for phase-2 multi-FPS curriculum, invariance, and tau clamp."""
from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from poc.lnn.custom_cfc_cell import CustomCfCCell, CustomCfCCellConfig
from poc.lnn.invariance_loss import invariance_loss
from poc.lnn.ode_dynamics_loss import clamp_tau
from poc.training.dataset import VEATICDataset
from poc.training.evaluate import run_evaluation
from poc.training.multifps_dataset import MultiFpsDataset, VEATICDownsampled
from poc.training.multifps_sampler import EquallyWeightedFpsBucketSampler, linear_probability_ramp
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


class _DictDataset(Dataset):
    """Simple dict dataset for sampler/dataset tests."""

    def __init__(self, n: int, fps: float, seq_len: int = 8) -> None:
        self.items = []
        for i in range(n):
            self.items.append(
                {
                    "video_id": f"v{i}",
                    "event_tensor": torch.randn(seq_len, 52),
                    "labels_va": torch.randn(seq_len, 2),
                    "fps": float(fps),
                }
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def test_multifps_dataset_yields_correct_fps() -> None:
    ds30 = _DictDataset(3, fps=30.0)
    ds1 = _DictDataset(2, fps=1.0)
    multi = MultiFpsDataset([(ds30, 30.0), (ds1, 1.0)])

    fps_values = [multi[i]["fps"] for i in range(len(multi))]
    assert fps_values.count(30.0) == 3
    assert fps_values.count(1.0) == 2
    assert sorted(multi.get_fps_to_indices().keys()) == [1.0, 30.0]


def test_veatic_downsampled_min_seq_len_enforced() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp_path, n_videos=2, n_frames=360)
        base = VEATICDataset(npz, ratings, seq_len=30)
        ds_1hz = VEATICDownsampled(base=base, target_fps=1.0, min_seq_len=8)

        assert len(ds_1hz) > 0
        sample = ds_1hz[0]
        assert sample["event_tensor"].shape[0] >= 8
        assert sample["labels_va"].shape[0] >= 8


def test_equally_weighted_sampler_homogeneous_batch() -> None:
    fps_to_indices = {
        30.0: list(range(0, 20)),
        5.0: list(range(20, 40)),
        1.0: list(range(40, 60)),
    }

    def ramp(_: int) -> dict[float, float]:
        return {30.0: 1.0, 5.0: 1.0, 1.0: 1.0}

    sampler = EquallyWeightedFpsBucketSampler(
        fps_to_indices=fps_to_indices,
        batch_size=4,
        curriculum_ramp=ramp,
        seed=7,
    )
    idx_to_fps = {idx: fps for fps, arr in fps_to_indices.items() for idx in arr}
    for batch in sampler:
        batch_fps = {idx_to_fps[i] for i in batch}
        assert len(batch_fps) == 1


def test_equally_weighted_sampler_balanced_exposure() -> None:
    fps_to_indices = {
        30.0: list(range(0, 300)),
        5.0: list(range(300, 600)),
        1.0: list(range(600, 900)),
    }

    def ramp(_: int) -> dict[float, float]:
        return {30.0: 1.0, 5.0: 1.0, 1.0: 1.0}

    sampler = EquallyWeightedFpsBucketSampler(
        fps_to_indices=fps_to_indices,
        batch_size=10,
        curriculum_ramp=ramp,
        seed=42,
    )
    idx_to_fps = {idx: fps for fps, arr in fps_to_indices.items() for idx in arr}

    exposure = {30.0: 0, 5.0: 0, 1.0: 0}
    for batch in sampler:
        fps = idx_to_fps[batch[0]]
        exposure[fps] += 1

    mean_exp = sum(exposure.values()) / len(exposure)
    for value in exposure.values():
        assert abs(value - mean_exp) <= mean_exp * 0.10


def test_curriculum_linear_ramp() -> None:
    assert linear_probability_ramp(39, 40, 60, 0.3) == pytest.approx(0.0)
    assert linear_probability_ramp(40, 40, 60, 0.3) == pytest.approx(0.0)
    assert linear_probability_ramp(50, 40, 60, 0.3) == pytest.approx(0.15)
    assert linear_probability_ramp(60, 40, 60, 0.3) == pytest.approx(0.3)
    assert linear_probability_ramp(80, 40, 60, 0.3) == pytest.approx(0.3)


def test_invariance_loss_math() -> None:
    h_high = torch.arange(1, 1 + 2 * 6 * 2, dtype=torch.float32).view(2, 6, 2)
    h_low = h_high[:, ::2, :].clone()
    loss = invariance_loss(h_high, h_low, fps_ratio=2)
    assert loss.item() == pytest.approx(0.0, abs=1e-8)


def test_tau_clamp_after_step() -> None:
    cell = CustomCfCCell(CustomCfCCellConfig(input_dim=4, hidden_dim=8, num_personas=2, tau_init=1.0))
    with torch.no_grad():
        cell.tau_base.fill_(100.0)
    changed = clamp_tau(cell, upper_bound=30.0)
    assert changed == cell.tau_base.numel()
    assert float(cell.tau_base.max().item()) <= np.log(30.0) + 1e-6


def test_variance_aware_eval_catches_lowpass() -> None:
    class _DummyOdeLoss:
        def __init__(self, hidden_dim: int):
            self.cfg = SimpleNamespace(input_stride=1)
            self.hidden_dim = hidden_dim

        def __call__(
            self,
            x: torch.Tensor,
            backbone_step_fn,
            persona_idx,
            init_hidden_fn,
            fps,
        ) -> dict:
            del backbone_step_fn, persona_idx, init_hidden_fn
            total = x.size(1)
            prefix_len = max(1, int(total * 0.2))
            rollout_len = max(1, total - prefix_len)
            x_roll = x[:, prefix_len : prefix_len + rollout_len, :]
            x_pred = torch.zeros_like(x_roll)
            h_prefix = torch.zeros(x.size(0), prefix_len, self.hidden_dim, device=x.device)
            return {
                "loss": F.mse_loss(x_pred, x_roll),
                "L_world_model": F.mse_loss(x_pred, x_roll),
                "L_recon_prefix": torch.tensor(0.0, device=x.device),
                "L_motion": torch.tensor(0.0, device=x.device),
                "prefix_len": prefix_len,
                "rollout_len": rollout_len,
                "x_pred_rollout": x_pred,
                "h_seq_prefix": h_prefix,
                "actual_dt": 1.0 / float(fps),
                "total_physical_horizon_s": float(rollout_len) / float(fps),
            }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp_path, n_videos=2, n_frames=90)
        dataset = VEATICDataset(npz, ratings, seq_len=30)
        cfg = TrainConfig(
            seq_len=30,
            batch_size=2,
            hidden_dim=8,
            num_layers=1,
            num_personas=2,
            freeze_f_net_steps=0,
            use_ode_loss=True,
            use_vicreg=False,
            seed=3,
            log_every_n=10,
        )
        trainer = TinyTrainer(cfg=cfg, dataset=dataset, device="cpu")
        trainer.ode_loss = _DummyOdeLoss(hidden_dim=cfg.hidden_dim)

        result = run_evaluation(trainer=trainer, dataset=dataset, sample_per_video=1)

    assert result["variance_ratio"] < 0.1
    assert result["shuffle_test_pass"] is False
    assert result["variance_aware_fail_reason"] == "low-pass filter fake positive"
