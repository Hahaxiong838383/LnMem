"""TDD tests for dt-aware ODE rollout path."""
from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.lnn.custom_cfc_cell import CustomCfCCell, CustomCfCCellConfig
from poc.lnn.ode_dynamics_loss import Decoder, OdeDynamicsLoss, OdeDynamicsLossConfig
from poc.training.daemon_dataset import DaemonAUDataset
from poc.training.dataset import VEATICDataset
from poc.training.evaluate import run_evaluation
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig


def _init_hidden(batch: int, device: torch.device) -> list[torch.Tensor]:
    return [torch.zeros(batch, 2, device=device)]


def _dt_sensitive_step(
    x_t: torch.Tensor,
    h_states: list[torch.Tensor],
    persona_idx: torch.Tensor,
    dt: float = 1.0,
) -> list[torch.Tensor]:
    del persona_idx
    h_prev = h_states[0]
    drive = x_t.mean(dim=1, keepdim=True).expand_as(h_prev)
    return [h_prev + dt * drive]


def _build_ode_loss(*, dt_target: float, use_dt_aware: bool, input_stride: int = 3) -> OdeDynamicsLoss:
    cfg = OdeDynamicsLossConfig(
        input_stride=input_stride,
        dt_target=dt_target,
        fps_default=30.0,
        use_dt_aware=use_dt_aware,
        prefix_ratio=0.2,
        rollout_horizon=24,
    )
    return OdeDynamicsLoss(cfg=cfg, decoder=Decoder(hidden_dim=2, input_dim=3, num_layers=1))


def test_dt_passes_through_to_cell() -> None:
    """dt=1.0 vs dt=0.1 should produce different hidden trajectory."""
    torch.manual_seed(7)
    x = torch.ones(1, 30, 3)
    persona_idx = torch.zeros(1, dtype=torch.long)
    ode_loss = _build_ode_loss(dt_target=0.1, use_dt_aware=True)

    out_30fps = ode_loss(
        x=x,
        backbone_step_fn=_dt_sensitive_step,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=30.0,
    )
    out_1fps = ode_loss(
        x=x,
        backbone_step_fn=_dt_sensitive_step,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=1.0,
    )

    assert out_30fps["actual_dt"] == pytest.approx(0.1, abs=1e-6)
    assert out_1fps["actual_dt"] == pytest.approx(1.0, abs=1e-6)

    h_last_30 = out_30fps["h_seq_prefix"][:, -1, :]
    h_last_1 = out_1fps["h_seq_prefix"][:, -1, :]
    assert not torch.allclose(h_last_30, h_last_1)


def test_ode_dt_math_smaller_dt_smaller_change() -> None:
    """Semi-implicit Euler should make very small dt produce much smaller state change."""
    cell = CustomCfCCell(
        CustomCfCCellConfig(input_dim=1, hidden_dim=1, num_personas=1, tau_init=1.0, f_net_hidden=2)
    )
    with torch.no_grad():
        cell.tau_base.fill_(1.0)
        cell.persona_tau_bias.weight.zero_()
        for p in cell.f_net.parameters():
            p.zero_()
        cell.f_net[-1].bias.fill_(2.0)

    x_t = torch.zeros(1, 1)
    h_prev = torch.ones(1, 1)
    persona_idx = torch.zeros(1, dtype=torch.long)

    h_small = cell(x_t, h_prev, persona_idx, dt=0.001)
    h_big = cell(x_t, h_prev, persona_idx, dt=1.0)

    diff_small = torch.norm(h_small - h_prev).item()
    diff_big = torch.norm(h_big - h_prev).item()
    assert diff_small < diff_big / 5.0


def test_stride_from_fps_correct() -> None:
    """Stride and actual_dt should follow fps and dt_target with 1/fps floor."""
    x = torch.randn(1, 30, 3)
    persona_idx = torch.zeros(1, dtype=torch.long)

    out_30 = _build_ode_loss(dt_target=0.1, use_dt_aware=True)(
        x=x,
        backbone_step_fn=_dt_sensitive_step,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=30.0,
    )
    assert out_30["actual_dt"] == pytest.approx(0.1, abs=1e-6)
    assert out_30["prefix_len"] == 2
    assert out_30["rollout_len"] == 8

    out_1hz = _build_ode_loss(dt_target=1.0, use_dt_aware=True)(
        x=x,
        backbone_step_fn=_dt_sensitive_step,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=1.0,
    )
    assert out_1hz["actual_dt"] == pytest.approx(1.0, abs=1e-6)
    assert out_1hz["prefix_len"] == 6
    assert out_1hz["rollout_len"] == 24

    out_1hz_small_target = _build_ode_loss(dt_target=0.1, use_dt_aware=True)(
        x=x,
        backbone_step_fn=_dt_sensitive_step,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=1.0,
    )
    assert out_1hz_small_target["actual_dt"] == pytest.approx(1.0, abs=1e-6)
    assert out_1hz_small_target["prefix_len"] == 6
    assert out_1hz_small_target["rollout_len"] == 24


def test_backward_compat_use_dt_aware_false() -> None:
    """Legacy input_stride behavior should stay unchanged when dt-aware is disabled."""
    x = torch.randn(1, 30, 3)
    persona_idx = torch.zeros(1, dtype=torch.long)
    dt_calls: list[float] = []

    def step_with_record(
        x_t: torch.Tensor,
        h_states: list[torch.Tensor],
        persona_idx_inner: torch.Tensor,
        dt: float = 1.0,
    ) -> list[torch.Tensor]:
        dt_calls.append(dt)
        return _dt_sensitive_step(x_t, h_states, persona_idx_inner, dt=dt)

    loss = _build_ode_loss(dt_target=0.1, use_dt_aware=False, input_stride=3)
    out_a = loss(
        x=x,
        backbone_step_fn=step_with_record,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=1.0,
    )
    out_b = loss(
        x=x,
        backbone_step_fn=step_with_record,
        persona_idx=persona_idx,
        init_hidden_fn=_init_hidden,
        fps=30.0,
    )

    assert "actual_dt" not in out_a
    assert out_a["prefix_len"] == 2 and out_a["rollout_len"] == 8
    assert out_b["prefix_len"] == 2 and out_b["rollout_len"] == 8
    assert torch.allclose(out_a["x_pred_rollout"], out_b["x_pred_rollout"], atol=1e-7)
    assert dt_calls and set(dt_calls) == {1.0}


def test_evaluate_surfaces_actual_dt() -> None:
    """run_evaluation should expose actual_dt and total physical horizon."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        npz, ratings = _create_mock_veatic(tmpdir, n_videos=2, n_frames=90)
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
            seed=7,
            log_every_n=10,
        )
        trainer = TinyTrainer(cfg=cfg, dataset=dataset, device="cpu")
        trainer.ode_loss.cfg = dataclasses.replace(
            trainer.ode_loss.cfg,
            use_dt_aware=True,
            dt_target=0.1,
            fps_default=30.0,
        )

        result = run_evaluation(trainer=trainer, dataset=dataset, sample_per_video=1)

    assert result["actual_dt"] == pytest.approx(0.1, abs=1e-6)
    assert result["total_physical_horizon_s"] == pytest.approx(
        result["actual_dt"] * result["rollout_len"], abs=1e-6
    )


def test_daemon_dataset_yields_fps_1() -> None:
    """Daemon dataset should mark 1Hz and VEATIC dataset should mark 30fps."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        daemon_cache = tmpdir / "daemon_cache.npz"
        np.savez_compressed(daemon_cache, au_s0=np.full((8, 20), 0.2, dtype=np.float32))
        daemon_ds = DaemonAUDataset(daemon_cache)
        assert daemon_ds[0]["fps"] == pytest.approx(1.0)

        npz, ratings = _create_mock_veatic(tmpdir / "veatic", n_videos=1, n_frames=40)
        veatic_ds = VEATICDataset(npz, ratings, seq_len=30)
        assert veatic_ds[0]["fps"] == pytest.approx(30.0)
