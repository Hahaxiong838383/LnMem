"""Tests for ADR-023 v2 latent world-model rollout loss."""
from __future__ import annotations

import torch
import torch.nn as nn

from poc.lnn import Decoder, MultiLayerCustomCfC, OdeDynamicsLoss, OdeDynamicsLossConfig


def _build_real_backbone_stack(hidden_dim: int = 64) -> tuple[MultiLayerCustomCfC, nn.Linear]:
    """Create fusion+backbone stack for rollout-step tests."""
    fusion = nn.Linear(52, hidden_dim)
    backbone = MultiLayerCustomCfC(
        input_dim=hidden_dim,
        hidden_dim=hidden_dim,
        num_layers=2,
        num_personas=4,
    )
    return backbone, fusion


def _build_real_step_fn(backbone: MultiLayerCustomCfC, fusion: nn.Linear):
    """Build step callable required by OdeDynamicsLoss.forward."""
    def _step(x_t: torch.Tensor, h_prev_list: list[torch.Tensor], persona_idx: torch.Tensor):
        x_fused = fusion(x_t)
        return backbone.step(x_fused, h_prev_list, persona_idx, dt=1.0)

    return _step


def _build_real_init_hidden_fn(backbone: MultiLayerCustomCfC):
    """Build hidden-state initializer for OdeDynamicsLoss.forward."""
    def _init_hidden(batch_size: int, device: torch.device):
        return [cell.init_hidden(batch_size, device=device) for cell in backbone.cells]

    return _init_hidden


def test_ode_dynamics_loss_smoke_forward():
    """Smoke: v2 forward should run and return required keys."""
    torch.manual_seed(0)
    x = torch.randn(4, 30, 52)
    persona_idx = torch.zeros(4, dtype=torch.long)
    backbone, fusion = _build_real_backbone_stack()
    decoder = Decoder(hidden_dim=64, input_dim=52, num_layers=1)
    loss_mod = OdeDynamicsLoss(OdeDynamicsLossConfig(), decoder=decoder)

    out = loss_mod(
        x=x,
        backbone_step_fn=_build_real_step_fn(backbone, fusion),
        persona_idx=persona_idx,
        init_hidden_fn=_build_real_init_hidden_fn(backbone),
    )

    assert "loss" in out
    assert "L_world_model" in out
    assert "L_recon_prefix" in out
    assert out["x_pred_rollout"].shape[0] == x.shape[0]
    assert out["h_seq_prefix"].shape[0] == x.shape[0]


def test_l_world_model_nonzero():
    """v2 anti-v1 check: world-model loss must be non-zero."""
    torch.manual_seed(1)
    x = torch.randn(4, 30, 52)
    persona_idx = torch.zeros(4, dtype=torch.long)
    backbone, fusion = _build_real_backbone_stack()
    loss_mod = OdeDynamicsLoss(OdeDynamicsLossConfig())

    out = loss_mod(
        x=x,
        backbone_step_fn=_build_real_step_fn(backbone, fusion),
        persona_idx=persona_idx,
        init_hidden_fn=_build_real_init_hidden_fn(backbone),
    )

    assert out["L_world_model"].item() > 0.01


def test_l_world_model_grad_reaches_backbone():
    """Backward should propagate gradient into backbone parameters."""
    torch.manual_seed(2)
    x = torch.randn(3, 30, 52)
    persona_idx = torch.zeros(3, dtype=torch.long)
    backbone, fusion = _build_real_backbone_stack()
    loss_mod = OdeDynamicsLoss(OdeDynamicsLossConfig())

    out = loss_mod(
        x=x,
        backbone_step_fn=_build_real_step_fn(backbone, fusion),
        persona_idx=persona_idx,
        init_hidden_fn=_build_real_init_hidden_fn(backbone),
    )
    out["loss"].backward()

    grad = backbone.cells[0].f_net[0].weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_decoder_grad_reaches_decoder_weight():
    """Backward should propagate gradient into decoder weights."""
    torch.manual_seed(3)
    x = torch.randn(3, 30, 52)
    persona_idx = torch.zeros(3, dtype=torch.long)
    backbone, fusion = _build_real_backbone_stack()
    decoder = Decoder(hidden_dim=64, input_dim=52, num_layers=1)
    loss_mod = OdeDynamicsLoss(OdeDynamicsLossConfig(), decoder=decoder)

    out = loss_mod(
        x=x,
        backbone_step_fn=_build_real_step_fn(backbone, fusion),
        persona_idx=persona_idx,
        init_hidden_fn=_build_real_init_hidden_fn(backbone),
    )
    out["loss"].backward()

    grad = decoder.net.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0


def test_rollout_autoregressive_no_future_teacher_forcing():
    """Rollout must depend on prior hidden, not future ground-truth x."""
    torch.manual_seed(4)
    cfg = OdeDynamicsLossConfig(prefix_ratio=0.5, rollout_horizon=5)
    decoder = Decoder(hidden_dim=4, input_dim=4, num_layers=1)
    with torch.no_grad():
        decoder.net.weight.copy_(torch.eye(4))
        decoder.net.bias.zero_()
    loss_mod = OdeDynamicsLoss(cfg=cfg, decoder=decoder)

    def step_fn(x_t: torch.Tensor, h_prev_list: list[torch.Tensor], persona_idx: torch.Tensor):
        h_prev = h_prev_list[0]
        h_new = 0.8 * h_prev + 0.2 * x_t
        return [h_new]

    def init_hidden_fn(batch_size: int, device: torch.device):
        return [torch.zeros(batch_size, 4, device=device)]

    base = torch.linspace(0.0, 1.0, 10).view(1, 10, 1).repeat(2, 1, 4)
    x_future_changed = base.clone()
    x_future_changed[:, 5:, :] += 9.0
    x_prefix_changed = base.clone()
    x_prefix_changed[:, :5, :] += 1.5
    persona_idx = torch.zeros(2, dtype=torch.long)

    out_base = loss_mod(base, step_fn, persona_idx, init_hidden_fn)
    out_future_changed = loss_mod(x_future_changed, step_fn, persona_idx, init_hidden_fn)
    out_prefix_changed = loss_mod(x_prefix_changed, step_fn, persona_idx, init_hidden_fn)

    assert torch.allclose(out_base["x_pred_rollout"], out_future_changed["x_pred_rollout"], atol=1e-6)
    assert not torch.allclose(out_base["x_pred_rollout"], out_prefix_changed["x_pred_rollout"], atol=1e-6)


def test_shuffle_raises_world_model_loss():
    """Shuffle rollout targets should significantly increase world-model loss."""
    torch.manual_seed(5)
    cfg = OdeDynamicsLossConfig(prefix_ratio=0.5, rollout_horizon=10, input_stride=1)
    decoder = Decoder(hidden_dim=4, input_dim=4, num_layers=1)
    with torch.no_grad():
        decoder.net.weight.copy_(torch.eye(4))
        decoder.net.bias.zero_()
    loss_mod = OdeDynamicsLoss(cfg=cfg, decoder=decoder)

    delta = torch.tensor([0.05, -0.03, 0.02, 0.01], dtype=torch.float32)

    def step_fn(x_t: torch.Tensor, h_prev_list: list[torch.Tensor], persona_idx: torch.Tensor):
        _ = h_prev_list  # deterministic transition with fixed drift
        h_new = x_t + delta.unsqueeze(0)
        return [h_new]

    def init_hidden_fn(batch_size: int, device: torch.device):
        return [torch.zeros(batch_size, 4, device=device)]

    n_steps = 20
    x0 = torch.tensor([0.2, -0.1, 0.05, 0.3], dtype=torch.float32)
    x_clean = torch.stack([x0 + delta * i for i in range(n_steps)], dim=0)
    x_clean = x_clean.unsqueeze(0).repeat(3, 1, 1)
    x_shuf = x_clean.clone()
    perm = torch.randperm(10)
    x_shuf[:, 10:, :] = x_shuf[:, 10 + perm, :]

    persona_idx = torch.zeros(3, dtype=torch.long)
    out_clean = loss_mod(x_clean, step_fn, persona_idx, init_hidden_fn)
    out_shuf = loss_mod(x_shuf, step_fn, persona_idx, init_hidden_fn)

    assert out_shuf["L_world_model"].item() >= out_clean["L_world_model"].item() * 1.5
