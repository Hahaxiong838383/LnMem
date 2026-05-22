"""ODE dynamics loss unit tests; 覆盖 ADR-023 的 6 个核心验收点."""
from __future__ import annotations

import torch
import torch.nn as nn

from poc.lnn.ode_dynamics_loss import Decoder, OdeDynamicsLoss, OdeDynamicsLossConfig


def _rand_seq(batch: int = 4, steps: int = 8, dim: int = 6) -> torch.Tensor:
    return torch.randn(batch, steps, dim)


def test_ode_dynamics_loss_smoke() -> None:
    """Smoke: forward 返回完整字典且标量可回传."""
    x = _rand_seq(dim=52)
    h = _rand_seq(dim=64)
    persona_idx = torch.zeros(x.size(0), dtype=torch.long)
    loss_fn = OdeDynamicsLoss()

    proj = nn.Linear(52, 64, bias=False)

    def step_fn(x_t: torch.Tensor, h_prev: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        _ = p_idx
        return h_prev + 0.1 * proj(x_t)

    out = loss_fn(x, h, step_fn, persona_idx)
    assert set(out.keys()) >= {"loss", "L_predict", "L_recon", "L_ode", "L_infonce_aux"}
    assert out["loss"].ndim == 0


def test_l_recon_zero_for_identity_decoder() -> None:
    """L_recon: 当 decoder=identity 且 h=x 时应为 0."""
    x = _rand_seq(dim=52)
    h = x.clone()
    persona_idx = torch.zeros(x.size(0), dtype=torch.long)
    decoder = Decoder(hidden_dim=52, input_dim=52, num_layers=1)
    with torch.no_grad():
        decoder.net.weight.copy_(torch.eye(52))
        decoder.net.bias.zero_()
    cfg = OdeDynamicsLossConfig(w_predict=0.0, w_recon=1.0, w_ode=0.0, w_infonce_aux=0.0)
    loss_fn = OdeDynamicsLoss(cfg=cfg, decoder=decoder)

    out = loss_fn(x, h, lambda _x, h_prev, _p: h_prev, persona_idx)
    assert torch.isclose(out["L_recon"], torch.tensor(0.0), atol=1e-6)


def test_l_predict_zero_for_perfect_rollout() -> None:
    """L_predict: step_fn 完美预测时应趋近 0."""
    batch, steps, dim = 3, 10, 4
    x = torch.randn(batch, steps, dim)
    h = torch.zeros(batch, steps, dim)
    for t in range(1, steps):
        h[:, t, :] = h[:, t - 1, :] + 0.5 * x[:, t, :]
    persona_idx = torch.zeros(batch, dtype=torch.long)
    cfg = OdeDynamicsLossConfig(w_predict=1.0, w_recon=0.0, w_ode=0.0, w_infonce_aux=0.0, predict_horizon=5)
    loss_fn = OdeDynamicsLoss(cfg=cfg, decoder=Decoder(hidden_dim=dim, input_dim=dim))

    def step_fn(x_t: torch.Tensor, h_prev: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        _ = p_idx
        return h_prev + 0.5 * x_t

    out = loss_fn(x, h, step_fn, persona_idx)
    assert out["L_predict"].item() < 1e-7


def test_decoder_receives_gradient() -> None:
    """梯度可达 Decoder 参数."""
    x = _rand_seq(dim=52)
    h = _rand_seq(dim=64).requires_grad_(True)
    persona_idx = torch.zeros(x.size(0), dtype=torch.long)
    loss_fn = OdeDynamicsLoss()

    proj = nn.Linear(52, 64, bias=False)

    def step_fn(x_t: torch.Tensor, h_prev: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        _ = p_idx
        return h_prev + 0.1 * proj(x_t)

    out = loss_fn(x, h, step_fn, persona_idx)
    out["loss"].backward()
    assert loss_fn.decoder.net.weight.grad is not None


def test_backbone_step_path_receives_gradient() -> None:
    """梯度可达 backbone_step_fn 闭包参数."""
    x = _rand_seq(dim=52)
    h = _rand_seq(dim=64)
    persona_idx = torch.zeros(x.size(0), dtype=torch.long)
    loss_fn = OdeDynamicsLoss()
    step_proj = nn.Linear(52, 64, bias=False)

    def step_fn(x_t: torch.Tensor, h_prev: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        _ = p_idx
        return h_prev + 0.05 * step_proj(x_t)

    out = loss_fn(x, h, step_fn, persona_idx)
    out["loss"].backward()
    assert step_proj.weight.grad is not None


def test_l_total_in_reasonable_range() -> None:
    """随机初始化下总 loss 量级不应爆炸或塌陷."""
    torch.manual_seed(7)
    x = _rand_seq(batch=8, steps=12, dim=52)
    h = _rand_seq(batch=8, steps=12, dim=64)
    persona_idx = torch.zeros(x.size(0), dtype=torch.long)
    loss_fn = OdeDynamicsLoss()
    proj = nn.Linear(52, 64, bias=False)

    def step_fn(x_t: torch.Tensor, h_prev: torch.Tensor, p_idx: torch.Tensor) -> torch.Tensor:
        _ = p_idx
        return h_prev + 0.1 * proj(x_t)

    out = loss_fn(x, h, step_fn, persona_idx)
    assert 1.0 <= out["loss"].item() <= 20.0
