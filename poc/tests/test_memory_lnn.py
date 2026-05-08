"""Tests for 5-stage memory LTC chain and gate dynamics."""
from __future__ import annotations

import math

import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.memory.memory_lnn import DEFAULT_TAU_SECONDS, MemoryLNN


def _build_model() -> MemoryLNN:
    torch.manual_seed(7)
    return MemoryLNN(LNNTrainConfig())


def _zero_inputs(batch: int, feature_dim: int, steps: int = 1) -> torch.Tensor:
    return torch.zeros(batch, steps, feature_dim)


def test_step_returns_five_stage_tensors() -> None:
    """step should return a 5-tuple of (B,N) tensors."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch = 3
    state = model.step(
        input_signal=torch.randn(batch, cfg.feature_dim),
        persona_idx=torch.tensor([0, 1, 2], dtype=torch.long),
        importance_gates=torch.rand(batch, 4),
        dt=1.0,
    )
    assert len(state) == 5
    for stage in state:
        assert stage.shape == (batch, cfg.hidden_dim)


def test_exponential_decay_matches_tau30_for_m1() -> None:
    """m1 with no input should decay close to exp(-t/tau) for tau=30s."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch = 2
    persona = torch.zeros(batch, dtype=torch.long)
    gates = torch.zeros(batch, 4)
    state = model.reset_state(batch)
    state = (torch.ones_like(state[0]), state[1], state[2], state[3], state[4])

    for _ in range(60):
        state = model.step(
            input_signal=torch.zeros(batch, cfg.feature_dim),
            persona_idx=persona,
            importance_gates=gates,
            dt=1.0,
            m_state_prev=state,
        )

    expected = math.exp(-60.0 / DEFAULT_TAU_SECONDS[0])
    assert torch.allclose(state[0], torch.full_like(state[0], expected), atol=8e-3)


def test_confirmed_prevents_m5_decay() -> None:
    """confirmed=True must zero out leak term at stage 5."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch = 2
    persona = torch.zeros(batch, dtype=torch.long)
    gates = torch.zeros(batch, 4)
    state = model.reset_state(batch)
    initial_m5 = torch.randn_like(state[4])
    state = (state[0], state[1], state[2], state[3], initial_m5.clone())

    for _ in range(30):
        state = model.step(
            input_signal=torch.zeros(batch, cfg.feature_dim),
            persona_idx=persona,
            importance_gates=gates,
            dt=86400.0,
            m_state_prev=state,
            confirmed_mask=torch.ones(batch, dtype=torch.bool),
        )

    assert torch.allclose(state[4], initial_m5, atol=1e-6)


def test_persona_tau_offset_changes_m5_trajectory() -> None:
    """Different persona offsets should produce different trajectories under same input."""
    cfg = LNNTrainConfig()
    model = _build_model()
    with torch.no_grad():
        model.tau_offset.weight[1, 4] = 1.0

    batch = 2
    persona = torch.tensor([0, 1], dtype=torch.long)
    gates = torch.zeros(batch, 4)
    state = model.reset_state(batch)
    ones = torch.ones_like(state[4])
    state = (state[0], state[1], state[2], state[3], ones)

    for _ in range(200):
        state = model.step(
            input_signal=torch.zeros(batch, cfg.feature_dim),
            persona_idx=persona,
            importance_gates=gates,
            dt=3600.0,
            m_state_prev=state,
        )

    assert torch.mean(torch.abs(state[4][0] - state[4][1])).item() > 1e-2


def test_gate_zero_blocks_penetration_to_stage2() -> None:
    """When g12=0, stage2 should not receive stage1 signal."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch = 2
    persona = torch.zeros(batch, dtype=torch.long)
    prev = model.reset_state(batch)
    prev = (torch.ones_like(prev[0]), torch.zeros_like(prev[1]), prev[2], prev[3], prev[4])
    gates = torch.zeros(batch, 4)

    state = model.step(
        input_signal=torch.zeros(batch, cfg.feature_dim),
        persona_idx=persona,
        importance_gates=gates,
        dt=1.0,
        m_state_prev=prev,
    )
    assert torch.allclose(state[1], torch.zeros_like(state[1]), atol=1e-7)


def test_gate_one_allows_formula_penetration_to_stage2() -> None:
    """When g12=1, stage2 update should follow semi-implicit Euler formula.

    半隐式: m[t+1] = (m[t] + g*u*dt/τ) / (1 + dt/τ)
    起始 m₂=0, g₁₂=1, u=m₁=1, dt=1, τ=300:
      expected = (0 + 1*1/300) / (1 + 1/300) = 1/301 ≈ 0.003322
    """
    cfg = LNNTrainConfig()
    model = _build_model()
    batch = 2
    persona = torch.zeros(batch, dtype=torch.long)
    prev = model.reset_state(batch)
    prev = (torch.ones_like(prev[0]), torch.zeros_like(prev[1]), prev[2], prev[3], prev[4])
    gates = torch.zeros(batch, 4)
    gates[:, 0] = 1.0

    state = model.step(
        input_signal=torch.zeros(batch, cfg.feature_dim),
        persona_idx=persona,
        importance_gates=gates,
        dt=1.0,
        m_state_prev=prev,
    )
    # Semi-implicit Euler: m_new = (0 + 1·dt/τ) / (1 + dt/τ)
    dt_over_tau = 1.0 / DEFAULT_TAU_SECONDS[1]
    expected = torch.full_like(state[1], dt_over_tau / (1.0 + dt_over_tau))
    assert torch.allclose(state[1], expected, atol=1e-6)


def test_forward_returns_full_trajectory_shape() -> None:
    """forward should return (B,T,5,64) for default config."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch, steps = 4, 60
    out = model(
        input_seq=_zero_inputs(batch, cfg.feature_dim, steps=steps),
        persona_idx=torch.arange(batch, dtype=torch.long) % cfg.n_personas,
        importance_gates_seq=torch.zeros(batch, steps, 4),
        dt=1.0,
    )
    assert out.shape == (batch, steps, 5, cfg.hidden_dim)


def test_reset_state_returns_all_zeros() -> None:
    """reset_state should create zero tensors for all stages."""
    cfg = LNNTrainConfig()
    model = _build_model()
    state = model.reset_state(batch_size=3)
    assert len(state) == 5
    for stage in state:
        assert torch.count_nonzero(stage).item() == 0
        assert stage.shape == (3, cfg.hidden_dim)


def test_get_tau_is_positive() -> None:
    """exp(log+offset) ensures positive tau for every persona/stage."""
    model = _build_model()
    tau = model.get_tau(torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert tau.shape == (4, 5)
    assert torch.all(tau > 0)


def test_backward_gradients_flow() -> None:
    """Autograd should backprop into projection, stage gain, and tau offset params."""
    cfg = LNNTrainConfig()
    model = _build_model()
    batch, steps = 3, 8
    input_seq = torch.randn(batch, steps, cfg.feature_dim)
    persona = torch.tensor([0, 1, 2], dtype=torch.long)
    gates = torch.rand(batch, steps, 4)
    out = model(input_seq=input_seq, persona_idx=persona, importance_gates_seq=gates, dt=1.0)
    loss = out.sum()
    loss.backward()

    assert model.input_projection.weight.grad is not None
    assert model.stages[0].input_gain.grad is not None
    assert model.tau_offset.weight.grad is not None
    assert torch.count_nonzero(model.input_projection.weight.grad).item() > 0
