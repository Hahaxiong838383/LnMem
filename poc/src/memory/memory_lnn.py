"""5-stage memory LTC chain with persona-aware tau and smooth ODE updates."""
from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor, nn

from poc.src.lnn.contracts import LNNTrainConfig

DEFAULT_TAU_SECONDS = (30.0, 300.0, 3600.0, 86400.0, 604800.0)
N_STAGES = 5


class MemoryStage(nn.Module):
    """Single LTC stage: dm/dt = (leak_factor*(-m) + gain*u) / tau.

    用半隐式欧拉离散化（参考 Hasani 2020 LTC 论文 Eq. 6），保证 dt > τ 时仍数值稳定：
      m[t+1] = (m[t] + g*u*dt/τ) / (1 + leak_factor * dt/τ)

    当 confirmed=True 时 leak_factor=0，公式退化为开环积分（终身记忆，不衰减）。
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.input_gain = nn.Parameter(torch.ones(hidden_dim))

    def forward(
        self,
        prev_m: Tensor,
        input_u: Tensor,
        dt: float,
        tau: Tensor,
        confirmed: Tensor | None = None,
    ) -> Tensor:
        """Semi-implicit Euler step (numerically stable for any dt/tau ratio)."""
        tau_safe = torch.clamp(tau, min=1e-6)
        dt_over_tau = (dt / tau_safe).unsqueeze(-1)  # (B, 1)

        # leak_factor: 1 = leak active（标准 LTC），0 = leak shutdown（confirmed 终身记忆）
        if confirmed is None:
            leak_factor = torch.ones_like(dt_over_tau)
        else:
            leak_factor = (1.0 - confirmed.float().unsqueeze(-1))

        numerator = prev_m + input_u * self.input_gain * dt_over_tau
        denominator = 1.0 + leak_factor * dt_over_tau
        return numerator / denominator


class MemoryLNN(nn.Module):
    """Five cascaded LTC stages with per-persona per-stage time constants."""

    def __init__(self, config: LNNTrainConfig, n_stages: int = N_STAGES) -> None:
        super().__init__()
        if n_stages <= 0 or n_stages > len(DEFAULT_TAU_SECONDS):
            raise ValueError("n_stages must be in [1, 5]")
        self.config = config
        self.n_stages = n_stages
        self.hidden_dim = config.hidden_dim

        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim)
        nn.init.zeros_(self.input_projection.bias)

        self.stages = nn.ModuleList(MemoryStage(config.hidden_dim) for _ in range(n_stages))
        self.tau_offset = nn.Embedding(config.n_personas, n_stages)
        nn.init.zeros_(self.tau_offset.weight)

        self.register_buffer(
            "default_tau",
            torch.tensor(DEFAULT_TAU_SECONDS[:n_stages], dtype=torch.float32),
            persistent=False,
        )

    def get_tau(self, persona_idx: Tensor) -> Tensor:
        """Return tau(B,S): tau_i = exp(log(tau_default_i) + offset(persona,i))."""
        offset = self.tau_offset(persona_idx)
        base_log_tau = self.default_tau.to(device=offset.device, dtype=offset.dtype).log()
        return torch.exp(base_log_tau.unsqueeze(0) + offset)

    def reset_state(
        self,
        batch_size: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> tuple[Tensor, ...]:
        """Create zero state tuple with shape ((B,N), ... x n_stages)."""
        ref = self.input_projection.weight
        dev = device if device is not None else ref.device
        dtp = dtype if dtype is not None else ref.dtype
        return tuple(torch.zeros(batch_size, self.hidden_dim, device=dev, dtype=dtp) for _ in range(self.n_stages))

    def step(
        self,
        input_signal: Tensor,
        persona_idx: Tensor,
        importance_gates: Tensor,
        dt: float,
        m_state_prev: tuple[Tensor, ...] | None = None,
        confirmed_mask: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """One-step evolution over five cascaded stages using explicit Euler discretization."""
        batch_size = input_signal.shape[0]
        if m_state_prev is None:
            m_state_prev = self.reset_state(batch_size, device=input_signal.device, dtype=input_signal.dtype)
        if len(m_state_prev) != self.n_stages:
            raise ValueError("m_state_prev stage count mismatch")
        if importance_gates.shape[1] != self.n_stages - 1:
            raise ValueError("importance_gates must be (B, n_stages-1)")

        taus = self.get_tau(persona_idx)
        projected = self.input_projection(input_signal)
        gates = torch.clamp(importance_gates, min=0.0, max=1.0)

        stage_inputs: list[Tensor] = [projected]
        for idx in range(1, self.n_stages):
            stage_inputs.append(gates[:, idx - 1 : idx] * m_state_prev[idx - 1])

        new_states: list[Tensor] = []
        for idx, stage in enumerate(self.stages):
            is_last = idx == self.n_stages - 1
            current_confirmed = confirmed_mask if is_last else None
            new_state = stage(
                prev_m=m_state_prev[idx],
                input_u=stage_inputs[idx],
                dt=dt,
                tau=taus[:, idx],
                confirmed=current_confirmed,
            )
            new_states.append(new_state)
        return tuple(new_states)

    def forward(
        self,
        input_seq: Tensor,
        persona_idx: Tensor,
        importance_gates_seq: Tensor,
        dt: float = 1.0,
        confirmed_mask: Tensor | None = None,
    ) -> Tensor:
        """Roll out T steps and return trajectory tensor (B,T,S,N)."""
        batch_size, steps, _ = input_seq.shape
        state = self.reset_state(batch_size, device=input_seq.device, dtype=input_seq.dtype)

        trajectories: list[Tensor] = []
        for t in range(steps):
            step_confirmed: Tensor | None
            if confirmed_mask is None:
                step_confirmed = None
            elif confirmed_mask.dim() == 2:
                step_confirmed = confirmed_mask[:, t]
            else:
                step_confirmed = confirmed_mask
            state = self.step(
                input_signal=input_seq[:, t, :],
                persona_idx=persona_idx,
                importance_gates=importance_gates_seq[:, t, :],
                dt=dt,
                m_state_prev=state,
                confirmed_mask=step_confirmed,
            )
            trajectories.append(torch.stack(state, dim=1))

        return torch.stack(trajectories, dim=1)


__all__: Sequence[str] = ["MemoryLNN", "MemoryStage", "DEFAULT_TAU_SECONDS", "N_STAGES"]
