"""LTC-driven expression dynamics for smooth evolution.
用液态吸引子动力学替代表情静态查表。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

EXPRESSION_DIMS = (
    "smile_intensity",
    "brow_furrow",
    "cheek_raise",
    "lip_corner_depress",
    "jaw_drop",
    "mouth_open_ratio",
)

STATE_LABELS = (
    "low_energy",
    "focused",
    "distracted",
    "fatigued",
    "emotional_positive",
    "emotional_negative",
    "neutral",
    "transitioning",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _freeze_vector(values: Mapping[str, float], *, low: float, high: float) -> Mapping[str, float]:
    keys = set(values)
    expected = set(EXPRESSION_DIMS)
    if keys != expected:
        raise ValueError(f"expression keys mismatch: {sorted(keys)}")
    out: dict[str, float] = {}
    for dim in EXPRESSION_DIMS:
        val = float(values[dim])
        if not low <= val <= high:
            raise ValueError(f"{dim} out of range [{low}, {high}]: {val}")
        out[dim] = val
    return MappingProxyType(out)


@dataclass(frozen=True)
class ExpressionProfile:
    """Persona expression config with base, offsets and taus.

    persona_id: profile id；base 对应公式 base_i；state_offset 对应 Δ_i(state)；
    tau 对应公式 τ_i，且必须为正数。
    """

    persona_id: str
    base: Mapping[str, float]
    state_offset: Mapping[str, Mapping[str, float]]
    tau: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.persona_id:
            raise ValueError("persona_id must be non-empty")
        frozen_base = _freeze_vector(self.base, low=0.0, high=1.0)
        frozen_tau = _freeze_vector(self.tau, low=0.5, high=20.0)
        state_keys = set(self.state_offset)
        if state_keys != set(STATE_LABELS):
            raise ValueError(f"state labels mismatch: {sorted(state_keys)}")
        frozen_offsets: dict[str, Mapping[str, float]] = {}
        for state in STATE_LABELS:
            frozen_offsets[state] = _freeze_vector(self.state_offset[state], low=-1.0, high=1.0)
        object.__setattr__(self, "base", frozen_base)
        object.__setattr__(self, "tau", frozen_tau)
        object.__setattr__(self, "state_offset", MappingProxyType(frozen_offsets))

    def attractor(self, state: str) -> dict[str, float]:
        """Compute attractor A_i(state)=base_i+offset_i(state), then clamp [0,1]."""
        if state not in self.state_offset:
            raise ValueError(f"unknown state: {state}")
        offsets = self.state_offset[state]
        return {dim: _clamp01(self.base[dim] + offsets[dim]) for dim in EXPRESSION_DIMS}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ExpressionProfile":
        """Build immutable profile from plain dict-like input."""
        return cls(
            persona_id=str(d["persona_id"]),
            base=d["base"],
            state_offset=d["state_offset"],
            tau=d["tau"],
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert profile into json-serializable plain dict."""
        return {
            "persona_id": self.persona_id,
            "base": {dim: float(self.base[dim]) for dim in EXPRESSION_DIMS},
            "state_offset": {
                state: {dim: float(self.state_offset[state][dim]) for dim in EXPRESSION_DIMS}
                for state in STATE_LABELS
            },
            "tau": {dim: float(self.tau[dim]) for dim in EXPRESSION_DIMS},
        }


class ExpressionDynamics:
    """Euler-discretized LTC expression integrator.

    d(expr_i)/dt=(A_i(state)-expr_i)/tau_i, expr[t+1]=expr[t]+(A-expr)*dt/tau。
    """

    def __init__(self, profile: ExpressionProfile) -> None:
        self.profile = profile

    def step(self, current_expr: Mapping[str, float], state: str, dt: float = 1.0) -> dict[str, float]:
        """Single-step evolution with clamp on each expression dimension."""
        if dt < 0.0:
            raise ValueError("dt must be non-negative")
        attractor = self.profile.attractor(state)
        if set(current_expr) != set(EXPRESSION_DIMS):
            raise ValueError("current_expr must contain all expression dimensions")
        next_expr: dict[str, float] = {}
        for dim in EXPRESSION_DIMS:
            x_t = _clamp01(float(current_expr[dim]))
            tau = float(self.profile.tau[dim])
            x_next = x_t + (attractor[dim] - x_t) * dt / tau
            next_expr[dim] = _clamp01(x_next)
        return next_expr

    def evolve_sequence(
        self,
        states: list[str],
        initial_expr: Mapping[str, float] | None = None,
        dt: float = 1.0,
    ) -> list[dict[str, float]]:
        """Run N sequential steps; initial_expr defaults to profile.base."""
        expr = dict(self.profile.base if initial_expr is None else initial_expr)
        frames: list[dict[str, float]] = []
        for state in states:
            expr = self.step(expr, state, dt=dt)
            frames.append(expr)
        return frames


def load_profile(path: Path) -> ExpressionProfile:
    """Load profile from JSON path."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ExpressionProfile.from_dict(payload)


def save_profile(profile: ExpressionProfile, path: Path) -> None:
    """Save profile to JSON path with UTF-8 and stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(profile.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


__all__ = [
    "EXPRESSION_DIMS",
    "STATE_LABELS",
    "ExpressionProfile",
    "ExpressionDynamics",
    "load_profile",
    "save_profile",
]
