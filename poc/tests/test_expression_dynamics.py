"""Tests for LTC expression dynamics smoothness and stability."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from poc.src.lnn.expression_dynamics import (
    EXPRESSION_DIMS,
    STATE_LABELS,
    ExpressionDynamics,
    ExpressionProfile,
)


def _profile(tau: float = 4.0, base_shift: float = 0.0) -> ExpressionProfile:
    base = {d: 0.30 + base_shift for d in EXPRESSION_DIMS}
    offsets = {s: {d: 0.0 for d in EXPRESSION_DIMS} for s in STATE_LABELS}
    offsets["focused"] = {d: 0.40 for d in EXPRESSION_DIMS}
    offsets["fatigued"] = {d: -0.20 for d in EXPRESSION_DIMS}
    offsets["emotional_positive"] = {d: 0.55 for d in EXPRESSION_DIMS}
    return ExpressionProfile(
        persona_id=f"p-{tau}-{base_shift}",
        base=base,
        state_offset=offsets,
        tau={d: tau for d in EXPRESSION_DIMS},
    )


def test_attractor_shape_and_range() -> None:
    out = _profile().attractor("focused")
    assert set(out) == set(EXPRESSION_DIMS)
    assert all(0.0 <= out[d] <= 1.0 for d in EXPRESSION_DIMS)


def test_step_short_tau_converges_faster_than_long_tau() -> None:
    fast, slow = ExpressionDynamics(_profile(tau=1.0)), ExpressionDynamics(_profile(tau=10.0))
    init = {d: 0.1 for d in EXPRESSION_DIMS}
    a = fast.profile.attractor("focused")
    xf = fast.step(init, "focused")
    xs = slow.step(init, "focused")
    assert abs(a["smile_intensity"] - xf["smile_intensity"]) < abs(a["smile_intensity"] - xs["smile_intensity"])


def test_state_switch_is_smooth_not_step_jump() -> None:
    dyn = ExpressionDynamics(_profile(tau=4.0))
    frames = dyn.evolve_sequence(["focused", "focused", "fatigued"], dt=1.0)
    prev_expr, next_expr = frames[1], frames[2]
    delta = abs(next_expr["smile_intensity"] - prev_expr["smile_intensity"])
    a_prev = dyn.profile.attractor("focused")["smile_intensity"]
    a_next = dyn.profile.attractor("fatigued")["smile_intensity"]
    bound = abs(a_next - a_prev) * 1.0 / dyn.profile.tau["smile_intensity"]
    assert delta < bound


def test_long_run_converges_to_attractor() -> None:
    dyn = ExpressionDynamics(_profile(tau=3.0))
    final = dyn.evolve_sequence(["focused"] * 80)[-1]
    target = dyn.profile.attractor("focused")
    assert abs(final["jaw_drop"] - target["jaw_drop"]) < 0.01


def test_different_personas_evolve_differently() -> None:
    p1, p2 = ExpressionDynamics(_profile(tau=2.0, base_shift=0.0)), ExpressionDynamics(_profile(tau=8.0, base_shift=0.1))
    init = {d: 0.2 for d in EXPRESSION_DIMS}
    s1 = p1.step(init, "focused")
    s2 = p2.step(init, "focused")
    assert s1 != s2


def test_clamp_works_for_extreme_offsets_and_inputs() -> None:
    profile = _profile()
    profile = ExpressionProfile.from_dict(profile.to_dict() | {"state_offset": {**profile.to_dict()["state_offset"], "emotional_positive": {d: 1.0 for d in EXPRESSION_DIMS}}})
    dyn = ExpressionDynamics(profile)
    out = dyn.step({d: -0.5 for d in EXPRESSION_DIMS}, "emotional_positive", dt=5.0)
    assert all(0.0 <= out[d] <= 1.0 for d in EXPRESSION_DIMS)


def test_profile_dict_roundtrip_is_consistent() -> None:
    profile = _profile(tau=5.0, base_shift=0.05)
    assert ExpressionProfile.from_dict(profile.to_dict()).to_dict() == profile.to_dict()


def test_dt_half_and_one_have_different_paths_but_same_terminal() -> None:
    dyn = ExpressionDynamics(_profile(tau=4.0))
    path_1s = dyn.evolve_sequence(["focused"] * 20, dt=1.0)
    path_05 = dyn.evolve_sequence(["focused"] * 40, dt=0.5)
    assert path_1s[0]["brow_furrow"] != path_05[0]["brow_furrow"]
    assert abs(path_1s[-1]["brow_furrow"] - path_05[-1]["brow_furrow"]) < 0.01


def test_unknown_state_raises() -> None:
    dyn = ExpressionDynamics(_profile())
    with pytest.raises(ValueError):
        dyn.step({d: 0.2 for d in EXPRESSION_DIMS}, "unknown")


def test_profile_immutability() -> None:
    profile = _profile()
    with pytest.raises(FrozenInstanceError):
        profile.persona_id = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        profile.base["smile_intensity"] = 0.9  # type: ignore[index]


def test_smoke_sequence_length_matches_states() -> None:
    dyn = ExpressionDynamics(_profile())
    seq = dyn.evolve_sequence(["neutral", "focused", "fatigued"], dt=1.0)
    assert len(seq) == 3
