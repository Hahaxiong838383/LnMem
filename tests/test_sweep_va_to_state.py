"""Tests for V-A sweep script; 覆盖均匀分布、全 neutral、决策树选优。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "sweep_va_to_state.py"
_spec = importlib.util.spec_from_file_location("sweep_va_to_state", _SCRIPT)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _frame_df(valence: np.ndarray, arousal: np.ndarray, filename: str = "0.mp4") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "filename": [filename] * len(valence),
            "frame_idx": np.arange(len(valence), dtype=int),
            "valence": valence,
            "arousal": arousal,
        }
    )


def test_sweep_balanced_data() -> None:
    """360 个均匀角度点应让 10 组候选都接近 coverage_balance=1。"""
    angles = np.radians(np.arange(360, dtype=float) + 0.5)
    df = _frame_df(0.9 * np.cos(angles), 0.9 * np.sin(angles))
    report = mod.build_report(df, [0.1, 0.15, 0.2, 0.25, 0.3], [0.0, 22.5], n_states=8)
    assert len(report["candidate_schemes"]) == 10
    for candidate in report["candidate_schemes"]:
        assert candidate["metrics"]["coverage_balance"] == pytest.approx(1.0, rel=0.05)
        assert candidate["metrics"]["boundary_stability"] > 0.0


def test_sweep_neutral_concentration() -> None:
    """全在中性圆内时应全部映射到 state=0，coverage_balance 记为 infinity。"""
    df = _frame_df(np.full(64, 0.03), np.full(64, 0.04))
    candidate = mod.evaluate_scheme(df, 0.1, 0.0, n_states=8)
    assert candidate["state_counts"][0] == 64
    assert candidate["state_counts"][1:] == [0] * 8
    assert candidate["metrics"]["coverage_balance"] == "inf"
    assert candidate["metrics"]["coverage_unbounded"] is True
    assert candidate["metrics"]["boundary_stability"] == 0.0


def test_sweep_recommendation_decision_tree() -> None:
    """通过 coverage 且 boundary 最低的方案应优先于 fail coverage 或更高 boundary。"""
    recommendation = mod.pick_recommendation(
        [
                {
                    "scheme_id": "scheme_a",
                    "metrics": {
                        "coverage_balance": 2.0,
                        "boundary_stability": 0.05,
                        "va_reconstruction_mse": 0.30,
                        "state_entropy": 2.0,
                    },
                },
                {
                    "scheme_id": "scheme_b",
                    "metrics": {
                        "coverage_balance": 6.2,
                        "boundary_stability": 0.01,
                        "va_reconstruction_mse": 0.10,
                        "state_entropy": 2.9,
                    },
                },
                {
                    "scheme_id": "scheme_c",
                    "metrics": {
                        "coverage_balance": 1.8,
                        "boundary_stability": 0.09,
                        "va_reconstruction_mse": 0.28,
                        "state_entropy": 2.1,
                    },
                },
            ]
        )
    assert recommendation["scheme_id"] == "scheme_a"
    assert "boundary_stability" in recommendation["reason"]


def test_sweep_state_entropy_and_decision_priority() -> None:
    """当 boundary 略高但 entropy 明显更高时应优先 entropy 候选。"""
    recommendation = mod.pick_recommendation(
        [
            {
                "scheme_id": "scheme_a",
                "metrics": {
                    "coverage_balance": 2.0,
                    "boundary_stability": 0.05,
                    "va_reconstruction_mse": 0.30,
                    "state_entropy": 2.0,
                },
            },
            {
                "scheme_id": "scheme_b",
                "metrics": {
                    "coverage_balance": 2.0,
                    "boundary_stability": 0.06,
                    "va_reconstruction_mse": 0.30,
                    "state_entropy": 2.5,
                },
            },
        ]
    )
    assert recommendation["scheme_id"] == "scheme_b"
    assert "state_entropy" in recommendation["reason"]
