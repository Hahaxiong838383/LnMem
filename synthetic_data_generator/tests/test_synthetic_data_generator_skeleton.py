"""Placeholder cases for contracts/protocols.py::SyntheticDataGenerator.

Case names document M0 W2-W4 behavior while collection stays green.
"""
from __future__ import annotations

import pytest


def test_type_1_llm_generation_fills_p0_slots() -> None:
    """Type 1 should create schema-guided EventTensor samples. 类型 1 需覆盖 P0。"""
    pytest.skip("M0 W2 implementation pending: type_1 LLM prompt and parser")


def test_type_2_persona_simulation_spans_50_personas_30_days() -> None:
    """Type 2 should simulate 50 owners for 30 days. 类型 2 需覆盖人群和时间线。"""
    pytest.skip("M0 W2 implementation pending: persona pool and timeline simulator")


def test_type_3_real_log_injection_enforces_boss_share_cap() -> None:
    """Type 3 should pause when boss share exceeds cap. 类型 3 必须受占比红线约束。"""
    pytest.skip("M0 W2 implementation pending: real-log monitor and pause logic")


def test_pipeline_stats_exposes_monitor_snapshot() -> None:
    """Stats should include monitor data. stats 需暴露 evals 可读的占比统计。"""
    pytest.skip("M0 W2 implementation pending: pipeline stats aggregation")
