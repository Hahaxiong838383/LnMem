"""Collectable placeholder tests for the evals skeleton.

M0 W2-W4 用真实 backend 和 metric 替换这些 skip。
"""
from __future__ import annotations

import pytest


def test_p0_recall_metric_exact_match_case() -> None:
    """P0 recall should be exact and order-insensitive. P0 召回应精确且顺序无关。"""
    pytest.skip("M0 W2 implementation pending: exact-match P0 recall not implemented")


def test_longmem_accuracy_refusal_case() -> None:
    """LongMem accuracy should reward evidence-based refusal. 无证据时应拒绝幻觉。"""
    pytest.skip("M0 W2 implementation pending: refusal-aware longmem judge not implemented")


def test_latency_metric_uses_milliseconds() -> None:
    """Latency metric should use milliseconds consistently. 延迟单位必须统一毫秒。"""
    pytest.skip("M0 W2 implementation pending: latency measurement not implemented")


def test_traffic_light_m1_threshold_boundaries() -> None:
    """M1 traffic light should follow roadmap §3 thresholds. M1 阈值来自路线图。"""
    pytest.skip("M0 W2 implementation pending: traffic-light aggregation not implemented")
