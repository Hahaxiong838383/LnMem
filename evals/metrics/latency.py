"""Latency metric skeleton for contracts/protocols.py EvalResult.

端到端 latency_ms 真实计算在 M0 W2 填。
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.protocols import EvalCase, EvalResult


@dataclass(frozen=True)
class LatencyMetric:
    """End-to-end latency metric in milliseconds. 端到端延迟指标。"""

    threshold_ms: float = 5000.0
    name: str = "latency_ms"

    def score(self, case: EvalCase, elapsed_ms: float) -> EvalResult:
        """Score elapsed milliseconds against budget. 评估毫秒级延迟。"""
        raise NotImplementedError("M0 W2 todo: compare elapsed_ms with latency budget")
