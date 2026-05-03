"""LongMem accuracy metric skeleton for contracts/protocols.py EvalResult.

中文 LongMemEval 准确率真实计算在 M0 W3 填。
"""
from __future__ import annotations

from dataclasses import dataclass

from contracts.protocols import EvalCase, EvalResult


@dataclass(frozen=True)
class LongMemAccuracyMetric:
    """Chinese LongMemEval accuracy metric. 中文长期记忆准确率指标。"""

    threshold: float = 0.60
    name: str = "longmem_acc"

    def score(self, case: EvalCase, answer: str) -> EvalResult:
        """Score answer against the case oracle. 根据标准答案评估回答。"""
        raise NotImplementedError("M0 W3 todo: implement answer judging and refusal handling")
