"""P0 recall metric skeleton for contracts/protocols.py EvalResult.

P0 召回真实计算在 M0 W2 填。
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from contracts.protocols import EvalCase, EvalResult
from contracts.schema_v1 import P0_SLOTS

P0_REQUIRED_SLOT_IDS = tuple(int(slot_id) for slot_id in P0_SLOTS)


@dataclass(frozen=True)
class P0RecallMetric:
    """P0 exact-match recall metric. P0 槽位精确召回指标。"""

    threshold: float = 1.0
    name: str = "p0_recall"

    def score(self, case: EvalCase, retrieved_event_ids: Sequence[str]) -> EvalResult:
        """Score retrieved ids against expected ids. 计算期望事件召回率。"""
        raise NotImplementedError("M0 W2 todo: compute P0 recall using expected_event_ids")
