"""Stub case loader for contracts/protocols.py EvalCase.

JSONL 占位数据转成 EventTensor，真实 100 题 M0 W3 填。
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import EvalCase
from contracts.schema_v1 import SlotID

DEFAULT_CASE_PATH = Path(__file__).with_name("data") / "cases.jsonl"


class EvalCaseKind(str, Enum):
    """Case kind taxonomy for Chinese LongMemEval. 中文题型分类。"""

    SINGLE_EVENT = "single_event"
    MULTI_HOP = "multi_hop"
    TIME_AWARE = "time_aware"
    HALLUCINATION_REFUSAL = "hallucination_refusal"


@dataclass(frozen=True)
class RawEvalCase:
    """Raw JSONL case envelope before conversion. JSONL 原始记录。"""

    case_id: str
    kind: EvalCaseKind
    description: str
    setup_events: List[Mapping[str, Any]]
    query: str
    expected_event_ids: List[str]
    metric: str


def load_cases(path: str | Path = DEFAULT_CASE_PATH) -> List[EvalCase]:
    """Load stub eval cases from JSONL. 从 JSONL 加载占位评估题。"""
    case_path = Path(path)
    cases: List[EvalCase] = []
    with case_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = _raw_case_from_dict(json.loads(line))
            cases.append(
                EvalCase(
                    case_id=raw.case_id,
                    description=f"[{raw.kind.value}] {raw.description}",
                    setup_events=tuple(_event_from_dict(item) for item in raw.setup_events),
                    query=raw.query,
                    expected_event_ids=tuple(raw.expected_event_ids),
                    metric=raw.metric,
                )
            )
    return cases


def _raw_case_from_dict(data: Mapping[str, Any]) -> RawEvalCase:
    return RawEvalCase(
        case_id=str(data["case_id"]),
        kind=EvalCaseKind(str(data["kind"])),
        description=str(data["description"]),
        setup_events=list(data.get("setup_events", [])),
        query=str(data["query"]),
        expected_event_ids=list(data.get("expected_event_ids", [])),
        metric=str(data["metric"]),
    )


def _event_from_dict(data: Mapping[str, Any]) -> EventTensor:
    slots = {
        SlotID(int(slot_id)): SlotValue(
            value=slot_payload.get("value"),
            confidence=float(slot_payload.get("confidence", 1.0)),
            span=tuple(slot_payload["span"]) if slot_payload.get("span") else None,
            raw_text=slot_payload.get("raw_text"),
        )
        for slot_id, slot_payload in dict(data.get("slots", {})).items()
    }
    return EventTensor(
        user_id=str(data["user_id"]),
        timestamp=datetime.fromisoformat(str(data["timestamp"])),
        slots=slots,
        event_id=str(data["event_id"]),
        schema_version=str(data.get("schema_version", "v1")),
    )
