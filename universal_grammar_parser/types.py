"""Shared parser DTOs for contracts/protocols.py UniversalGrammarParser.

Keeps raw backend outputs separate from EventTensor contract mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from contracts.event_tensor import IntentType  # re-export 单一来源（ADR-018）
from contracts.schema_v1 import SlotID

__all__ = ["IntentType", "RawSlotCandidate", "ParseBackendResult"]


@dataclass(frozen=True)
class RawSlotCandidate:
    """Raw slot proposal. 后端候选槽位，尚未写入 SlotValue。"""

    slot_id: SlotID
    value: Any
    confidence: float
    span: Optional[Tuple[int, int]] = None
    raw_text: Optional[str] = None
    source: str = "unknown"


@dataclass(frozen=True)
class ParseBackendResult:
    """Backend parse result. 后端原始解析结果，等待 postprocess。"""

    candidates: Tuple[RawSlotCandidate, ...] = ()
    raw_payload: Mapping[str, Any] = field(default_factory=dict)
    backend_name: str = "unknown"
    diagnostics: Dict[str, Any] = field(default_factory=dict)
