"""Postprocess JSON into EventTensor per contracts/protocols.py.

Maps raw slot candidates to SlotValue and enforces span/priority gates.
"""
from __future__ import annotations

from typing import Mapping, Sequence

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import ParseRequest
from contracts.schema_v1 import SlotID, SlotPriority

from .types import ParseBackendResult, RawSlotCandidate


class JsonPostprocessor:
    """JSON-to-EventTensor mapper. JSON 后处理与槽位门控骨架。"""

    def decode_backend_payload(
        self,
        payload: Mapping[str, object],
        *,
        backend_name: str,
    ) -> ParseBackendResult:
        """Decode backend JSON. 解码后端 JSON，但 M0 不做真实解析。"""
        raise NotImplementedError("M0 W2 todo: robust JSON schema decoding")

    def candidate_to_slot_value(
        self,
        candidate: RawSlotCandidate,
        source_text: str,
    ) -> SlotValue:
        """Convert one candidate. 候选槽位转 SlotValue 并校验 span。"""
        raise NotImplementedError("M0 W2 todo: map candidate to SlotValue")

    def validate_span(self, source_text: str, span: tuple[int, int]) -> None:
        """Validate character span. 校验中文字符偏移和 raw_text 对齐。"""
        raise NotImplementedError("M0 W2 todo: validate span offsets")

    def require_priority_slots(
        self,
        event: EventTensor,
        priorities: Sequence[int],
    ) -> EventTensor:
        """Apply priority gate. 按 require_priorities 过滤或回退。"""
        raise NotImplementedError("M0 W2 todo: enforce required priorities")

    def build_event(
        self,
        req: ParseRequest,
        backend_result: ParseBackendResult,
    ) -> EventTensor:
        """Build EventTensor. 从后端结果构造 EventTensor。"""
        raise NotImplementedError("M0 W2 todo: build EventTensor from slots")

    def priority_to_slot_ids(self, priority: SlotPriority) -> frozenset[SlotID]:
        """Resolve priority slots. 优先级转 schema_v1 槽位集合。"""
        raise NotImplementedError("M0 W2 todo: route priority to slot ids")
