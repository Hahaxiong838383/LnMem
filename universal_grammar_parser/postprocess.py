"""Postprocess JSON into EventTensor per contracts/protocols.py.

Maps raw slot candidates to SlotValue and enforces span/priority gates.
M0 W2 实现（2026-05-12，spike PASS by gemini-3.1-flash-lite 后填实）。
"""
from __future__ import annotations

import json
import logging
from typing import Any, Mapping, Sequence

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import ParseRequest
from contracts.schema_v1 import (
    NUM_SLOTS,
    P0_SLOTS,
    P1_SLOTS,
    P2_SLOTS,
    P3_SLOTS,
    SCHEMA_V1,
    SlotID,
    SlotPriority,
)

from .types import ParseBackendResult, RawSlotCandidate


_log = logging.getLogger(__name__)


# slot 英文 name → SlotID 的逆向映射 + 中文 cn_name alias（F3-04 codex 一审）
_NAME_TO_SLOT_ID: dict[str, SlotID] = {meta.name: sid for sid, meta in SCHEMA_V1.items()}
# 中文 cn_name 也作 alias，应对 LLM 偶尔返中文 key
_CN_NAME_TO_SLOT_ID: dict[str, SlotID] = {meta.cn_name: sid for sid, meta in SCHEMA_V1.items()}


def _resolve_slot_name(name: str) -> SlotID | None:
    """F3-04: 兼容英文 / 中文 key（LLM 偶尔返中文）。

    优先英文 name，其次中文 cn_name，最后大小写不敏感的英文匹配。
    """
    if name in _NAME_TO_SLOT_ID:
        return _NAME_TO_SLOT_ID[name]
    if name in _CN_NAME_TO_SLOT_ID:
        return _CN_NAME_TO_SLOT_ID[name]
    # 大小写不敏感的英文 fallback
    name_lower = name.lower()
    for k, sid in _NAME_TO_SLOT_ID.items():
        if k.lower() == name_lower:
            return sid
    return None


class JsonPostprocessor:
    """JSON-to-EventTensor mapper. JSON 后处理与槽位门控。

    配合 schema-guided LLM prompt 使用：LLM 输出
    `{"slots": {slot_name: {value, confidence, raw_text}}}`
    本类负责 JSON 解析 → SlotValue → EventTensor。
    """

    def decode_backend_payload(
        self,
        payload: Mapping[str, object] | str,
        *,
        backend_name: str,
    ) -> ParseBackendResult:
        """Decode backend JSON.

        Accept dict or raw string (含 markdown fence)。失败返带 diagnostics 的空结果。
        """
        if isinstance(payload, str):
            cleaned = self._strip_markdown_fence(payload)
            try:
                payload_dict = json.loads(cleaned)
            except json.JSONDecodeError as e:
                _log.warning("decode_backend_payload JSON parse failed: %s", e)
                return ParseBackendResult(
                    candidates=(), raw_payload={"_raw_text": payload[:500]},
                    backend_name=backend_name,
                    diagnostics={"parse_error": str(e), "truncated": _looks_truncated(cleaned)},
                )
        elif isinstance(payload, Mapping):
            payload_dict = dict(payload)
        else:
            return ParseBackendResult(
                candidates=(), raw_payload={},
                backend_name=backend_name,
                diagnostics={"parse_error": f"unsupported payload type {type(payload).__name__}"},
            )

        slots_obj = payload_dict.get("slots")
        if not isinstance(slots_obj, dict):
            return ParseBackendResult(
                candidates=(), raw_payload=payload_dict,
                backend_name=backend_name,
                diagnostics={"parse_error": "top-level 'slots' missing or not dict"},
            )

        candidates: list[RawSlotCandidate] = []
        unknown_slots: list[str] = []
        non_dict_slots: list[str] = []
        for slot_name, slot_data in slots_obj.items():
            sid = _resolve_slot_name(slot_name)
            if sid is None:
                unknown_slots.append(slot_name)
                continue
            # F3-07 (codex 一审): slot 局部污染 — 跟踪非 dict slot 数（不静默吞）
            if not isinstance(slot_data, dict):
                non_dict_slots.append(slot_name)
                continue
            value = slot_data.get("value")
            # 显式 null / 空 list / 空 dict 视为未填
            if value is None or (isinstance(value, (list, dict)) and len(value) == 0):
                continue
            confidence_raw = slot_data.get("confidence", 1.0)
            try:
                confidence = float(confidence_raw)
            except (TypeError, ValueError):
                confidence = 0.5
            confidence = max(0.0, min(1.0, confidence))
            raw_text = slot_data.get("raw_text")
            if raw_text is not None and not isinstance(raw_text, str):
                raw_text = str(raw_text)
            span = slot_data.get("span")
            if isinstance(span, (list, tuple)) and len(span) == 2:
                try:
                    span_t: tuple[int, int] | None = (int(span[0]), int(span[1]))
                except (TypeError, ValueError):
                    span_t = None
            else:
                span_t = None
            candidates.append(
                RawSlotCandidate(
                    slot_id=sid, value=value, confidence=confidence,
                    span=span_t, raw_text=raw_text, source=backend_name,
                )
            )

        diagnostics: dict[str, Any] = {}
        if unknown_slots:
            diagnostics["unknown_slot_names"] = unknown_slots
        if non_dict_slots:
            diagnostics["non_dict_slots"] = non_dict_slots
        # F3-04 (codex 一审): unknown_slots 全占 + 0 candidate → 升 hard error 信号
        # （LLM 完全没用 schema_v1 key 时不该静默"成功"）
        if not candidates and unknown_slots and len(unknown_slots) == len(slots_obj):
            diagnostics["all_unknown"] = True
        return ParseBackendResult(
            candidates=tuple(candidates), raw_payload=payload_dict,
            backend_name=backend_name, diagnostics=diagnostics,
        )

    def candidate_to_slot_value(
        self,
        candidate: RawSlotCandidate,
        source_text: str,
    ) -> SlotValue:
        """Convert one candidate. 候选槽位转 SlotValue。

        span 不合法时不抛异常，只 log + 设为 None（fail-soft）。
        """
        span = candidate.span
        if span is not None:
            try:
                self.validate_span(source_text, span)
            except ValueError as e:
                _log.warning("candidate_to_slot_value invalid span %s: %s", span, e)
                span = None
        # raw_text 与 span 对齐校验（仅当 span 非 None 时）
        raw_text = candidate.raw_text
        if span is not None and raw_text is not None:
            actual = source_text[span[0]:span[1]]
            if actual != raw_text:
                _log.debug(
                    "raw_text mismatch with span: span=%s actual=%r vs raw_text=%r",
                    span, actual, raw_text,
                )
                # 不替换（信 LLM 的 raw_text，便于 debug）
        return SlotValue(
            value=candidate.value,
            confidence=candidate.confidence,
            span=span,
            raw_text=raw_text,
        )

    def validate_span(self, source_text: str, span: tuple[int, int]) -> None:
        """Validate character span. 校验中文字符偏移。

        简单版：[start, end) 在 [0, len(source_text)] 内 + start < end。
        中文按 Python str 字符索引（一字符 = len(s)=1），LLM 通常按字符数返回。
        """
        if not isinstance(span, tuple) or len(span) != 2:
            raise ValueError(f"span must be (start, end) tuple, got {span!r}")
        start, end = span
        if not (isinstance(start, int) and isinstance(end, int)):
            raise ValueError(f"span elements must be int, got {type(start).__name__}/{type(end).__name__}")
        if start < 0 or end > len(source_text) or start >= end:
            raise ValueError(
                f"span ({start},{end}) out of range [0,{len(source_text)}) or empty"
            )

    def require_priority_slots(
        self,
        event: EventTensor,
        priorities: Sequence[int],
    ) -> EventTensor:
        """Apply priority gate. 按 require_priorities 检查必填槽位是否存在。

        缺失任一必填 priority 槽时 log warning，但 **不删 event**（fail-soft）—
        调用方根据 event.filled_priority_count() 决定是否使用。
        """
        present = set(event.slots.keys())
        for prio_int in priorities:
            try:
                prio = SlotPriority(prio_int)
            except ValueError:
                _log.warning("require_priority_slots: invalid priority %r", prio_int)
                continue
            required = self.priority_to_slot_ids(prio)
            missing = required - present
            if missing:
                missing_names = sorted(SCHEMA_V1[s].name for s in missing)
                _log.info(
                    "require_priority_slots: P%d missing %d slots: %s",
                    int(prio), len(missing), missing_names,
                )
        return event

    def build_event(
        self,
        req: ParseRequest,
        backend_result: ParseBackendResult,
    ) -> EventTensor:
        """Build EventTensor from candidates. 从后端结果构造 EventTensor。"""
        slots: dict[SlotID, SlotValue] = {}
        for cand in backend_result.candidates:
            try:
                sv = self.candidate_to_slot_value(cand, req.text)
                slots[cand.slot_id] = sv
            except (ValueError, TypeError) as e:
                _log.warning(
                    "build_event skip slot %s: %s", SCHEMA_V1[cand.slot_id].name, e,
                )
        event = EventTensor(
            user_id=req.user_id,
            timestamp=req.timestamp,
            slots=slots,
        )
        # 优先级门控（仅 log 不变更）
        if req.require_priorities:
            event = self.require_priority_slots(event, req.require_priorities)
        return event

    def priority_to_slot_ids(self, priority: SlotPriority) -> frozenset[SlotID]:
        """Resolve priority slots. 优先级转 schema_v1 槽位集合。"""
        return {
            SlotPriority.P0: P0_SLOTS,
            SlotPriority.P1: P1_SLOTS,
            SlotPriority.P2: P2_SLOTS,
            SlotPriority.P3: P3_SLOTS,
        }[priority]

    # ---------- 内部辅助 ----------

    @staticmethod
    def _strip_markdown_fence(text: str) -> str:
        """Strip ```json ... ``` markdown fence if present."""
        cleaned = text.strip()
        if not cleaned.startswith("```"):
            return cleaned
        lines = cleaned.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].rstrip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()


def _looks_truncated(text: str) -> bool:
    """启发式判断 JSON 是否被截断（max_tokens 超出）。"""
    s = text.strip()
    if not s:
        return False
    # 完整 JSON 应以 } 或 ] 结尾
    if s[-1] not in ("}", "]"):
        return True
    # 大括号配对
    open_braces = s.count("{")
    close_braces = s.count("}")
    return open_braces != close_braces
