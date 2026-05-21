"""UI helper formatters for EventTensor rendering."""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from contracts.event_tensor import EventTensor, SlotValue
from contracts.schema_v1 import SCHEMA_V1, SlotID


def event_to_dict(event: EventTensor) -> Dict[str, Any]:
    """Convert EventTensor into grouped dict by slot priority for UI."""
    grouped: Dict[str, Dict[str, Any]] = {"P0": {}, "P1": {}, "P2": {}, "P3": {}}
    for slot_id, slot_val in event.slots.items():
        meta = SCHEMA_V1[slot_id]
        grouped[meta.priority.name][meta.name] = _to_plain_value(slot_val)
    return {
        "event_id": event.event_id,
        "user_id": event.user_id,
        "timestamp": event.timestamp.isoformat(),
        "valid_until": event.valid_until.isoformat() if event.valid_until else None,
        # v1.1 新字段（ADR-018）
        "intent_type": event.intent_type.value,
        "schema_version": event.schema_version,
        **grouped,
    }


def event_to_complete_state(event: EventTensor, text_raw: Optional[str] = None) -> Dict[str, Any]:
    """完整状态描述 JSON（川哥钦定 ADR-016 Phase β 输出格式）。

    把 EventTensor 拆成川哥要的 6 维全景：
    - text_raw（原文）
    - universal_grammar_20_slots（万能语法 20 槽 by priority）
    - intent_type（句式 5 类）
    - emotional_state（多模态 V-A + 9-state + AU）
    - lnn_hidden_state（Phase γ base v3 训完后真实化）
    - follow_up_suggestions（M1+ AI 建议）

    示例：
        昨晚 11 点和小李在公司吃了越南粉
        → 川哥能直观看到 AI 对这句话的完整理解
    """
    slots_grouped: Dict[str, Dict[str, Any]] = {"P0": {}, "P1": {}, "P2": {}, "P3": {}}
    for slot_id, slot_val in event.slots.items():
        meta = SCHEMA_V1[slot_id]
        slots_grouped[meta.priority.name][meta.name] = _to_plain_value(slot_val)

    return {
        "text_raw": text_raw or _reconstruct_text_from_slots(event),
        "event_id": event.event_id,
        "user_id": event.user_id,
        "timestamp": event.timestamp.isoformat(),
        "schema_version": event.schema_version,
        "universal_grammar_20_slots": slots_grouped,
        "intent_type": event.intent_type.value,
        "emotional_state": event.multimodal_context if event.multimodal_context else {
            "_status": "无多模态信号 · 用户未抓帧",
            "_note": "ADR-016 base v3 input signal 暂缺",
        },
        "lnn_hidden_state": {
            "_status": "placeholder",
            "_note": "Phase γ（5/27+）base v3 训完后会出 64-dim 真表征",
            "_planned_dim": 64,
        },
        "follow_up_suggestions": {
            "_status": "placeholder",
            "_note": "M1+ 主动建议层（产品定义 §10.1 主动推理）实现",
        },
    }


def _reconstruct_text_from_slots(event: EventTensor) -> str:
    """从 20 槽反推原文（兜底，缺 raw_text 时用）。"""
    pieces = []
    for slot_id in [SlotID.WHEN_TIME, SlotID.WHO_ACTOR, SlotID.WHERE_LOCATION, SlotID.WHAT_ACTION]:
        slot = event.get(slot_id)
        if slot and slot.raw_text:
            pieces.append(slot.raw_text)
        elif slot and slot.value:
            pieces.append(str(slot.value))
    return " ".join(pieces) if pieces else "（无原文）"


def events_to_dataframe(events: Iterable[EventTensor]):
    """Convert EventTensor list into a concise pandas dataframe."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    for event in events:
        priority = event.filled_priority_count()
        rows.append(
            {
                "event_id": event.event_id[:8],
                "timestamp": event.timestamp.strftime("%H:%M:%S"),
                "WHO": _get_slot(event, SlotID.WHO_ACTOR),
                "WHAT": _get_slot(event, SlotID.WHAT_ACTION),
                "WHEN": _get_slot(event, SlotID.WHEN_TIME),
                "WHERE": _get_slot(event, SlotID.WHERE_LOCATION),
                "P0": f"{priority['P0']}/4",
            }
        )
    return pd.DataFrame(rows)


def _get_slot(event: EventTensor, slot_id: SlotID) -> Any:
    slot = event.get(slot_id)
    return _to_plain_value(slot)


def _to_plain_value(slot_value: SlotValue | None) -> Any:
    if slot_value is None:
        return None
    value = slot_value.value
    if isinstance(value, (list, dict, str, int, float, bool)) or value is None:
        return value
    return str(value)
