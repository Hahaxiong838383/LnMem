"""
EventTensor 数据契约 — 万能语法 20 槽 schema 的运行时载体

设计来源：[`docs/source-research.md`](../docs/source-research.md) §3.6 接口标准化
设计原则：所有跨模块数据交换都用 EventTensor，模块内部可自由表示。
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .schema_v1 import NUM_SLOTS, SCHEMA_V1, SlotID


@dataclass(frozen=True)
class SlotValue:
    """单槽位赋值 — 含 value / 置信度 / 原文 span 位置（可选）"""
    value: Any                           # 槽位值（字符串、数值、entity_id、link 等）
    confidence: float = 1.0              # [0, 1]
    span: Optional[tuple] = None         # 原文 (start, end) 字符偏移，便于回溯
    raw_text: Optional[str] = None       # 原文片段（debug / UI 用）

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")


@dataclass
class EventTensor:
    """
    单事件张量 — 万能语法 20 槽 schema_v1 的运行时载体

    schema_v1 由 contracts.schema_v1.SCHEMA_V1 冻结
    新增字段必须保持向后兼容（加默认值，不改默认行为）
    """
    user_id: str
    timestamp: datetime
    slots: Dict[SlotID, SlotValue] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    valid_until: Optional[datetime] = None         # bi-temporal: t_invalid，None=valid forever
    raw_modalities: Dict[str, bytes] = field(default_factory=dict)  # 原始音视频/截图 reference
    embedding: Optional[List[float]] = None        # 用于向量检索的事件级 embedding（可选）
    schema_version: str = "v1"                     # 用于未来 schema 升级时的迁移

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be tz-aware (use datetime.now(timezone.utc) etc.)")
        for sid in self.slots:
            if sid not in SCHEMA_V1:
                raise ValueError(f"unknown slot id: {sid} (schema_v1 has {NUM_SLOTS} slots)")

    def get(self, slot: SlotID) -> Optional[SlotValue]:
        return self.slots.get(slot)

    def set(self, slot: SlotID, value: SlotValue) -> "EventTensor":
        """不可变更新 — 返回新 EventTensor，保留原对象。"""
        new_slots = dict(self.slots)
        new_slots[slot] = value
        return EventTensor(
            user_id=self.user_id,
            timestamp=self.timestamp,
            slots=new_slots,
            event_id=self.event_id,
            valid_until=self.valid_until,
            raw_modalities=dict(self.raw_modalities),
            embedding=list(self.embedding) if self.embedding else None,
            schema_version=self.schema_version,
        )

    def filled_priority_count(self) -> Dict[str, int]:
        """统计每个 priority 档已填的槽位数 — evals 用"""
        from .schema_v1 import P0_SLOTS, P1_SLOTS, P2_SLOTS, P3_SLOTS
        present = set(self.slots.keys())
        return {
            "P0": len(present & P0_SLOTS),
            "P1": len(present & P1_SLOTS),
            "P2": len(present & P2_SLOTS),
            "P3": len(present & P3_SLOTS),
        }

    def to_dict(self) -> Dict[str, Any]:
        """JSON-friendly 表示，用于落盘 / 跨进程传输 / W&B logging"""
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "slots": {int(sid): {"value": sv.value, "confidence": sv.confidence,
                                 "span": sv.span, "raw_text": sv.raw_text}
                      for sid, sv in self.slots.items()},
            "schema_version": self.schema_version,
            "has_embedding": self.embedding is not None,
            "raw_modalities_keys": list(self.raw_modalities.keys()),
        }


def empty_event(user_id: str) -> EventTensor:
    """工厂方法 — 当前 UTC 时间 + 空槽位"""
    return EventTensor(user_id=user_id, timestamp=datetime.now(timezone.utc))
