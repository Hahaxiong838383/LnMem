"""
EventTensor 数据契约 — 万能语法 20 槽 schema 的运行时载体

设计来源：[`docs/source-research.md`](../docs/source-research.md) §3.6 接口标准化
设计原则：所有跨模块数据交换都用 EventTensor，模块内部可自由表示。

schema 演进记录：
- v1.0（M0 草案，2026-05-04 封版）：20 槽 + slots + raw_modalities + embedding
- v1.1（2026-05-21，ADR-018 升级）：加 intent_type（5 类句式）+ multimodal_context
  （两个均为可选字段，默认值兜底，旧代码不传不报错）
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .schema_v1 import NUM_SLOTS, SCHEMA_V1, SlotID


class IntentType(str, Enum):
    """
    v1.1 新增：句式意图类型（5 类）

    与 SlotID.INTENT（slot 7 语义意图，如 "分享回忆" / "道歉"）正交：
    - intent_type 是**句式层面**的意图（陈述还是疑问）
    - SlotID.INTENT 是**语义层面**的意图（这句话想达成什么目的）

    川哥 2026-05-21 钦定：「我是疑问句还是反问句？我是哭着说还是笑着说？」
    LnMem 要让 LNN 学到主人的心理活动，必须保留句式信号。
    """
    DECLARATIVE = "陈述"      # 陈述事实/经历："昨晚吃了越南粉"
    QUESTION = "疑问"          # 真问问题："今天几号"
    RHETORICAL = "反问"        # 反问语气，不期待答案："这能怪我吗"
    EXCLAMATION = "感叹"       # 情绪宣泄："太累了！"
    REQUEST = "请求"           # 请求行动："帮我订一下票"


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

    schema 由 contracts.schema_v1.SCHEMA_V1 冻结
    新增字段必须保持向后兼容（加默认值，不改默认行为）

    v1.1 新增（2026-05-21 ADR-018）：
    - intent_type: 句式意图 5 类（默认 DECLARATIVE 兜底）
    - multimodal_context: V-A + AU + 9-state（ADR-016 base v3 input signal）
    """
    user_id: str
    timestamp: datetime
    slots: Dict[SlotID, SlotValue] = field(default_factory=dict)
    event_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    valid_until: Optional[datetime] = None         # bi-temporal: t_invalid，None=valid forever
    raw_modalities: Dict[str, bytes] = field(default_factory=dict)  # 原始音视频/截图 reference
    embedding: Optional[List[float]] = None        # 用于向量检索的事件级 embedding（可选）

    # v1.1 新字段（向后兼容，旧代码不传不报错）
    intent_type: IntentType = IntentType.DECLARATIVE
    multimodal_context: Optional[Dict[str, Any]] = None  # {"v_a": [...], "state_label": ..., "au_vector": {...}, "source": ..., "confidence": ...}

    schema_version: str = "v1.1"                   # 用于未来 schema 升级时的迁移

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("timestamp must be tz-aware (use datetime.now(timezone.utc) etc.)")
        for sid in self.slots:
            if sid not in SCHEMA_V1:
                raise ValueError(f"unknown slot id: {sid} (schema_v1 has {NUM_SLOTS} slots)")
        # 容错：如果上游传字符串而不是 IntentType enum，自动转换
        if isinstance(self.intent_type, IntentType):
            pass  # 已是 enum，OK
        elif isinstance(self.intent_type, str):
            try:
                object.__setattr__(self, "intent_type", IntentType(self.intent_type))
            except ValueError:
                raise ValueError(
                    f"intent_type must be one of {[t.value for t in IntentType]}, "
                    f"got {self.intent_type!r}"
                )
        else:
            raise TypeError(
                f"intent_type must be IntentType or str, got {type(self.intent_type).__name__}"
            )

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
            intent_type=self.intent_type,
            multimodal_context=dict(self.multimodal_context) if self.multimodal_context else None,
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
            # v1.1 新字段
            "intent_type": self.intent_type.value,
            "multimodal_context": self.multimodal_context,
            "schema_version": self.schema_version,
            "has_embedding": self.embedding is not None,
            "raw_modalities_keys": list(self.raw_modalities.keys()),
        }


def empty_event(user_id: str) -> EventTensor:
    """工厂方法 — 当前 UTC 时间 + 空槽位"""
    return EventTensor(user_id=user_id, timestamp=datetime.now(timezone.utc))
