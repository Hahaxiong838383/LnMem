"""
万能语法 20 槽 schema_v1（M0 草案，M1 末冻结）

冻结时间点：[`roadmap/milestones.md`](../roadmap/milestones.md) M1 末
变更规则：M1 末之前可以加字段不能减；M1 末后任何变更必须经老板签字 + ADR 记录

P0 核心骨架（4 槽，召回 100%）   — slots[1..4]
P1 交互上下文（6 槽，召回 ≥90%）— slots[5..10]
P2 关系网络（6 槽，召回 ≥70%）  — slots[11..16]
P3 元信息（4 槽，召回 ≥40%）    — slots[17..20]
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Dict, FrozenSet


class SlotPriority(IntEnum):
    """槽位优先级。数字越小越关键，召回门槛越高。"""
    P0 = 0
    P1 = 1
    P2 = 2
    P3 = 3


class SlotID(IntEnum):
    """20 槽枚举（1..20，对齐 EventTensor 索引）。"""
    # P0 — 核心事件骨架
    WHO_ACTOR = 1
    WHAT_ACTION = 2
    WHEN_TIME = 3
    WHERE_LOCATION = 4
    # P1 — 交互上下文
    LISTENER = 5
    TOPIC = 6
    INTENT = 7
    SENTIMENT = 8
    MODALITY = 9
    SESSION_ID = 10
    # P2 — 关系网络
    RELATED_ENTITIES = 11
    CAUSAL_LINK = 12
    TEMPORAL_LINK = 13
    SPATIAL_LINK = 14
    OBJECT_ATTRS = 15
    SOCIAL_LINK = 16
    # P3 — 元信息
    SOURCE_MODALITY = 17
    CONFIDENCE = 18
    IS_INFERRED = 19
    PRIVACY_LEVEL = 20


@dataclass(frozen=True)
class SlotMeta:
    """槽位元数据。值类型描述给 parser / synth 用，不限制存储格式。"""
    slot_id: SlotID
    name: str           # 英文短名，对齐 source-research.md 命名
    cn_name: str        # 中文释义，给非工程师看
    priority: SlotPriority
    value_type: str     # entity / span / categorical / scalar / enum / link
    description: str    # 一句话描述，作 parser/synth 的 schema-guided prompt


SCHEMA_V1: Dict[SlotID, SlotMeta] = {
    SlotID.WHO_ACTOR: SlotMeta(SlotID.WHO_ACTOR, "who_actor", "行为主体", SlotPriority.P0, "entity",
        "事件的主语/施动者（用户、AI、第三人；指代消解后）"),
    SlotID.WHAT_ACTION: SlotMeta(SlotID.WHAT_ACTION, "what_action", "动作", SlotPriority.P0, "span",
        "事件谓词或核心动词短语（动宾/动补结构）"),
    SlotID.WHEN_TIME: SlotMeta(SlotID.WHEN_TIME, "when_time", "时间", SlotPriority.P0, "scalar",
        "绝对或相对时间，归一化为 ISO8601 或 timedelta（昨晚 → -1d）"),
    SlotID.WHERE_LOCATION: SlotMeta(SlotID.WHERE_LOCATION, "where_location", "地点", SlotPriority.P0, "entity",
        "物理或逻辑位置（家/办公室/微信群/某 app 内）"),

    SlotID.LISTENER: SlotMeta(SlotID.LISTENER, "listener", "听者", SlotPriority.P1, "entity",
        "对话另一方，可能为多人或群组"),
    SlotID.TOPIC: SlotMeta(SlotID.TOPIC, "topic", "话题", SlotPriority.P1, "span",
        "话题简述，用于检索 / 主题聚合"),
    SlotID.INTENT: SlotMeta(SlotID.INTENT, "intent", "意图", SlotPriority.P1, "categorical",
        "ask / inform / request / complain / share / plan / reflect / other"),
    SlotID.SENTIMENT: SlotMeta(SlotID.SENTIMENT, "sentiment", "情绪", SlotPriority.P1, "scalar",
        "情绪极性 [-1, 1]，默认 0"),
    SlotID.MODALITY: SlotMeta(SlotID.MODALITY, "modality", "输入模态", SlotPriority.P1, "enum",
        "speech / text / vision / multimodal"),
    SlotID.SESSION_ID: SlotMeta(SlotID.SESSION_ID, "session_id", "会话 id", SlotPriority.P1, "scalar",
        "会话级唯一 ID（与 user_id × time 绑定）"),

    SlotID.RELATED_ENTITIES: SlotMeta(SlotID.RELATED_ENTITIES, "related_entities", "相关实体", SlotPriority.P2, "entity",
        "本事件提到的其他人/物/项目（list of entity ids）"),
    SlotID.CAUSAL_LINK: SlotMeta(SlotID.CAUSAL_LINK, "causal_link", "因果链", SlotPriority.P2, "link",
        "因果关联到的 event_id（cause-of / caused-by）"),
    SlotID.TEMPORAL_LINK: SlotMeta(SlotID.TEMPORAL_LINK, "temporal_link", "时序链", SlotPriority.P2, "link",
        "时序关联到的 event_id（before / during / after）"),
    SlotID.SPATIAL_LINK: SlotMeta(SlotID.SPATIAL_LINK, "spatial_link", "空间链", SlotPriority.P2, "link",
        "空间共现 event_id"),
    SlotID.OBJECT_ATTRS: SlotMeta(SlotID.OBJECT_ATTRS, "object_attrs", "对象属性", SlotPriority.P2, "scalar",
        "对象的关键属性（颜色 / 大小 / 状态 等结构化键值）"),
    SlotID.SOCIAL_LINK: SlotMeta(SlotID.SOCIAL_LINK, "social_link", "社交关系", SlotPriority.P2, "categorical",
        "亲密度 / 角色（family / friend / colleague / stranger / ai_self）"),

    SlotID.SOURCE_MODALITY: SlotMeta(SlotID.SOURCE_MODALITY, "source_modality", "原始模态", SlotPriority.P3, "enum",
        "原始信号通道（mic / camera / typed / api_event / synthetic）"),
    SlotID.CONFIDENCE: SlotMeta(SlotID.CONFIDENCE, "confidence", "整体置信度", SlotPriority.P3, "scalar",
        "整个事件的整体置信度 [0, 1]，区别于单槽位 confidence"),
    SlotID.IS_INFERRED: SlotMeta(SlotID.IS_INFERRED, "is_inferred", "是否推断", SlotPriority.P3, "categorical",
        "true = LLM 推断而非直接观测"),
    SlotID.PRIVACY_LEVEL: SlotMeta(SlotID.PRIVACY_LEVEL, "privacy_level", "隐私级别", SlotPriority.P3, "enum",
        "public / private / sensitive（影响 KG 治理 + UI 展示策略）"),
}


def slots_by_priority(priority: SlotPriority) -> FrozenSet[SlotID]:
    return frozenset(s for s, m in SCHEMA_V1.items() if m.priority == priority)


P0_SLOTS: FrozenSet[SlotID] = slots_by_priority(SlotPriority.P0)
P1_SLOTS: FrozenSet[SlotID] = slots_by_priority(SlotPriority.P1)
P2_SLOTS: FrozenSet[SlotID] = slots_by_priority(SlotPriority.P2)
P3_SLOTS: FrozenSet[SlotID] = slots_by_priority(SlotPriority.P3)
NUM_SLOTS = 20

assert len(SCHEMA_V1) == NUM_SLOTS, "schema_v1 槽位数必须严格等于 20"
assert len(P0_SLOTS) == 4 and len(P1_SLOTS) == 6 and len(P2_SLOTS) == 6 and len(P3_SLOTS) == 4, \
    "P0/P1/P2/P3 槽位数必须为 4/6/6/4"
