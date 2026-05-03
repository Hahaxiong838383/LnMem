"""contracts/ — LnMem 公共数据契约（4 路 codex 唯一可依赖的共享 API）

任何模块跨边界的数据交换都必须用 contracts.* 中定义的类型。
模块内部实现（私有类型 / DataFrame / pickle 缓存）随便。
"""
from .event_tensor import EventTensor, SlotValue, empty_event
from .protocols import (
    EvalCase,
    EvalResult,
    EvalSuite,
    GraphitiZhAdapter,
    KGHit,
    KGQuery,
    ParseRequest,
    SyntheticDataGenerator,
    SynthSpec,
    TrafficLight,
    UniversalGrammarParser,
)
from .schema_v1 import (
    NUM_SLOTS,
    P0_SLOTS,
    P1_SLOTS,
    P2_SLOTS,
    P3_SLOTS,
    SCHEMA_V1,
    SlotID,
    SlotMeta,
    SlotPriority,
)

__version__ = "0.1.0"  # contracts schema version — 与 schema_v1 解耦

__all__ = [
    # event tensor
    "EventTensor", "SlotValue", "empty_event",
    # schema
    "SlotID", "SlotMeta", "SlotPriority", "SCHEMA_V1",
    "P0_SLOTS", "P1_SLOTS", "P2_SLOTS", "P3_SLOTS", "NUM_SLOTS",
    # parser
    "UniversalGrammarParser", "ParseRequest",
    # synth
    "SyntheticDataGenerator", "SynthSpec",
    # graphiti
    "GraphitiZhAdapter", "KGQuery", "KGHit",
    # evals
    "EvalSuite", "EvalCase", "EvalResult", "TrafficLight",
]
