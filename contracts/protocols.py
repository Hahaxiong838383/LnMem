"""
4 个 M0 模块的 Protocol 接口契约 — 4 路 codex 实现的边界

设计原则：
- 接口先行（[`CLAUDE.md`](../CLAUDE.md) §3 三道门第 1 道）
- 模块之间唯一交换数据：EventTensor + 显式 dataclass 入参
- Protocol 用 typing.Protocol（duck typing），实现类不需要 import 这个文件
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Protocol, Sequence, Tuple

from .event_tensor import EventTensor
from .schema_v1 import SlotID


# ============================================================================
# 1. universal_grammar_parser  —  自然语言 → 20 槽 EventTensor
# ============================================================================

@dataclass(frozen=True)
class ParseRequest:
    text: str                              # 原始中/英文句子
    user_id: str
    timestamp: datetime
    context_window: Sequence[str] = ()     # 前 N 句对话上下文（可选）
    require_priorities: Tuple[int, ...] = (0,)  # 至少要填的优先级，默认 P0


class UniversalGrammarParser(Protocol):
    """中文文本 → EventTensor。"""

    def parse(self, req: ParseRequest) -> EventTensor: ...

    def parse_batch(self, reqs: Sequence[ParseRequest]) -> List[EventTensor]: ...


# ============================================================================
# 2. synthetic_data_generator  —  合成 EventTensor 样本（类型 1/2/3）
# ============================================================================

@dataclass(frozen=True)
class SynthSpec:
    """
    类型 1 = LLM 自动生成万能语法槽位填充对话样本（v0.3 §6.1）
    类型 2 = 50 虚拟主人 ×30 天日志 → 多轮场景模拟
    类型 3 = 老板 + 种子用户真实日志（占比受 roadmap §3.2 监控限制）
    """
    kind: str                              # "type_1" / "type_2" / "type_3"
    n_samples: int
    seed: int = 42
    persona_pool: Optional[Sequence[str]] = None    # type_2 用
    real_log_path: Optional[str] = None             # type_3 用
    boss_share_cap: float = 0.5                     # 老板贡献占比上限（roadmap §3.2）


class SyntheticDataGenerator(Protocol):
    """合成数据 pipeline — 输出对齐万能语法 schema_v1 的 EventTensor 序列。"""

    def generate(self, spec: SynthSpec) -> Iterable[EventTensor]: ...

    def stats(self) -> dict:
        """返回当前累计的占比统计 — evals + roadmap §3.2 老板占比监控用"""
        ...


# ============================================================================
# 3. graphiti_zh_adapter  —  EventTensor ↔ Graphiti bi-temporal KG
# ============================================================================

@dataclass(frozen=True)
class KGQuery:
    user_id: str
    text_query: Optional[str] = None
    slot_filter: Optional[dict] = None     # {SlotID: value} 精确匹配
    valid_at: Optional[datetime] = None    # bi-temporal: t_valid 查询
    top_k: int = 10


@dataclass(frozen=True)
class KGHit:
    event: EventTensor
    score: float                           # 召回分数（向量 + bm25 + temporal 综合）
    explanation: Optional[str] = None      # 命中槽位 / 路径


class GraphitiZhAdapter(Protocol):
    """
    Graphiti 中文适配 — 把 EventTensor 写入 KG，并支持时间感知召回。

    M0 末验收：
    - P0 槽位 100% 入图
    - 中文 entity F1 ≥ 0.5（roadmap §3.2 红线）
    - 30 分钟窗口内逐句精确匹配召回 100%
    """

    def upsert(self, event: EventTensor) -> str:                  # 返回 graph node id
        ...

    def query(self, q: KGQuery) -> List[KGHit]: ...

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """bi-temporal 失效（不删除节点，只标记 t_invalid）"""
        ...


# ============================================================================
# 4. evals  —  评估器 / 三色判定 / 中文 LongMemEval
# ============================================================================

@dataclass(frozen=True)
class EvalCase:
    case_id: str
    description: str                       # 场景描述（"30 分钟前提到的人现在询问"）
    setup_events: Sequence[EventTensor]    # 注入到记忆系统的事件
    query: str                             # 自然语言查询
    expected_event_ids: Sequence[str]      # 标准答案的 event_id 集合
    metric: str                            # "p0_recall" / "longmem_acc" / "latency_ms"


@dataclass(frozen=True)
class EvalResult:
    case_id: str
    metric: str
    score: float
    passed: bool
    threshold: float
    details: Optional[dict] = None


@dataclass(frozen=True)
class TrafficLight:
    """三色判定结果（roadmap §3）"""
    color: str                             # "RED" / "YELLOW" / "GREEN"
    metric_summary: dict
    triggered_rules: Sequence[str]
    next_action: str


class EvalSuite(Protocol):
    """评估套件 — 跑 100 case 中文 LongMemEval + 自动三色判定"""

    def run(self, cases: Sequence[EvalCase]) -> List[EvalResult]: ...

    def traffic_light(self, results: Sequence[EvalResult], milestone: str) -> TrafficLight:
        """根据 milestone（M1/M2/M3 末）和 results 生成红/黄/绿"""
        ...
