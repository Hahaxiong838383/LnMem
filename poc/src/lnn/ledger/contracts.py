"""F1 Attribution Ledger 契约 — frozen dataclass + Protocol

实现 ADR-038 §F1 三层 schema (codex 钦定 v1.1):
- fit_runs: 一次 online fit_step 的 lineage 锚 (ckpt_in/out_sha8 + event_uuid_list_hash)
- event_attribution: 该 fit_run 吃了哪些 event (per-event membership, 不是 per-event 梯度账本)
- delta_summary: 该 fit_run 的整体参数变化摘要 (L2 / topk dims, 不分 event)

设计原则
--------
1. **接口先行** (CLAUDE.md §3): 本文件零 sqlite/torch 依赖, 下游 sqlite_writer.py 实施.
2. **immutability** (~/.claude/rules/coding-style.md): 所有 dataclass frozen=True.
3. **LNN-FIRST P-1**: lineage 是治理基础设施, 不替代 LNN 主路径. 不违 P-1.
4. **可逆性 = Surviving Feedback Replay**: F1 Phase A 不做反向抑制, 只记录 lineage 供
   Phase B 重放使用. 这是 codex+gemini 二轮评审共识的数学合法路径.

Phase A DoD (ADR-038 §F1):
- 任意 ckpt_out_sha8 可反查 fit_run_id → event_uuid 列表 → ckpt_in_sha8
- CLI `lnmem ledger lineage <ckpt_sha8>` 输出完整 lineage 链
- 断电恢复后 ledger consistency (0 孤儿记录)

下游消费方
----------
- sqlite_writer.py: SQLite 真源 (Phase A 主路径)
- parquet_exporter.py: 异步分析导出
- fit_orchestrator.py: 集成点 (写 ckpt 后事务写 ledger)
- cli.py: lnmem ledger lineage / list / inspect
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

LedgerSchemaVersion = "ledger_v1"


# ---------- 记录类型 (frozen dataclass) ----------


@dataclass(frozen=True)
class FitRunRecord:
    """一次 online fit_step 的 lineage 锚.

    主键: fit_run_id (UUID4 string).
    Lineage: ckpt_in_sha8 → ckpt_out_sha8.
    Event membership: event_uuid_list_hash (sha256(sorted(event_uuids)) 前 16 字符).

    ADR-038 v1.1 §F1 codex schema:
        fit_runs(fit_run_id, ckpt_in_sha8, ckpt_out_sha8, fired_at_utc,
                 persona_idx, event_uuid_list_hash, batch_size,
                 config_snapshot_json, success_flag)
    """

    fit_run_id: str
    ckpt_in_sha8: str
    ckpt_out_sha8: str
    fired_at_utc: str  # ISO 8601 UTC
    persona_idx: int
    event_uuid_list_hash: str  # sha256(sorted event_uuids)[:16]
    batch_size: int
    config_snapshot_json: str  # OnlineAdapterConfig dataclass → json.dumps
    success_flag: bool

    def __post_init__(self) -> None:
        if not self.fit_run_id:
            raise ValueError("fit_run_id required")
        if len(self.ckpt_in_sha8) != 8 or len(self.ckpt_out_sha8) != 8:
            raise ValueError("ckpt_*_sha8 must be 8-char hex")
        if len(self.event_uuid_list_hash) != 16:
            raise ValueError("event_uuid_list_hash must be 16-char hex")


@dataclass(frozen=True)
class EventAttributionRecord:
    """该 fit_run 吃了哪个 event (membership, 不含梯度归因).

    ADR-038 v1.1 §F1: per-fit-run + event membership, 不做 per-event per-dim attribution
    (codex 钦定: 3 天工期不足以做 dense 归因, 且 replay 比反向抑制更可信).

    复合主键: (fit_run_id, event_uuid).
    """

    fit_run_id: str
    event_uuid: str
    event_order: int  # 在 batch 内的顺序 (0-indexed), 关系 replay 重放顺序
    reward: float | None
    label_source: str  # user_active / user_passive_response / system_inferred
    window_frame_ids: str  # JSON list of frame ids in alignment window
    contribution_mode: str  # "membership_only" for Phase A; future: "delta_attributed"

    def __post_init__(self) -> None:
        if not self.event_uuid:
            raise ValueError("event_uuid required")
        if self.event_order < 0:
            raise ValueError("event_order must be >= 0")
        if self.contribution_mode not in {"membership_only", "delta_attributed"}:
            raise ValueError(
                f"contribution_mode unknown: {self.contribution_mode}; "
                f"allowed: membership_only / delta_attributed"
            )


@dataclass(frozen=True)
class DeltaSummaryRecord:
    """该 fit_run 的整体参数变化摘要 (不分 event).

    用于快速回答 "这个 fit_run 改了多少 / 改了哪些维度"
    不替代 per-event 归因 (Phase A 不做).

    主键 = fit_run_id (1:1).
    """

    fit_run_id: str
    delta_l2_total: float
    delta_l2_by_persona_json: str  # {persona_idx: l2_value} JSON
    topk_dims_json: str  # top-k 维度索引 + delta 值 JSON list


# ---------- Writer / Reader Protocol ----------


@runtime_checkable
class LedgerWriter(Protocol):
    """Ledger 写入接口.

    Fail-safe 约定 (ADR-038 v1.1):
    1. 先原子写 ckpt (调用方负责)
    2. 再调 write_fit_run() — 事务写 SQLite 真源
    3. parquet 异步 materialize (write_fit_run 不阻塞)

    实现要求:
    - write_fit_run 是事务的 (要么三表全成功, 要么全回滚)
    - 写入失败 raise LedgerWriteError, 不静默吞错
    - 不依赖 torch (Phase A schema 是纯 Python 数据)
    """

    def write_fit_run(
        self,
        fit_run: FitRunRecord,
        events: list[EventAttributionRecord],
        delta_summary: DeltaSummaryRecord,
    ) -> None:
        """事务写一次 fit_run 全套记录. 失败 raise."""
        ...

    def ensure_schema(self) -> None:
        """初始化或迁移 schema. 幂等."""
        ...


@runtime_checkable
class LedgerReader(Protocol):
    """Ledger 读取接口 (CLI / dashboard 用).

    实现要求:
    - 只读, 不修改
    - lineage 查询 O(log N), 必须建索引
    - 支持断电后 consistency check (孤儿记录检测)
    """

    def lineage_by_ckpt_out(self, ckpt_out_sha8: str) -> FitRunRecord | None:
        """根据 ckpt_out_sha8 反查 fit_run. 找不到返回 None."""
        ...

    def lineage_by_ckpt_in(self, ckpt_in_sha8: str) -> list[FitRunRecord]:
        """根据 ckpt_in_sha8 查所有以它为输入的 fit_run (1:N)."""
        ...

    def events_of_run(self, fit_run_id: str) -> list[EventAttributionRecord]:
        """该 fit_run 吃了哪些 event."""
        ...

    def delta_of_run(self, fit_run_id: str) -> DeltaSummaryRecord | None:
        """该 fit_run 的参数变化摘要."""
        ...

    def consistency_check(self) -> ConsistencyReport:
        """断电恢复用. 检测孤儿记录 (event_attribution 引用不存在的 fit_run_id 等)."""
        ...


@dataclass(frozen=True)
class ConsistencyReport:
    """Ledger 完整性报告 (consistency_check 输出).

    DoD: Phase A 模拟断电 3 次后, 所有指标必须为 0.
    """

    total_fit_runs: int
    total_event_records: int
    total_delta_records: int
    orphan_events: int  # event_attribution.fit_run_id 不在 fit_runs
    orphan_deltas: int  # delta_summary.fit_run_id 不在 fit_runs
    runs_without_events: int  # fit_runs 没有任何 event_attribution
    runs_without_delta: int  # fit_runs 没有 delta_summary

    @property
    def is_clean(self) -> bool:
        return (
            self.orphan_events == 0
            and self.orphan_deltas == 0
            and self.runs_without_events == 0
            and self.runs_without_delta == 0
        )


# ---------- Exceptions ----------


class LedgerWriteError(RuntimeError):
    """Ledger 写入失败. Caller 必须处理 (不静默吞)."""


class LedgerReadError(RuntimeError):
    """Ledger 读取失败 (DB corrupt / schema mismatch 等)."""


class LedgerSchemaMismatchError(LedgerReadError):
    """Schema 版本不匹配, 需要 migration."""
