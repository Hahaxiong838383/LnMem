"""F1 Attribution Ledger — Phase A lineage 优先

实现 ADR-038 §F1: 每次 online fit_run 记录 lineage 三表
(fit_runs / event_attribution / delta_summary)。

模块结构:
- contracts.py: Protocol + frozen dataclass schema (零外部依赖)
- sqlite_writer.py: SQLite 真源实现 (Phase A, codex 实施)
- parquet_exporter.py: 异步分析导出 (Phase A 后, 不阻塞主路径)
- cli.py: `lnmem ledger lineage <ckpt_sha8>` 命令 (Phase A)

ADR-038 v1.1 决策权: schema 决策权归 codex (5.4 high), cc 实施集成.
"""

from poc.src.lnn.ledger.contracts import (
    FitRunRecord,
    EventAttributionRecord,
    DeltaSummaryRecord,
    LedgerWriter,
    LedgerReader,
    LedgerSchemaVersion,
)

__all__ = [
    "FitRunRecord",
    "EventAttributionRecord",
    "DeltaSummaryRecord",
    "LedgerWriter",
    "LedgerReader",
    "LedgerSchemaVersion",
]
