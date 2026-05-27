"""F1 Attribution Ledger 契约测试 (ADR-038 v1.1 §F1 Phase A)

测试 contracts.py 的 frozen dataclass 验证逻辑 + Protocol 类型可识别性.
不依赖 sqlite 实现 (那由 codex 在 sqlite_writer.py 实施).

ADR-038 v1.1 DoD 覆盖:
- 任意 ckpt_out_sha8 可反查 fit_run_id → event_uuid 列表 (接口形状测试)
- 断电恢复后 ledger consistency (ConsistencyReport.is_clean 测试)
- CLI lineage 输出非空 (lineage_by_ckpt_out 接口测试)
"""
from __future__ import annotations

import json
import uuid

import pytest

from poc.src.lnn.ledger import (
    DeltaSummaryRecord,
    EventAttributionRecord,
    FitRunRecord,
    LedgerReader,
    LedgerSchemaVersion,
    LedgerWriter,
)
from poc.src.lnn.ledger.contracts import ConsistencyReport


# ---------- FitRunRecord 验证 ----------


def _valid_fit_run(**overrides) -> FitRunRecord:
    base = dict(
        fit_run_id=str(uuid.uuid4()),
        ckpt_in_sha8="abcd1234",
        ckpt_out_sha8="ef567890",
        fired_at_utc="2026-05-27T01:23:45Z",
        persona_idx=30,
        event_uuid_list_hash="0123456789abcdef",
        batch_size=8,
        config_snapshot_json=json.dumps({"delta_lr": 1e-4}),
        success_flag=True,
    )
    base.update(overrides)
    return FitRunRecord(**base)


def test_fit_run_valid_construct() -> None:
    rec = _valid_fit_run()
    assert rec.ckpt_in_sha8 != rec.ckpt_out_sha8
    assert rec.success_flag is True


def test_fit_run_rejects_short_sha8() -> None:
    with pytest.raises(ValueError, match="ckpt_.*_sha8"):
        _valid_fit_run(ckpt_in_sha8="abc")


def test_fit_run_rejects_wrong_hash_len() -> None:
    with pytest.raises(ValueError, match="event_uuid_list_hash"):
        _valid_fit_run(event_uuid_list_hash="short")


def test_fit_run_rejects_empty_id() -> None:
    with pytest.raises(ValueError, match="fit_run_id"):
        _valid_fit_run(fit_run_id="")


def test_fit_run_is_frozen() -> None:
    rec = _valid_fit_run()
    with pytest.raises((AttributeError, Exception)):
        rec.success_flag = False  # type: ignore[misc]


# ---------- EventAttributionRecord 验证 ----------


def _valid_event(**overrides) -> EventAttributionRecord:
    base = dict(
        fit_run_id=str(uuid.uuid4()),
        event_uuid=str(uuid.uuid4()),
        event_order=0,
        reward=1.0,
        label_source="user_active",
        window_frame_ids=json.dumps([100, 101, 102]),
        contribution_mode="membership_only",
    )
    base.update(overrides)
    return EventAttributionRecord(**base)


def test_event_phase_a_membership_only() -> None:
    """Phase A 只允许 membership_only; delta_attributed 是未来扩展."""
    rec = _valid_event()
    assert rec.contribution_mode == "membership_only"


def test_event_rejects_unknown_contribution_mode() -> None:
    with pytest.raises(ValueError, match="contribution_mode unknown"):
        _valid_event(contribution_mode="reverse_inhibit")  # ADR-038 v1.1 拒绝


def test_event_rejects_negative_order() -> None:
    with pytest.raises(ValueError, match="event_order"):
        _valid_event(event_order=-1)


def test_event_allows_none_reward() -> None:
    """有些 event 可能没 reward (passive_response 等), 允许 None."""
    rec = _valid_event(reward=None)
    assert rec.reward is None


# ---------- DeltaSummaryRecord 验证 ----------


def test_delta_summary_valid() -> None:
    rec = DeltaSummaryRecord(
        fit_run_id=str(uuid.uuid4()),
        delta_l2_total=0.0123,
        delta_l2_by_persona_json=json.dumps({"30": 0.0123}),
        topk_dims_json=json.dumps([[5, 0.008], [12, 0.003]]),
    )
    assert rec.delta_l2_total > 0


# ---------- ConsistencyReport 验证 ----------


def test_consistency_report_clean() -> None:
    rep = ConsistencyReport(
        total_fit_runs=10,
        total_event_records=80,
        total_delta_records=10,
        orphan_events=0,
        orphan_deltas=0,
        runs_without_events=0,
        runs_without_delta=0,
    )
    assert rep.is_clean is True


def test_consistency_report_dirty_orphan_events() -> None:
    rep = ConsistencyReport(
        total_fit_runs=10,
        total_event_records=85,
        total_delta_records=10,
        orphan_events=5,
        orphan_deltas=0,
        runs_without_events=0,
        runs_without_delta=0,
    )
    assert rep.is_clean is False


def test_consistency_report_dirty_run_without_events() -> None:
    rep = ConsistencyReport(
        total_fit_runs=10,
        total_event_records=72,
        total_delta_records=10,
        orphan_events=0,
        orphan_deltas=0,
        runs_without_events=1,
        runs_without_delta=0,
    )
    assert rep.is_clean is False


# ---------- Protocol 类型检查 ----------


class _StubWriter:
    """最小合规 writer (满足 Protocol 接口形状, 不做真实写入)."""

    def write_fit_run(
        self,
        fit_run: FitRunRecord,
        events: list[EventAttributionRecord],
        delta_summary: DeltaSummaryRecord,
    ) -> None:
        return None

    def ensure_schema(self) -> None:
        return None


class _StubReader:
    """最小合规 reader."""

    def lineage_by_ckpt_out(self, ckpt_out_sha8: str) -> FitRunRecord | None:
        return None

    def lineage_by_ckpt_in(self, ckpt_in_sha8: str) -> list[FitRunRecord]:
        return []

    def events_of_run(self, fit_run_id: str) -> list[EventAttributionRecord]:
        return []

    def delta_of_run(self, fit_run_id: str) -> DeltaSummaryRecord | None:
        return None

    def consistency_check(self) -> ConsistencyReport:
        return ConsistencyReport(0, 0, 0, 0, 0, 0, 0)


def test_writer_protocol_runtime_checkable() -> None:
    writer = _StubWriter()
    assert isinstance(writer, LedgerWriter)


def test_reader_protocol_runtime_checkable() -> None:
    reader = _StubReader()
    assert isinstance(reader, LedgerReader)


# ---------- Schema version ----------


def test_schema_version_is_v1() -> None:
    """Phase A schema 版本锚定. 任何 schema 变更必须升版本 + 写 migration ADR."""
    assert LedgerSchemaVersion == "ledger_v1"


# ---------- ADR-038 LNN-FIRST 兼容性 (元测试) ----------


def test_no_reverse_inhibit_mode_in_phase_a() -> None:
    """ADR-038 v1.1 §F1: Phase A 不做反向抑制 (数学非法).

    contribution_mode 在 Phase A 只允许 'membership_only'.
    'delta_attributed' 是未来扩展占位 (Phase B+).
    'reverse_inhibit' / 'unlearn' 等永远不允许.
    """
    allowed_phase_a = {"membership_only"}
    allowed_future = {"delta_attributed"}

    with pytest.raises(ValueError):
        _valid_event(contribution_mode="reverse_inhibit")
    with pytest.raises(ValueError):
        _valid_event(contribution_mode="unlearn")

    # 验证占位扩展未来可用
    rec = _valid_event(contribution_mode="delta_attributed")
    assert rec.contribution_mode in allowed_future
