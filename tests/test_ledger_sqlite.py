"""SQLite-backed ledger tests for ADR-038 F1 Phase A."""
from __future__ import annotations

import json
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

from poc.src.lnn.fit_orchestrator import try_fit_step
from poc.src.lnn.ledger import (
    DeltaSummaryRecord,
    EventAttributionRecord,
    FitRunRecord,
    LedgerWriteError,
    SQLiteLedgerReader,
    SQLiteLedgerWriter,
)
from poc.tests.test_fit_orchestrator import (
    _REAL_CKPT,
    _make_feedback_event,
    _seed_feedback_jsonl,
    _seed_signals_jsonl,
)


def _fit_run(
    idx: int,
    *,
    ckpt_in_sha8: str = "aaaabbbb",
    ckpt_out_sha8: str | None = None,
) -> FitRunRecord:
    return FitRunRecord(
        fit_run_id=str(uuid.uuid4()),
        ckpt_in_sha8=ckpt_in_sha8,
        ckpt_out_sha8=ckpt_out_sha8 or f"{idx:08x}",
        fired_at_utc=f"2026-05-28T10:00:{idx:02d}Z",
        persona_idx=30,
        event_uuid_list_hash=f"{idx + 1:016x}",
        batch_size=2,
        config_snapshot_json=json.dumps({"delta_lr": 1e-4, "idx": idx}),
        success_flag=True,
    )


def _events(fit_run_id: str, idx: int) -> list[EventAttributionRecord]:
    return [
        EventAttributionRecord(
            fit_run_id=fit_run_id,
            event_uuid=f"event-{idx:02d}-{order}",
            event_order=order,
            reward=1.0 - order,
            label_source="user_active",
            window_frame_ids=json.dumps([idx * 10 + order, idx * 10 + order + 1]),
            contribution_mode="membership_only",
        )
        for order in range(2)
    ]


def _delta(fit_run_id: str, idx: int) -> DeltaSummaryRecord:
    return DeltaSummaryRecord(
        fit_run_id=fit_run_id,
        delta_l2_total=0.01 * (idx + 1),
        delta_l2_by_persona_json=json.dumps({"30": 0.01 * (idx + 1)}),
        topk_dims_json=json.dumps([{"persona_idx": 30, "dim_idx": idx, "signed_delta": 0.1}]),
    )


class _FailingEventsWriter(SQLiteLedgerWriter):
    """Inject an event insert failure after fit_runs has been inserted."""

    def _insert_events(
        self,
        conn: sqlite3.Connection,
        events: list[EventAttributionRecord],
    ) -> None:
        raise sqlite3.IntegrityError("synthetic event insert failure")


class _FailSoftWriter:
    """Protocol-compatible writer used to verify fit_orchestrator fail-soft semantics."""

    def ensure_schema(self) -> None:
        return None

    def write_fit_run(
        self,
        fit_run: FitRunRecord,
        events: list[EventAttributionRecord],
        delta_summary: DeltaSummaryRecord,
    ) -> None:
        raise LedgerWriteError(f"synthetic fail-soft error for {fit_run.fit_run_id}")


def test_ensure_schema_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = SQLiteLedgerWriter(db_path)
    for _ in range(3):
        writer.ensure_schema()
    conn = sqlite3.connect(db_path)
    try:
        row_count = conn.execute("SELECT COUNT(*) FROM schema_meta").fetchone()[0]
        version = conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'version'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert row_count == 1
    assert version == "ledger_v1"


def test_three_table_write_is_atomic(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = _FailingEventsWriter(db_path)
    fit_run = _fit_run(1)
    with pytest.raises(LedgerWriteError, match="fit_run_id="):
        writer.write_fit_run(fit_run, _events(fit_run.fit_run_id, 1), _delta(fit_run.fit_run_id, 1))
    conn = sqlite3.connect(db_path)
    try:
        fit_runs = conn.execute("SELECT COUNT(*) FROM fit_runs").fetchone()[0]
        events = conn.execute("SELECT COUNT(*) FROM event_attribution").fetchone()[0]
        deltas = conn.execute("SELECT COUNT(*) FROM delta_summary").fetchone()[0]
    finally:
        conn.close()
    assert fit_runs == 0
    assert events == 0
    assert deltas == 0


def test_consistency_check_reports_dirty_but_does_not_crash(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = SQLiteLedgerWriter(db_path)
    writer.ensure_schema()
    conn = sqlite3.connect(db_path)
    try:
        dirty_run = _fit_run(2, ckpt_in_sha8="ccccdddd")
        conn.execute(
            """
            INSERT INTO fit_runs(
                fit_run_id, ckpt_in_sha8, ckpt_out_sha8, fired_at_utc,
                persona_idx, event_uuid_list_hash, batch_size,
                config_snapshot_json, success_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dirty_run.fit_run_id,
                dirty_run.ckpt_in_sha8,
                dirty_run.ckpt_out_sha8,
                dirty_run.fired_at_utc,
                dirty_run.persona_idx,
                dirty_run.event_uuid_list_hash,
                dirty_run.batch_size,
                dirty_run.config_snapshot_json,
                int(dirty_run.success_flag),
            ),
        )
        conn.execute(
            """
            INSERT INTO event_attribution(
                fit_run_id, event_uuid, event_order, reward,
                label_source, window_frame_ids, contribution_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                "missing-fit-run",
                "orphan-event",
                0,
                1.0,
                "user_active",
                "[1,2]",
                "membership_only",
            ),
        )
        conn.execute(
            """
            INSERT INTO delta_summary(
                fit_run_id, delta_l2_total, delta_l2_by_persona_json, topk_dims_json
            ) VALUES (?, ?, ?, ?)
            """,
            ("missing-fit-run", 0.1, '{"30": 0.1}', "[]"),
        )
        conn.commit()
    finally:
        conn.close()
    report = SQLiteLedgerReader(db_path).consistency_check()
    assert report.is_clean is False
    assert report.orphan_events == 1
    assert report.orphan_deltas == 1
    assert report.runs_without_events == 1
    assert report.runs_without_delta == 1


def test_lineage_queries_and_cli_roundtrip(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = SQLiteLedgerWriter(db_path)
    run_a = _fit_run(10, ckpt_in_sha8="11112222", ckpt_out_sha8="aaaab001")
    run_b = _fit_run(11, ckpt_in_sha8="11112222", ckpt_out_sha8="aaaab002")
    run_c = _fit_run(12, ckpt_in_sha8="aaaab001", ckpt_out_sha8="aaaab003")
    for idx, run in enumerate((run_a, run_b, run_c), start=10):
        writer.write_fit_run(run, _events(run.fit_run_id, idx), _delta(run.fit_run_id, idx))

    reader = SQLiteLedgerReader(db_path)
    fit_run = reader.lineage_by_ckpt_out("aaaab003")
    upstream = reader.lineage_by_ckpt_in("11112222")
    assert fit_run is not None
    assert fit_run.fit_run_id == run_c.fit_run_id
    assert [run.fit_run_id for run in upstream] == [run_a.fit_run_id, run_b.fit_run_id]
    assert len(reader.events_of_run(run_a.fit_run_id)) == 2
    assert reader.delta_of_run(run_b.fit_run_id) is not None

    cli = subprocess.run(
        [
            sys.executable,
            "-m",
            "poc.src.lnn.ledger.cli",
            "--db",
            str(db_path),
            "lineage",
            "aaaab003",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert cli.returncode == 0, cli.stderr
    assert run_c.fit_run_id in cli.stdout
    assert "aaaab001 → aaaab003" in cli.stdout


def test_wal_mode_is_enabled(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = SQLiteLedgerWriter(db_path)
    writer.ensure_schema()
    conn = sqlite3.connect(db_path)
    try:
        journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        synchronous = conn.execute("PRAGMA synchronous").fetchone()[0]
    finally:
        conn.close()
    assert str(journal_mode).lower() == "wal"
    assert int(synchronous) == 1


def test_three_runs_plus_three_power_failures_report_precise_dirty_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.sqlite"
    writer = SQLiteLedgerWriter(db_path)
    for idx in range(3):
        run = _fit_run(20 + idx, ckpt_in_sha8=f"feed{idx:04x}", ckpt_out_sha8=f"beef{idx:04x}")
        writer.write_fit_run(run, _events(run.fit_run_id, 20 + idx), _delta(run.fit_run_id, 20 + idx))

    conn = sqlite3.connect(db_path)
    try:
        for idx in range(3):
            crashed = _fit_run(30 + idx, ckpt_in_sha8=f"dead{idx:04x}", ckpt_out_sha8=f"cafe{idx:04x}")
            conn.execute(
                """
                INSERT INTO fit_runs(
                    fit_run_id, ckpt_in_sha8, ckpt_out_sha8, fired_at_utc,
                    persona_idx, event_uuid_list_hash, batch_size,
                    config_snapshot_json, success_flag
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    crashed.fit_run_id,
                    crashed.ckpt_in_sha8,
                    crashed.ckpt_out_sha8,
                    crashed.fired_at_utc,
                    crashed.persona_idx,
                    crashed.event_uuid_list_hash,
                    crashed.batch_size,
                    crashed.config_snapshot_json,
                    int(crashed.success_flag),
                ),
            )
            conn.commit()
    finally:
        conn.close()

    report = SQLiteLedgerReader(db_path).consistency_check()
    assert report.total_fit_runs == 6
    assert report.total_event_records == 6
    assert report.total_delta_records == 3
    assert report.orphan_events == 0
    assert report.orphan_deltas == 0
    assert report.runs_without_events == 3
    assert report.runs_without_delta == 3
    assert report.is_clean is False


@pytest.mark.skipif(not _REAL_CKPT.exists(), reason="day12_with_self ckpt not available")
def test_try_fit_step_ledger_failure_is_fail_soft(tmp_path: Path) -> None:
    feedback_dir = tmp_path / "feedback"
    signals_dir = tmp_path / "signals"
    out_dir = tmp_path / "out"
    events = [_make_feedback_event(i + 100, label="mismatch") for i in range(3)]
    _seed_feedback_jsonl(feedback_dir, events)
    _seed_signals_jsonl(signals_dir, list(range(1, 200)))

    result = try_fit_step(
        feedback_dir=feedback_dir,
        signals_dir=signals_dir,
        ckpt_in_path=_REAL_CKPT,
        ckpt_out_dir=out_dir,
        persona_idx=30,
        force=True,
        ledger_writer=_FailSoftWriter(),
    )
    assert result.triggered is True
    assert result.ckpt_out_path is not None and result.ckpt_out_path.exists()
    assert result.metadata["ledger_write_failed"] is True
    assert result.metrics["ledger_write_failed"] == 1.0
