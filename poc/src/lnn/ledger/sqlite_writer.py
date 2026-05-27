"""SQLite writer for Phase A attribution ledger."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

from poc.src.lnn.ledger.contracts import (
    DeltaSummaryRecord,
    EventAttributionRecord,
    FitRunRecord,
    LedgerWriteError,
)
from poc.src.lnn.ledger.sqlite_common import ensure_schema, open_rw_connection, resolve_db_path


class SQLiteLedgerWriter:
    """Persist one fit run into the three-table SQLite lineage store."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = resolve_db_path(db_path)
        self._lock = threading.RLock()

    def ensure_schema(self) -> None:
        """Create the Phase A schema if it is not already present."""
        with self._lock:
            conn = open_rw_connection(self.db_path)
            try:
                ensure_schema(conn)
            except Exception as exc:  # noqa: BLE001
                raise LedgerWriteError(
                    f"ensure_schema failed at {self.db_path}: {exc}"
                ) from exc
            finally:
                conn.close()

    def write_fit_run(
        self,
        fit_run: FitRunRecord,
        events: list[EventAttributionRecord],
        delta_summary: DeltaSummaryRecord,
    ) -> None:
        """Write fit, event, and delta records in one IMMEDIATE transaction."""
        with self._lock:
            conn = open_rw_connection(self.db_path)
            try:
                ensure_schema(conn)
                conn.execute("BEGIN IMMEDIATE")
                try:
                    self._insert_fit_run(conn, fit_run)
                    self._insert_events(conn, events)
                    self._insert_delta(conn, delta_summary)
                    conn.execute("COMMIT")
                except Exception as exc:  # noqa: BLE001
                    conn.execute("ROLLBACK")
                    raise self._wrap_write_error(
                        fit_run_id=fit_run.fit_run_id,
                        stage="write_fit_run",
                        exc=exc,
                    ) from exc
            finally:
                conn.close()

    def _insert_fit_run(self, conn: sqlite3.Connection, fit_run: FitRunRecord) -> None:
        """Insert the fit_runs anchor row."""
        sql = """
            INSERT INTO fit_runs(
                fit_run_id, ckpt_in_sha8, ckpt_out_sha8, fired_at_utc,
                persona_idx, event_uuid_list_hash, batch_size,
                config_snapshot_json, success_flag
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        conn.execute(
            sql,
            (
                fit_run.fit_run_id,
                fit_run.ckpt_in_sha8,
                fit_run.ckpt_out_sha8,
                fit_run.fired_at_utc,
                fit_run.persona_idx,
                fit_run.event_uuid_list_hash,
                fit_run.batch_size,
                fit_run.config_snapshot_json,
                int(fit_run.success_flag),
            ),
        )

    def _insert_events(
        self,
        conn: sqlite3.Connection,
        events: list[EventAttributionRecord],
    ) -> None:
        """Insert all event membership rows for one fit run."""
        sql = """
            INSERT INTO event_attribution(
                fit_run_id, event_uuid, event_order, reward,
                label_source, window_frame_ids, contribution_mode
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        conn.executemany(
            sql,
            [
                (
                    event.fit_run_id,
                    event.event_uuid,
                    event.event_order,
                    event.reward,
                    event.label_source,
                    event.window_frame_ids,
                    event.contribution_mode,
                )
                for event in events
            ],
        )

    def _insert_delta(
        self,
        conn: sqlite3.Connection,
        delta_summary: DeltaSummaryRecord,
    ) -> None:
        """Insert the 1:1 delta summary row."""
        sql = """
            INSERT INTO delta_summary(
                fit_run_id, delta_l2_total, delta_l2_by_persona_json, topk_dims_json
            ) VALUES (?, ?, ?, ?)
        """
        conn.execute(
            sql,
            (
                delta_summary.fit_run_id,
                delta_summary.delta_l2_total,
                delta_summary.delta_l2_by_persona_json,
                delta_summary.topk_dims_json,
            ),
        )

    def _wrap_write_error(
        self,
        *,
        fit_run_id: str,
        stage: str,
        exc: Exception,
    ) -> LedgerWriteError:
        """Return a contextual writer error without losing root cause."""
        if isinstance(exc, LedgerWriteError):
            return exc
        return LedgerWriteError(
            f"{stage} failed for fit_run_id={fit_run_id} at {self.db_path}: "
            f"{type(exc).__name__}: {exc}"
        )


__all__ = ["SQLiteLedgerWriter"]
