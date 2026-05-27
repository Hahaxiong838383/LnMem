"""SQLite reader for Phase A attribution ledger."""
from __future__ import annotations

from pathlib import Path

from poc.src.lnn.ledger.contracts import (
    ConsistencyReport,
    DeltaSummaryRecord,
    EventAttributionRecord,
    FitRunRecord,
)
from poc.src.lnn.ledger.sqlite_common import (
    delta_from_row,
    event_from_row,
    fit_run_from_row,
    open_ro_connection,
    read_consistency_report,
    resolve_db_path,
    validate_schema,
    wrap_read_error,
)


class SQLiteLedgerReader:
    """Read fit lineage records from the SQLite ledger in strict read-only mode."""

    def __init__(self, db_path: Path | None = None) -> None:
        self.db_path = resolve_db_path(db_path)

    def lineage_by_ckpt_out(self, ckpt_out_sha8: str) -> FitRunRecord | None:
        """Return the fit run that produced the given output checkpoint hash."""
        return self._fetch_fit_run(
            "SELECT * FROM fit_runs WHERE ckpt_out_sha8 = ?",
            (ckpt_out_sha8,),
        )

    def lineage_by_ckpt_in(self, ckpt_in_sha8: str) -> list[FitRunRecord]:
        """Return all fit runs that used the given input checkpoint hash."""
        return self._fetch_fit_runs(
            """
            SELECT * FROM fit_runs
            WHERE ckpt_in_sha8 = ?
            ORDER BY fired_at_utc ASC, fit_run_id ASC
            """,
            (ckpt_in_sha8,),
        )

    def events_of_run(self, fit_run_id: str) -> list[EventAttributionRecord]:
        """Return batch membership rows ordered by original batch order."""
        return self._fetch_events(
            """
            SELECT * FROM event_attribution
            WHERE fit_run_id = ?
            ORDER BY event_order ASC, event_uuid ASC
            """,
            (fit_run_id,),
        )

    def delta_of_run(self, fit_run_id: str) -> DeltaSummaryRecord | None:
        """Return the 1:1 delta summary row for one fit run."""
        try:
            with self._open() as conn:
                row = conn.execute(
                    "SELECT * FROM delta_summary WHERE fit_run_id = ?",
                    (fit_run_id,),
                ).fetchone()
                return None if row is None else delta_from_row(row)
        except Exception as exc:  # noqa: BLE001
            raise wrap_read_error(self.db_path, exc)

    def consistency_check(self) -> ConsistencyReport:
        """Scan all tables and return orphan/missing-link counts."""
        try:
            with self._open() as conn:
                return read_consistency_report(conn)
        except Exception as exc:  # noqa: BLE001
            raise wrap_read_error(self.db_path, exc)

    def list_fit_runs(self, limit: int = 20) -> list[FitRunRecord]:
        """Return recent fit runs for CLI listing."""
        return self._fetch_fit_runs(
            """
            SELECT * FROM fit_runs
            ORDER BY fired_at_utc DESC, fit_run_id DESC
            LIMIT ?
            """,
            (int(limit),),
        )

    def fit_run_by_id(self, fit_run_id: str) -> FitRunRecord | None:
        """Return one fit run by primary key."""
        return self._fetch_fit_run(
            "SELECT * FROM fit_runs WHERE fit_run_id = ?",
            (fit_run_id,),
        )

    def _open(self):
        conn = open_ro_connection(self.db_path)
        try:
            validate_schema(conn)
        except Exception:
            conn.close()
            raise
        return conn

    def _fetch_fit_run(self, sql: str, params: tuple[object, ...]) -> FitRunRecord | None:
        try:
            with self._open() as conn:
                row = conn.execute(sql, params).fetchone()
                return None if row is None else fit_run_from_row(row)
        except Exception as exc:  # noqa: BLE001
            raise wrap_read_error(self.db_path, exc)

    def _fetch_fit_runs(
        self,
        sql: str,
        params: tuple[object, ...],
    ) -> list[FitRunRecord]:
        try:
            with self._open() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [fit_run_from_row(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            raise wrap_read_error(self.db_path, exc)

    def _fetch_events(
        self,
        sql: str,
        params: tuple[object, ...],
    ) -> list[EventAttributionRecord]:
        try:
            with self._open() as conn:
                rows = conn.execute(sql, params).fetchall()
                return [event_from_row(row) for row in rows]
        except Exception as exc:  # noqa: BLE001
            raise wrap_read_error(self.db_path, exc)


__all__ = ["SQLiteLedgerReader"]
