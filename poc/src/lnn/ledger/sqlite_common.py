"""Shared SQLite helpers for the attribution ledger."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from poc.src.lnn.ledger.contracts import (
    ConsistencyReport,
    DeltaSummaryRecord,
    EventAttributionRecord,
    FitRunRecord,
    LedgerReadError,
    LedgerSchemaMismatchError,
    LedgerSchemaVersion,
)

DEFAULT_LEDGER_DB_PATH = Path.home() / ".lnmem" / "ledger" / "ledger.sqlite"

_SCHEMA_SQL = (
    """
    CREATE TABLE IF NOT EXISTS schema_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fit_runs (
        fit_run_id TEXT PRIMARY KEY,
        ckpt_in_sha8 TEXT NOT NULL,
        ckpt_out_sha8 TEXT NOT NULL,
        fired_at_utc TEXT NOT NULL,
        persona_idx INTEGER NOT NULL,
        event_uuid_list_hash TEXT NOT NULL,
        batch_size INTEGER NOT NULL,
        config_snapshot_json TEXT NOT NULL,
        success_flag INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS event_attribution (
        fit_run_id TEXT NOT NULL,
        event_uuid TEXT NOT NULL,
        event_order INTEGER NOT NULL,
        reward REAL,
        label_source TEXT NOT NULL,
        window_frame_ids TEXT NOT NULL,
        contribution_mode TEXT NOT NULL,
        PRIMARY KEY (fit_run_id, event_uuid)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS delta_summary (
        fit_run_id TEXT PRIMARY KEY,
        delta_l2_total REAL NOT NULL,
        delta_l2_by_persona_json TEXT NOT NULL,
        topk_dims_json TEXT NOT NULL
    )
    """,
    "CREATE UNIQUE INDEX IF NOT EXISTS idx_fit_runs_ckpt_out ON fit_runs(ckpt_out_sha8)",
    "CREATE INDEX IF NOT EXISTS idx_fit_runs_ckpt_in ON fit_runs(ckpt_in_sha8)",
    "CREATE INDEX IF NOT EXISTS idx_event_attribution_event_uuid ON event_attribution(event_uuid)",
)


def resolve_db_path(db_path: Path | None) -> Path:
    """Return configured DB path or the project default."""
    return Path(db_path) if db_path is not None else DEFAULT_LEDGER_DB_PATH


def open_rw_connection(db_path: Path) -> sqlite3.Connection:
    """Open a writer connection with WAL-friendly pragmas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        str(db_path),
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def open_ro_connection(db_path: Path) -> sqlite3.Connection:
    """Open a read-only connection via SQLite URI mode."""
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(
        uri,
        uri=True,
        isolation_level=None,
        check_same_thread=False,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create Phase A tables and indexes if they do not yet exist."""
    for statement in _SCHEMA_SQL:
        conn.execute(statement)
    conn.execute(
        "INSERT OR IGNORE INTO schema_meta(key, value) VALUES('version', ?)",
        (LedgerSchemaVersion,),
    )
    validate_schema(conn)


def validate_schema(conn: sqlite3.Connection) -> None:
    """Check schema version and fail if it drifts from the contract."""
    row = conn.execute(
        "SELECT value FROM schema_meta WHERE key = 'version'"
    ).fetchone()
    actual = row["value"] if row is not None else None
    if actual != LedgerSchemaVersion:
        raise LedgerSchemaMismatchError(
            f"ledger schema mismatch: expected {LedgerSchemaVersion}, got {actual!r}"
        )


def fit_run_from_row(row: sqlite3.Row) -> FitRunRecord:
    """Convert a SQLite row into a fit-run record."""
    return FitRunRecord(
        fit_run_id=row["fit_run_id"],
        ckpt_in_sha8=row["ckpt_in_sha8"],
        ckpt_out_sha8=row["ckpt_out_sha8"],
        fired_at_utc=row["fired_at_utc"],
        persona_idx=int(row["persona_idx"]),
        event_uuid_list_hash=row["event_uuid_list_hash"],
        batch_size=int(row["batch_size"]),
        config_snapshot_json=row["config_snapshot_json"],
        success_flag=bool(row["success_flag"]),
    )


def event_from_row(row: sqlite3.Row) -> EventAttributionRecord:
    """Convert a SQLite row into an event attribution record."""
    return EventAttributionRecord(
        fit_run_id=row["fit_run_id"],
        event_uuid=row["event_uuid"],
        event_order=int(row["event_order"]),
        reward=row["reward"],
        label_source=row["label_source"],
        window_frame_ids=row["window_frame_ids"],
        contribution_mode=row["contribution_mode"],
    )


def delta_from_row(row: sqlite3.Row) -> DeltaSummaryRecord:
    """Convert a SQLite row into a delta summary record."""
    return DeltaSummaryRecord(
        fit_run_id=row["fit_run_id"],
        delta_l2_total=float(row["delta_l2_total"]),
        delta_l2_by_persona_json=row["delta_l2_by_persona_json"],
        topk_dims_json=row["topk_dims_json"],
    )


def read_consistency_report(conn: sqlite3.Connection) -> ConsistencyReport:
    """Build the O(N) consistency report using table scans and joins."""
    total_fit_runs = int(conn.execute("SELECT COUNT(*) FROM fit_runs").fetchone()[0])
    total_event_records = int(
        conn.execute("SELECT COUNT(*) FROM event_attribution").fetchone()[0]
    )
    total_delta_records = int(
        conn.execute("SELECT COUNT(*) FROM delta_summary").fetchone()[0]
    )
    orphan_events = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM event_attribution AS ea
            LEFT JOIN fit_runs AS fr ON fr.fit_run_id = ea.fit_run_id
            WHERE fr.fit_run_id IS NULL
            """
        ).fetchone()[0]
    )
    orphan_deltas = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM delta_summary AS ds
            LEFT JOIN fit_runs AS fr ON fr.fit_run_id = ds.fit_run_id
            WHERE fr.fit_run_id IS NULL
            """
        ).fetchone()[0]
    )
    runs_without_events = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM fit_runs AS fr
            LEFT JOIN event_attribution AS ea ON ea.fit_run_id = fr.fit_run_id
            WHERE ea.fit_run_id IS NULL
            """
        ).fetchone()[0]
    )
    runs_without_delta = int(
        conn.execute(
            """
            SELECT COUNT(*)
            FROM fit_runs AS fr
            LEFT JOIN delta_summary AS ds ON ds.fit_run_id = fr.fit_run_id
            WHERE ds.fit_run_id IS NULL
            """
        ).fetchone()[0]
    )
    return ConsistencyReport(
        total_fit_runs=total_fit_runs,
        total_event_records=total_event_records,
        total_delta_records=total_delta_records,
        orphan_events=orphan_events,
        orphan_deltas=orphan_deltas,
        runs_without_events=runs_without_events,
        runs_without_delta=runs_without_delta,
    )


def wrap_read_error(db_path: Path, exc: Exception) -> LedgerReadError:
    """Attach DB path context to reader failures."""
    if isinstance(exc, LedgerReadError):
        return exc
    return LedgerReadError(f"ledger read failed at {db_path}: {exc}")
