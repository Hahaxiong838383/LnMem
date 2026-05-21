"""SQLite backend for Graphiti adapter; persists bi-temporal KG records."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional

from contracts.protocols import KGQuery
from contracts.schema_v1 import SlotID

from .mapper import KGEdge, KGNode


class SQLiteGraphitiBackend:
    """SQLite-backed KG storage; 与 InMemoryGraphitiBackend 接口等价."""

    _DDL_NODES = """
    CREATE TABLE IF NOT EXISTS kg_nodes (
        node_id TEXT PRIMARY KEY,
        event_id TEXT NOT NULL,
        user_id TEXT NOT NULL,
        t_valid TEXT NOT NULL,
        t_invalid TEXT,
        slots_json TEXT NOT NULL,
        intent_type TEXT,
        multimodal_context_json TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """
    _DDL_EDGES = """
    CREATE TABLE IF NOT EXISTS kg_edges (
        edge_id TEXT PRIMARY KEY,
        src_node_id TEXT NOT NULL,
        dst_node_id TEXT NOT NULL,
        relation TEXT NOT NULL,
        evidence_event_id TEXT NOT NULL,
        t_valid TEXT NOT NULL,
        t_invalid TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )
    """

    def __init__(self, db_path: Path):
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = Lock()
        self._bootstrap()

    def upsert_node(self, node: KGNode) -> str:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO kg_nodes (node_id,event_id,user_id,t_valid,t_invalid,slots_json,intent_type,multimodal_context_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    node.node_id,
                    node.event_id,
                    node.user_id,
                    self._dt_to_text(node.t_valid),
                    self._dt_to_text(node.t_invalid),
                    self._slots_to_json(node.slots),
                    getattr(node, "intent_type", None),
                    self._json_or_none(getattr(node, "multimodal_context", None)),
                ),
            )
            self._conn.commit()
        return node.node_id

    def add_edges(self, edges: List[KGEdge]) -> None:
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO kg_edges (edge_id,src_node_id,dst_node_id,relation,evidence_event_id,t_valid,t_invalid) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        edge.edge_id,
                        edge.source_id,
                        edge.target_id,
                        edge.relation,
                        edge.evidence_event_id or "",
                        self._dt_to_text(edge.t_valid),
                        self._dt_to_text(edge.t_invalid),
                    )
                    for edge in edges
                ],
            )
            self._conn.commit()

    def get_node(self, node_id: str) -> Optional[KGNode]:
        row = self._conn.execute(
            "SELECT node_id,event_id,user_id,t_valid,t_invalid,slots_json FROM kg_nodes WHERE node_id = ?",
            (node_id,),
        ).fetchone()
        return self._row_to_node(row) if row else None

    def query_candidates(self, q: KGQuery) -> List[KGNode]:
        rows = self._conn.execute(
            "SELECT node_id,event_id,user_id,t_valid,t_invalid,slots_json FROM kg_nodes WHERE user_id = ?",
            (q.user_id,),
        ).fetchall()
        nodes = [self._row_to_node(row) for row in rows]
        return [node for node in nodes if self._matches_slot_filter(node, q.slot_filter) and self._matches_valid_at(node, q.valid_at)]

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        invalid_text = self._dt_to_text(t_invalid)
        with self._lock:
            self._conn.execute("UPDATE kg_nodes SET t_invalid = ? WHERE event_id = ?", (invalid_text, event_id))
            self._conn.execute("UPDATE kg_edges SET t_invalid = ? WHERE evidence_event_id = ?", (invalid_text, event_id))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _bootstrap(self) -> None:
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute(self._DDL_NODES)
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_user_time ON kg_nodes(user_id, t_valid)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_id ON kg_nodes(event_id)")
            self._conn.execute(self._DDL_EDGES)
            self._conn.commit()

    def _row_to_node(self, row: sqlite3.Row) -> KGNode:
        slots: Dict[SlotID, Any] = {SlotID(int(key)): value for key, value in json.loads(row["slots_json"]).items()}
        return KGNode(
            node_id=row["node_id"],
            event_id=row["event_id"],
            user_id=row["user_id"],
            t_valid=self._text_to_dt(row["t_valid"]),
            t_invalid=self._text_to_dt(row["t_invalid"]),
            slots=slots,
        )

    def _matches_slot_filter(self, node: KGNode, slot_filter: Optional[dict]) -> bool:
        if not slot_filter:
            return True
        for slot_key, expected in slot_filter.items():
            slot_id = slot_key if isinstance(slot_key, SlotID) else SlotID(int(slot_key))
            if node.slots.get(slot_id) != expected:
                return False
        return True

    def _matches_valid_at(self, node: KGNode, valid_at: Optional[datetime]) -> bool:
        if valid_at is None:
            return True
        if node.t_valid > valid_at:
            return False
        return node.t_invalid is None or valid_at < node.t_invalid

    @staticmethod
    def _dt_to_text(value: Optional[datetime]) -> Optional[str]:
        return value.isoformat() if value is not None else None

    @staticmethod
    def _text_to_dt(value: Optional[str]) -> Optional[datetime]:
        return datetime.fromisoformat(value) if value is not None else None

    @staticmethod
    def _slots_to_json(slots: Dict[SlotID, Any]) -> str:
        return json.dumps({str(int(slot_id)): value for slot_id, value in slots.items()}, ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _json_or_none(value: Optional[dict]) -> Optional[str]:
        return None if value is None else json.dumps(value, ensure_ascii=False, separators=(",", ":"))
