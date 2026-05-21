"""Tests for SQLiteGraphitiBackend parity with in-memory semantics."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import KGQuery
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter.mapper import EventTensorMapper
from graphiti_zh_adapter.sqlite_backend import SQLiteGraphitiBackend


def _event(*, event_id: str, user_id: str, ts: datetime, action: str = "喝咖啡") -> EventTensor:
    return EventTensor(
        user_id=user_id,
        event_id=event_id,
        timestamp=ts,
        slots={
            SlotID.WHO_ACTOR: SlotValue("Rayman"),
            SlotID.WHAT_ACTION: SlotValue(action),
            SlotID.TOPIC: SlotValue("咖啡"),
            SlotID.RELATED_ENTITIES: SlotValue(["拿铁"]),
        },
    )


def test_upsert_and_get_node_roundtrip(tmp_path: Path) -> None:
    """Persisted node should roundtrip back to KGNode with slot data intact."""
    backend = SQLiteGraphitiBackend(tmp_path / "kg.sqlite")
    mapper = EventTensorMapper()
    event = _event(event_id="evt_roundtrip", user_id="u1", ts=datetime(2026, 5, 21, 8, 0, tzinfo=timezone.utc))

    node_id = backend.upsert_node(mapper.to_kg_node(event))
    loaded = backend.get_node(node_id)
    backend.close()

    assert loaded is not None
    assert loaded.node_id == node_id
    assert loaded.event_id == event.event_id
    assert loaded.user_id == event.user_id
    assert loaded.slots[SlotID.WHAT_ACTION] == "喝咖啡"


def test_query_candidates_user_filter(tmp_path: Path) -> None:
    """Query should only return nodes for requested user_id."""
    backend = SQLiteGraphitiBackend(tmp_path / "kg.sqlite")
    mapper = EventTensorMapper()
    ts = datetime(2026, 5, 21, 9, 0, tzinfo=timezone.utc)
    backend.upsert_node(mapper.to_kg_node(_event(event_id="evt_u1", user_id="u1", ts=ts)))
    backend.upsert_node(mapper.to_kg_node(_event(event_id="evt_u2", user_id="u2", ts=ts)))

    results = backend.query_candidates(KGQuery(user_id="u1"))
    backend.close()

    assert [node.event_id for node in results] == ["evt_u1"]


def test_query_candidates_time_window(tmp_path: Path) -> None:
    """valid_at should enforce [t_valid, t_invalid) bi-temporal window."""
    backend = SQLiteGraphitiBackend(tmp_path / "kg.sqlite")
    mapper = EventTensorMapper()
    base = datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc)
    active = _event(event_id="evt_active", user_id="u1", ts=base - timedelta(minutes=5), action="写周报")
    expired = _event(event_id="evt_expired", user_id="u1", ts=base - timedelta(minutes=15), action="喝拿铁")
    backend.upsert_node(mapper.to_kg_node(active))
    backend.upsert_node(mapper.to_kg_node(expired))
    backend.invalidate("evt_expired", base - timedelta(minutes=10))

    results = backend.query_candidates(KGQuery(user_id="u1", valid_at=base))
    backend.close()

    assert [node.event_id for node in results] == ["evt_active"]


def test_invalidate_keeps_data_sets_t_invalid(tmp_path: Path) -> None:
    """Invalidate updates t_invalid instead of physically deleting node."""
    backend = SQLiteGraphitiBackend(tmp_path / "kg.sqlite")
    mapper = EventTensorMapper()
    ts = datetime(2026, 5, 21, 11, 0, tzinfo=timezone.utc)
    event = _event(event_id="evt_inv", user_id="u1", ts=ts)
    node_id = backend.upsert_node(mapper.to_kg_node(event))
    invalid_at = ts + timedelta(minutes=3)

    backend.invalidate(event.event_id, invalid_at)
    loaded = backend.get_node(node_id)
    backend.close()

    assert loaded is not None
    assert loaded.event_id == event.event_id
    assert loaded.t_invalid == invalid_at


def test_restart_persistence(tmp_path: Path) -> None:
    """Data survives process-style restart by reopening a new backend connection."""
    db_path = tmp_path / "kg.sqlite"
    mapper = EventTensorMapper()
    ts = datetime(2026, 5, 21, 12, 0, tzinfo=timezone.utc)
    event = _event(event_id="evt_restart", user_id="u1", ts=ts)

    backend1 = SQLiteGraphitiBackend(db_path)
    backend1.upsert_node(mapper.to_kg_node(event))
    backend1.close()

    backend2 = SQLiteGraphitiBackend(db_path)
    node = backend2.get_node("node_evt_restart")
    backend2.close()

    assert node is not None
    assert node.event_id == "evt_restart"


def test_intent_type_multimodal_optional_fields(tmp_path: Path) -> None:
    """v1.1 nullable columns accept legacy node writes and keep NULL values."""
    db_path = tmp_path / "kg.sqlite"
    backend = SQLiteGraphitiBackend(db_path)
    mapper = EventTensorMapper()
    event = _event(event_id="evt_optional", user_id="u1", ts=datetime(2026, 5, 21, 13, 0, tzinfo=timezone.utc))

    backend.upsert_node(mapper.to_kg_node(event))
    backend.close()

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT intent_type, multimodal_context_json FROM kg_nodes WHERE node_id = ?",
        ("node_evt_optional",),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] is None
    assert row[1] is None
