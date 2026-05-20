"""End-to-end tests for the Phase A in-memory graphiti adapter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import KGQuery
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter import GraphitiZhAdapterSkeleton


def _build_event(event_id: str, timestamp: datetime, action: str) -> EventTensor:
    return EventTensor(
        user_id="u_e2e",
        timestamp=timestamp,
        event_id=event_id,
        slots={
            SlotID.WHO_ACTOR: SlotValue("Rayman"),
            SlotID.WHAT_ACTION: SlotValue(action),
            SlotID.WHEN_TIME: SlotValue(timestamp.isoformat()),
            SlotID.WHERE_LOCATION: SlotValue("上海办公室"),
            SlotID.TOPIC: SlotValue("咖啡计划"),
        },
    )


def test_end_to_end_upsert_query_returns_event_within_30min() -> None:
    """Upsert then query should recall at least 90% matching events in 30 minutes."""
    adapter = GraphitiZhAdapterSkeleton()
    valid_at = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)
    inserted_ids = []
    for index in range(10):
        event = _build_event(
            event_id=f"evt_{index}",
            timestamp=valid_at - timedelta(minutes=index * 3),
            action=f"喝咖啡讨论路线图 #{index}",
        )
        inserted_ids.append(event.event_id)
        adapter.upsert(event)

    hits = adapter.query(KGQuery(user_id="u_e2e", text_query="咖啡", valid_at=valid_at, top_k=10))
    recalled_ids = {hit.event.event_id for hit in hits}

    assert len(recalled_ids & set(inserted_ids)) >= 9


def test_end_to_end_invalidate_then_query_excludes_event() -> None:
    """Invalidated events should be excluded from later valid_at queries."""
    adapter = GraphitiZhAdapterSkeleton()
    base = datetime(2026, 5, 20, 13, 0, tzinfo=timezone.utc)
    event = _build_event("evt_to_remove", base - timedelta(minutes=5), "喝咖啡后离开")
    adapter.upsert(event)
    adapter.invalidate(event.event_id, base)

    hits = adapter.query(KGQuery(user_id="u_e2e", text_query="咖啡", valid_at=base + timedelta(minutes=1)))

    assert event.event_id not in [hit.event.event_id for hit in hits]
