"""Phase A behavior tests for the in-memory graphiti_zh_adapter."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import KGQuery
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter import GraphitiZhAdapterSkeleton


def _event(
    *,
    event_id: str,
    timestamp: datetime,
    action: str,
    location: str = "上海",
    user_id: str = "u_graphiti",
    topic: str = "咖啡",
) -> EventTensor:
    return EventTensor(
        user_id=user_id,
        timestamp=timestamp,
        event_id=event_id,
        slots={
            SlotID.WHO_ACTOR: SlotValue("Rayman"),
            SlotID.WHAT_ACTION: SlotValue(action),
            SlotID.WHEN_TIME: SlotValue(timestamp.isoformat()),
            SlotID.WHERE_LOCATION: SlotValue(location),
            SlotID.TOPIC: SlotValue(topic),
            SlotID.RELATED_ENTITIES: SlotValue(["拿铁", "燕麦奶"]),
        },
    )


def test_upsert_maps_p0_slots_to_graph_node() -> None:
    """Upsert should persist all P0 slots as node properties."""
    adapter = GraphitiZhAdapterSkeleton()
    event = _event(
        event_id="evt_p0",
        timestamp=datetime(2026, 5, 20, 8, 0, tzinfo=timezone.utc),
        action="喝咖啡",
    )

    node_id = adapter.upsert(event)
    node = adapter.backend.get_node(node_id)

    assert node is not None
    assert node.node_id == "node_evt_p0"
    assert node.slots[SlotID.WHO_ACTOR] == "Rayman"
    assert node.slots[SlotID.WHAT_ACTION] == "喝咖啡"
    assert node.slots[SlotID.WHEN_TIME] == event.timestamp.isoformat()
    assert node.slots[SlotID.WHERE_LOCATION] == "上海"
    assert any(edge.relation == "mentions" for edge in adapter.backend.edges.values())


def test_query_fuses_vector_bm25_and_temporal_scores() -> None:
    """Query should combine text, slot-filter, and temporal evidence in Phase A."""
    adapter = GraphitiZhAdapterSkeleton()
    valid_at = datetime(2026, 5, 20, 9, 0, tzinfo=timezone.utc)
    adapter.upsert(_event(event_id="best", timestamp=valid_at - timedelta(minutes=2), action="喝咖啡并讨论路线图"))
    adapter.upsert(_event(event_id="older", timestamp=valid_at - timedelta(minutes=25), action="喝咖啡"))
    adapter.upsert(_event(event_id="other", timestamp=valid_at - timedelta(minutes=1), action="写周报"))

    hits = adapter.query(
        KGQuery(
            user_id="u_graphiti",
            text_query="路线图",
            slot_filter={SlotID.TOPIC: "咖啡"},
            valid_at=valid_at,
            top_k=3,
        )
    )

    assert [hit.event.event_id for hit in hits] == ["best", "other", "older"]
    assert "text-substring" in (hits[0].explanation or "")
    assert "slot-filter" in (hits[0].explanation or "")
    assert "temporal-near" in (hits[0].explanation or "")
    assert hits[0].score > hits[1].score > hits[2].score


def test_invalidate_marks_event_without_deleting_node() -> None:
    """Invalidate should set t_invalid while retaining historical records."""
    adapter = GraphitiZhAdapterSkeleton()
    timestamp = datetime(2026, 5, 20, 10, 0, tzinfo=timezone.utc)
    event = _event(event_id="evt_invalid", timestamp=timestamp, action="提交周报")
    node_id = adapter.upsert(event)

    invalid_at = timestamp + timedelta(minutes=5)
    adapter.invalidate(event.event_id, invalid_at)
    node = adapter.backend.get_node(node_id)

    assert node is not None
    assert node.event_id == event.event_id
    assert node.t_invalid == invalid_at


def test_valid_at_filters_expired_events() -> None:
    """valid_at should exclude records outside their valid-time window."""
    adapter = GraphitiZhAdapterSkeleton()
    base = datetime(2026, 5, 20, 11, 0, tzinfo=timezone.utc)
    expired = _event(event_id="evt_expired", timestamp=base - timedelta(minutes=20), action="喝拿铁")
    active = _event(event_id="evt_active", timestamp=base - timedelta(minutes=5), action="喝美式")
    adapter.upsert(expired)
    adapter.upsert(active)
    adapter.invalidate(expired.event_id, base - timedelta(minutes=10))

    hits = adapter.query(KGQuery(user_id="u_graphiti", text_query="喝", valid_at=base, top_k=10))

    assert [hit.event.event_id for hit in hits] == ["evt_active"]
