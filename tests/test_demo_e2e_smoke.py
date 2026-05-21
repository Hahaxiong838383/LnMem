"""Smoke tests for Streamlit E2E demo and parser-to-KG closure."""
from __future__ import annotations

from datetime import datetime, timezone

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import KGQuery, ParseRequest
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter.adapter import GraphitiZhAdapterSkeleton


def test_streamlit_app_imports() -> None:
    """App module should import and expose core entrypoint helpers."""
    import apps.demo_e2e.main as app

    assert hasattr(app, "main")
    assert hasattr(app, "init_state")
    assert hasattr(app, "left_panel")
    assert hasattr(app, "right_panel")


def test_e2e_parser_to_kg_pipeline() -> None:
    """Mock parser output should be upserted and then query-hit from KG."""

    class DummyParser:
        def parse(self, req: ParseRequest) -> EventTensor:
            return EventTensor(
                user_id=req.user_id,
                timestamp=req.timestamp,
                slots={
                    SlotID.WHO_ACTOR: SlotValue("我"),
                    SlotID.WHAT_ACTION: SlotValue("吃饭"),
                    SlotID.WHERE_LOCATION: SlotValue("公司"),
                },
            )

    parser = DummyParser()
    adapter = GraphitiZhAdapterSkeleton()

    req = ParseRequest(
        text="昨晚和小李在公司吃饭",
        user_id="chuange",
        timestamp=datetime.now(timezone.utc),
    )
    event = parser.parse(req)
    node_id = adapter.upsert(event)

    hits = adapter.query(
        KGQuery(
            user_id="chuange",
            text_query="吃饭",
            valid_at=datetime.now(timezone.utc),
            top_k=5,
        )
    )

    assert node_id.startswith("node_")
    assert hits
    assert hits[0].event.event_id == event.event_id
