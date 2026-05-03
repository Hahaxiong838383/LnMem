"""Smoke tests for graphiti_zh_adapter skeleton collection."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from contracts.event_tensor import EventTensor
from contracts.protocols import GraphitiZhAdapter, KGHit, KGQuery
from contracts.schema_v1 import SlotID
from graphiti_zh_adapter import GraphitiZhAdapterSkeleton


@runtime_checkable
class RuntimeGraphitiZhAdapter(GraphitiZhAdapter, Protocol):
    """Runtime-checkable mirror for Protocol smoke validation."""


def test_contract_imports_are_available() -> None:
    """contracts.* graphiti types import cleanly."""
    assert EventTensor(user_id="u1", timestamp=datetime.now(timezone.utc)).user_id == "u1"
    assert KGQuery(user_id="u1").top_k == 10
    assert KGHit is not None
    assert SlotID.WHO_ACTOR.value == 1


def test_adapter_satisfies_graphiti_protocol_shape() -> None:
    """Skeleton exposes the GraphitiZhAdapter method surface."""
    adapter = GraphitiZhAdapterSkeleton()
    assert isinstance(adapter, RuntimeGraphitiZhAdapter)
