"""Collect-only placeholders for future graphiti_zh_adapter behavior tests."""
from __future__ import annotations

import pytest


def test_upsert_maps_p0_slots_to_graph_node() -> None:
    """Upsert should persist all P0 slots as node properties."""
    pytest.skip("M0 W2 implementation pending: P0 slots must be projected into KGNode")


def test_query_fuses_vector_bm25_and_temporal_scores() -> None:
    """Query should combine vector, BM25, and temporal evidence."""
    pytest.skip("M0 W2 implementation pending: hybrid retrieval scoring is not implemented")


def test_invalidate_marks_event_without_deleting_node() -> None:
    """Invalidate should set t_invalid while retaining historical records."""
    pytest.skip("M0 W2 implementation pending: bi-temporal invalidation is not implemented")


def test_valid_at_filters_expired_events() -> None:
    """valid_at should exclude records outside their valid-time window."""
    pytest.skip("M0 W2 implementation pending: temporal window filtering is not implemented")
