"""Protocol adapter for the Graphiti Zh module.

Wires EventTensor mapping, in-memory storage, temporal filtering, and ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List

from contracts.event_tensor import EventTensor
from contracts.protocols import GraphitiZhAdapter, KGHit, KGQuery

from .client import GraphitiClient
from .mapper import EventTensorMapper
from .retriever import GraphitiRetriever
from .stub_backend import InMemoryGraphitiBackend
from .temporal import BiTemporalOperator


@dataclass
class GraphitiZhAdapterSkeleton(GraphitiZhAdapter):
    """GraphitiZhAdapter entrypoint; 中文 EventTensor 到 bi-temporal KG 的入口."""

    mapper: EventTensorMapper = field(default_factory=EventTensorMapper)
    retriever: GraphitiRetriever = field(default_factory=GraphitiRetriever)
    temporal: BiTemporalOperator = field(default_factory=BiTemporalOperator)
    backend: InMemoryGraphitiBackend = field(default_factory=InMemoryGraphitiBackend)
    client: GraphitiClient = field(default_factory=GraphitiClient)

    def upsert(self, event: EventTensor) -> str:
        """Upsert one EventTensor into the in-memory KG."""
        node = self.mapper.to_kg_node(event)
        edges = self.mapper.to_kg_edges(event)
        node_id = self.backend.upsert_node(node)
        self.backend.add_edges(edges)
        return node_id

    def query(self, q: KGQuery) -> List[KGHit]:
        """Query KG hits with backend prefilter, valid-time gating, and ranking."""
        candidates = self.backend.query_candidates(q)
        if q.valid_at is not None:
            candidates = self.temporal.filter_active(candidates, q.valid_at)
        return self.retriever.rank(candidates, q)

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """Mark one event invalid without deleting the stored node."""
        self.backend.invalidate(event_id, t_invalid)
