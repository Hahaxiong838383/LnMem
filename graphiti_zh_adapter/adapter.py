"""Protocol adapter for the Graphiti Zh module.

Wires EventTensor mapping, backend storage, temporal filtering, and ranking.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Union

from contracts.event_tensor import EventTensor
from contracts.protocols import GraphitiZhAdapter, KGHit, KGQuery

from .client import GraphitiClient
from .mapper import EventTensorMapper
from .retriever import GraphitiRetriever
from .sqlite_backend import SQLiteGraphitiBackend
from .stub_backend import InMemoryGraphitiBackend
from .temporal import BiTemporalOperator


@dataclass
class GraphitiZhAdapterSkeleton(GraphitiZhAdapter):
    """GraphitiZhAdapter entrypoint; 中文 EventTensor 到 bi-temporal KG 的入口."""

    mapper: EventTensorMapper = field(default_factory=EventTensorMapper)
    retriever: GraphitiRetriever = field(default_factory=GraphitiRetriever)
    temporal: BiTemporalOperator = field(default_factory=BiTemporalOperator)
    backend_type: Literal["memory", "sqlite"] = "memory"
    db_path: Optional[Path] = None
    backend: Optional[Union[InMemoryGraphitiBackend, SQLiteGraphitiBackend]] = None
    client: GraphitiClient = field(default_factory=GraphitiClient)

    def __post_init__(self) -> None:
        if self.backend is not None:
            return
        if self.backend_type == "sqlite":
            if self.db_path is None:
                raise ValueError("db_path is required when backend_type='sqlite'")
            self.backend = SQLiteGraphitiBackend(self.db_path)
            return
        self.backend = InMemoryGraphitiBackend()

    def upsert(self, event: EventTensor) -> str:
        """Upsert one EventTensor into the configured KG backend."""
        if self.backend is None:
            raise RuntimeError("backend not initialized")
        node = self.mapper.to_kg_node(event)
        edges = self.mapper.to_kg_edges(event)
        node_id = self.backend.upsert_node(node)
        self.backend.add_edges(edges)
        return node_id

    def query(self, q: KGQuery) -> List[KGHit]:
        """Query KG hits with backend prefilter, valid-time gating, and ranking."""
        if self.backend is None:
            raise RuntimeError("backend not initialized")
        candidates = self.backend.query_candidates(q)
        if q.valid_at is not None:
            candidates = self.temporal.filter_active(candidates, q.valid_at)
        return self.retriever.rank(candidates, q)

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """Mark one event invalid without deleting the stored node."""
        if self.backend is None:
            raise RuntimeError("backend not initialized")
        self.backend.invalidate(event_id, t_invalid)
