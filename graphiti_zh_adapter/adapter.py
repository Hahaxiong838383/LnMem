"""Protocol skeleton for contracts/protocols.py::GraphitiZhAdapter.

This module wires EventTensor, mapping, retrieval, temporal, and backend stubs.
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
    """GraphitiZhAdapter M0 skeleton; 中文 EventTensor 到 bi-temporal KG 的入口."""

    mapper: EventTensorMapper = field(default_factory=EventTensorMapper)
    retriever: GraphitiRetriever = field(default_factory=GraphitiRetriever)
    temporal: BiTemporalOperator = field(default_factory=BiTemporalOperator)
    backend: InMemoryGraphitiBackend = field(default_factory=InMemoryGraphitiBackend)
    client: GraphitiClient = field(default_factory=GraphitiClient)

    def upsert(self, event: EventTensor) -> str:
        """Upsert one EventTensor into KG; M0 W2 填入 P0 槽位入图逻辑."""
        raise NotImplementedError("M0 W2 todo: map EventTensor to Graphiti nodes and edges")

    def query(self, q: KGQuery) -> List[KGHit]:
        """Query temporal KG hits; M0 W3 填入 vector/BM25/temporal 融合召回."""
        raise NotImplementedError("M0 W2 todo: retrieve KGHit list from Chinese Graphiti index")

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """Mark event invalid without deletion; M0 W2 填入 bi-temporal 失效写入."""
        raise NotImplementedError("M0 W2 todo: write t_invalid for event node and edges")
