"""In-memory backend for contracts/protocols.py::GraphitiZhAdapter.

This mirrors future Graphiti storage boundaries without graphiti-core dependency.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from contracts.protocols import KGQuery

from .mapper import KGEdge, KGNode


@dataclass
class InMemoryGraphitiBackend:
    """Dict-backed backend placeholder; 只声明 fake Graphiti 行为边界."""

    nodes: Dict[str, KGNode] = field(default_factory=dict)
    edges: Dict[str, KGEdge] = field(default_factory=dict)

    def upsert_node(self, node: KGNode) -> str:
        """Store or replace one node; M0 W2 填 dict 写入占位逻辑."""
        raise NotImplementedError("M0 W2 todo: store KGNode in in-memory backend")

    def add_edges(self, edges: List[KGEdge]) -> None:
        """Store temporal edges; M0 W2 填边去重和 t_valid/t_invalid 写入."""
        raise NotImplementedError("M0 W2 todo: store KGEdge records in in-memory backend")

    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Fetch one node by id; M0 W2 填 node lookup."""
        raise NotImplementedError("M0 W2 todo: fetch KGNode by node_id")

    def query_candidates(self, q: KGQuery) -> List[KGNode]:
        """Return rough candidates before ranking; M0 W2 填 filter/temporal 初筛."""
        raise NotImplementedError("M0 W2 todo: filter candidate KGNodes")

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """Set invalidation time for an event; M0 W2 填不删除的失效标记."""
        raise NotImplementedError("M0 W2 todo: update t_invalid for matching records")
