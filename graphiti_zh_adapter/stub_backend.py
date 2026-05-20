"""In-memory backend for the Graphiti adapter.

Stores KG nodes and edges in dicts for Phase A validation.
"""
from __future__ import annotations

from dataclasses import replace
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from contracts.schema_v1 import SlotID
from contracts.protocols import KGQuery

from .mapper import KGEdge, KGNode


@dataclass
class InMemoryGraphitiBackend:
    """Dict-backed backend used before graphiti-core integration."""

    nodes: Dict[str, KGNode] = field(default_factory=dict)
    edges: Dict[str, KGEdge] = field(default_factory=dict)

    def upsert_node(self, node: KGNode) -> str:
        """Store or replace one node by node_id."""
        self.nodes[node.node_id] = node
        return node.node_id

    def add_edges(self, edges: List[KGEdge]) -> None:
        """Store edges by edge_id and naturally dedupe repeated inserts."""
        for edge in edges:
            self.edges[edge.edge_id] = edge

    def get_node(self, node_id: str) -> Optional[KGNode]:
        """Fetch one node by node_id."""
        return self.nodes.get(node_id)

    def query_candidates(self, q: KGQuery) -> List[KGNode]:
        """Return rough candidates filtered by user, time, and exact slots."""
        return [
            node
            for node in self.nodes.values()
            if node.user_id == q.user_id
            and self._matches_slot_filter(node, q.slot_filter)
            and self._matches_valid_at(node, q.valid_at)
        ]

    def invalidate(self, event_id: str, t_invalid: datetime) -> None:
        """Set t_invalid for matching nodes and keep history intact."""
        for node_id, node in list(self.nodes.items()):
            if node.event_id == event_id:
                self.nodes[node_id] = replace(node, t_invalid=t_invalid)
        for edge_id, edge in list(self.edges.items()):
            if edge.evidence_event_id == event_id:
                self.edges[edge_id] = replace(edge, t_invalid=t_invalid)

    def _matches_slot_filter(self, node: KGNode, slot_filter: Optional[dict]) -> bool:
        if not slot_filter:
            return True
        for slot_key, expected in slot_filter.items():
            slot_id = slot_key if isinstance(slot_key, SlotID) else SlotID(slot_key)
            if node.slots.get(slot_id) != expected:
                return False
        return True

    def _matches_valid_at(self, node: KGNode, valid_at: Optional[datetime]) -> bool:
        if valid_at is None:
            return True
        if node.t_valid > valid_at:
            return False
        return node.t_invalid is None or valid_at < node.t_invalid
