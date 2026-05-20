"""EventTensor mapper for the Graphiti adapter.

Maps EventTensor payloads to in-memory KG node and edge records.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional

from contracts.event_tensor import EventTensor, SlotValue
from contracts.schema_v1 import SlotID


@dataclass(frozen=True)
class KGNode:
    """KG node shape for one EventTensor; 节点属性保留 20 槽."""

    node_id: str
    event_id: str
    user_id: str
    t_valid: datetime
    t_invalid: Optional[datetime]
    slots: Dict[SlotID, Any] = field(default_factory=dict)
    schema_version: str = "v1"


@dataclass(frozen=True)
class KGEdge:
    """KG edge shape with bi-temporal bounds; 边记录有效和失效时间."""

    edge_id: str
    source_id: str
    target_id: str
    relation: str
    t_valid: datetime
    t_invalid: Optional[datetime] = None
    evidence_event_id: Optional[str] = None


@dataclass
class EventTensorMapper:
    """Mapper between EventTensor and KG records; 关系归一化留到 Phase B."""

    node_prefix: str = "node"

    def to_kg_node(self, event: EventTensor) -> KGNode:
        """Convert one EventTensor into a KGNode; 当前只存 value 层。"""
        return KGNode(
            node_id=f"{self.node_prefix}_{event.event_id}",
            event_id=event.event_id,
            user_id=event.user_id,
            t_valid=event.timestamp,
            t_invalid=event.valid_until,
            slots={slot_id: slot_value.value for slot_id, slot_value in event.slots.items()},
            schema_version=event.schema_version,
        )

    def event_to_node(self, event: EventTensor) -> KGNode:
        """Backward-compatible alias for the Protocol skeleton docs."""
        return self.to_kg_node(event)

    def node_to_event(self, node: KGNode) -> EventTensor:
        """Convert one KGNode back into an EventTensor."""
        slots = {slot_id: SlotValue(value=value) for slot_id, value in node.slots.items()}
        return EventTensor(
            user_id=node.user_id,
            timestamp=node.t_valid,
            slots=slots,
            event_id=node.event_id,
            valid_until=node.t_invalid,
            schema_version=node.schema_version,
        )

    def to_kg_edges(self, event: EventTensor) -> List[KGEdge]:
        """Extract simple relation edges from one EventTensor."""
        node_id = f"{self.node_prefix}_{event.event_id}"
        edges: List[KGEdge] = []
        related = event.get(SlotID.RELATED_ENTITIES)
        if related is not None:
            edges.extend(self._build_mentions_edges(node_id, event, related.value))
        causal = event.get(SlotID.CAUSAL_LINK)
        if causal is not None and causal.value:
            target_id = f"{self.node_prefix}_{causal.value}"
            edges.append(
                KGEdge(
                    edge_id=f"edge_{event.event_id}_causal_{causal.value}",
                    source_id=node_id,
                    target_id=target_id,
                    relation="causal_link",
                    t_valid=event.timestamp,
                    t_invalid=event.valid_until,
                    evidence_event_id=event.event_id,
                )
            )
        return edges

    def event_to_edges(self, event: EventTensor, node: KGNode) -> List[KGEdge]:
        """Backward-compatible alias for the Protocol skeleton docs."""
        del node
        return self.to_kg_edges(event)

    def _build_mentions_edges(
        self,
        node_id: str,
        event: EventTensor,
        raw_entities: Any,
    ) -> List[KGEdge]:
        entities = raw_entities if isinstance(raw_entities, Iterable) and not isinstance(raw_entities, (str, bytes, dict)) else [raw_entities]
        edges: List[KGEdge] = []
        for index, entity in enumerate(entities):
            if entity in (None, ""):
                continue
            edges.append(
                KGEdge(
                    edge_id=f"edge_{event.event_id}_mentions_{index}",
                    source_id=node_id,
                    target_id=f"entity_{entity}",
                    relation="mentions",
                    t_valid=event.timestamp,
                    t_invalid=event.valid_until,
                    evidence_event_id=event.event_id,
                )
            )
        return edges
