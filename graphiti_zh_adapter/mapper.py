"""EventTensor mapper for contracts/protocols.py::GraphitiZhAdapter.

Maps 20-slot EventTensor payloads to KG node and bi-temporal edge shapes.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from contracts.event_tensor import EventTensor
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
    """Mapper between EventTensor and KG records; 真实实体归一化留到 M0 W2."""

    node_prefix: str = "event"

    def event_to_node(self, event: EventTensor) -> KGNode:
        """Convert EventTensor to KGNode; M0 W2 填 20 槽属性投影."""
        raise NotImplementedError("M0 W2 todo: project EventTensor slots into KGNode")

    def node_to_event(self, node: KGNode) -> EventTensor:
        """Convert KGNode back to EventTensor; M0 W2 填 slot value 还原."""
        raise NotImplementedError("M0 W2 todo: rebuild EventTensor from KGNode")

    def event_to_edges(self, event: EventTensor, node: KGNode) -> List[KGEdge]:
        """Extract KG edges from EventTensor links; M0 W2 填 temporal/causal/social 边."""
        raise NotImplementedError("M0 W2 todo: map link slots to bi-temporal KG edges")
