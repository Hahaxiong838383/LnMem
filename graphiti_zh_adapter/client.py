"""Client placeholder for contracts/protocols.py::GraphitiZhAdapter.

Real graphiti-core integration is intentionally deferred to M0 W3.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from contracts.event_tensor import EventTensor

from .mapper import KGEdge, KGNode


@dataclass(frozen=True)
class GraphitiClientConfig:
    """Connection config for future graphiti-core; 中文环境下先保留占位参数."""

    endpoint: Optional[str] = None
    namespace: str = "lnmem_zh"
    timeout_seconds: float = 10.0
    dry_run: bool = True


@dataclass
class GraphitiClient:
    """Thin wrapper over future graphiti-core APIs; 当前只定义调用边界."""

    config: GraphitiClientConfig = GraphitiClientConfig()

    def connect(self) -> None:
        """Open a future graphiti-core connection; M0 W3 接真实客户端."""
        raise NotImplementedError("M0 W2 todo: connect to graphiti-core client")

    def upsert_episode(self, node: KGNode, edges: List[KGEdge]) -> str:
        """Write one event node and its temporal edges; M0 W3 对齐 Graphiti API."""
        raise NotImplementedError("M0 W2 todo: call graphiti-core upsert episode")

    def search(self, text: str, valid_at: Optional[datetime]) -> List[str]:
        """Search future graphiti-core index; M0 W3 返回候选 node ids."""
        raise NotImplementedError("M0 W2 todo: call graphiti-core hybrid search")

    def invalidate_episode(self, event_id: str, t_invalid: datetime) -> None:
        """Invalidate future graphiti-core episode; M0 W3 写 bi-temporal end time."""
        raise NotImplementedError("M0 W2 todo: call graphiti-core temporal invalidation")
