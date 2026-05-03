"""Hybrid retriever for contracts/protocols.py::GraphitiZhAdapter.

Defines vector, BM25, and temporal fusion surfaces for KGQuery to KGHit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from contracts.protocols import KGHit, KGQuery

from .mapper import KGNode


@dataclass(frozen=True)
class RetrievalWeights:
    """Fusion weights for candidate scoring; 中文召回默认三路同权偏 temporal."""

    vector: float = 0.34
    bm25: float = 0.33
    temporal: float = 0.33


@dataclass(frozen=True)
class CandidateScore:
    """Intermediate score for one KG candidate; 便于解释命中路径."""

    node_id: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    temporal_score: float = 0.0
    explanation: Optional[str] = None


@dataclass
class GraphitiRetriever:
    """Query planner for KGQuery to KGHit; M0 W3 填三路融合策略."""

    weights: RetrievalWeights = RetrievalWeights()

    def score_candidate(self, q: KGQuery, node: KGNode) -> CandidateScore:
        """Score one candidate node; M0 W3 填 vector/BM25/temporal 归一化."""
        raise NotImplementedError("M0 W2 todo: compute hybrid candidate score")

    def query(self, q: KGQuery, candidates: List[KGNode]) -> List[KGHit]:
        """Rank candidates into KGHit list; M0 W3 填 top_k 和 explanation."""
        raise NotImplementedError("M0 W2 todo: rank candidates and build KGHit objects")
