"""Hybrid retriever for the Graphiti adapter.

Phase A uses text, slot, and temporal proximity as simple scoring signals.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from contracts.protocols import KGHit, KGQuery

from .mapper import EventTensorMapper, KGNode


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
    """Query planner for KGQuery to KGHit; Phase A keeps scoring explicit."""

    weights: RetrievalWeights = RetrievalWeights()
    mapper: EventTensorMapper = field(default_factory=EventTensorMapper)

    def score_candidate(self, q: KGQuery, node: KGNode) -> CandidateScore:
        """Score one candidate with simple substring, filter, and time signals."""
        text_score = self._text_score(q.text_query, node)
        filter_score = self._slot_filter_score(q.slot_filter, node)
        temporal_score = self._temporal_score(q.valid_at, node.t_valid)
        return CandidateScore(
            node_id=node.node_id,
            vector_score=text_score,
            bm25_score=filter_score,
            temporal_score=temporal_score,
            explanation=self._build_explanation(text_score, filter_score, temporal_score),
        )

    def query(self, q: KGQuery, candidates: List[KGNode]) -> List[KGHit]:
        """Backward-compatible alias for rank()."""
        return self.rank(candidates, q)

    def rank(self, candidates: List[KGNode], q: KGQuery) -> List[KGHit]:
        """Rank candidates and return top-k KGHit records."""
        scored = []
        for node in candidates:
            candidate_score = self.score_candidate(q, node)
            total = (
                candidate_score.vector_score * self.weights.vector
                + candidate_score.bm25_score * self.weights.bm25
                + candidate_score.temporal_score * self.weights.temporal
            )
            scored.append((total, candidate_score, node))
        scored.sort(key=lambda item: item[0], reverse=True)
        hits: List[KGHit] = []
        for total, candidate_score, node in scored[: q.top_k]:
            hits.append(
                KGHit(
                    event=self.mapper.node_to_event(node),
                    score=total,
                    explanation=candidate_score.explanation,
                )
            )
        return hits

    def _text_score(self, text_query: Optional[str], node: KGNode) -> float:
        if not text_query:
            return 0.0
        query = text_query.lower()
        for value in node.slots.values():
            if query in str(value).lower():
                return 1.0
        return 0.0

    def _slot_filter_score(self, slot_filter: Optional[dict], node: KGNode) -> float:
        if not slot_filter:
            return 0.0
        return 0.5 if all(node.slots.get(slot_id) == expected for slot_id, expected in slot_filter.items()) else 0.0

    def _temporal_score(self, valid_at: Optional[datetime], t_valid: datetime) -> float:
        if valid_at is None:
            return 0.0
        minutes = abs((valid_at - t_valid).total_seconds()) / 60.0
        return 0.3 / (1.0 + minutes)

    def _build_explanation(self, text_score: float, filter_score: float, temporal_score: float) -> str:
        parts = []
        if text_score:
            parts.append("text-substring")
        if filter_score:
            parts.append("slot-filter")
        if temporal_score:
            parts.append("temporal-near")
        return ", ".join(parts) or "no-match-signals"
