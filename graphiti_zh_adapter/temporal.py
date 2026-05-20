"""Bi-temporal operators for the Graphiti adapter.

Provides closed-open valid-time checks for KG records.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import List, Optional

from .mapper import KGNode


@dataclass(frozen=True)
class TemporalWindow:
    """Closed-open valid-time window; 用于 valid_at 时间点查询."""

    t_valid: datetime
    t_invalid: Optional[datetime] = None


@dataclass(frozen=True)
class BiTemporalRecord:
    """Temporal metadata for KG records; 后续可扩展 transaction time."""

    record_id: str
    window: TemporalWindow
    recorded_at: datetime


@dataclass
class BiTemporalOperator:
    """Bi-temporal helper surface for Phase A valid-time filtering."""

    require_timezone: bool = True

    def normalize_valid_time(self, value: datetime) -> datetime:
        """Normalize a datetime into UTC and enforce timezone awareness."""
        if self.require_timezone and value.tzinfo is None:
            raise ValueError("valid time must be timezone-aware")
        return value.astimezone(timezone.utc)

    def is_valid_at(self, window: TemporalWindow, valid_at: datetime) -> bool:
        """Check the closed-open interval t_valid <= at < t_invalid."""
        normalized_at = self.normalize_valid_time(valid_at)
        start = self.normalize_valid_time(window.t_valid)
        if normalized_at < start:
            return False
        if window.t_invalid is None:
            return True
        return normalized_at < self.normalize_valid_time(window.t_invalid)

    def invalidate_window(self, window: TemporalWindow, t_invalid: datetime) -> TemporalWindow:
        """Return one invalidated window and guard monotonicity."""
        normalized_invalid = self.normalize_valid_time(t_invalid)
        if normalized_invalid < self.normalize_valid_time(window.t_valid):
            raise ValueError("t_invalid must be greater than or equal to t_valid")
        return TemporalWindow(t_valid=window.t_valid, t_invalid=normalized_invalid)

    def filter_active(self, nodes: List[KGNode], at_time: datetime) -> List[KGNode]:
        """Filter nodes that are active at one valid-time instant."""
        return [
            node
            for node in nodes
            if self.is_valid_at(
                TemporalWindow(t_valid=node.t_valid, t_invalid=node.t_invalid),
                at_time,
            )
        ]
