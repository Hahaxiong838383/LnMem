"""Bi-temporal operators for contracts/protocols.py::GraphitiZhAdapter.

Provides valid-at query surfaces and invalidation records for Chinese KG memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


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
    """Bi-temporal helper surface; 时间归一化和时区校验留到 M0 W2."""

    require_timezone: bool = True

    def normalize_valid_time(self, value: datetime) -> datetime:
        """Normalize valid time to canonical timezone; M0 W2 填 UTC 策略."""
        raise NotImplementedError("M0 W2 todo: normalize timezone-aware datetime")

    def is_valid_at(self, window: TemporalWindow, valid_at: datetime) -> bool:
        """Check if a record is valid at a point; M0 W2 填闭开区间判断."""
        raise NotImplementedError("M0 W2 todo: evaluate valid_at against temporal window")

    def invalidate_window(self, window: TemporalWindow, t_invalid: datetime) -> TemporalWindow:
        """Return invalidated window; M0 W2 填单调时间保护."""
        raise NotImplementedError("M0 W2 todo: apply bi-temporal invalidation")
