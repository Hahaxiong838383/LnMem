"""boss_share_cap monitor for contracts/protocols.py::SyntheticDataGenerator.

M0 W1 exposes API only; roadmap §3.2 logic is pending.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec


@dataclass(frozen=True)
class BossShareSnapshot:
    """Current source-share snapshot. 当前老板真实日志占比快照。"""

    total_events: int
    boss_events: int
    boss_share: float
    cap: float
    paused: bool


class BossShareMonitor:
    """Monitor real-log contribution share. 监控老板日志占比红线。"""

    def observe(self, events: Iterable[EventTensor], source: str) -> BossShareSnapshot:
        """Record generated events by source. 按来源记录未来的累计占比。"""
        raise NotImplementedError("M0 W2 todo: record source counts and return snapshot")

    def assert_within_cap(self, spec: SynthSpec) -> None:
        """Raise when boss share cap is exceeded. 超过 boss_share_cap 时暂停注入。"""
        raise NotImplementedError("M0 W2 todo: enforce boss_share_cap before type_3 injection")

    def stats(self) -> dict:
        """Return monitor stats for pipeline.stats. 返回 evals 可读的占比统计。"""
        raise NotImplementedError("M0 W2 todo: expose cumulative boss share stats")
