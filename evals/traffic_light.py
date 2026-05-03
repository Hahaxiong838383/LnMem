"""Traffic-light skeleton for contracts/protocols.py TrafficLight.

阈值来自 roadmap/milestones.md §3，真实判定 M0 W2-W4 填。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from contracts.protocols import EvalResult, TrafficLight


class LightColor(str, Enum):
    """Traffic light color values. 三色枚举。"""

    RED = "RED"
    YELLOW = "YELLOW"
    GREEN = "GREEN"


@dataclass(frozen=True)
class MilestoneThresholds:
    """Roadmap threshold placeholder. roadmap §3 阈值占位。"""

    milestone: str
    red_rules: Sequence[str]
    yellow_rules: Sequence[str]
    green_rule: str


ROADMAP_RULEBOOK: Mapping[str, MilestoneThresholds] = {
    "M1": MilestoneThresholds(
        milestone="M1",
        red_rules=("p0_recall < 0.60", "graphiti_unavailable", "pipeline_not_run"),
        yellow_rules=("0.60 <= p0_recall < 0.80", "graphiti_f1 < 0.50", "pipeline_unstable"),
        green_rule="all_m1_targets_met",
    ),
    "M2": MilestoneThresholds(
        milestone="M2",
        red_rules=("lnn_diverged", "event_segmentation_f1 < 0.30"),
        yellow_rules=("lnn_weakly_usable", "0.30 <= event_segmentation_f1 < 0.50"),
        green_rule="all_m2_targets_met",
    ),
    "M3": MilestoneThresholds(
        milestone="M3",
        red_rules=("schedule_lag > 0.50",),
        yellow_rules=("0.30 <= schedule_lag <= 0.50",),
        green_rule="schedule_lag < 0.30",
    ),
}


def build_traffic_light(results: Sequence[EvalResult], milestone: str) -> TrafficLight:
    """Aggregate eval results into TrafficLight. 聚合结果为三色判定。"""
    raise NotImplementedError("M0 W2 todo: implement roadmap §3 aggregation")
