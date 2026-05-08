"""Day 3 剧本校验器（v0.1 / 2026-05-06）。

校验单个 Script（单画像 × 单天）的字段完整性、时间连续性、状态分布、duration 区间。

不依赖 LLM，纯逻辑校验——为下游 ``script_to_signals.py`` 提供可信输入门槛。

多画像聚合统计在 ``dataset_report.py``，本模块只管单 Script。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from .event_schema import (
    DURATION_MAX_S,
    DURATION_MIN_S,
    Activity,
    Script,
    ScriptEvent,
    State,
)


# ---------- 校验结果数据结构 ----------


@dataclass(frozen=True)
class ValidationIssue:
    """单条校验问题。"""

    severity: str          # "error" / "warning"
    code: str              # e.g. "TIME_OVERLAP" / "DURATION_OUT_OF_RANGE"
    message: str
    event_id: str = ""     # 关联 event（empty 表示 script 级问题）


@dataclass(frozen=True)
class ScriptValidationReport:
    """单 Script 校验报告。"""

    persona_id: str
    day_index: int
    n_events: int
    issues: tuple[ValidationIssue, ...]
    state_distribution: dict[str, int] = field(default_factory=dict)
    activity_distribution: dict[str, int] = field(default_factory=dict)
    total_duration_s: float = 0.0
    duration_coverage_h: float = 0.0     # total_duration / 3600

    @property
    def passed(self) -> bool:
        return not any(i.severity == "error" for i in self.issues)

    @property
    def n_errors(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def n_warnings(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    def summary_line(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {self.persona_id} day{self.day_index}: "
            f"{self.n_events} events, {self.duration_coverage_h:.1f}h, "
            f"{self.n_errors} errors, {self.n_warnings} warnings"
        )


# ---------- 校验逻辑 ----------


def _parse_iso(ts: str) -> datetime | None:
    try:
        return datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return None


def _validate_event(event: ScriptEvent) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if not event.event_id:
        issues.append(
            ValidationIssue("error", "EMPTY_EVENT_ID", "event_id 不能为空", event.event_id)
        )
    if not event.persona_id:
        issues.append(
            ValidationIssue("error", "EMPTY_PERSONA_ID", "persona_id 不能为空", event.event_id)
        )
    if _parse_iso(event.ts) is None:
        issues.append(
            ValidationIssue(
                "error",
                "BAD_TIMESTAMP",
                f"ts 不是合法 ISO8601: {event.ts!r}",
                event.event_id,
            )
        )
    if not (DURATION_MIN_S <= event.duration_s <= DURATION_MAX_S):
        issues.append(
            ValidationIssue(
                "error",
                "DURATION_OUT_OF_RANGE",
                f"duration_s={event.duration_s:.1f} 不在 [{DURATION_MIN_S}, {DURATION_MAX_S}]",
                event.event_id,
            )
        )

    hint = event.multimodal_hint
    for name, val in [
        ("eye_closure", hint.eye_closure),
        ("head_pitch_norm", hint.head_pitch_norm),
        ("head_yaw_norm", hint.head_yaw_norm),
        ("gaze_stability", hint.gaze_stability),
        ("voice_present", hint.voice_present),
        ("utterance_density", hint.utterance_density),
    ]:
        if not (0.0 <= val <= 1.0):
            issues.append(
                ValidationIssue(
                    "error",
                    "HINT_OUT_OF_RANGE",
                    f"multimodal_hint.{name}={val:.3f} 不在 [0,1]",
                    event.event_id,
                )
            )
    if hint.app_switch_count_5min < 0:
        issues.append(
            ValidationIssue(
                "error",
                "NEGATIVE_APP_SWITCH",
                f"app_switch_count_5min={hint.app_switch_count_5min} 不能为负",
                event.event_id,
            )
        )

    return issues


def _validate_temporal_continuity(events: tuple[ScriptEvent, ...]) -> list[ValidationIssue]:
    """时间序列连续性：ts 严格递增 + 相邻事件可以有 ≤ 30s 间隙（允许小停顿）。"""
    issues: list[ValidationIssue] = []
    if len(events) < 2:
        return issues

    parsed: list[tuple[ScriptEvent, datetime]] = []
    for ev in events:
        dt = _parse_iso(ev.ts)
        if dt is not None:
            parsed.append((ev, dt))

    for i in range(1, len(parsed)):
        prev_ev, prev_dt = parsed[i - 1]
        curr_ev, curr_dt = parsed[i]
        prev_end = prev_dt + timedelta(seconds=prev_ev.duration_s)

        if curr_dt < prev_end - timedelta(seconds=1):  # 1s 容差
            issues.append(
                ValidationIssue(
                    "error",
                    "TIME_OVERLAP",
                    f"事件 {curr_ev.event_id} ({curr_ev.ts}) 与上一事件 {prev_ev.event_id} 时间重叠",
                    curr_ev.event_id,
                )
            )
        elif curr_dt > prev_end + timedelta(seconds=30):
            gap_s = (curr_dt - prev_end).total_seconds()
            issues.append(
                ValidationIssue(
                    "warning",
                    "LARGE_TIME_GAP",
                    f"事件 {curr_ev.event_id} 与上一事件间隔 {gap_s:.0f}s（> 30s）",
                    curr_ev.event_id,
                )
            )

    return issues


def _validate_persona_consistency(script: Script) -> list[ValidationIssue]:
    """所有事件的 persona_id / day_index 必须与 Script 一致。"""
    issues: list[ValidationIssue] = []
    for ev in script.events:
        if ev.persona_id != script.persona_id:
            issues.append(
                ValidationIssue(
                    "error",
                    "PERSONA_MISMATCH",
                    f"事件 persona_id={ev.persona_id} 不等于 script.persona_id={script.persona_id}",
                    ev.event_id,
                )
            )
        if ev.day_index != script.day_index:
            issues.append(
                ValidationIssue(
                    "error",
                    "DAY_INDEX_MISMATCH",
                    f"事件 day_index={ev.day_index} 不等于 script.day_index={script.day_index}",
                    ev.event_id,
                )
            )
    return issues


def _validate_state_diversity(events: tuple[ScriptEvent, ...]) -> list[ValidationIssue]:
    """状态多样性：8 类至少出现 3 类（避免 LLM 一边倒）。"""
    issues: list[ValidationIssue] = []
    states = Counter(ev.state for ev in events)
    if len(states) < 3:
        issues.append(
            ValidationIssue(
                "warning",
                "LOW_STATE_DIVERSITY",
                f"仅出现 {len(states)} 类状态（< 3）：{sorted(s.value for s in states)}",
            )
        )
    return issues


def validate_script(script: Script) -> ScriptValidationReport:
    """对单个 Script 跑全量校验。"""
    issues: list[ValidationIssue] = []

    if len(script.events) == 0:
        issues.append(
            ValidationIssue(
                "error",
                "EMPTY_EVENTS",
                f"script {script.persona_id} day{script.day_index} 事件列表为空",
            )
        )

    for ev in script.events:
        issues.extend(_validate_event(ev))

    issues.extend(_validate_persona_consistency(script))
    issues.extend(_validate_temporal_continuity(script.events))
    issues.extend(_validate_state_diversity(script.events))

    state_dist = Counter(ev.state.value for ev in script.events)
    activity_dist = Counter(ev.activity.value for ev in script.events)
    total_dur = sum(ev.duration_s for ev in script.events)

    return ScriptValidationReport(
        persona_id=script.persona_id,
        day_index=script.day_index,
        n_events=len(script.events),
        issues=tuple(issues),
        state_distribution=dict(state_dist),
        activity_distribution=dict(activity_dist),
        total_duration_s=total_dur,
        duration_coverage_h=total_dur / 3600.0,
    )


def validate_script_dict(d: dict[str, Any]) -> ScriptValidationReport:
    """直接对 JSON dict 校验（不构造 Script，便于 LLM 输出实时校验）。"""
    return validate_script(Script.from_dict(d))


__all__ = [
    "ValidationIssue",
    "ScriptValidationReport",
    "validate_script",
    "validate_script_dict",
]
