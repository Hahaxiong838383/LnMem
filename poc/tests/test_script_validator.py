"""Day 3 Lane A 测试 — script_validator 校验逻辑（v0.1）。"""
from __future__ import annotations

from poc.src.data.event_schema import (
    Activity,
    MultimodalHint,
    Script,
    ScriptEvent,
    State,
)
from poc.src.data.script_validator import validate_script


def _make_event(
    idx: int,
    persona_id: str = "vboss_001",
    day_index: int = 0,
    ts: str = "2026-05-06T09:00:00",
    duration_s: float = 300.0,
    activity: Activity = Activity.WORK_FOCUS,
    state: State = State.FOCUSED,
    hint: MultimodalHint | None = None,
) -> ScriptEvent:
    return ScriptEvent(
        event_id=f"{persona_id}_d{day_index}_e{idx:03d}",
        ts=ts,
        persona_id=persona_id,
        day_index=day_index,
        duration_s=duration_s,
        activity=activity,
        state=state,
        multimodal_hint=hint or MultimodalHint(),
    )


def _seq_script(states: list[State], activities: list[Activity] | None = None) -> Script:
    """构造一个时间连续 + 状态多样的 Script，每事件 600s，从 9:00 起。"""
    if activities is None:
        activities = [Activity.WORK_FOCUS] * len(states)
    events: list[ScriptEvent] = []
    base_minute = 0
    for i, (st, act) in enumerate(zip(states, activities)):
        ts = f"2026-05-06T09:{base_minute:02d}:00"
        events.append(
            _make_event(idx=i, ts=ts, duration_s=600.0, state=st, activity=act)
        )
        base_minute += 10
    return Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=tuple(events),
    )


# ---------- 通过场景 ----------


def test_validate_passes_clean_script() -> None:
    s = _seq_script(
        states=[State.FOCUSED, State.NEUTRAL, State.DISTRACTED, State.LOW_ENERGY],
    )
    rep = validate_script(s)
    assert rep.passed, f"should pass, got: {[i.message for i in rep.issues]}"
    assert rep.n_errors == 0
    assert rep.n_events == 4


def test_state_distribution_counted() -> None:
    s = _seq_script(states=[State.FOCUSED, State.FOCUSED, State.NEUTRAL, State.LOW_ENERGY])
    rep = validate_script(s)
    assert rep.state_distribution["focused"] == 2
    assert rep.state_distribution["neutral"] == 1


def test_activity_distribution_counted() -> None:
    s = _seq_script(
        states=[State.FOCUSED, State.NEUTRAL, State.DISTRACTED],
        activities=[Activity.WORK_FOCUS, Activity.BREAK_SHORT, Activity.WORK_MEETING],
    )
    rep = validate_script(s)
    assert rep.activity_distribution["work_focus"] == 1
    assert rep.activity_distribution["break_short"] == 1
    assert rep.activity_distribution["work_meeting"] == 1


def test_duration_coverage_hours() -> None:
    s = _seq_script(states=[State.FOCUSED] * 3)  # 3 × 600s = 1800s = 0.5h
    rep = validate_script(s)
    # state 多样性会触发 warning，但 errors 应为 0
    assert rep.n_errors == 0
    assert abs(rep.duration_coverage_h - 0.5) < 0.01


# ---------- 必填字段错误 ----------


def test_validate_catches_empty_event_id() -> None:
    bad_ev = ScriptEvent(
        event_id="",
        ts="2026-05-06T09:00:00",
        persona_id="vboss_001",
        day_index=0,
        duration_s=300.0,
        activity=Activity.WORK_FOCUS,
        state=State.FOCUSED,
        multimodal_hint=MultimodalHint(),
    )
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(bad_ev,),
    )
    rep = validate_script(s)
    assert any(i.code == "EMPTY_EVENT_ID" for i in rep.issues)
    assert not rep.passed


def test_validate_catches_bad_timestamp() -> None:
    bad_ev = _make_event(idx=0, ts="not-a-date")
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(bad_ev,),
    )
    rep = validate_script(s)
    assert any(i.code == "BAD_TIMESTAMP" for i in rep.issues)


def test_validate_catches_duration_out_of_range() -> None:
    too_short = _make_event(idx=0, duration_s=2.0)
    too_long = _make_event(idx=1, ts="2026-05-06T11:00:00", duration_s=5000.0)
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(too_short, too_long),
    )
    rep = validate_script(s)
    codes = {i.code for i in rep.issues}
    assert "DURATION_OUT_OF_RANGE" in codes


def test_validate_catches_hint_out_of_range() -> None:
    bad_hint = MultimodalHint(eye_closure=1.5, gaze_stability=-0.2)
    bad_ev = _make_event(idx=0, hint=bad_hint)
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(bad_ev,),
    )
    rep = validate_script(s)
    assert sum(1 for i in rep.issues if i.code == "HINT_OUT_OF_RANGE") >= 2


# ---------- 时间连续性 ----------


def test_validate_catches_time_overlap() -> None:
    """ev1 9:00 时长 600s，ev2 9:05（提前 5min）→ 重叠。"""
    ev1 = _make_event(idx=0, ts="2026-05-06T09:00:00", duration_s=600.0)
    ev2 = _make_event(idx=1, ts="2026-05-06T09:05:00", duration_s=300.0)
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev1, ev2),
    )
    rep = validate_script(s)
    assert any(i.code == "TIME_OVERLAP" for i in rep.issues)


def test_validate_warns_large_time_gap() -> None:
    """ev1 9:00 时长 60s 结束 9:01，ev2 9:30 → 间隔 29 分钟。"""
    ev1 = _make_event(idx=0, ts="2026-05-06T09:00:00", duration_s=60.0)
    ev2 = _make_event(idx=1, ts="2026-05-06T09:30:00", duration_s=300.0)
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev1, ev2),
    )
    rep = validate_script(s)
    gap_warns = [i for i in rep.issues if i.code == "LARGE_TIME_GAP"]
    assert len(gap_warns) == 1
    assert gap_warns[0].severity == "warning"


# ---------- 一致性 + 多样性 ----------


def test_validate_catches_persona_mismatch() -> None:
    ev = _make_event(idx=0, persona_id="vboss_002")  # 与 script.persona_id 不符
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev,),
    )
    rep = validate_script(s)
    assert any(i.code == "PERSONA_MISMATCH" for i in rep.issues)


def test_validate_warns_low_state_diversity() -> None:
    """全部 FOCUSED 状态 → 多样性警告。"""
    s = _seq_script(states=[State.FOCUSED] * 5)
    rep = validate_script(s)
    assert any(i.code == "LOW_STATE_DIVERSITY" for i in rep.issues)
    assert rep.n_errors == 0  # 多样性是 warning 不是 error


def test_summary_line_format() -> None:
    s = _seq_script(states=[State.FOCUSED, State.NEUTRAL, State.DISTRACTED])
    rep = validate_script(s)
    line = rep.summary_line()
    assert "vboss_001" in line
    assert "day0" in line
    assert "PASS" in line or "FAIL" in line
