"""Day 3 Lane A 测试 — event_schema 序列化与枚举（v0.1）。"""
from __future__ import annotations

import pytest

from poc.src.data.event_schema import (
    Activity,
    MultimodalHint,
    Script,
    ScriptEvent,
    State,
)


def _sample_event(event_id: str = "vboss_001_d0_e000") -> ScriptEvent:
    return ScriptEvent(
        event_id=event_id,
        ts="2026-05-06T09:00:00",
        persona_id="vboss_001",
        day_index=0,
        duration_s=600.0,
        activity=Activity.WORK_FOCUS,
        state=State.FOCUSED,
        multimodal_hint=MultimodalHint(
            eye_closure=0.18,
            gaze_stability=0.85,
            voice_present=0.0,
            active_app="VSCode",
            app_switch_count_5min=1,
        ),
        description="陈墨在 VSCode 写文档",
    )


# ---------- ScriptEvent 序列化 ----------


def test_script_event_to_jsonl_dict_has_required_fields() -> None:
    ev = _sample_event()
    d = ev.to_jsonl_dict()
    for k in (
        "event_id",
        "ts",
        "persona_id",
        "day_index",
        "duration_s",
        "activity",
        "state",
        "multimodal_hint",
        "description",
        "meta",
    ):
        assert k in d, f"missing field: {k}"


def test_script_event_enum_serialised_as_str() -> None:
    ev = _sample_event()
    d = ev.to_jsonl_dict()
    assert d["activity"] == "work_focus"
    assert d["state"] == "focused"


def test_script_event_round_trip() -> None:
    ev1 = _sample_event()
    ev2 = ScriptEvent.from_jsonl_dict(ev1.to_jsonl_dict())
    assert ev1 == ev2


def test_script_event_frozen_immutable() -> None:
    ev = _sample_event()
    with pytest.raises(Exception):  # FrozenInstanceError or TypeError
        ev.duration_s = 999.0  # type: ignore[misc]


# ---------- MultimodalHint 序列化 ----------


def test_multimodal_hint_round_trip_preserves_values() -> None:
    h1 = MultimodalHint(
        eye_closure=0.42,
        head_pitch_norm=0.3,
        gaze_stability=0.6,
        voice_present=1.0,
        utterance_density=0.55,
        active_app="Chrome",
        app_switch_count_5min=12,
    )
    h2 = MultimodalHint.from_dict(h1.to_dict())
    assert h1 == h2


def test_multimodal_hint_default_values() -> None:
    h = MultimodalHint()
    assert h.eye_closure == 0.15
    assert h.gaze_stability == 0.7
    assert h.voice_present == 0.0
    assert h.active_app == ""


# ---------- Script 容器 ----------


def test_script_to_json_round_trip() -> None:
    ev1 = _sample_event("vboss_001_d0_e000")
    ev2 = ScriptEvent(
        event_id="vboss_001_d0_e001",
        ts="2026-05-06T09:10:00",
        persona_id="vboss_001",
        day_index=0,
        duration_s=300.0,
        activity=Activity.BREAK_SHORT,
        state=State.NEUTRAL,
        multimodal_hint=MultimodalHint(),
    )
    s1 = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev1, ev2),
    )
    s2 = Script.from_json(s1.to_json())
    assert s1 == s2
    assert len(s2.events) == 2


def test_script_events_is_tuple() -> None:
    s = Script(
        persona_id="vboss_001",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(_sample_event(),),
    )
    assert isinstance(s.events, tuple)


def test_state_enum_8_classes() -> None:
    """state 必须是 LNN 训练目标的 8 类（与 cognitive.md 对齐）。"""
    expected = {
        "low_energy",
        "focused",
        "distracted",
        "fatigued",
        "emotional_positive",
        "emotional_negative",
        "neutral",
        "transitioning",
    }
    actual = {s.value for s in State}
    assert actual == expected


def test_activity_enum_8_classes() -> None:
    expected = {
        "work_focus",
        "work_meeting",
        "break_short",
        "break_long",
        "commute",
        "leisure",
        "sleep",
        "transition",
    }
    actual = {a.value for a in Activity}
    assert actual == expected
