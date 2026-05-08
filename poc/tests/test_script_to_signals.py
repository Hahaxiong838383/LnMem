"""Day 3 Lane C 测试 — script_to_signals 转换器（v0.1）。"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from poc.src.data.event_schema import (
    Activity,
    MultimodalHint,
    Script,
    ScriptEvent,
    State,
)
from poc.src.fusion import FEATURE_DIM
from poc.src.script_to_signals import (
    SignalFrame,
    script_to_signals,
    write_signals_jsonl,
    script_file_to_signals_file,
)


# ---------- 辅助 ----------


def _make_script(
    n_events: int = 3,
    duration_s: float = 10.0,
    persona_id: str = "vboss_001",
) -> Script:
    states = [State.FOCUSED, State.NEUTRAL, State.LOW_ENERGY, State.DISTRACTED]
    events: list[ScriptEvent] = []
    base_minute = 0
    for i in range(n_events):
        ts = f"2026-05-06T09:{base_minute:02d}:00"
        events.append(
            ScriptEvent(
                event_id=f"{persona_id}_d0_e{i:03d}",
                ts=ts,
                persona_id=persona_id,
                day_index=0,
                duration_s=duration_s,
                activity=Activity.WORK_FOCUS,
                state=states[i % len(states)],
                multimodal_hint=MultimodalHint(
                    eye_closure=0.2 + 0.1 * i,
                    gaze_stability=0.85,
                    voice_present=0.0,
                    active_app="VSCode",
                    app_switch_count_5min=2,
                ),
            )
        )
        base_minute += 5
    return Script(
        persona_id=persona_id,
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=tuple(events),
    )


# ---------- 向量维度 + 范围 ----------


def test_vector_dim_is_18() -> None:
    """Day 6: 12→18 维（+6 AU 表情特征）。"""
    s = _make_script(n_events=2, duration_s=5.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    assert all(len(f.vector) == FEATURE_DIM for f in frames)
    assert FEATURE_DIM == 18


def test_vector_values_in_unit_interval_with_noise() -> None:
    """高斯扰动后必须仍在 [0,1]（clamp 生效）。"""
    s = _make_script(n_events=3, duration_s=20.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.5, seed=42))
    for f in frames:
        for v in f.vector:
            assert 0.0 <= v <= 1.0, f"vector value {v} out of [0,1]"


def test_vector_values_in_unit_interval_without_noise() -> None:
    s = _make_script(n_events=2, duration_s=10.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    for f in frames:
        for v in f.vector:
            assert 0.0 <= v <= 1.0


# ---------- 帧数 + 时序 ----------


def test_frame_count_matches_total_duration() -> None:
    """3 events × 10s × 1Hz = 30 frames."""
    s = _make_script(n_events=3, duration_s=10.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    assert len(frames) == 30


def test_frame_idx_monotonic() -> None:
    s = _make_script(n_events=4, duration_s=5.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    indices = [f.frame_idx for f in frames]
    assert indices == sorted(indices)
    assert indices == list(range(len(frames)))


def test_frame_rate_2hz_doubles_frame_count() -> None:
    s = _make_script(n_events=2, duration_s=10.0)
    frames_1hz = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    frames_2hz = list(script_to_signals(s, frame_rate_hz=2.0, noise_sigma=0.0))
    assert len(frames_2hz) == 2 * len(frames_1hz)


# ---------- 可重现性 ----------


def test_same_seed_yields_identical_vectors() -> None:
    s = _make_script(n_events=3, duration_s=10.0)
    frames_a = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.1, seed=7))
    frames_b = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.1, seed=7))
    assert [f.vector for f in frames_a] == [f.vector for f in frames_b]


def test_different_seeds_differ() -> None:
    s = _make_script(n_events=3, duration_s=10.0)
    frames_a = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.1, seed=7))
    frames_b = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.1, seed=8))
    assert [f.vector for f in frames_a] != [f.vector for f in frames_b]


# ---------- ground truth + 溯源 ----------


def test_ground_truth_state_propagated() -> None:
    s = _make_script(n_events=4, duration_s=5.0)
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    states_seen = {f.ground_truth_state for f in frames}
    assert "focused" in states_seen
    assert "neutral" in states_seen


def test_source_event_id_traceable() -> None:
    s = _make_script(n_events=3, duration_s=5.0, persona_id="vboss_007")
    frames = list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    src_ids = {f.source_event_id for f in frames}
    assert "vboss_007_d0_e000" in src_ids
    assert "vboss_007_d0_e002" in src_ids
    for f in frames:
        assert f.persona_id == "vboss_007"


# ---------- JSONL 写入 + 端到端 ----------


def test_write_signals_jsonl_writes_one_line_per_frame() -> None:
    s = _make_script(n_events=2, duration_s=5.0)
    frames = script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0)
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "signals.jsonl"
        n = write_signals_jsonl(frames, out)
        lines = out.read_text(encoding="utf-8").splitlines()
        assert len(lines) == n == 10
        first = json.loads(lines[0])
        assert "vector" in first
        assert len(first["vector"]) == 18  # Day 6: 12→18
        assert "ground_truth_state" in first


def test_script_file_to_signals_file_roundtrip() -> None:
    s = _make_script(n_events=2, duration_s=5.0)
    with tempfile.TemporaryDirectory() as td:
        script_path = Path(td) / "script.json"
        script_path.write_text(s.to_json(), encoding="utf-8")
        out = Path(td) / "out" / "signals.jsonl"
        n = script_file_to_signals_file(script_path, out, frame_rate_hz=1.0, noise_sigma=0.0)
        assert n == 10
        assert out.exists()


# ---------- 错误输入 ----------


def test_voice_present_zero_outputs_zero() -> None:
    """codex F1-04 review fix: voice_present=0 必须显式输出 0.0，不能 fallback 0.5。"""
    h = MultimodalHint(
        voice_present=0.0,
        utterance_density=0.9,
        active_app="VSCode",
        gaze_stability=0.85,
    )
    ev = ScriptEvent(
        event_id="e1",
        ts="2026-05-06T09:00:00",
        persona_id="p",
        day_index=0,
        duration_s=5.0,
        activity=Activity.WORK_FOCUS,
        state=State.FOCUSED,
        multimodal_hint=h,
    )
    s = Script(
        persona_id="p",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev,),
    )
    frame = next(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    # vector index 14 = voice_present（Day 6: 8→14，表情 6 维插在中间）
    assert frame.vector[14] == 0.0
    # missing 应不含 audio（因为我们显式构造了 audio dict）
    assert "audio" not in frame.missing


def test_voice_present_one_outputs_one() -> None:
    """对照测试：voice_present=1 时仍正确输出 1.0。"""
    h = MultimodalHint(voice_present=1.0, utterance_density=0.5, active_app="Slack")
    ev = ScriptEvent(
        event_id="e2",
        ts="2026-05-06T09:00:00",
        persona_id="p",
        day_index=0,
        duration_s=5.0,
        activity=Activity.WORK_MEETING,
        state=State.FOCUSED,
        multimodal_hint=h,
    )
    s = Script(
        persona_id="p",
        day_index=0,
        generated_at="2026-05-06T08:00:00",
        events=(ev,),
    )
    frame = next(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=0.0))
    # Day 6: voice_present 从 index 8 → 14（表情 6 维插在中间）
    assert frame.vector[14] == 1.0


def test_invalid_frame_rate_raises() -> None:
    s = _make_script(n_events=1, duration_s=5.0)
    with pytest.raises(ValueError):
        list(script_to_signals(s, frame_rate_hz=0.0, noise_sigma=0.0))


def test_negative_noise_raises() -> None:
    s = _make_script(n_events=1, duration_s=5.0)
    with pytest.raises(ValueError):
        list(script_to_signals(s, frame_rate_hz=1.0, noise_sigma=-0.1))
