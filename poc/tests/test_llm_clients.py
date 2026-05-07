"""Day 4 LLM 客户端测试 — 解析/拼装/降级链/HTTP 错误（mock urllib）。"""
from __future__ import annotations

import json
from typing import Any
from unittest.mock import patch

import pytest

from poc.src.data.event_schema import Activity, ScriptEvent, State
from poc.src.data.llm_clients import (
    FallbackLLMClient,
    GeminiProxyLLMClient,
    LLMOutputError,
    MiniMaxLLMClient,
    _coerce_event_dict,
    _events_from_payload,
    _extract_json_object,
    _persona_id_from_brief,
    _strip_thinking,
    _truncate_to_target,
)


# ---------- 文本剥离 + JSON 抠取 ----------


def test_strip_thinking_drops_think_tags() -> None:
    raw = "<think>internal monologue</think>{\"events\":[]}"
    assert _strip_thinking(raw) == "{\"events\":[]}"


def test_strip_thinking_drops_markdown_fences() -> None:
    raw = "```json\n{\"events\":[]}\n```"
    assert _strip_thinking(raw).strip() == "{\"events\":[]}"


def test_extract_json_object_handles_prefix_text() -> None:
    raw = "Here is the JSON:\n{\"events\":[{\"x\":1}]}"
    obj = _extract_json_object(raw)
    assert obj == {"events": [{"x": 1}]}


def test_extract_json_object_raises_on_no_json() -> None:
    with pytest.raises(LLMOutputError):
        _extract_json_object("just text no json")


def test_extract_json_object_handles_nested_braces() -> None:
    raw = '{"events":[{"meta":{"k":"v"}}]}'
    obj = _extract_json_object(raw)
    assert obj == {"events": [{"meta": {"k": "v"}}]}


# ---------- 拼装 event_id + ts ----------


def _good_event() -> dict[str, Any]:
    return {
        "duration_s": 600,
        "activity": "work_focus",
        "state": "focused",
        "description": "写代码",
        "multimodal_hint": {
            "eye_closure": 0.12,
            "head_pitch_norm": 0.08,
            "head_yaw_norm": 0.05,
            "gaze_stability": 0.85,
            "voice_present": 0.0,
            "utterance_density": 0.0,
            "active_app": "VSCode",
            "app_switch_count_5min": 1,
        },
    }


def test_coerce_event_attaches_event_id_and_ts() -> None:
    from datetime import datetime

    ts = datetime(2026, 5, 6, 9, 0, 0)
    out = _coerce_event_dict(_good_event(), "vboss_001", 0, 7, ts)
    assert out["event_id"] == "vboss_001_d0_e007"
    assert out["ts"] == "2026-05-06T09:00:00"
    assert out["persona_id"] == "vboss_001"
    assert out["day_index"] == 0


def test_coerce_event_clamps_duration() -> None:
    from datetime import datetime

    raw = _good_event()
    raw["duration_s"] = 99999  # 超上限
    out = _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))
    assert out["duration_s"] == 3600.0


def test_coerce_event_rejects_unknown_activity() -> None:
    from datetime import datetime

    raw = _good_event()
    raw["activity"] = "weird_activity"
    with pytest.raises(LLMOutputError):
        _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))


def test_coerce_event_rejects_unknown_state() -> None:
    from datetime import datetime

    raw = _good_event()
    raw["state"] = "weird_state"
    with pytest.raises(LLMOutputError):
        _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))


def test_coerce_event_aliases_state_transition() -> None:
    """LLM 偶尔写 'transition' 当 state；alias 应映射到 transitioning."""
    from datetime import datetime

    raw = _good_event()
    raw["state"] = "transition"  # 应映射到 transitioning
    out = _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))
    assert out["state"] == "transitioning"


def test_coerce_event_aliases_activity_meeting() -> None:
    from datetime import datetime

    raw = _good_event()
    raw["activity"] = "meeting"  # alias → work_meeting
    out = _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))
    assert out["activity"] == "work_meeting"


def test_coerce_event_clamps_hint_values() -> None:
    from datetime import datetime

    raw = _good_event()
    raw["multimodal_hint"]["eye_closure"] = 1.5  # 超上限
    raw["multimodal_hint"]["gaze_stability"] = -0.2  # 低于下限
    raw["multimodal_hint"]["app_switch_count_5min"] = -3
    out = _coerce_event_dict(raw, "vboss_001", 0, 0, datetime(2026, 5, 6, 9, 0, 0))
    assert out["multimodal_hint"]["eye_closure"] == 1.0
    assert out["multimodal_hint"]["gaze_stability"] == 0.0
    assert out["multimodal_hint"]["app_switch_count_5min"] == 0


def test_events_from_payload_chains_timestamps() -> None:
    payload = {"events": [_good_event(), _good_event(), _good_event()]}
    events = _events_from_payload(payload, "vboss_001", 0)
    assert len(events) == 3
    # ts 应该按 duration_s 累加
    assert events[0]["ts"] == "2026-05-06T09:00:00"
    assert events[1]["ts"] == "2026-05-06T09:10:00"  # +600s
    assert events[2]["ts"] == "2026-05-06T09:20:00"  # +600s
    # 全部能反序列化为 ScriptEvent
    for e in events:
        ScriptEvent.from_jsonl_dict(e)


def test_events_from_payload_rejects_empty() -> None:
    with pytest.raises(LLMOutputError):
        _events_from_payload({"events": []}, "vboss_001", 0)


def test_events_from_payload_rejects_missing_events_key() -> None:
    with pytest.raises(LLMOutputError):
        _events_from_payload({"foo": []}, "vboss_001", 0)


# ---------- persona_id 抽取 ----------


def test_persona_id_from_brief_extracts_id() -> None:
    brief = """[frontmatter]
id: vboss_001
name: 测试
hexaco:
  honesty_humility: 0.7
"""
    assert _persona_id_from_brief(brief) == "vboss_001"


def test_persona_id_from_brief_raises_when_missing() -> None:
    with pytest.raises(LLMOutputError):
        _persona_id_from_brief("no id field here")


# ---------- MiniMax 端到端（mock HTTP） ----------


_FAKE_MINIMAX_RESPONSE = {
    "choices": [
        {
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "events": [
                            _good_event(),
                            {
                                "duration_s": 300,
                                "activity": "break_short",
                                "state": "neutral",
                                "description": "起身喝水",
                                "multimodal_hint": {
                                    "eye_closure": 0.2,
                                    "head_pitch_norm": 0.1,
                                    "head_yaw_norm": 0.1,
                                    "gaze_stability": 0.5,
                                    "voice_present": 0.0,
                                    "utterance_density": 0.0,
                                    "active_app": "Chrome",
                                    "app_switch_count_5min": 2,
                                },
                            },
                        ]
                    }
                ),
            }
        }
    ],
    "usage": {"prompt_tokens": 100, "completion_tokens": 200},
}


def test_minimax_client_parses_mock_response(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    with patch(
        "poc.src.data.llm_clients._retry_post_json",
        return_value=_FAKE_MINIMAX_RESPONSE,
    ):
        # target_hours=0.25 (900s) 刚好被 2 events (600+300=900) 覆盖，不触发多轮
        client = MiniMaxLLMClient(max_fillin_rounds=0)
        events = client.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=0.25,
            seed=42,
        )
    assert len(events) == 2
    assert events[0]["activity"] == "work_focus"
    assert events[1]["activity"] == "break_short"
    assert events[0]["event_id"].startswith("vboss_001_d0_e")


def test_minimax_client_runs_fillin_when_short(monkeypatch: pytest.MonkeyPatch) -> None:
    """target=2400s (40min) 但 mock 每次只返 900s，应触发 2 轮 fillin."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    call_count = {"n": 0}

    def fake_post(*_a: Any, **_k: Any) -> dict[str, Any]:
        call_count["n"] += 1
        return _FAKE_MINIMAX_RESPONSE  # 每次 900s

    with patch("poc.src.data.llm_clients._retry_post_json", side_effect=fake_post):
        client = MiniMaxLLMClient(max_fillin_rounds=2, min_coverage_ratio=0.95)
        events = client.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=2400 / 3600.0,
            seed=42,
        )
    # 第 1 轮 900s + 第 2 轮 900s + 第 3 轮 900s = 2700s ≥ 2400×0.95 → 应停在第 3 次
    assert call_count["n"] == 3
    assert len(events) == 6  # 2 events × 3 轮
    # event_id 序号应连续 0..5
    expected_ids = [f"vboss_001_d0_e{i:03d}" for i in range(6)]
    assert [e["event_id"] for e in events] == expected_ids


def test_minimax_client_retries_on_empty_content(monkeypatch: pytest.MonkeyPatch) -> None:
    """前 2 次 empty content，第 3 次成功——验证 first_call_retries 生效."""
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")
    call_count = {"n": 0}

    def fake_post(*_a: Any, **_k: Any) -> dict[str, Any]:
        call_count["n"] += 1
        if call_count["n"] < 3:
            return {"choices": [{"message": {"role": "assistant", "content": ""}}]}
        return _FAKE_MINIMAX_RESPONSE

    with patch("poc.src.data.llm_clients._retry_post_json", side_effect=fake_post):
        client = MiniMaxLLMClient(max_fillin_rounds=0, first_call_retries=2)
        events = client.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=0.25,
            seed=42,
        )
    assert len(events) == 2
    assert call_count["n"] == 3  # 2 次空 + 1 次成功


def test_minimax_client_raises_when_all_first_call_retries_fail(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MINIMAX_API_KEY", "sk-test")

    def fake_post(*_a: Any, **_k: Any) -> dict[str, Any]:
        return {"choices": [{"message": {"role": "assistant", "content": ""}}]}

    with patch("poc.src.data.llm_clients._retry_post_json", side_effect=fake_post):
        client = MiniMaxLLMClient(max_fillin_rounds=0, first_call_retries=2)
        with pytest.raises(LLMOutputError):
            client.generate_script(
                persona_brief="[frontmatter]\nid: vboss_001\n",
                day_index=0,
                target_hours=0.25,
                seed=42,
            )


def test_minimax_client_refuses_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MINIMAX_API_KEY", raising=False)
    client = MiniMaxLLMClient()
    with pytest.raises(RuntimeError, match="MINIMAX_API_KEY"):
        client.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=8.0,
            seed=42,
        )


def test_gemini_proxy_client_refuses_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_PROXY_KEY", raising=False)
    client = GeminiProxyLLMClient()
    with pytest.raises(RuntimeError, match="GEMINI_PROXY_KEY"):
        client.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=8.0,
            seed=42,
        )


# ---------- truncate ----------


def _ev(duration: int) -> dict[str, Any]:
    return {
        "event_id": f"e{duration}",
        "ts": "2026-05-06T09:00:00",
        "persona_id": "vboss_001",
        "day_index": 0,
        "duration_s": float(duration),
        "activity": "work_focus",
        "state": "focused",
        "multimodal_hint": _good_event()["multimodal_hint"],
        "description": "",
        "meta": {},
    }


def test_truncate_does_nothing_when_within_bounds() -> None:
    # 5 × 600 = 3000s, target = 3000，原样返回
    events = [_ev(600) for _ in range(5)]
    out = _truncate_to_target(events, 3000)
    assert len(out) == 5


def test_truncate_does_nothing_when_minor_overflow() -> None:
    # 总 4000s, target 3000, 超 1000 < 1200 容差，原样返回
    events = [_ev(800) for _ in range(5)]
    out = _truncate_to_target(events, 3000)
    assert len(out) == 5


def test_truncate_drops_excess_when_clean_cut() -> None:
    # 总 6000s, target 3000；cum 累到 3000 时正好命中 → drop 后续
    events = [_ev(1000) for _ in range(6)]
    out = _truncate_to_target(events, 3000)
    total = sum(int(e["duration_s"]) for e in out)
    assert total <= 3000 + 1200
    assert total >= 3000 - 300
    assert len(out) <= 4


def test_truncate_keeps_clean_cut_when_close_enough() -> None:
    # 总 5400s, target 2000; 前 3 条 1800 已 ≥ target-300=1700 → 干净截断到 3 条
    events = [_ev(600) for _ in range(9)]
    out = _truncate_to_target(events, 2000)
    total = sum(int(e["duration_s"]) for e in out)
    assert total == 1800
    assert len(out) == 3


def test_truncate_shrinks_when_first_event_overshoots() -> None:
    # 第 1 条就 5000s > target 2000 且 cum=0 < target-300 → 裁剪第 1 条到 2000
    events = [_ev(5000), _ev(1000)]
    out = _truncate_to_target(events, 2000)
    assert len(out) == 1
    assert int(out[0]["duration_s"]) == 2000


# ---------- Fallback 链 ----------


class _AlwaysFailClient:
    def __init__(self, name: str) -> None:
        self.name = name

    def generate_script(self, **_: Any) -> list[dict[str, Any]]:
        raise RuntimeError(f"{self.name} failed")


class _AlwaysSucceedClient:
    def __init__(self, n_events: int = 2) -> None:
        self.n_events = n_events

    def generate_script(self, persona_brief: str, day_index: int, **_: Any) -> list[dict[str, Any]]:
        # 返回最少 1 个有效 event
        from datetime import datetime, timedelta

        ts = datetime(2026, 5, 6, 9, 0, 0) + timedelta(days=day_index)
        out = []
        for i in range(self.n_events):
            ev = _coerce_event_dict(_good_event(), "vboss_001", day_index, i, ts)
            out.append(ev)
            ts += timedelta(seconds=600)
        return out


def test_fallback_uses_primary_when_works() -> None:
    primary = _AlwaysSucceedClient(n_events=3)
    fallback = _AlwaysFailClient("fallback")
    chain = FallbackLLMClient(primary=primary, fallback=fallback)
    events = chain.generate_script(
        persona_brief="[frontmatter]\nid: vboss_001\n",
        day_index=0,
        target_hours=8.0,
        seed=0,
    )
    assert len(events) == 3
    assert chain.failures == []  # primary 成功 → 无降级


def test_fallback_falls_through_to_secondary_on_failure() -> None:
    primary = _AlwaysFailClient("primary")
    fallback = _AlwaysSucceedClient(n_events=2)
    chain = FallbackLLMClient(primary=primary, fallback=fallback)
    events = chain.generate_script(
        persona_brief="[frontmatter]\nid: vboss_001\n",
        day_index=0,
        target_hours=8.0,
        seed=0,
    )
    assert len(events) == 2
    # primary 失败 + fallback 成功 → 失败 trace 含 primary 错误 + used:fallback 标记
    assert any("primary failed" in f for f in chain.failures)
    assert any("used:_AlwaysSucceedClient" in f for f in chain.failures)


def test_fallback_falls_through_to_final() -> None:
    primary = _AlwaysFailClient("primary")
    fallback = _AlwaysFailClient("fallback")
    final = _AlwaysSucceedClient(n_events=1)
    chain = FallbackLLMClient(primary=primary, fallback=fallback, final=final)
    events = chain.generate_script(
        persona_brief="[frontmatter]\nid: vboss_001\n",
        day_index=0,
        target_hours=8.0,
        seed=0,
    )
    assert len(events) == 1
    assert any("used:_AlwaysSucceedClient" in f for f in chain.failures)


def test_fallback_raises_when_all_fail() -> None:
    primary = _AlwaysFailClient("primary")
    fallback = _AlwaysFailClient("fallback")
    chain = FallbackLLMClient(primary=primary, fallback=fallback)
    with pytest.raises(RuntimeError, match="fallback failed"):
        chain.generate_script(
            persona_brief="[frontmatter]\nid: vboss_001\n",
            day_index=0,
            target_hours=8.0,
            seed=0,
        )
