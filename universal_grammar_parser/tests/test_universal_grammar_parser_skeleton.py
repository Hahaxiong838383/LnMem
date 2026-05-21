"""Behavior tests for SchemaGuidedLLMParser. 模块内 P0/诊断/span 验证。"""
from __future__ import annotations

from datetime import datetime, timezone

from contracts.protocols import ParseRequest
from contracts.schema_v1 import SlotID
from universal_grammar_parser import SchemaGuidedLLMParser


def _req(text: str) -> ParseRequest:
    """Build parser request. 构造最小 ParseRequest。"""
    return ParseRequest(
        text=text,
        user_id="u-parser-test",
        timestamp=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
    )


def test_parse_p0_slots_from_simple_chinese_sentence() -> None:
    """P0 slots should cover who/what/when/where for simple Chinese text."""
    parser = SchemaGuidedLLMParser(
        llm_client=lambda prompt: (
            '{"slots":{"who_actor":{"value":"陈川","confidence":0.95,"raw_text":"陈川"},'
            '"what_action":{"value":"吃饭","confidence":0.93,"raw_text":"吃饭"},'
            '"when_time":{"value":"昨天","confidence":0.88,"raw_text":"昨天"},'
            '"where_location":{"value":"家","confidence":0.9,"raw_text":"家"}}}'
        )
    )
    event = parser.parse(_req("昨天陈川在家吃饭。"))
    assert event.get(SlotID.WHO_ACTOR).value == "陈川"
    assert event.get(SlotID.WHAT_ACTION).value == "吃饭"
    assert event.get(SlotID.WHEN_TIME).value == "昨天"
    assert event.get(SlotID.WHERE_LOCATION).value == "家"
    assert event.filled_priority_count()["P0"] == 4


def test_reject_malformed_llm_json_without_hard_parse() -> None:
    """Malformed LLM JSON should be rejected before EventTensor mapping."""
    parser = SchemaGuidedLLMParser(
        llm_client=lambda prompt: "这不是 JSON",
        sleep_fn=lambda seconds: None,
    )
    event = parser.parse(_req("昨晚我和小李吃饭。"))
    assert event.filled_priority_count()["P0"] == 0
    assert "parse_error" in parser.last_diagnostics


def test_validate_span_offsets_for_chinese_entities() -> None:
    """Chinese entity spans should align to Python string offsets."""
    text = "昨晚陈川在家吃饭。"
    parser = SchemaGuidedLLMParser(
        llm_client=lambda prompt: (
            '{"slots":{"who_actor":{"value":"陈川","confidence":0.95,"raw_text":"陈川","span":[2,4]},'
            '"where_location":{"value":"家","confidence":0.9,"raw_text":"家","span":[5,6]}}}'
        )
    )
    event = parser.parse(_req(text))
    who_span = event.get(SlotID.WHO_ACTOR).span
    where_span = event.get(SlotID.WHERE_LOCATION).span
    assert text[who_span[0]:who_span[1]] == "陈川"
    assert text[where_span[0]:where_span[1]] == "家"


def test_require_priorities_triggers_fallback_policy() -> None:
    """Missing required priorities should use the configured fallback order."""
    parser = SchemaGuidedLLMParser(
        llm_client=lambda prompt: (
            '{"slots":{"who_actor":{"value":"我","confidence":0.95,"raw_text":"我"},'
            '"what_action":{"value":"开会","confidence":0.91,"raw_text":"开会"}}}'
        )
    )
    event = parser.parse(_req("我明天在会议室开会。"))
    assert event.filled_priority_count()["P0"] < 4
    assert parser.last_diagnostics["missing_required_slots"]["P0"] == [
        "when_time",
        "where_location",
    ]
