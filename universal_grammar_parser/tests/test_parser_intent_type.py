"""Intent type extraction tests for SchemaGuidedLLMParser."""
from __future__ import annotations

from datetime import datetime, timezone

from contracts.protocols import ParseRequest
from universal_grammar_parser import SchemaGuidedLLMParser


def _req(text: str) -> ParseRequest:
    """Build parse request. 构造最小 ParseRequest。"""
    return ParseRequest(
        text=text,
        user_id="u-intent-type-test",
        timestamp=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
    )


def _mock_payload(intent_type: str) -> str:
    """Build mock LLM payload. 仅返回 intent_type + 空 slots。"""
    return f'{{"intent_type":"{intent_type}","slots":{{}}}}'


def _intent(event: object) -> str:
    """Read intent type from EventTensor. 兼容动态字段注入。"""
    return getattr(event, "intent_type")


def test_intent_declarative_yesterday_ate() -> None:
    """Declarative text should map to 陈述."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("陈述"))
    event = parser.parse(_req("昨晚吃了越南粉。"))
    assert _intent(event) == "陈述"


def test_intent_question_what_time() -> None:
    """Question text should map to 疑问."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("疑问"))
    event = parser.parse(_req("今天几号？"))
    assert _intent(event) == "疑问"


def test_intent_rhetorical_can_blame() -> None:
    """Rhetorical text should map to 反问."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("反问"))
    event = parser.parse(_req("这能怪我吗？"))
    assert _intent(event) == "反问"


def test_intent_exclamation_too_tired() -> None:
    """Exclamation text should map to 感叹."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("感叹"))
    event = parser.parse(_req("太累了！"))
    assert _intent(event) == "感叹"


def test_intent_request_book_ticket() -> None:
    """Request text should map to 请求."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("请求"))
    event = parser.parse(_req("帮我订一下票。"))
    assert _intent(event) == "请求"


def test_intent_fallback_unknown_defaults_to_declarative() -> None:
    """Unknown intent_type should fallback to 陈述."""
    parser = SchemaGuidedLLMParser(llm_client=lambda prompt: _mock_payload("未知"))
    event = parser.parse(_req("这句让模型没识别。"))
    assert _intent(event) == "陈述"

