"""Real and benchmark-style tests for parser + Gemini proxy client."""
from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest

from contracts.protocols import ParseRequest
from universal_grammar_parser import GeminiProxyClient, ParserConfig, SchemaGuidedLLMParser
from universal_grammar_parser.tests.fixtures.mock_llm_responses import build_mock_response
from universal_grammar_parser.tests.fixtures.parser_test_sentences import SENTENCES


def _req(text: str) -> ParseRequest:
    """Build parser request. 统一测试请求。"""
    return ParseRequest(
        text=text,
        user_id="u-real-test",
        timestamp=datetime(2026, 5, 21, 10, 0, tzinfo=timezone.utc),
    )


def test_gemini_proxy_client_smoke() -> None:
    """Real smoke, gated by env var. 默认跳过，不进 CI。"""
    if not os.getenv("LNMEM_PARSER_REAL_LLM_TEST"):
        pytest.skip("set LNMEM_PARSER_REAL_LLM_TEST=1 to enable real Gemini proxy smoke")
    if not os.getenv("GEMINI_PROXY_TOKEN"):
        pytest.skip("GEMINI_PROXY_TOKEN is required for real Gemini proxy smoke")
    client = GeminiProxyClient()
    parser = SchemaGuidedLLMParser(
        llm_client=client,
        config=ParserConfig(llm_model=client.model),
    )
    event = parser.parse(_req("昨晚和小李在公司吃饭。"))
    assert event.slots
    assert event.filled_priority_count()["P0"] >= 1
    assert not parser.last_diagnostics.get("exhausted")


def test_parse_10_sentences_p0_fill_rate() -> None:
    """10 句中文样本的 mock P0 平均填充率应达到 80%。"""
    partial_cases = {
        SENTENCES[3]["text"]: ("where_location",),
        SENTENCES[7]["text"]: ("when_time",),
    }

    def mock_llm(prompt: str) -> str:
        for sample in SENTENCES:
            if sample["text"] in prompt:
                return build_mock_response(
                    sample["text"],
                    missing_slots=partial_cases.get(sample["text"], ()),
                )
        raise AssertionError("prompt did not include any known test sentence")

    parser = SchemaGuidedLLMParser(llm_client=mock_llm)
    total_present = 0
    for sample in SENTENCES:
        total_present += parser.parse(_req(sample["text"])).filled_priority_count()["P0"]
    fill_rate = total_present / (len(SENTENCES) * 4)
    assert fill_rate >= 0.8
