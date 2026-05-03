"""Collect-only placeholder cases for M0 W2 parser implementation."""
from __future__ import annotations

import pytest


def test_parse_p0_slots_from_simple_chinese_sentence() -> None:
    """P0 slots should cover who/what/when/where for simple Chinese text."""
    pytest.skip("M0 W2 implementation pending: simple P0 slot extraction")


def test_reject_malformed_llm_json_without_hard_parse() -> None:
    """Malformed LLM JSON should be rejected before EventTensor mapping."""
    pytest.skip("M0 W2 implementation pending: robust JSON validation")


def test_validate_span_offsets_for_chinese_entities() -> None:
    """Chinese entity spans should align to Python string offsets."""
    pytest.skip("M0 W2 implementation pending: span offset validation")


def test_require_priorities_triggers_fallback_policy() -> None:
    """Missing required priorities should use the configured fallback order."""
    pytest.skip("M0 W2 implementation pending: priority fallback policy")
