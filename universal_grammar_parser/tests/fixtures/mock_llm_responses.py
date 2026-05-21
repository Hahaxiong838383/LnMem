"""Pre-recorded mock responses for parser tests. 模拟 LLM JSON。"""
from __future__ import annotations

import json
from typing import Iterable

from .parser_test_sentences import SENTENCES

P0_SLOT_NAMES = ("who_actor", "what_action", "when_time", "where_location")

_BY_TEXT = {sample["text"]: sample for sample in SENTENCES}


def build_mock_response(text: str, missing_slots: Iterable[str] = ()) -> str:
    """Build one JSON payload for a sample text. 可按需删掉部分 P0 槽位。"""
    sample = _BY_TEXT[text]
    missing = set(missing_slots)
    slots: dict[str, dict[str, object]] = {}
    for slot_name in P0_SLOT_NAMES:
        if slot_name in missing:
            continue
        value = sample["expected_p0"][slot_name]
        span = _find_span(text, value)
        slot_payload: dict[str, object] = {
            "value": value,
            "confidence": 0.92,
            "raw_text": value,
        }
        if span is not None:
            slot_payload["span"] = list(span)
        slots[slot_name] = slot_payload
    return json.dumps({"slots": slots}, ensure_ascii=False)


def _find_span(text: str, raw_text: str) -> tuple[int, int] | None:
    """Return [start, end) when raw_text exists in source. 找不到则返回 None。"""
    start = text.find(raw_text)
    if start < 0:
        return None
    return start, start + len(raw_text)
