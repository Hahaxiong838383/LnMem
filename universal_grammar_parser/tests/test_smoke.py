"""Smoke tests for UniversalGrammarParser skeleton collection."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Protocol, Sequence, runtime_checkable

from contracts.event_tensor import EventTensor, SlotValue
from contracts.protocols import ParseRequest, UniversalGrammarParser
from contracts.schema_v1 import SCHEMA_V1, SlotID
from universal_grammar_parser import SchemaGuidedLLMParser


@runtime_checkable
class RuntimeUniversalGrammarParser(Protocol):
    """Runtime-checkable mirror. 仅用于骨架鸭子类型验证。"""

    def parse(self, req: ParseRequest) -> EventTensor:
        """Parse one request."""

    def parse_batch(self, reqs: Sequence[ParseRequest]) -> List[EventTensor]:
        """Parse request sequence."""


def test_contracts_importable() -> None:
    """contracts.* imports expose parser inputs and EventTensor slots."""
    req = ParseRequest(
        text="昨天我在办公室和小王讨论了 LnMem。",
        user_id="u-smoke",
        timestamp=datetime.now(timezone.utc),
    )
    slot = SlotValue(value="小王", confidence=0.9, span=(7, 9), raw_text="小王")
    event = EventTensor(user_id=req.user_id, timestamp=req.timestamp)
    assert req.require_priorities == (0,)
    assert SlotID.WHO_ACTOR in SCHEMA_V1
    assert slot.confidence == 0.9
    assert event.schema_version == "v1"


def test_parser_signature_compatible_with_protocol() -> None:
    """Skeleton parser has the UniversalGrammarParser method surface."""
    parser = SchemaGuidedLLMParser()
    typed_parser: UniversalGrammarParser = parser
    assert isinstance(parser, RuntimeUniversalGrammarParser)
    assert callable(typed_parser.parse)
    assert callable(typed_parser.parse_batch)
