"""Batch wrapper for contracts/protocols.py UniversalGrammarParser.

Sequential for-loop orchestration only; no concurrency in the skeleton.
"""
from __future__ import annotations

from typing import List, Sequence

from contracts.event_tensor import EventTensor
from contracts.protocols import ParseRequest, UniversalGrammarParser


class SequentialBatchParser(UniversalGrammarParser):
    """Sequential batch adapter. 批量调用单句 parser 的骨架。"""

    def __init__(self, single_parser: UniversalGrammarParser) -> None:
        """Create adapter. 注入一个单句 parser，不跨模块依赖。"""
        self.single_parser = single_parser

    def parse(self, req: ParseRequest) -> EventTensor:
        """Parse one request. 透传单句解析，真实实现阶段补齐。"""
        raise NotImplementedError("M0 W2 todo: delegate to single parser")

    def parse_batch(self, reqs: Sequence[ParseRequest]) -> List[EventTensor]:
        """Parse requests in order. 内部 for-loop 串行调用 parse。"""
        raise NotImplementedError("M0 W2 todo: sequential for-loop parse")
