"""v1 parser skeleton for contracts/protocols.py UniversalGrammarParser.

GLiNER multilingual entity extraction fused with LLM postprocess.
"""
from __future__ import annotations

from typing import List, Sequence

from contracts.event_tensor import EventTensor
from contracts.protocols import ParseRequest, UniversalGrammarParser

from .config import ParserConfig
from .postprocess import JsonPostprocessor
from .types import ParseBackendResult, RawSlotCandidate


class GlinerHybridParser(UniversalGrammarParser):
    """GLiNER hybrid parser. v1 实体抽取加 LLM 融合骨架。"""

    def __init__(
        self,
        config: ParserConfig | None = None,
        postprocessor: JsonPostprocessor | None = None,
    ) -> None:
        """Create parser. 注入配置与后处理器，真实模型延后加载。"""
        self.config = config or ParserConfig()
        self.postprocessor = postprocessor or JsonPostprocessor()

    def parse(self, req: ParseRequest) -> EventTensor:
        """Parse one request. 单句解析到 EventTensor。"""
        raise NotImplementedError("M0 W2 todo: extract entities then fuse slots")

    def parse_batch(self, reqs: Sequence[ParseRequest]) -> List[EventTensor]:
        """Parse sequentially. 批量接口保持 Protocol 签名。"""
        raise NotImplementedError("M0 W2 todo: for-loop parse each request")

    def extract_entities(self, text: str) -> tuple[RawSlotCandidate, ...]:
        """Extract entities. GLiNER 中文实体候选抽取。"""
        raise NotImplementedError("M0 W2 todo: run multilingual GLiNER")

    def fuse_with_llm(
        self,
        req: ParseRequest,
        entities: Sequence[RawSlotCandidate],
    ) -> ParseBackendResult:
        """Fuse candidates. 将实体候选与 LLM 槽位判断融合。"""
        raise NotImplementedError("M0 W2 todo: fuse GLiNER and LLM output")
