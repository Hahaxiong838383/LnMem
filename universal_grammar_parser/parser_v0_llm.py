"""v0 parser skeleton for contracts/protocols.py UniversalGrammarParser.

Schema-guided LLM prompt plus strict JSON postprocess, no real parsing yet.
"""
from __future__ import annotations

from typing import List, Sequence

from contracts.event_tensor import EventTensor
from contracts.protocols import ParseRequest, UniversalGrammarParser

from .config import ParserConfig
from .postprocess import JsonPostprocessor
from .types import ParseBackendResult


class SchemaGuidedLLMParser(UniversalGrammarParser):
    """Schema-guided LLM parser. v0 LLM+JSON 后处理骨架。"""

    def __init__(
        self,
        config: ParserConfig | None = None,
        postprocessor: JsonPostprocessor | None = None,
    ) -> None:
        """Create parser. 注入配置与后处理器，避免绑定 SDK。"""
        self.config = config or ParserConfig()
        self.postprocessor = postprocessor or JsonPostprocessor()

    def parse(self, req: ParseRequest) -> EventTensor:
        """Parse one request. 单句解析到 EventTensor。"""
        raise NotImplementedError("M0 W2 todo: call schema-guided LLM parser")

    def parse_batch(self, reqs: Sequence[ParseRequest]) -> List[EventTensor]:
        """Parse sequentially. 批量接口保持 Protocol 签名。"""
        raise NotImplementedError("M0 W2 todo: for-loop parse each request")

    def build_prompt(self, req: ParseRequest) -> str:
        """Build schema prompt. 生成 schema_v1 引导 prompt。"""
        raise NotImplementedError("M0 W2 todo: construct schema-guided prompt")

    def call_backend(self, prompt: str) -> ParseBackendResult:
        """Call LLM backend. 调用 LLM 并返回未信任 payload。"""
        raise NotImplementedError("M0 W2 todo: call LLM backend safely")
