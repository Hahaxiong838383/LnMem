"""Parser configuration for contracts/protocols.py UniversalGrammarParser.

Defines model, retry, temperature, and fallback order placeholders.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple


class BackendKind(str, Enum):
    """Backend selector. 后端枚举，用于 v0/v1/fallback 编排。"""

    SCHEMA_GUIDED_LLM = "schema_guided_llm"
    GLINER_HYBRID = "gliner_hybrid"


@dataclass(frozen=True)
class ParserConfig:
    """Parser runtime options. 解析器运行配置，不绑定具体 SDK。"""

    llm_model: str = "placeholder-schema-guided-llm"
    gliner_model: str = "urchade/gliner_multi-v2.1"
    temperature: float = 0.0
    max_retries: int = 2
    timeout_seconds: float = 30.0
    fallback_order: Tuple[BackendKind, ...] = field(
        default_factory=lambda: (
            BackendKind.SCHEMA_GUIDED_LLM,
            BackendKind.GLINER_HYBRID,
        )
    )

    def validate(self) -> None:
        """Validate config invariants. 校验配置边界，真实实现阶段补齐。"""
        raise NotImplementedError("M0 W2 todo: validate parser config bounds")
