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
        """Validate config invariants. 校验配置边界。"""
        if not self.llm_model or not isinstance(self.llm_model, str):
            raise ValueError(f"llm_model must be non-empty str, got {self.llm_model!r}")
        if not self.gliner_model or not isinstance(self.gliner_model, str):
            raise ValueError(f"gliner_model must be non-empty str")
        if not (0.0 <= self.temperature <= 2.0):
            raise ValueError(f"temperature must be in [0, 2], got {self.temperature}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be ≥ 0, got {self.max_retries}")
        if self.timeout_seconds <= 0:
            raise ValueError(f"timeout_seconds must be > 0, got {self.timeout_seconds}")
        if not self.fallback_order:
            raise ValueError("fallback_order must contain at least one BackendKind")
        seen: set[BackendKind] = set()
        for k in self.fallback_order:
            if not isinstance(k, BackendKind):
                raise ValueError(f"fallback_order items must be BackendKind, got {type(k).__name__}")
            if k in seen:
                raise ValueError(f"fallback_order has duplicate {k}")
            seen.add(k)
