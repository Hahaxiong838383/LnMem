"""Shared skeleton config for contracts/protocols.py::SyntheticDataGenerator.

Defines control-plane types only; no real generation lives here.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional


class SynthKind(str, Enum):
    """Supported SynthSpec.kind values. 支持的三类合成数据来源。"""

    TYPE_1 = "type_1"
    TYPE_2 = "type_2"
    TYPE_3 = "type_3"


LLMFetchImpl = Callable[[Mapping[str, Any]], Mapping[str, Any]]


@dataclass(frozen=True)
class SyntheticGeneratorConfig:
    """Pipeline knobs for future generators. 后续真实现共用的控制参数。"""

    schema_version: str = "v1"
    batch_size: int = 16
    strict_schema: bool = True
    default_user_prefix: str = "synthetic_user"
    fetch_impl: Optional[LLMFetchImpl] = None


@dataclass(frozen=True)
class GenerationPlan:
    """Resolved generation plan. 未来把 SynthSpec 归一化后的内部计划。"""

    kind: SynthKind
    n_samples: int
    seed: int
    boss_share_cap: float
