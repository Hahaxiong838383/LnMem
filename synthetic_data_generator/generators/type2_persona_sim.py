"""Type 2 persona simulator for contracts/protocols.py::SyntheticDataGenerator.

Models 50 virtual owners x 30 days as a future backend skeleton.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec

from synthetic_data_generator.config import SyntheticGeneratorConfig


@dataclass(frozen=True)
class VirtualPersona:
    """Virtual owner profile. 虚拟主人的稳定画像字段。"""

    persona_id: str
    display_name: str
    age_band: str
    work_style: str
    relationship_focus: str


@dataclass(frozen=True)
class PersonaDayPlan:
    """One persona day plan. 单个虚拟主人一天的行为计划。"""

    persona_id: str
    day_index: int
    scene_tags: Sequence[str]


class Type2PersonaSimulator:
    """Simulate persona timelines. 生成多天多轮中文行为场景。"""

    def __init__(self, config: Optional[SyntheticGeneratorConfig] = None) -> None:
        """Create a type_2 backend. 创建 persona simulation 后端骨架。"""
        self.config = config or SyntheticGeneratorConfig()

    def build_personas(self, spec: SynthSpec) -> Sequence[VirtualPersona]:
        """Build virtual persona pool. 未来生成 50 个虚拟主人。"""
        raise NotImplementedError("M0 W2 todo: build deterministic virtual persona pool")

    def generate(self, spec: SynthSpec) -> Iterable[EventTensor]:
        """Generate type_2 EventTensor samples. 未来模拟 30 天行为日志。"""
        raise NotImplementedError("M0 W2 todo: generate type_2 persona simulation samples")
