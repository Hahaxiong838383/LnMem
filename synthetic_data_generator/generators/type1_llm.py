"""Type 1 LLM backend for contracts/protocols.py::SyntheticDataGenerator.

LLM client is injected through fetch_impl; no provider import here.
"""
from __future__ import annotations

from typing import Iterable, Optional

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec
from contracts.schema_v1 import SCHEMA_V1, SlotID

from synthetic_data_generator.config import LLMFetchImpl, SyntheticGeneratorConfig


class Type1LLMGenerator:
    """Generate schema-guided LLM samples. 生成万能语法槽位填充样本。"""

    def __init__(
        self,
        config: Optional[SyntheticGeneratorConfig] = None,
        fetch_impl: Optional[LLMFetchImpl] = None,
    ) -> None:
        """Create a type_1 backend. 创建 LLM 自动生成后端骨架。"""
        self.config = config or SyntheticGeneratorConfig()
        self.fetch_impl = fetch_impl or self.config.fetch_impl
        self.schema = SCHEMA_V1

    def required_slots(self) -> tuple[SlotID, ...]:
        """Return slots targeted by W2 prompts. 返回未来 prompt 必须覆盖的槽位。"""
        raise NotImplementedError("M0 W2 todo: derive required slots from schema priority")

    def generate(self, spec: SynthSpec) -> Iterable[EventTensor]:
        """Generate type_1 EventTensor samples. 未来调用注入的 LLM fetch_impl。"""
        raise NotImplementedError("M0 W2 todo: generate type_1 LLM samples")
