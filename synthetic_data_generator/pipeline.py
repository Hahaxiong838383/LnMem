"""Pipeline skeleton for contracts/protocols.py::SyntheticDataGenerator.

Routes SynthSpec.kind to type_1/type_2/type_3 backends.
"""
from __future__ import annotations

from typing import Iterable, Optional

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec

from synthetic_data_generator.config import LLMFetchImpl, SynthKind, SyntheticGeneratorConfig
from synthetic_data_generator.generators.type1_llm import Type1LLMGenerator
from synthetic_data_generator.generators.type2_persona_sim import Type2PersonaSimulator
from synthetic_data_generator.generators.type3_real_log import Type3RealLogInjector
from synthetic_data_generator.monitor import BossShareMonitor


class LnMemSyntheticDataGenerator:
    """SyntheticDataGenerator implementation. 类型 1/2/3 合成数据统一入口。"""

    def __init__(
        self,
        config: Optional[SyntheticGeneratorConfig] = None,
        fetch_impl: Optional[LLMFetchImpl] = None,
        monitor: Optional[BossShareMonitor] = None,
        type1: Optional[Type1LLMGenerator] = None,
        type2: Optional[Type2PersonaSimulator] = None,
        type3: Optional[Type3RealLogInjector] = None,
    ) -> None:
        """Create the pipeline. 创建可注入 backend 的 pipeline 骨架。"""
        self.config = config or SyntheticGeneratorConfig(fetch_impl=fetch_impl)
        self.monitor = monitor or BossShareMonitor()
        self.type1 = type1 or Type1LLMGenerator(self.config, fetch_impl)
        self.type2 = type2 or Type2PersonaSimulator(self.config)
        self.type3 = type3 or Type3RealLogInjector(self.config, self.monitor)

    def generate(self, spec: SynthSpec) -> Iterable[EventTensor]:
        """Route generation by SynthSpec.kind. 根据 kind 路由到三类后端。"""
        if spec.kind == SynthKind.TYPE_1:
            return self.type1.generate(spec)
        if spec.kind == SynthKind.TYPE_2:
            return self.type2.generate(spec)
        if spec.kind == SynthKind.TYPE_3:
            return self.type3.generate(spec)
        raise NotImplementedError(f"M0 W2 todo: unsupported SynthSpec.kind {spec.kind!r}")

    def stats(self) -> dict:
        """Return cumulative generator stats. 返回当前累计占比统计。"""
        raise NotImplementedError("M0 W2 todo: aggregate generator and monitor stats")
