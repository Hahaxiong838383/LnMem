"""Type 3 real-log injector for contracts/protocols.py::SyntheticDataGenerator.

Real log parsing and privacy controls are deferred to M0 W3-W4.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from contracts.event_tensor import EventTensor
from contracts.protocols import SynthSpec

from synthetic_data_generator.config import SyntheticGeneratorConfig
from synthetic_data_generator.monitor import BossShareMonitor


@dataclass(frozen=True)
class RealLogSource:
    """Real log source descriptor. 真实日志来源的最小描述。"""

    path: str
    owner_label: str
    requires_desensitization: bool = True


class Type3RealLogInjector:
    """Inject real logs under share caps. 在占比红线内注入真实日志。"""

    def __init__(
        self,
        config: Optional[SyntheticGeneratorConfig] = None,
        monitor: Optional[BossShareMonitor] = None,
    ) -> None:
        """Create a type_3 backend. 创建真实日志注入后端骨架。"""
        self.config = config or SyntheticGeneratorConfig()
        self.monitor = monitor or BossShareMonitor()

    def resolve_source(self, spec: SynthSpec) -> RealLogSource:
        """Resolve SynthSpec.real_log_path. 未来解析真实日志路径与来源。"""
        raise NotImplementedError("M0 W2 todo: resolve and validate real_log_path")

    def generate(self, spec: SynthSpec) -> Iterable[EventTensor]:
        """Generate type_3 EventTensor samples. 未来执行脱敏和占比检查。"""
        raise NotImplementedError("M0 W2 todo: generate type_3 real-log samples")
