"""EvalSuite skeleton for contracts/protocols.py EvalSuite.

Runner 只固定入口签名；真实执行在 M0 W2-W4 填。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from contracts.protocols import EvalCase, EvalResult, EvalSuite, TrafficLight

from .cases import DEFAULT_CASE_PATH


@dataclass(frozen=True)
class EvalRunConfig:
    """Runtime config for eval runs. 评估运行配置占位。"""

    case_path: Path = DEFAULT_CASE_PATH
    milestone: str = "M1"
    latency_budget_ms: float = 5000.0
    wandb_project: Optional[str] = None


class LnMemEvalSuite(EvalSuite):
    """Chinese LongMemEval suite skeleton. 中文长期记忆评估套件骨架。"""

    def __init__(self, config: Optional[EvalRunConfig] = None) -> None:
        self.config = config or EvalRunConfig()

    def run(self, cases: Sequence[EvalCase]) -> List[EvalResult]:
        """Run eval cases and return metric results. 执行 case 并返回指标结果。"""
        raise NotImplementedError("M0 W2 todo: connect backend and metric implementations")

    def traffic_light(self, results: Sequence[EvalResult], milestone: str) -> TrafficLight:
        """Build roadmap traffic light from results. 根据里程碑生成红黄绿。"""
        raise NotImplementedError("M0 W2 todo: implement roadmap §3 traffic-light rules")
