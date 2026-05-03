"""W&B hook skeleton for contracts/protocols.py eval outputs.

Self-hosted only; M0 W3 adds PII-safe whitelist logging.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from contracts.protocols import EvalResult, TrafficLight


@dataclass(frozen=True)
class WandbConfig:
    """W&B config placeholder. 自托管 W&B 配置占位。"""

    project: str
    base_url: Optional[str] = None
    entity: Optional[str] = None
    enabled: bool = False


class WandbHook:
    """PII-safe W&B logging skeleton. 隐私安全上报骨架。"""

    def __init__(self, config: WandbConfig) -> None:
        self.config = config

    def log_case_result(self, result: EvalResult) -> None:
        """Log one EvalResult. 上报单条脱敏指标。"""
        raise NotImplementedError("M0 W3 todo: log only whitelisted EvalResult fields")

    def log_traffic_light(self, light: TrafficLight) -> None:
        """Log milestone TrafficLight. 上报三色聚合结果。"""
        raise NotImplementedError("M0 W3 todo: log aggregate traffic-light summary")

    def sanitize_details(self, details: Optional[Mapping[str, object]]) -> Mapping[str, object]:
        """Return PII-safe details. 返回脱敏后的 details 白名单。"""
        raise NotImplementedError("M0 W3 todo: implement PII whitelist for W&B details")

    def close(self) -> None:
        """Close W&B run. 关闭 W&B run。"""
        raise NotImplementedError("M0 W3 todo: close self-hosted W&B run")
