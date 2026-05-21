"""Collapse monitoring for hidden representations.

Implements spec §6.1 four-signal collapse checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor


class CollapseMonitorProtocol(Protocol):
    """Collapse monitor protocol; 中文：训练步级监控接口。"""

    def step(self, hidden_state: Tensor, batch_metadata: dict) -> dict[str, float | str]:
        """Compute collapse metrics and severity for one batch."""


@dataclass(frozen=True)
class CollapseThresholds:
    """Threshold settings for collapse detection; 中文：四联阈值配置。"""

    var_threshold: float = 0.01
    dist_threshold: float = 0.1
    probe_eps: float = 0.05
    kg_acc_threshold: float = 0.20


class CollapseMonitor:
    """Monitor hidden collapse severity from multiple signals."""

    def __init__(self, var_threshold: float = 0.01, dist_threshold: float = 0.1) -> None:
        self.thresholds = CollapseThresholds(
            var_threshold=var_threshold,
            dist_threshold=dist_threshold,
        )

    def step(self, hidden_state: Tensor, batch_metadata: dict) -> dict[str, float | str]:
        """Return var/dist/probe/kg metrics and collapse severity."""
        self._validate_hidden(hidden_state)

        hs = hidden_state.float()
        var_metric = float(hs.var(dim=0, unbiased=False).mean().item())
        dist_metric = float(self._pairwise_distance(hs).item())
        probe_ratio = float(self._linear_probe_ratio(batch_metadata))
        kg_acc = float(self._kg_retrieval_acc(batch_metadata))

        severity = self._classify(var_metric, dist_metric, probe_ratio, kg_acc)
        metrics: dict[str, float | str] = {
            "var": var_metric,
            "pairwise_dist": dist_metric,
            "linear_probe_ratio": probe_ratio,
            "kg_retrieval_acc": kg_acc,
            "collapse_severity": severity,
        }

        if severity in {"HIGH", "CRITICAL"}:
            self.emit_warning(severity, metrics)
        return metrics

    def emit_warning(self, severity: str, metrics: dict[str, float | str]) -> None:
        """Emit warning side effect; 中文：预留给训练器日志/告警钩子。"""
        print(
            "[collapse-monitor]"
            f" severity={severity}"
            f" var={metrics['var']:.6f}"
            f" pairwise_dist={metrics['pairwise_dist']:.6f}",
        )

    def _validate_hidden(self, hidden_state: Tensor) -> None:
        if hidden_state.ndim != 2:
            raise ValueError("hidden_state must be 2D [batch, dim]")
        if hidden_state.size(0) < 2 or hidden_state.size(1) < 2:
            raise ValueError("hidden_state must have batch>=2 and dim>=2")

    def _pairwise_distance(self, hidden_state: Tensor) -> Tensor:
        # spec §6.1-2: batch 内 pairwise distance 过低意味着表示坍缩。
        if hidden_state.size(0) <= 1:
            return torch.tensor(0.0)
        dists = torch.pdist(hidden_state, p=2)
        if dists.numel() == 0:
            return torch.tensor(0.0)
        return dists.mean()

    def _linear_probe_ratio(self, batch_metadata: dict) -> float:
        # spec §6.1-3: probe 接近 random（ratio≈1）提示“无可分表征”。
        probe_acc = batch_metadata.get("probe_accuracy")
        random_acc = batch_metadata.get("probe_random_accuracy")
        if probe_acc is not None and random_acc not in (None, 0):
            return float(probe_acc) / float(random_acc)

        logits = batch_metadata.get("probe_logits")
        labels = batch_metadata.get("probe_labels")
        if isinstance(logits, Tensor) and isinstance(labels, Tensor) and logits.ndim == 2:
            pred = logits.argmax(dim=1)
            acc = float((pred == labels).float().mean().item())
            n_cls = max(int(logits.size(1)), 1)
            return acc / (1.0 / float(n_cls))

        return float("nan")

    def _kg_retrieval_acc(self, batch_metadata: dict) -> float:
        # spec §6.1-4: 当前阶段可占位；有真值时直接透传 top5 召回。
        val = batch_metadata.get("kg_top5_acc")
        if val is None:
            return float("nan")
        return float(val)

    def _classify(self, var_metric: float, dist_metric: float, probe_ratio: float, kg_acc: float) -> str:
        if var_metric < self.thresholds.var_threshold:
            return "CRITICAL"

        if dist_metric < self.thresholds.dist_threshold:
            return "HIGH"

        if not torch.isnan(torch.tensor(probe_ratio)) and abs(probe_ratio - 1.0) <= self.thresholds.probe_eps:
            return "HIGH"

        if not torch.isnan(torch.tensor(kg_acc)) and kg_acc < self.thresholds.kg_acc_threshold:
            return "HIGH"

        return "OK"
