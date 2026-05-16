"""RealtimeLNNInferencer (M0 W3 Phase 4.2, 2026-05-16)

封装 LNN ckpt 加载 + 实时 forward + margin/entropy 计算，给 signal_pipeline 用。

响应 codex Phase 4 finding：
- F-01: state_model.forward(x, persona_idx) 签名封装，禁止漏传
- F-02: 用 _signals_row_to_frame_dict + MultimodalFusion 转 18 维（不重新拼字典）
- F-03: 加载时检测 online_adapter_schema=phase2_v1 → OnlineLoRAAdapter wrap + load tau_drift
- F-04: ckpt 选择走显式 path（不再 max(glob) 选最新混乱）
- F-05: signal_path 元数据带入推理结果（A/B 归因）
- F-06: 失败模式独立 try/except，仅清空 state_* 字段，不影响 signal pipe 出队

设计原则：
- 实例化失败（ckpt 缺失 / 加载异常）→ logger.warning + 返 disabled instance
- infer 异常 → 返 dict 仅含 lnn_status=error，不抛
- 滑窗维护：60 帧 fusion vector，未满时返 lnn_status=warmup
"""
from __future__ import annotations

import logging
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from poc.src.fusion import MultimodalFusion
from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig
from poc.src.lnn.signals_loader import _signals_row_to_frame_dict


_DEFAULT_WINDOW = 60


@dataclass(frozen=True)
class InferenceResult:
    """LNN 单帧推理结果（不可变）"""
    lnn_status: str  # "ok" | "warmup" | "error" | "disabled"
    predicted_state: str | None = None
    state_probs: list[float] | None = None
    top_prob: float | None = None
    margin: float | None = None
    entropy: float | None = None
    error: str | None = None
    ckpt_sha8: str | None = None  # 推理结果可追溯到具体 ckpt
    signal_path_used: str | None = None  # F-05: 推理时的 signal_path 元数据


class RealtimeLNNInferencer:
    """Realtime LNN forward 封装。daemon 内单线程使用（signal_pipeline thread）。

    构造失败 → 返 disabled 实例（所有 infer 返 lnn_status=disabled），不阻止 daemon 启动。
    """

    def __init__(
        self,
        ckpt_path: Path | None = None,
        persona_idx: int = 30,  # vboss_self
        window: int = _DEFAULT_WINDOW,
        signal_path: str = "calibrated",
    ) -> None:
        self._persona_idx = persona_idx
        self._window = window
        self._signal_path = signal_path
        self._fusion = MultimodalFusion()
        self._history: deque[list[float]] = deque(maxlen=window)
        self._state_model = None
        self._ckpt_sha8: str | None = None
        self._enabled = False
        self._fail_count = 0
        self._fail_count_circuit_break = 100  # 连续 100 次失败 → 永久 disable
        self._logger = logging.getLogger(__name__)

        if ckpt_path is None:
            self._logger.warning("RealtimeLNNInferencer: ckpt_path=None → disabled")
            return

        try:
            self._load_ckpt(ckpt_path)
            self._enabled = True
            self._logger.info(
                "RealtimeLNNInferencer ready: ckpt=%s sha8=%s persona_idx=%d signal_path=%s",
                ckpt_path.name, self._ckpt_sha8, persona_idx, signal_path,
            )
        except Exception as exc:
            self._logger.warning(
                "RealtimeLNNInferencer init failed: %s — running in disabled mode", exc
            )
            self._state_model = None
            self._enabled = False

    def _load_ckpt(self, ckpt_path: Path) -> None:
        """加载 ckpt + F-03 OnlineLoRAAdapter unwrap"""
        import hashlib
        self._ckpt_sha8 = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()[:8]
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        factory = DefaultLNNFactory()
        sm = factory.create_state_model(ckpt["config"])
        sm.load_state_dict(ckpt["state_model"])
        sm.eval()

        # F-03: OnlineLoRAAdapter wrap（若 ckpt 有 online_tau_drift）
        if (
            ckpt.get("online_adapter_schema") == "phase2_v1"
            and "online_tau_drift" in ckpt
        ):
            base = getattr(sm, "personalization", None)
            if base is not None:
                online = OnlineLoRAAdapter(
                    base_adapter=base, online_config=OnlineAdapterConfig()
                )
                online.tau_drift.data.copy_(ckpt["online_tau_drift"])
                sm.personalization = online
                self._logger.info(
                    "  OnlineLoRAAdapter wrapped: tau_drift shape=%s",
                    list(ckpt["online_tau_drift"].shape),
                )

        self._state_model = sm

    def reset(self) -> None:
        """清空滑窗（如 daemon session 切换）"""
        self._history.clear()

    def infer(self, signal_row: dict[str, Any]) -> InferenceResult:
        """单帧推理 + 维护滑窗.

        Parameters
        ----------
        signal_row : dict
            signal_pipeline 输出的 dict（含 raw_au/calibrated_au/expression/head_pose）

        Returns
        -------
        InferenceResult
            lnn_status 区分 ok / warmup / error / disabled
        """
        if not self._enabled or self._state_model is None:
            return InferenceResult(
                lnn_status="disabled",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

        # F-06 fail-soft: 整个 infer 包独立 try
        try:
            # F-02: 用 signals_loader helper 转 fusion frame，禁止重新拼
            frame_dict = _signals_row_to_frame_dict(signal_row)
            fused = self._fusion.fuse(frame_dict)
            if len(fused.vector) != 18:
                return InferenceResult(
                    lnn_status="error",
                    error=f"fused vector dim {len(fused.vector)} != 18",
                    ckpt_sha8=self._ckpt_sha8,
                    signal_path_used=self._signal_path,
                )
            self._history.append(fused.vector)
        except Exception as exc:
            self._record_fail()
            return InferenceResult(
                lnn_status="error",
                error=f"fusion failed: {type(exc).__name__}: {exc}",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

        if len(self._history) < self._window:
            return InferenceResult(
                lnn_status="warmup",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

        # F-01: forward(x, persona_idx) 双参数
        try:
            x = torch.tensor(list(self._history), dtype=torch.float32).unsqueeze(0)  # (1, 60, 18)
            pid = torch.tensor([self._persona_idx], dtype=torch.long)
            with torch.no_grad():
                logits = self._state_model(x, pid)  # (1, T, n_states)
            last = logits[0, -1, :]  # (n_states,)
            probs = torch.softmax(last, dim=-1).tolist()
            sp = sorted(probs, reverse=True)
            entropy = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
            margin = sp[0] - sp[1]
            top_idx = int(last.argmax())
            self._fail_count = 0  # 成功重置
            return InferenceResult(
                lnn_status="ok",
                predicted_state=STATE_LABELS[top_idx],
                state_probs=probs,
                top_prob=sp[0],
                margin=margin,
                entropy=entropy,
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )
        except Exception as exc:
            self._record_fail()
            return InferenceResult(
                lnn_status="error",
                error=f"forward failed: {type(exc).__name__}: {exc}",
                ckpt_sha8=self._ckpt_sha8,
                signal_path_used=self._signal_path,
            )

    def _record_fail(self) -> None:
        """累计失败计数，超过阈值永久 disable（F-06 熔断）"""
        self._fail_count += 1
        if self._fail_count >= self._fail_count_circuit_break:
            self._logger.warning(
                "RealtimeLNNInferencer: %d consecutive failures, disabling for this session",
                self._fail_count,
            )
            self._enabled = False

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @property
    def ckpt_sha8(self) -> str | None:
        return self._ckpt_sha8

    def health_dict(self) -> dict[str, Any]:
        """暴露给 daemon /health endpoint"""
        return {
            "enabled": self._enabled,
            "ckpt_sha8": self._ckpt_sha8,
            "persona_idx": self._persona_idx,
            "signal_path": self._signal_path,
            "history_len": len(self._history),
            "fail_count": self._fail_count,
        }


__all__ = ["RealtimeLNNInferencer", "InferenceResult"]
