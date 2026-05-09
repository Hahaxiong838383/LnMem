"""ProactiveQueryTrigger — 监听信号判断是否触发询问。

触发条件（V0.1）：
1. brow_burst: 连续 5 帧（@1fps ≈ 5s）brow_furrow > 0.7（强烈皱眉）
2. low_vlm_conf: 当前帧 vlm_label.confidence < 0.65（VLM 自己不确定）

触发后：emit TriggerEvent，清空 brow 窗口防止连续触发。
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

_BROW_BURST_WINDOW = 5
_BROW_BURST_THRESHOLD = 0.7
_LOW_VLM_CONF_THRESHOLD = 0.65


@dataclass(frozen=True)
class TriggerEvent:
    """触发询问事件（不可变）。"""
    ts: str
    frame_id: int
    reason: str  # "brow_burst" | "low_vlm_conf"
    signal_snapshot: dict[str, float]
    vlm_label: dict[str, Any] | None


class ProactiveQueryTrigger:
    """监听 signal pipeline 输出 dict，判断是否应触发询问。"""

    def __init__(
        self,
        brow_threshold: float = _BROW_BURST_THRESHOLD,
        brow_window: int = _BROW_BURST_WINDOW,
        vlm_conf_threshold: float = _LOW_VLM_CONF_THRESHOLD,
    ) -> None:
        self._brow_threshold = brow_threshold
        self._brow_window_size = brow_window
        self._vlm_conf_threshold = vlm_conf_threshold
        self._brow_history: deque[float] = deque(maxlen=brow_window)

    def observe(self, item: dict[str, Any]) -> TriggerEvent | None:
        """处理一条 signal pipeline 输出，返回 TriggerEvent 或 None。"""
        if not item.get("face_detected"):
            self._brow_history.clear()
            return None

        expression = item.get("expression") or {}
        brow = float(expression.get("brow_furrow", 0.0))
        self._brow_history.append(brow)

        signal_snapshot = {
            "brow_furrow": brow,
            "smile_intensity": float(expression.get("smile_intensity", 0.0)),
            "eye_closure": float(expression.get("eye_closure", 0.0)),
            "cheek_raise": float(expression.get("cheek_raise", 0.0)),
        }
        vlm_label = item.get("vlm_label")

        # 优先级 1: low_vlm_conf（有 VLM 描述更具体）
        if isinstance(vlm_label, dict):
            conf = vlm_label.get("confidence")
            if isinstance(conf, (int, float)) and conf < self._vlm_conf_threshold:
                return TriggerEvent(
                    ts=str(item.get("ts", "")),
                    frame_id=int(item.get("frame_id", 0)),
                    reason="low_vlm_conf",
                    signal_snapshot=signal_snapshot,
                    vlm_label=dict(vlm_label),
                )

        # 优先级 2: brow_burst
        if (
            len(self._brow_history) >= self._brow_window_size
            and all(b > self._brow_threshold for b in self._brow_history)
        ):
            self._brow_history.clear()
            return TriggerEvent(
                ts=str(item.get("ts", "")),
                frame_id=int(item.get("frame_id", 0)),
                reason="brow_burst",
                signal_snapshot=signal_snapshot,
                vlm_label=dict(vlm_label) if isinstance(vlm_label, dict) else None,
            )

        return None
