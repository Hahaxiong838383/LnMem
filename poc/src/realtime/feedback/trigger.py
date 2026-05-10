"""ProactiveQueryTrigger — 监听信号判断是否触发询问。

触发条件（V0.1）：
1. brow_burst: 连续 5 帧（@1fps ≈ 5s）brow_furrow > 0.7（强烈皱眉）
2. low_vlm_conf: 当前帧 vlm_label.confidence < 0.65（VLM 自己不确定）

触发后：emit TriggerEvent，清空 brow 窗口防止连续触发。

FB-09 fix (V0.2)：
VLM 可能自报"未检测到人脸"等，此时 confidence=0 不应触发弹窗。
加双重过滤：au_description 命中关键词 AND confidence < 0.1 -> 跳过。
"""
from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# V0.2 阈值收紧（codex review FB-02 + 22/4min 过敏实测）：
# - window 5→10 帧 (10s @ 1fps)，过滤短暂瞬间皱眉
# - threshold 0.7→0.85，过滤老板自然工作脸 brow=0.5-0.8
_BROW_BURST_WINDOW = 10
_BROW_BURST_THRESHOLD = 0.85
_LOW_VLM_CONF_THRESHOLD = 0.65

# FB-09: VLM 自报无脸的关键词清单
VLM_NO_FACE_KEYWORDS = (
    "未检测到人脸",
    "图片中未出现人脸",
    "图中未出现人脸",
    "图片中未包含人脸",
    "图中未包含人脸",
    "无法识别",
    "无法分析",
    "无法检测",
    "无法进行",
)
VLM_NO_FACE_CONF_THRESHOLD = 0.1

_log = logging.getLogger(__name__)


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
        home: Path | None = None,
    ) -> None:
        self._brow_threshold = brow_threshold
        self._brow_window_size = brow_window
        self._vlm_conf_threshold = vlm_conf_threshold
        self._brow_history: deque[float] = deque(maxlen=brow_window)
        self._home = home or Path.home()

    def _is_vlm_self_report_no_face(self, vlm_label: dict[str, Any] | None) -> bool:
        """判断 VLM 是否自报"未检测到人脸"。

        两种 confidence=0 语义：
        (a) VLM 看到脸但拿不准状态 → 应该问用户（correct）
        (b) VLM 自报未检测到人脸 → 不该问用户（无效弹窗）

        双重过滤：au_description 命中关键词 AND confidence < 0.1
        """
        if not vlm_label:
            return False

        conf = vlm_label.get("confidence", 1.0)
        if not isinstance(conf, (int, float)) or conf >= VLM_NO_FACE_CONF_THRESHOLD:
            return False

        desc = (vlm_label.get("au_description") or "").strip()
        return any(kw in desc for kw in VLM_NO_FACE_KEYWORDS)

    def _log_vlm_no_face_skip(self, item: dict[str, Any], vlm_label: dict[str, Any] | None) -> None:
        """记录 VLM 自报无脸导致的跳过事件到 ~/.lnmem/feedback/skipped_<date>.jsonl。"""
        try:
            feedback_dir = self._home / ".lnmem" / "feedback"
            feedback_dir.mkdir(parents=True, exist_ok=True)

            day = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
            path = feedback_dir / f"skipped_{day}.jsonl"

            record = {
                "ts": str(item.get("ts", "")),
                "frame_id": int(item.get("frame_id", 0)),
                "reason": "vlm_no_face_skip",
                "vlm_label": vlm_label,
                "au_description": (vlm_label or {}).get("au_description", ""),
            }

            line = json.dumps(record, ensure_ascii=False) + "\n"
            with path.open("a", encoding="utf-8") as f:
                f.write(line)

        except Exception as e:
            _log.warning("Failed to log vlm_no_face_skip: %s", e)

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
                # FB-09: 检查 VLM 是否自报无脸，若是则跳过（不弹窗）
                if self._is_vlm_self_report_no_face(vlm_label):
                    self._log_vlm_no_face_skip(item, dict(vlm_label))
                    return None

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
