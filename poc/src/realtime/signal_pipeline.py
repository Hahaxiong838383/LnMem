"""Signal pipeline worker. / py-feat 信号提取线程（LNN-FIRST）。

LNN-FIRST 原则：py-feat 是唯一面部信号来源。
AUExtractor.extract() 同时返回 AU / 情绪 / 头部姿态（img2pose），
不再引入 MediaPipe 作为备用头姿估计路径。
"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import queue
import threading
from typing import Any

from .capture_worker import CapturedFrame
from .daemon_config import PipelineConfig


class SignalPipelineWorker(threading.Thread):
    def __init__(
        self,
        config: PipelineConfig,
        input_queue: "queue.Queue[CapturedFrame]",
        output_queue: "queue.Queue[dict[str, Any]]",
        stop_event: threading.Event,
        device_name: str,
    ) -> None:
        super().__init__(name="lnmem-signal", daemon=True)
        self._config = config
        self._input_queue = input_queue
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._device_name = device_name
        self._frames_processed = 0
        self._last_frame_ts: str | None = None
        self._au_extractor: Any | None = None
        self._caliper: Any | None = None

    def run(self) -> None:
        while not self._stop_event.is_set() or not self._input_queue.empty():
            try:
                item = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                result = self._process(item)
                _offer(self._output_queue, result)
                self._frames_processed += 1
                self._last_frame_ts = result["ts"]
            except Exception:
                logging.exception("signal pipeline failed for frame_id=%s", item.frame_id)
        self._close_resources()

    def _process(self, item: CapturedFrame) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ts": item.ts,
            "device": self._device_name,
            "frame_id": item.frame_id,
            "raw_video_offset_ms": item.raw_video_offset_ms,
            "face_detected": False,
            "raw_au": {},
            "calibrated_au": {},
            "expression": {},
            # head_pose: filled by py-feat img2pose below (yaw/pitch/roll in degrees)
            "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
            "session_id": item.session_id,
        }

        if self._config.enable_pyfeat:
            raw_au, _emo, pose = self._extract_pyfeat(item.frame)
            if raw_au:
                result["face_detected"] = True
                result["raw_au"] = _round_map(raw_au)
                result["head_pose"] = {k: round(v, 3) for k, v in pose.items()}
                caliper = self._ensure_caliper()
                calibrated = caliper.step(raw_au)
                result["calibrated_au"] = _round_map(calibrated)
                result["expression"] = _round_map(caliper.to_lnn_expression(calibrated))

        return result

    def _extract_pyfeat(
        self, frame: Any
    ) -> tuple[Mapping[str, float], dict[str, float], dict[str, float]]:
        """Call AUExtractor; returns (raw_au, emo, pose). / 调用 AUExtractor，返回三元组。"""
        if self._au_extractor is None:
            from poc.src.vision.au_extractor import AUExtractor

            self._au_extractor = AUExtractor()
        raw_au, emo, pose = self._au_extractor.extract(frame)
        return raw_au, emo, pose

    def _ensure_caliper(self) -> Any:
        if self._caliper is None:
            from poc.src.vision.feature_calibrator import AUCaliper

            self._caliper = AUCaliper()
        return self._caliper

    def _close_resources(self) -> None:
        self._au_extractor = None
        self._caliper = None


def _round_map(values: Mapping[str, float]) -> dict[str, float]:
    return {key: round(float(value), 4) for key, value in values.items()}


def _offer(target: "queue.Queue[dict[str, Any]]", item: dict[str, Any]) -> None:
    try:
        target.put_nowait(item)
    except queue.Full:
        try:
            target.get_nowait()
        except queue.Empty:
            pass
        target.put_nowait(item)


__all__ = ["SignalPipelineWorker"]
