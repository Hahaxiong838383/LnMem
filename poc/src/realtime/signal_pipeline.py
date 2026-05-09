"""Signal pipeline worker. / py-feat + MediaPipe 信号提取线程。"""

from __future__ import annotations

from collections.abc import Mapping
import logging
import queue
import threading
from typing import Any

import cv2  # type: ignore
import mediapipe  # type: ignore

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
        self._face_mesh: Any | None = None
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
        result = {
            "ts": item.ts,
            "device": self._device_name,
            "frame_id": item.frame_id,
            "raw_video_offset_ms": item.raw_video_offset_ms,
            "face_detected": False,
            "raw_au": {},
            "calibrated_au": {},
            "expression": {},
            "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0},
            "session_id": item.session_id,
        }
        landmarks = self._extract_landmarks(item.frame)
        if landmarks is not None:
            result["face_detected"] = True
            result["head_pose"] = _head_pose_from_landmarks(landmarks)

        raw_au: Mapping[str, float] = {}
        if self._config.enable_pyfeat:
            raw_au = self._extract_au(item.frame)
            if raw_au:
                result["face_detected"] = True
                result["raw_au"] = _round_map(raw_au)
                caliper = self._ensure_caliper()
                calibrated = caliper.step(raw_au)
                result["calibrated_au"] = _round_map(calibrated)
                result["expression"] = _round_map(caliper.to_lnn_expression(calibrated))
        return result

    def _extract_landmarks(self, frame: Any) -> Any | None:
        if not self._config.enable_mediapipe:
            return None
        if self._face_mesh is None:
            face_mesh = mediapipe.solutions.face_mesh
            self._face_mesh = face_mesh.FaceMesh(
                max_num_faces=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                refine_landmarks=True,
            )
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output = self._face_mesh.process(rgb)
        faces = getattr(output, "multi_face_landmarks", None)
        if not faces:
            return None
        return faces[0].landmark

    def _extract_au(self, frame: Any) -> Mapping[str, float]:
        if self._au_extractor is None:
            from poc.src.vision.au_extractor import AUExtractor

            self._au_extractor = AUExtractor()
        raw_au, _ = self._au_extractor.extract(frame)
        return raw_au

    def _ensure_caliper(self) -> Any:
        if self._caliper is None:
            from poc.src.vision.feature_calibrator import AUCaliper

            self._caliper = AUCaliper()
        return self._caliper

    def _close_resources(self) -> None:
        if self._face_mesh is not None and hasattr(self._face_mesh, "close"):
            self._face_mesh.close()
        self._face_mesh = None
        self._au_extractor = None
        self._caliper = None


def _head_pose_from_landmarks(landmarks: Any) -> dict[str, float]:
    left = landmarks[234]
    right = landmarks[454]
    nose = landmarks[1]
    up = landmarks[10]
    down = landmarks[152]
    center_x = (left.x + right.x) / 2.0
    center_y = (up.y + down.y) / 2.0
    width = max(1e-6, abs(right.x - left.x))
    height = max(1e-6, abs(down.y - up.y))
    yaw = (nose.x - center_x) / width * 45.0
    pitch = (nose.y - center_y) / height * 35.0
    return {"yaw_deg": round(float(yaw), 3), "pitch_deg": round(float(pitch), 3)}


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
