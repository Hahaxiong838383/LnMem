"""Camera capture worker. / 摄像头采集线程，负责重连与零积压送帧。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import queue
import threading
import time
from typing import Any

import cv2  # type: ignore

from .daemon_config import CaptureConfig


@dataclass
class CapturedFrame:
    frame_id: int
    ts: str
    frame: Any
    raw_video_offset_ms: int
    session_id: str


class CaptureWorker(threading.Thread):
    def __init__(
        self,
        config: CaptureConfig,
        output_queue: "queue.Queue[CapturedFrame]",
        stop_event: threading.Event,
        session_id: str,
    ) -> None:
        super().__init__(name="lnmem-capture", daemon=True)
        self._config = config
        self._output_queue = output_queue
        self._stop_event = stop_event
        self._session_id = session_id
        self._frame_id = 0
        self._connected = False
        self._start_monotonic = time.monotonic()
        self._last_frame_ts: str | None = None

    @property
    def last_frame_ts(self) -> str | None:
        return self._last_frame_ts

    def run(self) -> None:
        backoff_s = 1.0
        while not self._stop_event.is_set():
            cap = self._open_camera()
            if cap is None:
                logging.warning("camera open failed, retry in %.1fs", backoff_s)
                self._sleep_backoff(backoff_s)
                backoff_s = min(backoff_s * 2.0, 30.0)
                continue
            self._connected = True
            backoff_s = 1.0
            try:
                self._capture_loop(cap)
            except Exception:
                logging.exception("capture loop crashed, reconnecting")
            finally:
                self._connected = False
                cap.release()

    def _capture_loop(self, cap: Any) -> None:
        period_s = 1.0 / self._config.target_fps
        next_deadline = time.monotonic()
        while not self._stop_event.is_set():
            now = time.monotonic()
            if now < next_deadline:
                time.sleep(min(0.05, next_deadline - now))
                continue
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("camera read failed")
            ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
            item = CapturedFrame(
                frame_id=self._frame_id,
                ts=ts,
                frame=frame,
                raw_video_offset_ms=int((time.monotonic() - self._start_monotonic) * 1000),
                session_id=self._session_id,
            )
            self._last_frame_ts = ts
            self._offer_latest(item)
            self._frame_id += 1
            next_deadline += period_s

    def _offer_latest(self, item: CapturedFrame) -> None:
        try:
            self._output_queue.put_nowait(item)
        except queue.Full:
            try:
                self._output_queue.get_nowait()
            except queue.Empty:
                pass
            self._output_queue.put_nowait(item)

    def _open_camera(self) -> Any | None:
        backend = getattr(cv2, "CAP_AVFOUNDATION", 0)
        cap = cv2.VideoCapture(self._config.camera_index, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._config.camera_index)
        if not cap.isOpened():
            cap.release()
            return None
        width, height = self._config.resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, self._config.target_fps)
        return cap

    def _sleep_backoff(self, delay_s: float) -> None:
        deadline = time.monotonic() + delay_s
        while not self._stop_event.is_set() and time.monotonic() < deadline:
            time.sleep(0.1)


__all__ = ["CaptureWorker", "CapturedFrame"]
