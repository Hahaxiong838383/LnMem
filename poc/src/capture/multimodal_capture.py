"""Day 1 PoC skeleton for multimodal capture. Day 8+ will add real capture."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .app_context import MacOSAppContextTracker
from .audio_transcribe import WhisperCppTranscriber
from .face_signals import FaceSignalExtractor
from .signal_buffer import SignalBuffer, SignalBufferConfig


class MultimodalCapture:
    """Capture multimodal frame records. / 采集多模态单帧记录。"""

    def __init__(self, output_path: str | Path = "poc/data/capture/day1_capture.jsonl") -> None:
        self._face = FaceSignalExtractor()
        self._audio = WhisperCppTranscriber()
        self._context = MacOSAppContextTracker()
        self._buffer = SignalBuffer(SignalBufferConfig(output_path=Path(output_path)))
        self._frame_idx = 0
        self._opened = False

    def open(self) -> None:
        """Open all capture components. / 打开采集链路组件。"""
        self._face.open()
        self._audio.open()
        self._context.open()
        self._buffer.open()
        self._opened = True

    def close(self) -> None:
        """Close all capture components. / 关闭采集链路组件。"""
        if not self._opened:
            return
        self._buffer.close()
        self._context.close()
        self._audio.close()
        self._face.close()
        self._opened = False

    def capture_frame(self) -> dict[str, Any]:
        """Capture one frame record. / 采集一条符合契约的帧记录。"""
        if not self._opened:
            raise RuntimeError("MultimodalCapture is not opened")

        ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
        frame = {
            "ts": ts,
            "frame_idx": self._frame_idx,
            "face": self._face.capture_signals(frame=None),
            "audio": self._audio.transcribe(duration_s=0.0),
            "context": self._context.get_context(),
        }
        self._buffer.push(frame)
        self._frame_idx += 1
        return frame
