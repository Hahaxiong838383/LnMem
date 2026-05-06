"""Idempotent behavior tests for capture skeleton. / 采集骨架幂等行为测试。"""

from __future__ import annotations

import json
from pathlib import Path


def test_signal_buffer_open_open_close(monkeypatch) -> None:
    """Open twice should create one writer thread. / 连续 open 仅创建一次线程。"""
    from poc.src.capture import signal_buffer

    thread_create_count = 0

    class FakeThread:
        """Thread stub for deterministic idempotent test. / 线程替身避免并发抖动。"""

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
            nonlocal thread_create_count
            del args, kwargs
            thread_create_count += 1

        def start(self) -> None:
            return

        def join(self, timeout: float | None = None) -> None:
            del timeout
            return

    monkeypatch.setattr(signal_buffer.threading, "Thread", FakeThread)

    output_path = Path("/tmp/lnmem-fix-idempotent-open-open-close.jsonl")
    if output_path.exists():
        output_path.unlink()

    buffer = signal_buffer.SignalBuffer(signal_buffer.SignalBufferConfig(output_path=output_path))
    buffer.open()
    buffer.open()
    buffer.close()

    assert thread_create_count == 1


def test_multimodal_capture_open_open_close(monkeypatch, tmp_path) -> None:
    """Open twice should not reopen subcomponents. / 连续 open 不应重复打开子组件。"""
    from poc.src.capture import multimodal_capture

    class FakeComponent:
        """Simple open/close recorder. / 记录 open/close 次数的组件替身。"""

        def __init__(self) -> None:
            self._opened = False
            self.open_calls = 0
            self.close_calls = 0

        def open(self) -> None:
            self.open_calls += 1
            self._opened = True

        def close(self) -> None:
            self.close_calls += 1
            self._opened = False

    class FakeFace(FakeComponent):
        """Face stub. / 面部组件替身。"""

        def capture_signals(self, frame: object) -> dict[str, float | int]:
            del frame
            return {
                "eye_closure": 0.0,
                "head_pitch_deg": 0.0,
                "head_yaw_deg": 0.0,
                "gaze_stability": 0.0,
                "landmarks_n": 0,
            }

    class FakeAudio(FakeComponent):
        """Audio stub. / 音频组件替身。"""

        def transcribe(self, duration_s: float = 0.0) -> dict[str, str | float]:
            return {"transcript": "", "lang": "zh", "duration_s": duration_s}

    class FakeContext(FakeComponent):
        """Context stub. / 上下文组件替身。"""

        def get_context(self) -> dict[str, str | int]:
            return {"active_app": "unknown", "app_switch_count_5min": 0}

    class FakeBuffer(FakeComponent):
        """Buffer stub. / 缓冲组件替身。"""

        def __init__(self, _config: object) -> None:
            super().__init__()
            self.records: list[dict[str, object]] = []

        def push(self, frame_record: dict[str, object]) -> None:
            self.records.append(frame_record)

    monkeypatch.setattr(multimodal_capture, "FaceSignalExtractor", FakeFace)
    monkeypatch.setattr(multimodal_capture, "WhisperCppTranscriber", FakeAudio)
    monkeypatch.setattr(multimodal_capture, "MacOSAppContextTracker", FakeContext)
    monkeypatch.setattr(multimodal_capture, "SignalBuffer", FakeBuffer)

    capture = multimodal_capture.MultimodalCapture(output_path=tmp_path / "capture.jsonl")
    capture.open()
    capture.open()
    capture.close()

    assert capture._face.open_calls == 1
    assert capture._audio.open_calls == 1
    assert capture._context.open_calls == 1
    assert capture._buffer.open_calls == 1
    assert capture._face._opened is False
    assert capture._audio._opened is False
    assert capture._context._opened is False
    assert capture._buffer._opened is False


def test_signal_buffer_open_close_open_close(tmp_path) -> None:
    """Open-close cycles should append two jsonl lines. / 两轮开关后应累计两条 jsonl。"""
    from poc.src.capture.signal_buffer import SignalBuffer, SignalBufferConfig

    output_path = tmp_path / "idempotent-cycle.jsonl"
    buffer = SignalBuffer(SignalBufferConfig(output_path=output_path))

    first = {"ts": "2026-05-05T00:00:00Z", "frame_idx": 0}
    second = {"ts": "2026-05-05T00:00:01Z", "frame_idx": 1}

    buffer.open()
    buffer.push(first)
    buffer.close()

    buffer.open()
    buffer.push(second)
    buffer.close()

    lines = [line for line in output_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["frame_idx"] == 0
    assert parsed[1]["frame_idx"] == 1
