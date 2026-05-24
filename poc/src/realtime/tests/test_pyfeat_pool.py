"""Tests for py-feat worker pool behavior."""

from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any

import pytest

from poc.src.realtime.capture_worker import CapturedFrame
from poc.src.realtime.daemon_config import PipelineConfig
from poc.src.realtime.signal_pipeline import SignalPipelinePool, SignalPipelineWorker


def _cfg() -> PipelineConfig:
    return PipelineConfig(enable_pyfeat=True, enable_mediapipe=False, enable_vad=False, pyfeat_workers=1)


def _frame(fid: int) -> CapturedFrame:
    ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return CapturedFrame(fid, ts, frame={"fid": fid}, raw_video_offset_ms=fid, session_id="s")


def test_pool_init_validates_n_workers() -> None:
    for bad in (0, -1, 9):
        with pytest.raises(ValueError):
            SignalPipelinePool(
                n_workers=bad,
                config=_cfg(),
                input_queue=queue.Queue(),
                output_queue=queue.Queue(),
                stop_event=threading.Event(),
                device_name="dev",
            )


def test_pool_creates_n_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    started: list[str] = []

    def _fake_run(self: SignalPipelineWorker) -> None:
        started.append(self.name)

    monkeypatch.setattr(SignalPipelineWorker, "run", _fake_run)
    pool = SignalPipelinePool(2, _cfg(), queue.Queue(), queue.Queue(), threading.Event(), "dev")
    pool.start()
    pool.join(timeout=1.0)
    assert sorted(started) == ["lnmem-signal-0", "lnmem-signal-1"]


def test_pool_shares_caliper_state(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Caliper:
        def __init__(self) -> None:
            self.count = 0

        def step(self, _raw: dict[str, float]) -> dict[str, float]:
            self.count += 1
            return {"AU12": 0.5}

        def to_lnn_expression(self, _cal: dict[str, float]) -> dict[str, float]:
            return {"smile_intensity": 0.5}

    shared = _Caliper()
    lock = threading.Lock()
    out_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    w0 = SignalPipelineWorker(_cfg(), queue.Queue(), out_q, threading.Event(), "dev", shared_caliper=shared, shared_caliper_lock=lock)
    w1 = SignalPipelineWorker(_cfg(), queue.Queue(), out_q, threading.Event(), "dev", shared_caliper=shared, shared_caliper_lock=lock)
    monkeypatch.setattr(SignalPipelineWorker, "_annotate_vlm", lambda self, frame: None)
    monkeypatch.setattr(
        SignalPipelineWorker,
        "_extract_pyfeat",
        lambda self, frame: ({"AU12": 0.8}, {"happy": 0.1}, {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
    )

    t1 = threading.Thread(target=w0._process, args=(_frame(1),))
    t2 = threading.Thread(target=w1._process, args=(_frame(2),))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert shared.count == 2
    assert w0._caliper is w1._caliper


def test_pool_caliper_lock_prevents_race(monkeypatch: pytest.MonkeyPatch) -> None:
    class _RacyCaliper:
        def __init__(self) -> None:
            self.count = 0

        def step(self, _raw: dict[str, float]) -> dict[str, float]:
            seen = self.count
            time.sleep(0.0001)
            self.count = seen + 1
            return {"AU12": 0.7}

        def to_lnn_expression(self, _cal: dict[str, float]) -> dict[str, float]:
            return {"smile_intensity": 0.7}

    shared = _RacyCaliper()
    lock = threading.Lock()
    w0 = SignalPipelineWorker(_cfg(), queue.Queue(), queue.Queue(), threading.Event(), "dev", shared_caliper=shared, shared_caliper_lock=lock)
    w1 = SignalPipelineWorker(_cfg(), queue.Queue(), queue.Queue(), threading.Event(), "dev", shared_caliper=shared, shared_caliper_lock=lock)
    monkeypatch.setattr(SignalPipelineWorker, "_annotate_vlm", lambda self, frame: None)
    monkeypatch.setattr(
        SignalPipelineWorker,
        "_extract_pyfeat",
        lambda self, frame: ({"AU12": 0.9}, {"happy": 0.2}, {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0}),
    )

    def _runner(worker: SignalPipelineWorker, start_fid: int) -> None:
        for i in range(100):
            worker._process(_frame(start_fid + i))

    threads = [threading.Thread(target=_runner, args=((w0 if i % 2 == 0 else w1), i * 1000)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert shared.count == 1000


def test_pool_processes_all_frames(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SignalPipelineWorker,
        "_process",
        lambda self, item: {"ts": item.ts, "device": "dev", "frame_id": item.frame_id},
    )
    in_q: "queue.Queue[CapturedFrame]" = queue.Queue()
    out_q: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop = threading.Event()
    pool = SignalPipelinePool(2, _cfg(), in_q, out_q, stop, "dev")
    pool.start()
    for i in range(100):
        in_q.put(_frame(i))

    deadline = time.time() + 5.0
    while out_q.qsize() < 100 and time.time() < deadline:
        time.sleep(0.01)
    stop.set()
    pool.join(timeout=2.0)
    assert out_q.qsize() == 100
    assert pool.frames_processed == 100


def test_backward_compat_n_workers_1(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        SignalPipelineWorker,
        "_process",
        lambda self, item: {"ts": item.ts, "device": "dev", "frame_id": item.frame_id},
    )

    def _run_once(pipeline: Any, in_q: "queue.Queue[CapturedFrame]", out_q: "queue.Queue[dict[str, Any]]", stop: threading.Event) -> list[int]:
        pipeline.start()
        for i in range(5):
            in_q.put(_frame(i))
        deadline = time.time() + 2.0
        while out_q.qsize() < 5 and time.time() < deadline:
            time.sleep(0.01)
        stop.set()
        pipeline.join(timeout=1.0)
        return [out_q.get()["frame_id"] for _ in range(out_q.qsize())]

    in1: "queue.Queue[CapturedFrame]" = queue.Queue()
    out1: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop1 = threading.Event()
    single = SignalPipelineWorker(_cfg(), in1, out1, stop1, "dev")
    worker_ids = _run_once(single, in1, out1, stop1)

    in2: "queue.Queue[CapturedFrame]" = queue.Queue()
    out2: "queue.Queue[dict[str, Any]]" = queue.Queue()
    stop2 = threading.Event()
    pool1 = SignalPipelinePool(1, _cfg(), in2, out2, stop2, "dev")
    pool_ids = _run_once(pool1, in2, out2, stop2)
    assert worker_ids == [0, 1, 2, 3, 4]
    assert pool_ids == worker_ids

