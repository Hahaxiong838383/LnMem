"""Tests for multiprocessing py-feat pool and main-process router behavior."""

from __future__ import annotations

from datetime import datetime, timezone
import os
import queue
import threading
import time
from typing import Any

import numpy as np

from poc.src.realtime.capture_worker import CapturedFrame
from poc.src.realtime.pyfeat_mp_pool import PyfeatMpPool
from poc.src.realtime.signal_pipeline import SignalPipelineMpRouter


def _frame(fid: int) -> CapturedFrame:
    ts = datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z")
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    return CapturedFrame(
        frame_id=fid,
        ts=ts,
        frame=frame,
        raw_video_offset_ms=fid * 10,
        session_id="test-session",
    )


def _wait_result(pool: PyfeatMpPool, timeout_s: float = 10.0) -> dict[str, Any] | None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        result = pool.get_result(timeout=0.2)
        if result is not None:
            return result
    return None


def test_mp_pool_init_validates_n_workers() -> None:
    for bad in (0, -1, 5):
        try:
            PyfeatMpPool(bad)
        except ValueError:
            continue
        raise AssertionError(f"n_workers={bad} should raise ValueError")


def test_mp_pool_start_creates_n_processes() -> None:
    pool = PyfeatMpPool(2)
    try:
        pool.start()
        assert len(pool.workers) == 2
        assert all(worker.is_alive() for worker in pool.workers)
    finally:
        pool.stop(timeout=5.0)


def test_mp_pool_submit_get_result_roundtrip() -> None:
    pool = PyfeatMpPool(1)
    try:
        pool.start()
        assert pool.submit(_frame(1), block=True, timeout=1.0)
        result = _wait_result(pool, timeout_s=90.0)
        assert result is not None
        assert result["frame_id"] == 1
        assert "raw_au" in result
        assert "head_pose" in result
        assert "session_id" in result
    finally:
        pool.stop(timeout=5.0)


def test_mp_pool_independent_tmp_dirs() -> None:
    pool = PyfeatMpPool(2)
    try:
        pool.start()
        entries: list[str] = []
        deadline = time.time() + 5.0
        while time.time() < deadline:
            entries = [
                name for name in os.listdir("/tmp") if name.startswith("lnmem_pyfeat_w")
            ]
            if len(entries) >= 2:
                break
            time.sleep(0.1)
        assert len(entries) >= 2
        assert len(set(entries)) >= 2
    finally:
        pool.stop(timeout=5.0)


def test_mp_pool_stop_clean() -> None:
    pool = PyfeatMpPool(2)
    pool.start()
    pool.stop(timeout=5.0)
    assert all(not worker.is_alive() for worker in pool.workers)


def test_router_postprocess_face_detected_threshold() -> None:
    router = SignalPipelineMpRouter(
        mp_pool=PyfeatMpPool(1),
        output_queue=queue.Queue(),
        stop_event=threading.Event(),
        device_name="dev",
        lnn_inferencer=None,
    )
    base = {
        "ts": "2026-05-24T00:00:00Z",
        "frame_id": 1,
        "raw_video_offset_ms": 1,
        "session_id": "s",
        "head_pose": {"yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0},
    }
    low = dict(base)
    low["raw_au"] = {"AU01": 0.0}
    low_out = router._post_process(low)
    assert low_out["face_detected"] is False

    high = dict(base)
    high["frame_id"] = 2
    high["raw_au"] = {"AU01": 0.5}
    high_out = router._post_process(high)
    assert high_out["face_detected"] is True
    assert "calibrated_au" in high_out
    assert "expression" in high_out
