"""ADR-028 v0.1 方案 B: multiprocessing py-feat pool 真并行实现。"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import time
from typing import Any

from .capture_worker import CapturedFrame

_DEFAULT_START_METHOD = "spawn"  # macOS 强制，fork 容易触发 py-feat/torch 崩溃


def _set_worker_env() -> None:
    """Set worker process env vars for deterministic CPU threading behavior."""
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    mpl_dir = f"/tmp/lnmem_mpl_{os.getpid()}"
    os.makedirs(mpl_dir, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", mpl_dir)


def _pyfeat_worker_loop(
    worker_id: int,
    input_q: "mp.Queue[CapturedFrame]",
    output_q: "mp.Queue[dict[str, Any]]",
    stop_event: "mp.synchronize.Event",
) -> None:
    """Run a dedicated AUExtractor in one process and return lightweight dict output."""
    _set_worker_env()
    pid = os.getpid()
    tmp_dir = f"/tmp/lnmem_pyfeat_w{worker_id}_{pid}"
    os.makedirs(tmp_dir, exist_ok=True)
    extractor: Any | None = None
    extractor_err: str | None = None
    try:
        from poc.src.vision.au_extractor import AUExtractor

        extractor = AUExtractor(tmp_dir=tmp_dir)
        logging.info("[mp-worker-%s] pid=%s tmp_dir=%s initialized", worker_id, pid, tmp_dir)
    except Exception as exc:  # noqa: BLE001
        extractor_err = str(exc)
        logging.exception(
            "[mp-worker-%s] failed to initialize AUExtractor, fallback empty output",
            worker_id,
        )

    while not stop_event.is_set():
        try:
            captured = input_q.get(timeout=0.2)
        except queue.Empty:
            continue
        except Exception:
            logging.exception("[mp-worker-%s] input queue get failed", worker_id)
            continue

        if extractor is None:
            output_q.put(
                {
                    "worker_id": worker_id,
                    "frame_id": captured.frame_id,
                    "ts": captured.ts,
                    "session_id": captured.session_id,
                    "raw_video_offset_ms": captured.raw_video_offset_ms,
                    "raw_au": {},
                    "head_pose": {},
                    "error": f"au_extractor_init_failed:{extractor_err}",
                }
            )
            continue

        try:
            raw_au, _emo, pose = extractor.extract(captured.frame)
            output_q.put(
                {
                    "worker_id": worker_id,
                    "frame_id": captured.frame_id,
                    "ts": captured.ts,
                    "session_id": captured.session_id,
                    "raw_video_offset_ms": captured.raw_video_offset_ms,
                    "raw_au": dict(raw_au) if raw_au else {},
                    "head_pose": dict(pose) if pose else {},
                }
            )
        except Exception as exc:  # noqa: BLE001
            logging.exception(
                "[mp-worker-%s] extract failed frame_id=%s", worker_id, captured.frame_id
            )
            output_q.put(
                {
                    "worker_id": worker_id,
                    "frame_id": captured.frame_id,
                    "ts": captured.ts,
                    "session_id": captured.session_id,
                    "raw_video_offset_ms": captured.raw_video_offset_ms,
                    "raw_au": {},
                    "head_pose": {},
                    "error": str(exc),
                }
            )


class PyfeatMpPool:
    """Spawn-based process pool dedicated to py-feat extraction."""

    def __init__(self, n_workers: int) -> None:
        if n_workers < 1 or n_workers > 4:
            raise ValueError(f"n_workers must be in [1, 4] for mp pool, got {n_workers}")
        self._n_workers = n_workers
        self._ctx = mp.get_context(_DEFAULT_START_METHOD)
        self._input_q: "mp.Queue[CapturedFrame]" = self._ctx.Queue(maxsize=8)
        self._output_q: "mp.Queue[dict[str, Any]]" = self._ctx.Queue(maxsize=64)
        self._stop_event = self._ctx.Event()
        self._workers: list[mp.Process] = []

    def start(self) -> None:
        if self._workers:
            return
        for i in range(self._n_workers):
            process = self._ctx.Process(
                target=_pyfeat_worker_loop,
                args=(i, self._input_q, self._output_q, self._stop_event),
                name=f"lnmem-pyfeat-mp-{i}",
                daemon=True,
            )
            process.start()
            self._workers.append(process)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop_event.set()
        deadline = time.monotonic() + timeout
        for worker in self._workers:
            remaining = max(0.1, deadline - time.monotonic())
            worker.join(timeout=remaining)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)

    def submit(
        self, captured: CapturedFrame, block: bool = True, timeout: float | None = None
    ) -> bool:
        try:
            self._input_q.put(captured, block=block, timeout=timeout)
            return True
        except queue.Full:
            return False

    def get_result(self, timeout: float = 0.2) -> dict[str, Any] | None:
        try:
            return self._output_q.get(timeout=timeout)
        except queue.Empty:
            return None

    @property
    def n_workers(self) -> int:
        return self._n_workers

    @property
    def workers(self) -> list[mp.Process]:
        return self._workers

    def is_alive(self) -> bool:
        return any(worker.is_alive() for worker in self._workers)


__all__ = [
    "PyfeatMpPool",
    "_DEFAULT_START_METHOD",
    "_pyfeat_worker_loop",
    "_set_worker_env",
]
