"""Storage manager for raw video and signals. / 原始视频与衍生信号存储管理。"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any

import cv2  # type: ignore

from .capture_worker import CapturedFrame
from .daemon_config import StorageConfig


class StorageManager:
    def __init__(self, config: StorageConfig, stop_event: threading.Event) -> None:
        self._config = config
        self._stop_event = stop_event
        self._raw_dir = self._config.data_dir / "raw_video"
        self._raw_dir.mkdir(parents=True, exist_ok=True)
        self._archive_dir = self._config.data_dir / ".archive"
        self._archive_dir.mkdir(parents=True, exist_ok=True)
        self._frame_queue: "queue.Queue[CapturedFrame]" = queue.Queue(maxsize=32)
        self._signal_queue: "queue.Queue[dict[str, Any]]" = queue.Queue(maxsize=512)
        self._recent: deque[dict[str, Any]] = deque()
        self._recent_lock = threading.Lock()
        self._recent_cache_seconds = 7200
        self._raw_thread: threading.Thread | None = None
        self._signal_thread: threading.Thread | None = None
        self._video_writer: Any | None = None
        self._video_hour: str | None = None
        self._signal_handle: Any | None = None
        self._signal_day: str | None = None
        self._signal_hour: str | None = None
        self._signal_live_path: Path | None = None
        self._last_prune = 0.0

    def start(self) -> None:
        self._raw_thread = threading.Thread(target=self._raw_loop, name="lnmem-raw", daemon=True)
        self._signal_thread = threading.Thread(target=self._signal_loop, name="lnmem-signal-store", daemon=True)
        self._raw_thread.start()
        self._signal_thread.start()

    def stop(self) -> None:
        if self._raw_thread is not None:
            self._raw_thread.join(timeout=5.0)
        if self._signal_thread is not None:
            self._signal_thread.join(timeout=5.0)
        self._close_video_writer()
        self._close_signal_handle()

    def enqueue_frame(self, frame: CapturedFrame) -> None:
        self._offer(self._frame_queue, frame)

    def enqueue_signal(self, signal: dict[str, Any]) -> None:
        self._offer(self._signal_queue, signal)

    def get_recent(self, seconds: int) -> list[dict[str, Any]]:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=seconds)
        with self._recent_lock:
            cached = [item for item in self._recent if _parse_ts(item["ts"]) >= cutoff]
        if seconds <= self._recent_cache_seconds:
            return cached
        return self._scan_signal_files(cutoff)

    def _raw_loop(self) -> None:
        while not self._stop_event.is_set() or not self._frame_queue.empty():
            try:
                item = self._frame_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_frame(item)
                self._maybe_prune()
            except Exception:
                logging.exception("raw video write failed for frame_id=%s", item.frame_id)

    def _signal_loop(self) -> None:
        while not self._stop_event.is_set() or not self._signal_queue.empty():
            try:
                item = self._signal_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                self._write_signal(item)
                self._cache_recent(item)
                self._maybe_prune()
            except Exception:
                logging.exception("signal jsonl write failed for frame_id=%s", item.get("frame_id"))

    def _write_frame(self, item: CapturedFrame) -> None:
        hour_key = item.ts[:13].replace(":", "-")
        if self._video_hour != hour_key:
            self._rotate_video_writer(item, hour_key)
        if self._video_writer is not None:
            self._video_writer.write(item.frame)

    def _rotate_video_writer(self, item: CapturedFrame, hour_key: str) -> None:
        self._close_video_writer()
        height, width = item.frame.shape[:2]
        path = self._raw_dir / f"raw_{item.session_id}_{hour_key}.mp4"
        codec = self._config.compress_codec
        fourcc = cv2.VideoWriter_fourcc(*("avc1" if codec == "h264" else "mp4v"))
        writer = cv2.VideoWriter(str(path), fourcc, 1.0, (width, height))
        if not writer.isOpened():
            logging.warning("codec %s unavailable, fallback to mp4v", codec)
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (width, height))
        self._video_writer = writer if writer.isOpened() else None
        self._video_hour = hour_key

    def _write_signal(self, item: dict[str, Any]) -> None:
        dt = _parse_ts(item["ts"])
        day_key = dt.strftime("%Y-%m-%d")
        hour_key = dt.strftime("%Y-%m-%d-%H")
        if self._signal_handle is None:
            self._open_signal_handle(day_key=day_key, hour_key=hour_key)
        elif self._signal_day != day_key:
            self._close_signal_handle()
            self._open_signal_handle(day_key=day_key, hour_key=hour_key)
        elif self._signal_hour != hour_key:
            self._rotate_signal_hourly(archive_hour=self._signal_hour)
            self._open_signal_handle(day_key=day_key, hour_key=hour_key)
        if self._signal_handle is not None:
            self._signal_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._signal_handle.flush()

    def _open_signal_handle(self, day_key: str, hour_key: str) -> None:
        path = self._config.data_dir / f"signals_{day_key}.jsonl"
        self._signal_handle = path.open("a", encoding="utf-8")
        self._signal_day = day_key
        self._signal_hour = hour_key
        self._signal_live_path = path

    def _rotate_signal_hourly(self, archive_hour: str | None) -> None:
        if archive_hour is None:
            return
        live_path = self._signal_live_path
        if self._signal_handle is not None:
            self._signal_handle.flush()
            self._signal_handle.close()
        self._signal_handle = None
        if live_path is None or not live_path.exists():
            return
        archive_path = self._archive_dir / f"signals_{archive_hour}.jsonl"
        if archive_path.exists():
            stamp = datetime.now(tz=timezone.utc).strftime("%Y%m%d%H%M%S")
            archive_path = self._archive_dir / f"signals_{archive_hour}-{stamp}.jsonl"
        live_path.rename(archive_path)

    def _cache_recent(self, item: dict[str, Any]) -> None:
        cutoff = datetime.now(tz=timezone.utc) - timedelta(seconds=self._recent_cache_seconds)
        with self._recent_lock:
            self._recent.append(item)
            while self._recent and _parse_ts(self._recent[0]["ts"]) < cutoff:
                self._recent.popleft()

    def _scan_signal_files(self, cutoff: datetime) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        candidates = sorted(self._config.data_dir.glob("signals_*.jsonl"))
        candidates += sorted(self._archive_dir.glob("signals_*.jsonl"))
        for path in candidates:
            file_day = _file_day(path)
            if file_day is None or file_day < cutoff.date():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    item = json.loads(line)
                    if _parse_ts(item["ts"]) >= cutoff:
                        results.append(item)
        return results

    def _maybe_prune(self) -> None:
        now = time.monotonic()
        if now - self._last_prune < 300:
            return
        self._last_prune = now
        raw_cutoff = datetime.now(tz=timezone.utc) - timedelta(hours=self._config.raw_video_retention_hours)
        signal_cutoff = datetime.now(tz=timezone.utc).date() - timedelta(days=self._config.signal_jsonl_retention_days)
        for path in self._raw_dir.glob("raw_*.mp4"):
            if datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc) < raw_cutoff:
                path.unlink(missing_ok=True)
        candidates = list(self._config.data_dir.glob("signals_*.jsonl"))
        candidates.extend(self._archive_dir.glob("signals_*.jsonl"))
        for path in candidates:
            file_day = _file_day(path)
            if file_day is not None and file_day < signal_cutoff:
                path.unlink(missing_ok=True)

    def _close_video_writer(self) -> None:
        if self._video_writer is not None:
            self._video_writer.release()
        self._video_writer = None
        self._video_hour = None

    def _close_signal_handle(self) -> None:
        if self._signal_handle is not None:
            self._signal_handle.close()
        self._signal_handle = None
        self._signal_day = None
        self._signal_hour = None
        self._signal_live_path = None

    @staticmethod
    def _offer(target: queue.Queue[Any], item: Any) -> None:
        try:
            target.put_nowait(item)
        except queue.Full:
            try:
                target.get_nowait()
            except queue.Empty:
                pass
            target.put_nowait(item)


def _parse_ts(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _file_day(path: Path) -> Any:
    stem = path.stem.replace("signals_", "")
    day_part = stem.split("-")
    if len(day_part) < 3:
        return None
    day_text = "-".join(day_part[:3])
    try:
        return datetime.strptime(day_text, "%Y-%m-%d").date()
    except ValueError:
        return None


__all__ = ["StorageManager"]
