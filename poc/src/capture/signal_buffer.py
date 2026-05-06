"""Day 1 PoC skeleton for signal buffer. Day 8+ will add real capture."""

from __future__ import annotations

import json
import queue
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class SignalBufferConfig:
    """Buffer config. / 信号缓存与落盘参数占位。"""

    output_path: Path
    window_seconds: int = 300


class SignalBuffer:
    """5-minute buffer + async jsonl writer. / 5 分钟滑窗 + 异步 jsonl 落盘。"""

    def __init__(self, config: SignalBufferConfig) -> None:
        self._config = config
        self._window: deque[dict[str, Any]] = deque()
        self._writer_queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._writer_thread: threading.Thread | None = None
        self._file_handle: Any | None = None

    def open(self) -> None:
        """Start writer thread. / 启动异步写线程。"""
        if self._writer_thread is not None:
            return
        self._config.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_handle = self._config.output_path.open("a", encoding="utf-8")
        self._stop_event.clear()
        self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._writer_thread.start()

    def close(self) -> None:
        """Stop writer thread and flush. / 停止写线程并冲刷队列。"""
        self._stop_event.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2.0)
            self._writer_thread = None

        while not self._writer_queue.empty() and self._file_handle is not None:
            item = self._writer_queue.get_nowait()
            self._file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")

        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None

    def push(self, frame_record: dict[str, Any]) -> None:
        """Push one frame record. / 推入一条帧记录。"""
        self._window.append(frame_record)
        self._prune_window()
        self._writer_queue.put(frame_record)

    def get_window(self) -> list[dict[str, Any]]:
        """Get current window records. / 获取当前窗口记录。"""
        self._prune_window()
        return list(self._window)

    def _writer_loop(self) -> None:
        """Background writer. / 后台线程消费并写入 jsonl。"""
        while not self._stop_event.is_set() or not self._writer_queue.empty():
            try:
                item = self._writer_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            if self._file_handle is None:
                continue
            self._file_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            self._file_handle.flush()

    def _prune_window(self) -> None:
        """Keep only recent 5-minute records. / 仅保留最近 5 分钟记录。"""
        now_s = datetime.now(tz=timezone.utc).timestamp()
        cutoff = now_s - self._config.window_seconds
        while self._window:
            first = self._window[0]
            ts = _iso_to_epoch(str(first.get("ts", "")))
            if ts < cutoff:
                self._window.popleft()
                continue
            break


def _iso_to_epoch(iso_ts: str) -> float:
    """Convert ISO8601 string to epoch seconds. / ISO8601 转 epoch 秒。"""
    if not iso_ts:
        return 0.0
    normalized = iso_ts.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(normalized).timestamp()
    except ValueError:
        return 0.0
