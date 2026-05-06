"""Day 1 PoC skeleton for app context. Day 8+ will add real capture."""

from __future__ import annotations

import subprocess
import time
from collections import deque
from dataclasses import dataclass


@dataclass
class AppContextConfig:
    """App context config. / 前台应用上下文参数占位。"""

    window_seconds: int = 300


class MacOSAppContextTracker:
    """Track active app and switches. / 跟踪前台 app 与切换次数。"""

    def __init__(self, config: AppContextConfig | None = None) -> None:
        self._config = config or AppContextConfig()
        self._last_active_app = "unknown"
        self._switch_events: deque[float] = deque()

    def open(self) -> None:
        """Initialize tracker. / 初始化前台应用跟踪器。"""
        self._last_active_app = self._read_active_app()

    def close(self) -> None:
        """Close tracker. / 关闭跟踪器。"""
        self._switch_events.clear()

    def get_context(self) -> dict[str, str | int]:
        """Get current app context. / 获取当前上下文信息。"""
        now = time.time()
        active_app = self._read_active_app()

        if active_app != self._last_active_app:
            self._switch_events.append(now)
            self._last_active_app = active_app

        self._prune_old(now)
        return {
            "active_app": active_app,
            "app_switch_count_5min": len(self._switch_events),
        }

    def _read_active_app(self) -> str:
        """Read frontmost app name. / 读取当前前台 app 名称。"""
        script = (
            'tell application "System Events" '\
            'to get name of first application process whose frontmost is true'
        )
        try:
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            return "unknown"

        if result.returncode != 0:
            return "unknown"
        name = result.stdout.strip()
        return name or "unknown"

    def _prune_old(self, now_s: float) -> None:
        """Drop events out of the window. / 移除 5 分钟窗口外事件。"""
        min_ts = now_s - self._config.window_seconds
        while self._switch_events and self._switch_events[0] < min_ts:
            self._switch_events.popleft()
