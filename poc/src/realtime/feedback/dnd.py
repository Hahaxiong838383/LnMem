"""Do-not-disturb checks for proactive feedback. / 主动反馈勿扰检查。"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class DNDConfig:
    """Frontmost-app suppression list. / 前台应用勿扰列表。"""

    apps: tuple[str, ...]


def get_frontmost_app(timeout_s: float = 1.0) -> str | None:
    """Fetch frontmost app from macOS osascript. / 用 osascript 获取前台应用名。"""
    script = (
        'tell application "System Events" to get name of first application process '
        "whose frontmost is true"
    )
    try:
        completed = subprocess.run(
            ["osascript", "-e", script],
            timeout=timeout_s,
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
        return None
    output = completed.stdout.strip()
    return output or None


def should_suppress(config: DNDConfig, frontmost: str | None) -> bool:
    """Return suppress decision by case-insensitive substring match. / 子串匹配判断是否抑制。"""
    if frontmost is None:
        return False
    app_name = frontmost.lower()
    for candidate in config.apps:
        if candidate and candidate.lower() in app_name:
            return True
    return False
