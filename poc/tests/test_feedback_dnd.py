"""Tests for feedback DND checks. / 反馈勿扰检查测试。"""

from __future__ import annotations

import subprocess
import sys
import unittest

from poc.src.realtime.feedback.dnd import DNDConfig, get_frontmost_app, should_suppress


def test_none_frontmost_no_suppress() -> None:
    """None frontmost should not suppress. / 拿不到前台应用时不抑制。"""
    assert should_suppress(DNDConfig(apps=("zoom",)), None) is False


def test_exact_match_suppresses() -> None:
    """Exact app name should suppress. / 完全匹配应抑制。"""
    assert should_suppress(DNDConfig(apps=("zoom",)), "zoom") is True


def test_case_insensitive() -> None:
    """Matching should ignore case. / 匹配应大小写不敏感。"""
    assert should_suppress(DNDConfig(apps=("zoom",)), "ZOOM.us") is True


def test_substring_match() -> None:
    """Substring app name should suppress. / 子串命中应抑制。"""
    assert should_suppress(DNDConfig(apps=("loom",)), "Loom Recorder") is True


def test_no_match_passes() -> None:
    """No configured app matched should pass. / 未命中列表时应放行。"""
    assert should_suppress(DNDConfig(apps=("zoom", "loom")), "Cursor") is False


def test_get_frontmost_timeout_returns_none(monkeypatch) -> None:
    """Timeout should return None. / 超时异常应返回 None。"""

    def _fake_run(*_args, **_kwargs):
        raise subprocess.TimeoutExpired(cmd="osascript", timeout=0.1)

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert get_frontmost_app(timeout_s=0.1) is None


def test_get_frontmost_called_process_error_returns_none(monkeypatch) -> None:
    """CalledProcessError should return None. / 进程报错应返回 None。"""

    def _fake_run(*_args, **_kwargs):
        raise subprocess.CalledProcessError(returncode=1, cmd="osascript")

    monkeypatch.setattr(subprocess, "run", _fake_run)
    assert get_frontmost_app(timeout_s=0.1) is None


@unittest.skipUnless(sys.platform == "darwin", "requires macOS osascript")
def test_get_frontmost_app_integration_darwin() -> None:
    """Integration check should not raise on macOS. / macOS 集成调用不应抛异常。"""
    value = get_frontmost_app(timeout_s=1.0)
    assert value is None or isinstance(value, str)
