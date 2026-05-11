"""Tests for feedback rate limiter. / 反馈频率限制测试。"""

from __future__ import annotations

import json

from poc.src.realtime.feedback.rate_limiter import RateLimiterConfig, TokenBucketRateLimiter


def _new_limiter(tmp_path, max_per_hour: int = 3) -> TokenBucketRateLimiter:
    config = RateLimiterConfig(
        min_per_hour=1,
        max_per_hour=max_per_hour,
        state_path=tmp_path / "rate_state.json",
    )
    return TokenBucketRateLimiter(config)


def test_empty_state_allows(tmp_path) -> None:
    """Missing state file should allow first query. / 无状态文件应允许首次询问。"""
    limiter = _new_limiter(tmp_path)
    assert limiter.can_query(10_000.0) is True


def test_records_persist(tmp_path) -> None:
    """Recorded queries should be visible to a new instance. / 新实例应读到已记录历史。"""
    now = 10_000.0
    limiter = _new_limiter(tmp_path)
    limiter.record_query(now)

    reloaded = _new_limiter(tmp_path)
    assert reloaded.can_query(now + 1_201.0) is True

    payload = json.loads((tmp_path / "rate_state.json").read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert payload["queries"] == [now]


def test_max_per_hour_blocks(tmp_path) -> None:
    """Fourth query inside one hour should be blocked when max=3. / 一小时内第4次应被拒绝。"""
    base = 20_000.0
    limiter = _new_limiter(tmp_path, max_per_hour=3)
    limiter.record_query(base)
    limiter.record_query(base + 1_200.0)
    limiter.record_query(base + 2_400.0)

    assert limiter.can_query(base + 3_599.0) is False


def test_window_slides(tmp_path) -> None:
    """Query exactly 3600s old should not count in window. / 恰好3600秒前的记录不计入窗口。"""
    now = 30_000.0
    limiter = _new_limiter(tmp_path, max_per_hour=3)
    limiter.record_query(now - 3_600.0)
    limiter.record_query(now - 2_400.0)
    limiter.record_query(now - 1_200.0)

    assert limiter.can_query(now) is True


def test_min_interval_enforced(tmp_path) -> None:
    """Minimum spacing should be enforced by max_per_hour. / 按 max_per_hour 强制最小间隔。"""
    base = 40_000.0
    limiter = _new_limiter(tmp_path, max_per_hour=3)
    limiter.record_query(base)

    assert limiter.can_query(base + 1_199.0) is False
    assert limiter.can_query(base + 1_201.0) is True


def test_corrupted_state_recovers(tmp_path) -> None:
    """Corrupted json should self-heal as empty state. / 损坏状态文件应自愈为可用空状态。"""
    path = tmp_path / "rate_state.json"
    path.write_text("{not-json", encoding="utf-8")

    limiter = _new_limiter(tmp_path)
    assert limiter.can_query(50_000.0) is True
    assert path.exists() is False
