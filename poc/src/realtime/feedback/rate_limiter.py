"""Rate limiter for proactive feedback queries. / 主动反馈询问频率限制器。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

WINDOW_SECONDS = 3600.0
STATE_VERSION = 1


@dataclass(frozen=True)
class RateLimiterConfig:
    """Rate limiter config with persisted state path. / 频控配置与持久化路径。"""

    min_per_hour: int
    max_per_hour: int
    state_path: Path


class TokenBucketRateLimiter:
    """Sliding-window limiter with minimum spacing. / 滑窗+最小间隔限流。"""

    def __init__(self, config: RateLimiterConfig) -> None:
        self._config = config
        if config.max_per_hour <= 0:
            raise ValueError("max_per_hour must be > 0")

    def can_query(self, now_ts: float) -> bool:
        """Return whether a query is allowed at current timestamp. / 判断当前是否允许询问。"""
        queries = self._window_queries(self._load_queries(), now_ts)
        if len(queries) >= self._config.max_per_hour:
            return False
        if not queries:
            return True
        min_interval = WINDOW_SECONDS / float(self._config.max_per_hour)
        return (now_ts - queries[-1]) >= min_interval

    def record_query(self, now_ts: float) -> None:
        """Persist one query timestamp with atomic file write. / 原子落盘一次询问时间戳。"""
        queries = self._window_queries(self._load_queries(), now_ts)
        queries.append(float(now_ts))
        self._write_state(queries)

    def reset(self) -> None:
        """Delete persisted state for test/manual reset. / 删除持久化状态。"""
        try:
            self._config.state_path.unlink()
        except FileNotFoundError:
            return

    def _load_queries(self) -> list[float]:
        path = self._config.state_path
        if not path.exists():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return self._decode_queries(payload)
        except Exception:
            self._unlink_corrupted(path)
            return []

    def _decode_queries(self, payload: object) -> list[float]:
        if not isinstance(payload, dict):
            raise ValueError("state must be an object")
        queries = payload.get("queries")
        if not isinstance(queries, list):
            raise ValueError("queries must be a list")
        decoded: list[float] = []
        for item in queries:
            decoded.append(float(item))
        decoded.sort()
        return decoded

    def _window_queries(self, queries: list[float], now_ts: float) -> list[float]:
        lower = float(now_ts) - WINDOW_SECONDS
        upper = float(now_ts)
        return [q for q in queries if lower < q <= upper]

    def _write_state(self, queries: list[float]) -> None:
        path = self._config.state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"queries": queries, "version": STATE_VERSION}
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")
        tmp_path.replace(path)

    def _unlink_corrupted(self, path: Path) -> None:
        try:
            path.unlink()
        except OSError:
            return
