"""Feedback queries jsonl 落盘 — 原子写 + 容错读。

V0.3 schema（M0 W2 P0-1）：
  {
    "ts": ISO8601 string,
    "frame_id": int,
    "reason": "brow_burst" | "low_vlm_conf" | "user_triggered",  # ← user_triggered new
    "signal": {brow, smile, eye, cheek} | null,                  # ← null for user_triggered
    "vlm_label": {...} | null,
    "user_response": {"label": "match"|"mismatch"|"input"|"skip", "true_state": str|null, "note": str|null},
    "latency_s": float | null,
    # M0 W2 P0-1 V0.3 新增字段（响应 codex F3+F6）：
    "trigger_ref_frame_ts": ISO8601 | null,    # frame_id 对应帧的真实摄入时间戳
    "client_action_ts": ISO8601 | null,        # 用户点击/输入的时间戳
    "alignment_latency_s": float | null,       # client_action_ts - trigger_ref_frame_ts
    "caller_source": {"pid": int, "cmdline": str} | null  # user_triggered 审计字段
  }

V0.2 既有反馈（无 V0.3 新字段）仍兼容读取（V0.3 字段缺失 = None / 训练时按 None 处理）。
"""
from __future__ import annotations

import json
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------- 路径 ----------

def feedback_dir(home: Path | None = None) -> Path:
    base = home or Path.home()
    out = base / ".lnmem" / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    return out


def queries_path(day_iso: str | None = None, home: Path | None = None) -> Path:
    day = day_iso or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    return feedback_dir(home) / f"queries_{day}.jsonl"


# ---------- 全局 in-memory counter（响应 codex F7 统一口径） ----------

# Reason 类型计数器：统一在 append_query 入口埋点
# user_triggered 也走此入口，确保 sum(trigger_types) 一致
_TRIGGER_TYPES_COUNTER: Counter = Counter()

# user_triggered 训练入批指标（feedback_to_batch.py 写入）
_USER_TRIGGERED_METRICS: dict[str, int] = {
    "total": 0,
    "assembled": 0,
    "skipped_alignment": 0,
    "skipped_no_signal": 0,
}

_COUNTER_LOCK = threading.Lock()


def get_trigger_types() -> dict[str, int]:
    """返回 trigger_types 分布快照（dict）。daemon /health 调用。"""
    with _COUNTER_LOCK:
        return dict(_TRIGGER_TYPES_COUNTER)


def get_user_triggered_metrics() -> dict[str, int]:
    """返回 user_triggered_metrics 快照。daemon /health 调用。"""
    with _COUNTER_LOCK:
        return dict(_USER_TRIGGERED_METRICS)


def increment_user_triggered_metric(key: str, delta: int = 1) -> None:
    """供 feedback_to_batch.py 在解析时调用。"""
    if key not in _USER_TRIGGERED_METRICS:
        raise ValueError(f"unknown user_triggered metric key: {key}")
    with _COUNTER_LOCK:
        _USER_TRIGGERED_METRICS[key] += delta


def _reset_counters_for_test() -> None:
    """仅测试使用 — 清空 in-memory counter。"""
    with _COUNTER_LOCK:
        _TRIGGER_TYPES_COUNTER.clear()
        for k in _USER_TRIGGERED_METRICS:
            _USER_TRIGGERED_METRICS[k] = 0


# ---------- 写入 ----------

def append_query(record: dict[str, Any], home: Path | None = None) -> Path:
    """追加一条反馈记录到当日 jsonl。返回写入路径。

    V0.3：自动累加 trigger_types counter（按 record["reason"]）。
    """
    path = queries_path(home=home)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    # 统一口径埋点（响应 codex F7）
    reason = record.get("reason")
    if reason:
        with _COUNTER_LOCK:
            _TRIGGER_TYPES_COUNTER[reason] += 1
    return path


__all__ = [
    "append_query",
    "feedback_dir",
    "queries_path",
    "get_trigger_types",
    "get_user_triggered_metrics",
    "increment_user_triggered_metric",
    "_reset_counters_for_test",
]
