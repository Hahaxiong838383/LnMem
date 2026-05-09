"""Feedback queries jsonl 落盘 — 原子写 + 容错读。

落盘路径：~/.lnmem/feedback/queries_<YYYY-MM-DD>.jsonl
schema：
  {
    "ts": ISO8601 string,
    "frame_id": int,
    "reason": "brow_burst" | "low_vlm_conf",
    "signal": {brow, smile, eye, cheek},
    "vlm_label": {...} | null,
    "user_response": {"label": "match" | "mismatch" | "input" | "skip", "true_state": str | null, "note": str | null},
    "latency_s": float | null
  }
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def feedback_dir(home: Path | None = None) -> Path:
    base = home or Path.home()
    out = base / ".lnmem" / "feedback"
    out.mkdir(parents=True, exist_ok=True)
    return out


def queries_path(day_iso: str | None = None, home: Path | None = None) -> Path:
    day = day_iso or datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d")
    return feedback_dir(home) / f"queries_{day}.jsonl"


def append_query(record: dict[str, Any], home: Path | None = None) -> Path:
    """追加一条反馈记录到当日 jsonl。返回写入路径。"""
    path = queries_path(home=home)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with path.open("a", encoding="utf-8") as f:
        f.write(line)
    return path
