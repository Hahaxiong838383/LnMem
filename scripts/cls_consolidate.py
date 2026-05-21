#!/usr/bin/env python3
"""CLS v0 — 夜间整合最小可行（ADR-015 W+ 砖 2 / 不可妥协项 §10.1 #3）。

Daily 02:00 cron 触发：
  1. 从 SQLite events 表取当天 KG 节点
  2. LLM (gemini-3-flash) 提炼 "今日小结"（事实/情绪/未完成 3 条）
  3. 写入 SQLite daily_summaries 表（长期存储）
  4. 输出 markdown 报告

run:
    GEMINI_PROXY_TOKEN=... PYTHONPATH=. .venv-poc-fresh/bin/python scripts/cls_consolidate.py \\
      --db data/cls.sqlite --date 2026-05-21
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from datetime import datetime, date, timedelta, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from universal_grammar_parser.llm_client import GeminiProxyClient


CLS_PROMPT_TEMPLATE = """你是 LnMem 伴侣 AI 的"夜间整合"模块。把今天用户的事件列表提炼为 3 条"今日精华"。

【今日事件列表（最多 20 条）】
{events_jsonl}

【输出格式 — 严格 JSON，3 条精华】
{{
  "date": "{date_str}",
  "highlights": [
    {{
      "type": "事实",
      "summary": "<一句话总结最重要的客观事实>",
      "evidence_event_ids": ["<event_id1>", ...]
    }},
    {{
      "type": "情绪",
      "summary": "<一句话描述今天的情绪轨迹>",
      "evidence_event_ids": [...]
    }},
    {{
      "type": "未完成",
      "summary": "<一句话提示未完成事项或需关注的>",
      "evidence_event_ids": [...]
    }}
  ]
}}

只输出 JSON，不要其他文字。
"""


def init_db(db_path: Path) -> sqlite3.Connection:
    """初始化 SQLite，建 events + daily_summaries 表（如果不存在）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            event_id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text_raw TEXT,
            slots_json TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            date TEXT NOT NULL,
            highlights_json TEXT NOT NULL,
            n_events INTEGER NOT NULL,
            llm_model TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date)
        )
    """)
    conn.commit()
    return conn


def get_events_for_date(conn: sqlite3.Connection, user_id: str, target_date: date) -> list[dict]:
    """取该日所有 events。"""
    start = datetime.combine(target_date, datetime.min.time(), tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    cur = conn.execute(
        "SELECT event_id, timestamp, text_raw, slots_json FROM events "
        "WHERE user_id = ? AND timestamp >= ? AND timestamp < ? "
        "ORDER BY timestamp",
        (user_id, start.isoformat(), end.isoformat()),
    )
    rows = cur.fetchall()
    events = []
    for event_id, ts, text, slots_json in rows:
        events.append({
            "event_id": event_id[:8],
            "timestamp": ts,
            "text": text or "",
            "slots": json.loads(slots_json) if slots_json else {},
        })
    return events


def consolidate_one_day(
    conn: sqlite3.Connection,
    llm_client: GeminiProxyClient,
    user_id: str,
    target_date: date,
    *,
    max_events: int = 20,
) -> dict:
    """对一天做 CLS 整合，返回 highlights dict。"""
    events = get_events_for_date(conn, user_id, target_date)
    if not events:
        return {"date": target_date.isoformat(), "n_events": 0, "highlights": [], "status": "no_events"}

    events_jsonl = "\n".join(json.dumps(e, ensure_ascii=False) for e in events[:max_events])
    prompt = CLS_PROMPT_TEMPLATE.format(
        events_jsonl=events_jsonl,
        date_str=target_date.isoformat(),
    )

    t0 = time.time()
    try:
        raw = llm_client(prompt)
    except Exception as exc:
        return {
            "date": target_date.isoformat(),
            "n_events": len(events),
            "highlights": [],
            "status": f"llm_error: {exc}",
        }
    llm_s = time.time() - t0

    # 解析 JSON（容错）
    raw = raw.strip()
    if raw.startswith("```"):
        # markdown code block
        raw = raw.split("```", 2)[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.rsplit("```", 1)[0].strip()

    try:
        parsed = json.loads(raw)
        highlights = parsed.get("highlights", [])
    except Exception as exc:
        return {
            "date": target_date.isoformat(),
            "n_events": len(events),
            "highlights": [],
            "raw_llm": raw,
            "status": f"json_parse_error: {exc}",
        }

    # 写入 SQLite
    conn.execute(
        "INSERT OR REPLACE INTO daily_summaries (user_id, date, highlights_json, n_events, llm_model) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, target_date.isoformat(), json.dumps(parsed, ensure_ascii=False), len(events), "gemini-3-flash-preview"),
    )
    conn.commit()

    return {
        "date": target_date.isoformat(),
        "n_events": len(events),
        "highlights": highlights,
        "llm_latency_s": llm_s,
        "status": "ok",
    }


def write_report(path: Path, result: dict, user_id: str) -> None:
    lines = [
        f"# CLS 夜间整合报告 — {result['date']}",
        "",
        f"- user_id: `{user_id}`",
        f"- n_events: **{result['n_events']}**",
        f"- status: `{result['status']}`",
        f"- llm_latency: {result.get('llm_latency_s', 0):.2f}s",
        "",
        "## 今日精华",
        "",
    ]
    for h in result.get("highlights", []):
        lines.append(f"### 【{h.get('type', '?')}】 {h.get('summary', '?')}")
        evidence = h.get("evidence_event_ids", [])
        if evidence:
            lines.append(f"- 证据 event_ids: {', '.join(evidence)}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LnMem CLS v0 — 夜间整合最小可行")
    parser.add_argument("--db", type=Path, default=Path("data/cls.sqlite"))
    parser.add_argument("--user-id", default="chuange")
    parser.add_argument("--date", default=None, help="ISO 日期 (默认: 昨天)")
    parser.add_argument("--output-report", type=Path, default=None)
    args = parser.parse_args()

    target_date = (
        date.fromisoformat(args.date)
        if args.date
        else (datetime.now(timezone.utc).date() - timedelta(days=1))
    )
    output_report = args.output_report or Path(f"docs/reports/cls_{target_date.isoformat()}.md")

    conn = init_db(args.db)
    llm_client = GeminiProxyClient(model="gemini-3-flash-preview", max_tokens=4096, reasoning_effort="minimal")

    print(f"[{datetime.now().strftime('%H:%M:%S')}] CLS consolidating {args.user_id} for {target_date}", flush=True)
    result = consolidate_one_day(conn, llm_client, args.user_id, target_date)
    print(f"  n_events={result['n_events']} highlights={len(result.get('highlights', []))} status={result['status']}", flush=True)

    write_report(output_report, result, args.user_id)
    print(f"  report → {output_report}", flush=True)

    conn.close()
    return 0 if result["status"] in ("ok", "no_events") else 1


if __name__ == "__main__":
    raise SystemExit(main())
