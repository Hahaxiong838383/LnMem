#!/usr/bin/env python3
"""LnMem M0 端到端 demo — CLI 验证 parser → KG → query 闭环。

L4 砖 1（ADR-015 W+ 方案）。

run:
    GEMINI_PROXY_TOKEN=... PYTHONPATH=. .venv-poc-fresh/bin/python scripts/e2e_demo.py \\
      --output-report docs/reports/2026-05-21-e2e-demo.md
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from contracts.protocols import KGQuery, ParseRequest
from contracts.schema_v1 import SCHEMA_V1, SlotID
from graphiti_zh_adapter.adapter import GraphitiZhAdapterSkeleton
from universal_grammar_parser.config import ParserConfig
from universal_grammar_parser.llm_client import GeminiProxyClient
from universal_grammar_parser.parser_v0_llm import SchemaGuidedLLMParser


# M0 验收用 5 句中文样本（覆盖：事实记忆 / 计划 / 反思 / 情绪 / 查询）
DEMO_SENTENCES = [
    "昨晚 11 点和小李在公司吃了越南粉",
    "明天上午 10 点和王总开评审会，讨论 LnMem 产品定义",
    "今天下午 3 点要给爸妈打电话报平安",
    "上周三晚上和家人在家里看了《奥本海默》电影",
    "记一下：这周对自己写代码效率不太满意",
]

QUERIES = [
    "昨天和谁吃饭",
    "明天有什么会",
    "上周看了什么电影",
    "我最近心情如何",
]


def event_to_dict(event: Any) -> dict[str, Any]:
    grouped: dict[str, dict[str, Any]] = {"P0": {}, "P1": {}, "P2": {}, "P3": {}}
    for slot_id, val in event.slots.items():
        meta = SCHEMA_V1.get(slot_id)
        if not meta:
            continue
        v = val.value if hasattr(val, "value") else val
        grouped[meta.priority.name][meta.name] = v
    return {
        "event_id": event.event_id[:8],
        "timestamp": event.timestamp.isoformat(),
        **grouped,
    }


def write_report(
    path: Path,
    parse_results: list[dict[str, Any]],
    query_results: list[dict[str, Any]],
    overall_stats: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# LnMem M0 端到端 demo 验收报告")
    lines.append("")
    lines.append(f"- run_time: `{datetime.now(timezone.utc).isoformat()}`")
    lines.append(f"- sentences: **{overall_stats['n_sentences']}**")
    lines.append(f"- avg_parse_latency: **{overall_stats['avg_parse_s']:.2f}s**")
    lines.append(f"- avg_p0_fill: **{overall_stats['avg_p0_fill']:.2f}/4**")
    lines.append(f"- kg_nodes_written: **{overall_stats['kg_nodes']}**")
    lines.append(f"- queries: **{overall_stats['n_queries']}**")
    lines.append(f"- avg_query_latency: **{overall_stats['avg_query_s']:.3f}s**")
    lines.append(f"- avg_hits_per_query: **{overall_stats['avg_hits']:.2f}**")
    lines.append("")
    lines.append("## M0 验收门")
    lines.append("")
    parse_ok = overall_stats["avg_parse_s"] < 5.0
    p0_ok = overall_stats["avg_p0_fill"] >= 3.0
    query_ok = overall_stats["avg_query_s"] < 5.0
    hit_ok = overall_stats["avg_hits"] >= 1.0
    lines.append(f"- {'✅' if parse_ok else '❌'} parse latency < 5s (实测 {overall_stats['avg_parse_s']:.2f}s)")
    lines.append(f"- {'✅' if p0_ok else '⚠️'} P0 填充率 ≥ 3/4 (实测 {overall_stats['avg_p0_fill']:.2f}/4)")
    lines.append(f"- {'✅' if query_ok else '❌'} query latency < 5s (实测 {overall_stats['avg_query_s']:.3f}s)")
    lines.append(f"- {'✅' if hit_ok else '⚠️'} 平均命中数 ≥ 1 (实测 {overall_stats['avg_hits']:.2f})")
    lines.append("")
    lines.append("## Parse 结果（5 句中文）")
    lines.append("")
    for i, r in enumerate(parse_results, 1):
        lines.append(f"### #{i} `{r['text']}`")
        lines.append(f"- latency: {r['parse_s']:.2f}s, P0 填充: {r['p0_fill']}/4")
        lines.append(f"- event_id: `{r['event_id']}`, node_id: `{r['node_id']}`")
        lines.append("- slots:")
        for prio in ["P0", "P1"]:
            for k, v in r["slots"].get(prio, {}).items():
                lines.append(f"  - **{prio}** `{k}`: {v}")
        lines.append("")
    lines.append("## Query 结果（4 条自然语言查询）")
    lines.append("")
    for i, q in enumerate(query_results, 1):
        lines.append(f"### #{i} 查询: `{q['text']}`")
        lines.append(f"- latency: {q['query_s']:.3f}s, 命中: {q['n_hits']}")
        for j, h in enumerate(q["hits"][:3], 1):
            lines.append(f"  - hit #{j} score={h['score']:.3f}: {h['what_action']}")
            if h.get("explanation"):
                lines.append(f"    - {h['explanation']}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="LnMem M0 e2e demo")
    parser.add_argument("--output-report", type=Path, default=Path("docs/reports/2026-05-21-e2e-demo.md"))
    parser.add_argument("--user-id", default="chuange")
    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%H:%M:%S')}] init parser + adapter", flush=True)
    llm_client = GeminiProxyClient(model="gemini-3-flash-preview", max_tokens=4096, reasoning_effort="minimal")
    schema_parser = SchemaGuidedLLMParser(
        llm_client=llm_client,
        config=ParserConfig(llm_model="gemini-3-flash-preview", timeout_seconds=15.0),
    )
    adapter = GraphitiZhAdapterSkeleton()

    parse_results: list[dict[str, Any]] = []
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === parse {len(DEMO_SENTENCES)} sentences ===", flush=True)
    for text in DEMO_SENTENCES:
        t0 = time.time()
        # 用过去 7 天内随机一天的时间戳，制造时间分布
        req = ParseRequest(
            text=text, user_id=args.user_id,
            timestamp=datetime.now(timezone.utc) - timedelta(days=len(parse_results)),
        )
        event = schema_parser.parse(req)
        parse_s = time.time() - t0
        node_id = adapter.upsert(event)
        d = event_to_dict(event)
        p0_count = sum(1 for k in d["P0"].values() if k)
        parse_results.append({
            "text": text,
            "event_id": d["event_id"],
            "node_id": node_id[:12],
            "parse_s": parse_s,
            "p0_fill": p0_count,
            "slots": d,
        })
        print(f"  [{parse_s:.2f}s] P0={p0_count}/4 → {text}", flush=True)

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] === query {len(QUERIES)} sentences ===", flush=True)
    query_results: list[dict[str, Any]] = []
    for text in QUERIES:
        t0 = time.time()
        q = KGQuery(
            user_id=args.user_id, text_query=text,
            valid_at=datetime.now(timezone.utc), top_k=5,
        )
        hits = adapter.query(q)
        query_s = time.time() - t0
        hits_data = []
        for h in hits:
            wa = h.event.get(SlotID.WHAT_ACTION)
            hits_data.append({
                "score": h.score,
                "what_action": (wa.value if wa else "(none)") if hasattr(wa, "value") else (wa or "(none)"),
                "explanation": h.explanation or "",
            })
        query_results.append({
            "text": text, "query_s": query_s,
            "n_hits": len(hits), "hits": hits_data,
        })
        print(f"  [{query_s:.3f}s] hits={len(hits)} ← {text}", flush=True)

    overall = {
        "n_sentences": len(parse_results),
        "avg_parse_s": sum(r["parse_s"] for r in parse_results) / max(1, len(parse_results)),
        "avg_p0_fill": sum(r["p0_fill"] for r in parse_results) / max(1, len(parse_results)),
        "kg_nodes": len(parse_results),
        "n_queries": len(query_results),
        "avg_query_s": sum(r["query_s"] for r in query_results) / max(1, len(query_results)),
        "avg_hits": sum(r["n_hits"] for r in query_results) / max(1, len(query_results)),
    }
    write_report(args.output_report, parse_results, query_results, overall)
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] report → {args.output_report}", flush=True)
    print(f"avg_parse={overall['avg_parse_s']:.2f}s P0={overall['avg_p0_fill']:.2f}/4 hits={overall['avg_hits']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
