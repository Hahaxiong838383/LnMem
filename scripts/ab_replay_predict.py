#!/usr/bin/env python3
"""ADR-005 A/B 离线 dual replay — calibrated vs raw 路径 trigger 决策对照

回答 codex 二审 F-07：缺"同一时间线上双路径反事实触发决策"对账表。

输入：~/.lnmem/data/signals_*.jsonl（4 天 235K 帧）
输出：每帧两路 trigger 决策对照表 + 汇总 stats

A 路径（calibrated）：trigger 用 expression.brow_furrow，阈值 0.95
B 路径（raw）：       trigger 用 raw_au.AU04，          阈值 0.55

简化版（不全模拟 trigger 状态机，只统计"该帧 brow 值 > 阈值"分布）：
- 哪些帧 cal 会触发但 raw 不会（calibrated 假阳性候选）
- 哪些帧 raw 会触发但 cal 不会（raw 新增灵敏度）
- 哪些帧两路都触发（共识）
- 配合历史 144 user feedback events，看双路在真实 user true_state 上的对齐度
"""
from __future__ import annotations
import argparse
import json
import math
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import sys
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ADR-005 v0.3 双阈值（与 trigger.py 同步）
THR_CAL = 0.95
THR_RAW = 0.55
WINDOW = 15  # 连续 N 帧 > 阈值
COOLDOWN_FRAMES = 300  # 5min @ 1fps


def simulate_brow_burst(
    signals: list[dict], field: str, threshold: float
) -> list[dict]:
    """模拟 brow_burst 触发：滑窗 + cooldown。返回 trigger events 列表。"""
    triggers = []
    streak = 0
    cooldown_left = 0
    for s in signals:
        cooldown_left = max(0, cooldown_left - 1)
        v = _get_field(s, field)
        if v is None:
            streak = 0
            continue
        if v > threshold:
            streak += 1
            if streak >= WINDOW and cooldown_left == 0:
                triggers.append({
                    "ts": s.get("ts"),
                    "frame_id": s.get("frame_id"),
                    "value": v,
                    "field": field,
                    "threshold": threshold,
                })
                cooldown_left = COOLDOWN_FRAMES
                streak = 0
        else:
            streak = 0
    return triggers


def _get_field(s: dict, dotted: str) -> float | None:
    """从 signal dict 取嵌套字段（如 'expression.brow_furrow' 或 'raw_au.AU04'）"""
    cur = s
    for p in dotted.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    if cur is None or (isinstance(cur, float) and math.isnan(cur)):
        return None
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def load_signals(signals_dir: Path, days: list[str]) -> list[dict]:
    """加载多日 signals（按 ts 升序）"""
    out = []
    for d in days:
        f = signals_dir / f"signals_{d}.jsonl"
        if not f.exists():
            continue
        with f.open() as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    out.sort(key=lambda s: s.get("ts", ""))
    return out


def load_feedback_events(feedback_dir: Path) -> list[dict]:
    """加载所有 user feedback events（query + response）"""
    out = []
    for f in sorted(feedback_dir.glob("queries_*.jsonl")):
        with f.open() as fh:
            for line in fh:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    return out


def match_triggers_to_feedback(
    triggers: list[dict], events: list[dict], window_s: float = 60.0,
) -> list[tuple]:
    """对每个 trigger，找出 60s 内最近的真实 user feedback（看 mismatch 真实分布）"""
    # 索引 events by ts
    evs = sorted(events, key=lambda e: e.get("ts", ""))
    matched = []
    for trg in triggers:
        ts_trg = trg.get("ts", "")
        if not ts_trg:
            continue
        try:
            dt_trg = datetime.fromisoformat(ts_trg.replace("Z", "+00:00"))
        except Exception:
            continue
        # 找最近一个 ts 在 [trg_ts, trg_ts + window_s] 内的 event
        best = None
        best_delta = None
        for ev in evs:
            try:
                dt_ev = datetime.fromisoformat(ev["ts"].replace("Z", "+00:00"))
            except Exception:
                continue
            delta = (dt_ev - dt_trg).total_seconds()
            if 0 <= delta <= window_s:
                if best_delta is None or delta < best_delta:
                    best = ev
                    best_delta = delta
        matched.append((trg, best, best_delta))
    return matched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals-dir", type=Path,
                        default=Path.home() / ".lnmem/data")
    parser.add_argument("--feedback-dir", type=Path,
                        default=Path.home() / ".lnmem/feedback")
    parser.add_argument("--days", nargs="+",
                        default=["2026-05-12", "2026-05-13", "2026-05-14",
                                 "2026-05-15", "2026-05-16"])
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/lnmem-team-2026-05-16/ab_replay_report.md"))
    args = parser.parse_args()

    print(f"加载 signals: {args.days}", file=sys.stderr)
    signals = load_signals(args.signals_dir, args.days)
    print(f"  total frames: {len(signals):,}", file=sys.stderr)

    print(f"加载 feedback events ...", file=sys.stderr)
    events = load_feedback_events(args.feedback_dir)
    print(f"  total events: {len(events)}", file=sys.stderr)

    # 模拟两路 trigger
    print("\n=== 模拟 A 路径（calibrated, threshold=0.95）===", file=sys.stderr)
    trg_cal = simulate_brow_burst(signals, "expression.brow_furrow", THR_CAL)
    print(f"  触发: {len(trg_cal)}", file=sys.stderr)

    print("\n=== 模拟 B 路径（raw, threshold=0.55）===", file=sys.stderr)
    trg_raw = simulate_brow_burst(signals, "raw_au.AU04", THR_RAW)
    print(f"  触发: {len(trg_raw)}", file=sys.stderr)

    # 重叠分析
    cal_fids = {(t["ts"][:19], t["frame_id"]) for t in trg_cal}
    raw_fids = {(t["ts"][:19], t["frame_id"]) for t in trg_raw}
    both = cal_fids & raw_fids
    only_cal = cal_fids - raw_fids
    only_raw = raw_fids - cal_fids

    # 各路 trigger 对应的 user feedback
    matched_cal = match_triggers_to_feedback(trg_cal, events)
    matched_raw = match_triggers_to_feedback(trg_raw, events)

    def label_dist(matched):
        cnt = Counter()
        mm_states = Counter()
        for trg, ev, _ in matched:
            if ev is None:
                cnt["no_feedback"] += 1
            else:
                label = (ev.get("user_response") or {}).get("label", "?")
                cnt[label] += 1
                if label == "mismatch":
                    mm_states[(ev.get("user_response") or {}).get("true_state", "?")] += 1
        return cnt, mm_states

    cal_labels, cal_mm = label_dist(matched_cal)
    raw_labels, raw_mm = label_dist(matched_raw)

    # 输出报告
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# ADR-005 A/B 离线 Dual Replay 对账报告\n\n")
        f.write(f"**生成时间**：{datetime.now().isoformat()}\n")
        f.write(f"**数据范围**：{args.days[0]} → {args.days[-1]}，共 {len(signals):,} 帧 + {len(events)} user events\n\n")
        f.write(f"## A 路径（calibrated）vs B 路径（raw）触发对比\n\n")
        f.write(f"| 路径 | trigger 字段 | 阈值 | 4 天触发数 | 每天平均 |\n")
        f.write(f"|---|---|---|---|---|\n")
        days_n = len(args.days)
        f.write(f"| A calibrated | expression.brow_furrow | {THR_CAL} | {len(trg_cal)} | {len(trg_cal)/days_n:.1f} |\n")
        f.write(f"| B raw | raw_au.AU04 | {THR_RAW} | {len(trg_raw)} | {len(trg_raw)/days_n:.1f} |\n\n")

        f.write(f"### 重叠分析\n\n")
        f.write(f"- 两路都触发: **{len(both)}** ({len(both)/max(len(cal_fids|raw_fids),1)*100:.1f}%)\n")
        f.write(f"- 仅 A 触发（calibrated 独有）: **{len(only_cal)}** — *可能是 calibration 假阳性*\n")
        f.write(f"- 仅 B 触发（raw 独有）: **{len(only_raw)}** — *raw 新增灵敏度*\n\n")

        f.write(f"## 对应 user feedback 分布对比\n\n")
        f.write(f"### A 路径 trigger 配对 user feedback 60s 窗内\n\n")
        f.write(f"| label | count |\n|---|---|\n")
        for k, v in sorted(cal_labels.items(), key=lambda x: -x[1]):
            f.write(f"| {k} | {v} |\n")
        f.write(f"\nmismatch 时 true_state 分布: {dict(cal_mm)}\n\n")

        f.write(f"### B 路径 trigger 配对 user feedback 60s 窗内\n\n")
        f.write(f"| label | count |\n|---|---|\n")
        for k, v in sorted(raw_labels.items(), key=lambda x: -x[1]):
            f.write(f"| {k} | {v} |\n")
        f.write(f"\nmismatch 时 true_state 分布: {dict(raw_mm)}\n\n")

        # 计算关键 KPI
        def match_rate(labels):
            total = sum(labels.values()) - labels.get("no_feedback", 0)
            return labels.get("match", 0) / max(total, 1) * 100
        def mismatch_rate(labels):
            total = sum(labels.values()) - labels.get("no_feedback", 0)
            return labels.get("mismatch", 0) / max(total, 1) * 100
        def coverage(labels, total_trg):
            return (total_trg - labels.get("no_feedback", 0)) / max(total_trg, 1) * 100

        f.write(f"## 关键 KPI 对比\n\n")
        f.write(f"| 指标 | A 路径 (calibrated) | B 路径 (raw) | 差异 |\n")
        f.write(f"|---|---|---|---|\n")
        f.write(f"| trigger 频率 | {len(trg_cal)/days_n:.1f}/天 | {len(trg_raw)/days_n:.1f}/天 | {(len(trg_raw)-len(trg_cal))/days_n:+.1f} |\n")
        f.write(f"| match 率（trigger 后 user 确认对） | {match_rate(cal_labels):.1f}% | {match_rate(raw_labels):.1f}% | {match_rate(raw_labels)-match_rate(cal_labels):+.1f}% |\n")
        f.write(f"| mismatch 率（trigger 后 user 纠正） | {mismatch_rate(cal_labels):.1f}% | {mismatch_rate(raw_labels):.1f}% | {mismatch_rate(raw_labels)-mismatch_rate(cal_labels):+.1f}% |\n")
        f.write(f"| 60s 内有 user response 覆盖率 | {coverage(cal_labels, len(trg_cal)):.1f}% | {coverage(raw_labels, len(trg_raw)):.1f}% | - |\n\n")

        f.write(f"## 结论方向\n\n")
        if len(trg_raw) > len(trg_cal):
            f.write(f"- raw 路径触发量更高（+{len(trg_raw)-len(trg_cal)}），可能更适合 demo / 数据采集；work 节奏可调高 raw 阈值至 0.6+\n")
        else:
            f.write(f"- raw 路径触发量更低（-{len(trg_cal)-len(trg_raw)}），work 节奏更安静\n")
        if match_rate(raw_labels) > match_rate(cal_labels):
            f.write(f"- raw 路径 match 率更高（+{match_rate(raw_labels)-match_rate(cal_labels):.1f}%）→ **LNN 思想正面证据**\n")
        elif match_rate(raw_labels) < match_rate(cal_labels):
            f.write(f"- raw 路径 match 率反而更低（-{match_rate(cal_labels)-match_rate(raw_labels):.1f}%）→ 需要更多数据或检查 trigger 时机是否变差\n")
        else:
            f.write(f"- 两路 match 率接近，需要更长观察期或多维度对比\n")

    # 终端摘要
    print(f"\n=========================", file=sys.stderr)
    print(f"📊 A/B Dual Replay 结果", file=sys.stderr)
    print(f"=========================", file=sys.stderr)
    print(f"A (calibrated):  {len(trg_cal):>4} 触发 / {len(trg_cal)/days_n:.1f}/天", file=sys.stderr)
    print(f"B (raw):         {len(trg_raw):>4} 触发 / {len(trg_raw)/days_n:.1f}/天", file=sys.stderr)
    print(f"两路共识:        {len(both):>4} ({len(both)/max(len(cal_fids|raw_fids),1)*100:.1f}%)", file=sys.stderr)
    print(f"仅 A 独有:       {len(only_cal):>4}", file=sys.stderr)
    print(f"仅 B 独有:       {len(only_raw):>4}", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"A match 率:      {match_rate(cal_labels):.1f}% ({cal_labels.get('match',0)} / {sum(cal_labels.values()) - cal_labels.get('no_feedback',0)})", file=sys.stderr)
    print(f"B match 率:      {match_rate(raw_labels):.1f}% ({raw_labels.get('match',0)} / {sum(raw_labels.values()) - raw_labels.get('no_feedback',0)})", file=sys.stderr)
    print(f"", file=sys.stderr)
    print(f"完整报告: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
