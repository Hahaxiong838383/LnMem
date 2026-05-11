#!/usr/bin/env python3
"""离线回放 trigger 决策 — V0.2 vs V0.3 Phase B 对比。

用历史 signals_*.jsonl 重跑 ProactiveQueryTrigger，
评估不同阈值/通道配置下的触发分布（次数 / 通道占比 / 间隔 / 多样性）。

用法：
    .venv-poc-fresh/bin/python3 scripts/replay_triggers.py \\
        --signals ~/.lnmem/data \\
        --out /tmp/phase_b_replay_report.md

不副作用：纯读 signals jsonl，不写 feedback queries。
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# 加入 repo root 到 sys.path（脚本可直接跑）
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger  # noqa: E402


def load_signals(signals_dir: Path) -> list[dict]:
    """加载 signals_*.jsonl 并按 (date, frame_id) 排序。"""
    records: list[dict] = []
    for fp in sorted(signals_dir.glob("signals_*.jsonl")):
        with fp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    records.sort(key=lambda r: (r.get("ts", ""), r.get("frame_id", 0)))
    return records


def make_trigger_v02() -> ProactiveQueryTrigger:
    """V0.2 配置：brow 0.85 / 10 帧；无 pose_held；无 cooldown。

    F2-04 (codex 一审)：用 enable_pose_held=False 真实禁用 pose 通道，
    不再用 window=10**9 这种参数技巧（语义不等价 + 仍维护 deque）。
    """
    return ProactiveQueryTrigger(
        brow_threshold=0.85,
        brow_window=10,
        vlm_conf_threshold=0.65,
        enable_pose_held=False,
        cooldown_frames_brow=0,
        cooldown_frames_pose_held=0,
        cooldown_frames_low_vlm_conf=0,
    )


def make_trigger_v03() -> ProactiveQueryTrigger:
    """V0.3 Phase B 默认配置（trigger.py 模块级常量）。"""
    return ProactiveQueryTrigger()


def replay(trigger: ProactiveQueryTrigger, records: list[dict]) -> dict:
    """跑一遍 trigger，返回统计。"""
    events: list = []
    for r in records:
        ev = trigger.observe(r)
        if ev is not None:
            events.append(ev)

    by_reason: Counter = Counter(ev.reason for ev in events)

    # 按日期分桶（看每日分布）
    by_day: dict[str, Counter] = defaultdict(Counter)
    for ev in events:
        day = ev.ts[:10] if ev.ts else "unknown"
        by_day[day][ev.reason] += 1

    # 触发间隔（frame_id 差）
    intervals: list[int] = []
    for prev, curr in zip(events, events[1:]):
        intervals.append(curr.frame_id - prev.frame_id)

    if intervals:
        intervals_sorted = sorted(intervals)
        n = len(intervals_sorted)
        median = intervals_sorted[n // 2]
        p25 = intervals_sorted[n // 4]
        p75 = intervals_sorted[(3 * n) // 4]
        mean = sum(intervals) / n
    else:
        median = p25 = p75 = 0
        mean = 0.0

    # diversity = unique reason 数（弹窗多样性指标）
    diversity = len(by_reason)

    return {
        "total_events": len(events),
        "by_reason": dict(by_reason),
        "by_day": {d: dict(c) for d, c in sorted(by_day.items())},
        "interval_p25_frames": p25,
        "interval_median_frames": median,
        "interval_p75_frames": p75,
        "interval_mean_frames": mean,
        "diversity_unique_reasons": diversity,
        "events_sample_first_5": [
            {"ts": ev.ts, "fid": ev.frame_id, "reason": ev.reason}
            for ev in events[:5]
        ],
        "events_sample_last_5": [
            {"ts": ev.ts, "fid": ev.frame_id, "reason": ev.reason}
            for ev in events[-5:]
        ],
    }


def render_markdown(v02: dict, v03: dict, n_records: int, signals_dir: Path) -> str:
    """渲染对比报告。"""
    lines: list[str] = []
    lines.append("# Phase B 离线回放报告 — V0.2 vs V0.3")
    lines.append("")
    lines.append(f"**生成时间**：{datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**signals 目录**：`{signals_dir}`")
    lines.append(f"**总帧数**：{n_records}（约 {n_records / 3600:.1f}h @ 1fps）")
    lines.append("")
    lines.append("## 概览对比")
    lines.append("")
    lines.append("| 维度 | V0.2 (brow 0.85+10/无 pose/无 cooldown) | V0.3 Phase B (brow 0.95+15/pose_held+cooldown) |")
    lines.append("|------|-----------------------------------------|-----------------------------------------------|")
    lines.append(f"| 总触发次数 | {v02['total_events']} | {v03['total_events']} |")
    lines.append(f"| 触发频率（每帧/min @ 1fps） | 1/{n_records // max(v02['total_events'], 1)} 帧 ≈ {n_records / max(v02['total_events'], 1) / 60:.1f}min | 1/{n_records // max(v03['total_events'], 1)} 帧 ≈ {n_records / max(v03['total_events'], 1) / 60:.1f}min |")
    lines.append(f"| 触发通道多样性 (unique reasons) | {v02['diversity_unique_reasons']} | {v03['diversity_unique_reasons']} |")
    lines.append(f"| 触发间隔中位数 (帧) | {v02['interval_median_frames']} | {v03['interval_median_frames']} |")
    lines.append(f"| 触发间隔 P25/P75 (帧) | {v02['interval_p25_frames']} / {v02['interval_p75_frames']} | {v03['interval_p25_frames']} / {v03['interval_p75_frames']} |")
    lines.append("")

    lines.append("## V0.2 触发通道分布")
    lines.append("")
    total_v02 = max(v02["total_events"], 1)
    for reason, cnt in sorted(v02["by_reason"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{reason}`: {cnt} ({100 * cnt / total_v02:.1f}%)")
    lines.append("")

    lines.append("## V0.3 Phase B 触发通道分布")
    lines.append("")
    total_v03 = max(v03["total_events"], 1)
    for reason, cnt in sorted(v03["by_reason"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- `{reason}`: {cnt} ({100 * cnt / total_v03:.1f}%)")
    lines.append("")

    lines.append("## 每日触发分布")
    lines.append("")
    lines.append("### V0.2")
    for day, c in v02["by_day"].items():
        total = sum(c.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        lines.append(f"- {day}: total={total} ({breakdown})")
    lines.append("")
    lines.append("### V0.3 Phase B")
    for day, c in v03["by_day"].items():
        total = sum(c.values())
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(c.items()))
        lines.append(f"- {day}: total={total} ({breakdown})")
    lines.append("")

    lines.append("## 解读")
    lines.append("")
    delta_total = v03["total_events"] - v02["total_events"]
    delta_pct = 100 * delta_total / max(v02["total_events"], 1)
    lines.append(f"- **触发量变化**：{delta_total:+d} ({delta_pct:+.1f}%)。V0.3 通过 brow 调严（0.85→0.95）+ cooldown 抑制连发，整体频率下降；同时 pose_held 通道补充新触发场景。")
    if v03["diversity_unique_reasons"] > v02["diversity_unique_reasons"]:
        lines.append(f"- **多样性提升**：通道数 {v02['diversity_unique_reasons']} → {v03['diversity_unique_reasons']}，弹窗 reason 不再单一，UX 痛点'都是皱眉 1.0'缓解。")
    else:
        lines.append(f"- **多样性持平/退化**：通道数 {v02['diversity_unique_reasons']} → {v03['diversity_unique_reasons']}，需检查 pose_held 阈值是否过严。")
    lines.append("")

    lines.append("## 抽样事件（V0.3 前 5 后 5）")
    lines.append("")
    lines.append("```json")
    for ev in v03["events_sample_first_5"]:
        lines.append(json.dumps(ev, ensure_ascii=False))
    lines.append("...")
    for ev in v03["events_sample_last_5"]:
        lines.append(json.dumps(ev, ensure_ascii=False))
    lines.append("```")
    lines.append("")

    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description="V0.2 vs V0.3 Phase B trigger 离线回放对比")
    p.add_argument("--signals", type=Path, default=Path.home() / ".lnmem" / "data",
                   help="signals_*.jsonl 所在目录（默认 ~/.lnmem/data）")
    p.add_argument("--out", type=Path, default=Path("/tmp/phase_b_replay_report.md"),
                   help="markdown 报告输出路径")
    args = p.parse_args()

    if not args.signals.is_dir():
        print(f"❌ signals dir not found: {args.signals}", file=sys.stderr)
        return 2

    print(f"loading signals from {args.signals}...", file=sys.stderr)
    records = load_signals(args.signals)
    print(f"loaded {len(records)} records", file=sys.stderr)
    if not records:
        print("❌ no signals to replay", file=sys.stderr)
        return 3

    print("replaying V0.2 config...", file=sys.stderr)
    v02_stats = replay(make_trigger_v02(), records)
    print(f"  V0.2: {v02_stats['total_events']} events", file=sys.stderr)

    print("replaying V0.3 Phase B config...", file=sys.stderr)
    v03_stats = replay(make_trigger_v03(), records)
    print(f"  V0.3: {v03_stats['total_events']} events", file=sys.stderr)

    md = render_markdown(v02_stats, v03_stats, len(records), args.signals)
    args.out.write_text(md, encoding="utf-8")
    print(f"✅ report written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
