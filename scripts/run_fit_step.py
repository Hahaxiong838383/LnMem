#!/usr/bin/env python3
"""Phase 2 P0-3 fit_step CLI 入口（M0 W2 2026-05-12）。

用法：

    # 1) 检查 trigger 条件（min_total ≥10 AND mismatch ≥10），满足才跑
    .venv-poc-fresh/bin/python3 scripts/run_fit_step.py

    # 2) 强制 dry-run（跳过阈值，mismatch < 10 时也跑一次验证管线）
    .venv-poc-fresh/bin/python3 scripts/run_fit_step.py --force

    # 3) 指定 ckpt + 输出目录
    .venv-poc-fresh/bin/python3 scripts/run_fit_step.py \\
        --ckpt-in poc/data/checkpoints/day12_with_self/checkpoint_day12.pt \\
        --ckpt-out-dir poc/data/checkpoints/m0w2_phase2 \\
        --persona-idx 30

退出码（F4-01 codex 一审：cron 友好语义，"未触发但正常"归 0）：
- 0 = triggered=True 或 reason 合理未触发（below_threshold/no_feedback/no_batch/no_mismatch）
- 2 = error（reason=error / no_ckpt / adapter_mismatch）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.fit_orchestrator import try_fit_step  # noqa: E402
from poc.src.lnn.online_contracts import OnlineAdapterConfig  # noqa: E402


_OK_TRIGGERED = 0
_OK_NOT_FIRED = 0  # F4-01 (codex 一审): cron 友好，未触发但正常归 0
_ERROR = 2


def main() -> int:
    p = argparse.ArgumentParser(description="Phase 2 P0-3 fit_step CLI 入口")
    p.add_argument(
        "--feedback-dir", type=Path,
        default=Path.home() / ".lnmem" / "feedback",
        help="反馈 jsonl 所在目录（默认 ~/.lnmem/feedback）",
    )
    p.add_argument(
        "--signals-dir", type=Path,
        default=Path.home() / ".lnmem" / "data",
        help="signals jsonl 所在目录（默认 ~/.lnmem/data）",
    )
    p.add_argument(
        "--ckpt-in", type=Path,
        default=REPO_ROOT / "poc" / "data" / "checkpoints" / "day12_with_self" / "checkpoint_day12.pt",
        help="输入 ckpt 路径（默认 day12_with_self/checkpoint_day12.pt）",
    )
    p.add_argument(
        "--ckpt-out-dir", type=Path,
        default=REPO_ROOT / "poc" / "data" / "checkpoints" / "m0w2_phase2",
        help="输出 ckpt 目录（默认 poc/data/checkpoints/m0w2_phase2）",
    )
    p.add_argument(
        "--persona-idx", type=int, default=30,
        help="所有反馈视为来自此 persona（默认 30 = vboss_self）",
    )
    p.add_argument(
        "--force", action="store_true",
        help="跳过 trigger_if_ready 阈值（dry-run 用）",
    )
    p.add_argument(
        "--alignment-window-s", type=float, default=5.0,
        help="user_triggered alignment 硬过滤窗口（秒，默认 5.0）",
    )
    p.add_argument(
        "--min-event-ts", type=str, default=None,
        help=(
            "F-03 fix (codex 二审 2026-05-16): 增量水位线，仅 fit ts > 此 ISO8601 字符串的 events。"
            "通常传上次 fit 后 ckpt 的 phase2_metadata['fired_at_utc']，防 cron 反复训练同一批"
            "历史数据导致漂移偏置。"
        ),
    )
    args = p.parse_args()

    print(f"feedback_dir:  {args.feedback_dir}", file=sys.stderr)
    print(f"signals_dir:   {args.signals_dir}", file=sys.stderr)
    print(f"ckpt_in:       {args.ckpt_in}", file=sys.stderr)
    print(f"ckpt_out_dir:  {args.ckpt_out_dir}", file=sys.stderr)
    print(f"persona_idx:   {args.persona_idx} (vboss_self if 30)", file=sys.stderr)
    print(f"force:         {args.force}", file=sys.stderr)
    print(f"min_event_ts:  {args.min_event_ts or '<none, full feedback>'}", file=sys.stderr)
    print("", file=sys.stderr)

    result = try_fit_step(
        feedback_dir=args.feedback_dir,
        signals_dir=args.signals_dir,
        ckpt_in_path=args.ckpt_in,
        ckpt_out_dir=args.ckpt_out_dir,
        persona_idx=args.persona_idx,
        force=args.force,
        alignment_window_s=args.alignment_window_s,
        min_event_ts=args.min_event_ts,
    )

    # 结构化 stdout（便于 jq / 监控管道）
    report = {
        "triggered": result.triggered,
        "reason": result.reason,
        "counts": result.counts,
        "metrics": result.metrics,
        "ckpt_in_sha8": result.ckpt_in_sha8,
        "ckpt_out_path": str(result.ckpt_out_path) if result.ckpt_out_path else None,
        "error": result.error,
        "metadata": result.metadata,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))

    if result.triggered:
        print(f"\n✅ fit_step fired. ckpt_out={result.ckpt_out_path}", file=sys.stderr)
        return _OK_TRIGGERED

    if result.reason in ("below_threshold", "no_feedback", "no_batch", "no_mismatch"):
        print(f"\nℹ️  not fired: {result.reason}", file=sys.stderr)
        return _OK_NOT_FIRED

    print(f"\n❌ error: {result.reason} — {result.error}", file=sys.stderr)
    return _ERROR


if __name__ == "__main__":
    sys.exit(main())
