#!/usr/bin/env python3
"""TODO-1 (ADR-010) — Schmitt + EMA + dynamic quantile threshold sweep.

目的：用历史 raw_au.AU04 信号数据驱动决定 brow_burst 触发的 3 个参数：
  1. EMA α  ∈ {0.30, 0.35, 0.40, 0.45, 0.50}
  2. thr_high（进入触发） = quantile P85 / P90 / P91 / P95 of EMA(raw_AU04)
  3. thr_low（退出滞回）  = thr_high * gap_ratio，gap_ratio ∈ {0.50, 0.55, 0.60}

输出：
  A. 全量数据 quantile 表（按 α 分组）—— 让阈值"不再硬编码"
  B. 模拟当前 demo session（638 帧，2026-05-16T14:14:46Z→T14:25:40Z）每组参数下触发数
  C. 推荐 default：在 demo session 中触发 2-6 次 且 全量 7 日 trigger 密度合理 的组合

用法：
  PYTHONPATH=. ./.venv-poc-fresh/bin/python scripts/sweep_brow_threshold_quantile.py \
      --signals-glob '~/.lnmem/data/signals_2026-05-*.jsonl' \
      --demo-from-ts 2026-05-16T14:14:46 \
      --demo-to-ts   2026-05-16T14:25:40

参考：
  - ADR-005（dual signal_path raw vs calibrated）
  - ADR-006（threshold-auto-recalibration 占位）
  - ADR-010（脉冲信号 vs strict-deque 不匹配根因）
  - ROADMAP TODO-1
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
from collections import deque
from pathlib import Path
from typing import Iterable


def iter_signals(paths: list[Path]) -> Iterable[dict]:
    for p in paths:
        with p.open() as f:
            for line in f:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def ema_series(values: list[float], alpha: float) -> list[float]:
    """计算 EMA 序列；首帧用首值初始化（与 trigger 实现一致）。"""
    out: list[float] = []
    prev: float | None = None
    for v in values:
        prev = v if prev is None else alpha * v + (1.0 - alpha) * prev
        out.append(prev)
    return out


def quantile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if q <= 0.0:
        return sorted_vals[0]
    if q >= 1.0:
        return sorted_vals[-1]
    idx = q * (len(sorted_vals) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return sorted_vals[lo]
    frac = idx - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def simulate_schmitt(
    values: list[float],
    alpha: float,
    thr_high: float,
    thr_low: float,
    min_samples: int = 5,
    cooldown_frames: int = 300,
) -> int:
    """模拟 Schmitt + EMA 触发，返回触发次数。

    语义同 trigger.py.ProactiveQueryTrigger 的 use_schmitt_ema=True 分支：
      - EMA 每帧更新；
      - 上升沿 ema≥thr_high 且未在 cooldown 内 → fire；fire 后 cooldown_frames 不再 fire；
      - 下降沿 ema≤thr_low 解除 above 状态（允许下一次上升沿）。
    """
    if not values:
        return 0
    fires = 0
    ema: float | None = None
    above = False
    samples_since_reset = 0
    last_fire_idx = -10**9
    for i, v in enumerate(values):
        ema = v if ema is None else alpha * v + (1.0 - alpha) * ema
        samples_since_reset += 1
        in_cd = (i - last_fire_idx) < cooldown_frames
        if not above and ema >= thr_high:
            above = True
            if samples_since_reset >= min_samples and not in_cd:
                fires += 1
                last_fire_idx = i
                samples_since_reset = 0
        elif above and ema <= thr_low:
            above = False
    return fires


def extract_raw_au04(items: Iterable[dict], ts_from: str = "", ts_to: str = "9999") -> list[float]:
    out: list[float] = []
    for it in items:
        ts = it.get("ts", "")
        if ts < ts_from or ts >= ts_to:
            continue
        if not it.get("face_detected"):
            continue
        raw_au = it.get("raw_au") or {}
        v = raw_au.get("AU04")
        if isinstance(v, (int, float)) and math.isfinite(v):
            out.append(float(v))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--signals-glob",
        default="~/.lnmem/data/signals_2026-05-*.jsonl",
        help="历史 signals JSONL glob（用于全量 quantile 计算）",
    )
    p.add_argument(
        "--demo-from-ts",
        default="2026-05-16T14:14:46",
        help="demo session 起始 ts（ADR-010 锚定窗口）",
    )
    p.add_argument(
        "--demo-to-ts",
        default="2026-05-16T14:25:40",
        help="demo session 终止 ts（exclusive）",
    )
    p.add_argument("--alphas", type=str, default="0.30,0.35,0.40,0.45,0.50")
    p.add_argument("--quantiles", type=str, default="0.85,0.90,0.91,0.95")
    p.add_argument("--gap-ratios", type=str, default="0.50,0.55,0.60")
    p.add_argument("--cooldown-frames", type=int, default=300, help="模拟 cooldown（与 work 模式一致）")
    p.add_argument("--min-samples", type=int, default=5, help="最少样本数（与 demo brow_window=5 一致）")
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    quantiles_list = [float(x) for x in args.quantiles.split(",")]
    gap_ratios = [float(x) for x in args.gap_ratios.split(",")]

    paths = sorted(Path(p_).expanduser() for p_ in glob.glob(os.path.expanduser(args.signals_glob)))
    print(f"# sweep_brow_threshold_quantile — files: {len(paths)}")
    for path in paths:
        print(f"#   {path.name}")

    # 全量数据：每个 α 下 EMA 之后的 raw_AU04 分布
    all_raw = extract_raw_au04(iter_signals(paths))
    print(f"\n## 全量 face_detected 帧数: {len(all_raw)}")
    if not all_raw:
        print("FATAL: no face_detected frames; nothing to sweep")
        return
    raw_sorted = sorted(all_raw)
    print("\n## raw_AU04 全量分位数（无 EMA，作为参考）")
    for q in quantiles_list:
        print(f"   P{int(q*100):02d} = {quantile(raw_sorted, q):.4f}")

    print("\n## 各 α 下 EMA(raw_AU04) 分位数（决定 thr_high 候选）")
    print(f"{'α':>6} | " + " ".join(f"P{int(q*100):02d}".rjust(8) for q in quantiles_list))
    print("-" * 60)
    alpha_to_quantiles: dict[float, dict[float, float]] = {}
    for a in alphas:
        emas = ema_series(all_raw, a)
        emas_sorted = sorted(emas)
        alpha_to_quantiles[a] = {q: quantile(emas_sorted, q) for q in quantiles_list}
        row = " ".join(f"{alpha_to_quantiles[a][q]:.4f}".rjust(8) for q in quantiles_list)
        print(f"{a:>6.2f} | {row}")

    # demo session
    demo_raw = extract_raw_au04(iter_signals(paths), ts_from=args.demo_from_ts, ts_to=args.demo_to_ts)
    print(f"\n## demo session ({args.demo_from_ts} → {args.demo_to_ts}) face_detected 帧数: {len(demo_raw)}")
    if not demo_raw:
        print("WARN: no demo frames; trigger-count table skipped")

    if demo_raw:
        print("\n## 模拟触发表（demo session, cooldown=%d, min_samples=%d）" % (args.cooldown_frames, args.min_samples))
        header = f"{'α':>6} {'P':>5} {'thr_hi':>8} {'gap':>5} {'thr_lo':>8} | {'demo':>5} {'full':>6}"
        print(header)
        print("-" * len(header))
        candidates: list[tuple[float, float, float, float, int, int]] = []
        for a in alphas:
            for q in quantiles_list:
                thr_hi = alpha_to_quantiles[a][q]
                for g in gap_ratios:
                    thr_lo = thr_hi * g
                    n_demo = simulate_schmitt(
                        demo_raw, a, thr_hi, thr_lo,
                        min_samples=args.min_samples,
                        cooldown_frames=args.cooldown_frames,
                    )
                    n_full = simulate_schmitt(
                        all_raw, a, thr_hi, thr_lo,
                        min_samples=args.min_samples,
                        cooldown_frames=args.cooldown_frames,
                    )
                    candidates.append((a, q, thr_hi, thr_lo, n_demo, n_full))
                    print(f"{a:>6.2f} {int(q*100):>5d} {thr_hi:>8.4f} {g:>5.2f} {thr_lo:>8.4f} | {n_demo:>5d} {n_full:>6d}")

        # 推荐：demo 触发 2-6 次 且 全量触发率合理（按 7 日 ~30K 帧粗算每日 1-10 次）
        print("\n## 推荐组合（demo 2≤fires≤6 且 1≤full≤80）")
        rec = [c for c in candidates if 2 <= c[4] <= 6 and 1 <= c[5] <= 80]
        if rec:
            # 优先取 demo 触发 3-5 次 + 全量较稀疏的组合
            rec.sort(key=lambda c: (abs(c[4] - 4), c[5]))
            for c in rec[:10]:
                a, q, hi, lo, nd, nf = c
                print(f"   α={a:.2f}  P={int(q*100):02d}  thr_high={hi:.4f}  thr_low={lo:.4f}  demo={nd}  full={nf}")
        else:
            print("   (无符合区间的组合，请放宽 quantile 或 gap_ratio)")


if __name__ == "__main__":
    main()
