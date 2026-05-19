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

V0.3.7 ADR-010 codex F1 (P0) 修复（2026-05-17）：
  - 废弃原 simulate_schmitt 自写状态机（与 trigger.observe 语义不一致：
    缺 window_stable / latch 时机不同 / cooldown 内不 latch 等差异 → 阈值推荐系统性偏差）。
  - 改用 count_brow_burst_fires() 直接调 ProactiveQueryTrigger(use_schmitt_ema=True).observe()
    对每帧回放，与生产 trigger 状态机 1:1 同构。
  - 一致性回归测试见 poc/tests/test_feedback_trigger.py::TestSweepConsistency。

参考：
  - ADR-005（dual signal_path raw vs calibrated）
  - ADR-006（threshold-auto-recalibration 占位）
  - ADR-010（脉冲信号 vs strict-deque 不匹配根因 + F1 sweep-trigger 同构）
  - ROADMAP TODO-1
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import os
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


def _make_trigger(alpha, thr_high, thr_low, brow_window, cooldown_frames):
    """构造隔离 brow 通道的 ProactiveQueryTrigger（共享于两个 count_* 入口）。"""
    from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger
    return ProactiveQueryTrigger(
        brow_window=brow_window,
        signal_path="raw",
        use_schmitt_ema=True,
        brow_ema_alpha=alpha,
        brow_thr_high=thr_high,
        brow_thr_low=thr_low,
        cooldown_frames_brow=cooldown_frames,
        # - low_vlm_conf：item.vlm_label=None → 不进通道
        # - pose_held：enable_pose_held=False → 永远 max_std=None 不触发
        enable_pose_held=False,
    )


def count_brow_burst_fires_from_items(
    items: Iterable[dict],
    alpha: float,
    thr_high: float,
    thr_low: float,
    brow_window: int = 5,
    cooldown_frames: int = 300,
) -> int:
    """G1 (P0) ADR-010 V0.3.8 codex 二审修复（2026-05-17）：
    sweep 与生产 trigger 真正 1:1 同构必须喂**完整 items**，不能预压扁为 float。

    之前 `count_brow_burst_fires(values)` 把 no-face / NaN 帧剔除后再喂 trigger，
    每帧都是 face_detected=True → 抹掉了生产链路在 no-face/假阳性帧执行的
    `_brow_ema/_brow_above_high/_brow_samples_since_reset` 状态重置（trigger.py:520-527），
    以及 frame_id 真实跨度（影响 cooldown）。codex 最小复现：truth=0 sweep=1 等。

    本函数喂原 items（含 face_detected / raw_au / frame_id / ts），与生产 daemon
    通过 SignalPipeline 喂 trigger.observe 的链路完全一致。

    items: 完整 signal items 迭代器（不要先过滤 face_detected/NaN）
    返回：brow_burst 触发次数
    """
    trig = _make_trigger(alpha, thr_high, thr_low, brow_window, cooldown_frames)
    fires = 0
    for it in items:
        # 兜底 frame_id：缺失则按 enumerate 顺序模拟（避免 rollback 误检）
        ev = trig.observe(it)
        if ev is not None and ev.reason == "brow_burst":
            fires += 1
    return fires


def count_brow_burst_fires(
    values: list[float],
    alpha: float,
    thr_high: float,
    thr_low: float,
    brow_window: int = 5,
    cooldown_frames: int = 300,
) -> int:
    """**测试 / 合成数据专用**：把 list[float] 当作"所有帧都 face_detected=True"喂 trigger。

    F1 (P0) 原始入口；保留向后兼容 + 单元测试用（人工构造的纯净序列）。
    **生产数据 sweep 必须用 `count_brow_burst_fires_from_items` 喂完整 items**，
    否则会因 sweep 预过滤 no-face/NaN 帧而抹掉真 trigger 的状态重置语义
    （详见 codex 二审 G1 / ADR-010 V0.3.8）。
    """
    if not values:
        return 0
    items = [
        {
            "ts": "",
            "frame_id": i,
            "face_detected": True,
            "raw_au": {"AU04": float(v)},
            "expression": {},
            "vlm_label": None,
        }
        for i, v in enumerate(values)
    ]
    return count_brow_burst_fires_from_items(
        items, alpha, thr_high, thr_low,
        brow_window=brow_window,
        cooldown_frames=cooldown_frames,
    )


def extract_raw_au04(items: Iterable[dict], ts_from: str = "", ts_to: str = "9999") -> list[float]:
    """抽出有效 raw_AU04 浮点序列（剔 no-face / NaN），用于 quantile 计算。

    **注意**：此返回值仅用于 `ema_series` + quantile，不要再喂 trigger（会丢失
    no-face/NaN 帧的状态重置语义；G1 fix 改用 `materialize_items_for_sweep`）。
    """
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


def materialize_items_for_sweep(
    items: Iterable[dict],
    ts_from: str = "",
    ts_to: str = "9999",
) -> list[dict]:
    """G1 (P0) ADR-010 V0.3.8 codex 二审修复：

    sweep 现在喂 trigger 完整 items（保留 no-face / NaN 帧的状态重置语义），
    本函数把 jsonl 流物化成 trigger 可消费的轻量 dict 列表（只保留触发判定
    所需字段，省内存）：
      - ts: 用于 ts_from/ts_to 过滤（保留以匹配真 trigger 输入 schema）
      - frame_id: 用于 cooldown / rollback 检测
      - face_detected: bool（关键：no-face=False 触发 trigger 内部清状态）
      - raw_au: {AU04: float|NaN|None}（保留 NaN 以触发 _safe_float 路径）
      - expression: {}（走 raw 路径不读）
      - vlm_label: None（关闭 low_vlm_conf 通道）

    7 日全量 ~250K 帧物化后 ~25MB，离线 sweep 可接受。
    """
    out: list[dict] = []
    for it in items:
        ts = it.get("ts", "")
        if ts < ts_from or ts >= ts_to:
            continue
        raw_au_src = it.get("raw_au") if isinstance(it.get("raw_au"), dict) else {}
        au04 = raw_au_src.get("AU04") if raw_au_src else None
        out.append({
            "ts": ts,
            "frame_id": it.get("frame_id", 0),
            "face_detected": bool(it.get("face_detected")),
            # 保留 AU04 原值（含 NaN/None），让 trigger 内 _safe_float 走真路径
            "raw_au": {"AU04": au04} if au04 is not None else {},
            "expression": {},
            "vlm_label": None,
        })
    return out


def per_session_fires_stats(
    session_values: dict[str, list],
    alpha: float,
    thr_high: float,
    thr_low: float,
    brow_window: int,
    cooldown_frames: int,
    *,
    items_mode: bool | None = None,
) -> dict:
    """F2 (P1) ADR-010 codex review: 跨 session 触发数分布指标。

    G1 fix（codex 二审 V0.3.8）：支持两种 session 数据模式
      - `items_mode=True` 或自动探测 dict：value=完整 items list（保留 no-face/NaN）
        → 调 count_brow_burst_fires_from_items（生产数据用此路径）
      - `items_mode=False` 或自动探测 float：value=float list（合成数据 / 单测用）
        → 调 count_brow_burst_fires（向后兼容）

    返回字段：
      - per_session: [(name, fires), ...]
      - mean / std / cv / min / max / sessions_touched / n_sessions
    """
    fires_list: list[tuple[str, int]] = []
    for name, vals in session_values.items():
        # 自动探测：list 第一个元素若是 dict → items_mode；否则 float mode
        use_items = items_mode if items_mode is not None else (
            bool(vals) and isinstance(vals[0], dict)
        )
        if use_items:
            n = count_brow_burst_fires_from_items(
                vals, alpha, thr_high, thr_low,
                brow_window=brow_window,
                cooldown_frames=cooldown_frames,
            )
        else:
            n = count_brow_burst_fires(
                vals, alpha, thr_high, thr_low,
                brow_window=brow_window,
                cooldown_frames=cooldown_frames,
            )
        fires_list.append((name, n))
    counts = [n for _, n in fires_list]
    if not counts:
        return {
            "per_session": [], "mean": 0.0, "std": 0.0, "cv": float("inf"),
            "min": 0, "max": 0, "sessions_touched": 0, "n_sessions": 0,
        }
    n = len(counts)
    mean = sum(counts) / n
    var = sum((c - mean) ** 2 for c in counts) / n
    std = math.sqrt(var)
    cv = (std / mean) if mean > 0 else float("inf")
    return {
        "per_session": fires_list,
        "mean": mean,
        "std": std,
        "cv": cv,
        "min": min(counts),
        "max": max(counts),
        "sessions_touched": sum(1 for c in counts if c >= 1),
        "n_sessions": n,
    }


def leave_one_session_out_quantiles(
    session_values: dict[str, list[float]],
    alpha: float,
    quantile_list: list[float],
) -> dict[str, dict[float, float]]:
    """F2 (P1) leave-one-session-out: 对每个 session 留出做 validation，
    其他 sessions union 上计算 EMA quantile 作为 thr_high 候选。

    G2 (P1) ADR-010 V0.3.8 codex 二审修复（2026-05-17）：
      - **按 session 单独 ema_series**（每 session 首帧重置）→ 不让上一 session
        末尾的高 EMA "漂入" 下一 session 开头（生产 daemon 跨天 reset 同语义）。
      - **names 排序**（sorted 而非 dict.keys 顺序）→ 让结果与字典插入顺序解耦，
        保证 "同数据不同文件顺序得到相同阈值" 的可复现性。

    session_values: dict[session_name, list[float]]（仅 face_detected + 有限 raw_AU04）
    返回 {validation_session_name: {q: thr_hi}}。
    """
    out: dict[str, dict[float, float]] = {}
    names = sorted(session_values.keys())  # G2: 排序保证可复现
    for held_out in names:
        # G2: 每 session 单独 EMA（首帧 reset），再 concat 计算 quantile
        train_emas: list[float] = []
        for name in names:
            if name == held_out:
                continue
            train_emas.extend(ema_series(session_values[name], alpha))
        if not train_emas:
            out[held_out] = {q: 0.0 for q in quantile_list}
            continue
        emas_sorted = sorted(train_emas)
        out[held_out] = {q: quantile(emas_sorted, q) for q in quantile_list}
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
    # F2 (P1) ADR-010 codex review: 跨 session 稳健性参数
    p.add_argument(
        "--per-session-stats",
        action="store_true",
        default=True,
        help="按 file=session 切分计算跨 session 触发分布（mean/std/cv）",
    )
    p.add_argument(
        "--recommend-mean-min", type=float, default=2.0,
        help="推荐组合要求：跨 session 触发数 mean ≥ 此值",
    )
    p.add_argument(
        "--recommend-mean-max", type=float, default=8.0,
        help="推荐组合要求：跨 session 触发数 mean ≤ 此值",
    )
    p.add_argument(
        "--recommend-cv-max", type=float, default=1.0,
        help="推荐组合要求：触发数变异系数 std/mean ≤ 此值（越小越稳健）",
    )
    p.add_argument(
        "--recommend-min-touched-ratio", type=float, default=0.5,
        help="推荐组合要求：至少在此比例的 session 中触发 ≥ 1 次",
    )
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    quantiles_list = [float(x) for x in args.quantiles.split(",")]
    gap_ratios = [float(x) for x in args.gap_ratios.split(",")]

    paths = sorted(Path(p_).expanduser() for p_ in glob.glob(os.path.expanduser(args.signals_glob)))
    print(f"# sweep_brow_threshold_quantile — files: {len(paths)}")
    for path in paths:
        print(f"#   {path.name}")

    # G1 (P0) ADR-010 V0.3.8 codex 二审修复：
    #   sweep 必须喂完整 items 给 trigger（保留 no-face / NaN 帧的状态重置语义），
    #   不能预压扁为 float。下面 quantile 用 floats 路径不变（quantile 不需 frame schema），
    #   触发计数路径改用 items_full / items_demo。
    all_items_full = materialize_items_for_sweep(iter_signals(paths))
    all_raw = extract_raw_au04(all_items_full)
    print(f"\n## 全量 face_detected 帧数: {len(all_raw)} (raw items: {len(all_items_full)})")
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
    # G1: 触发计数喂完整 items；quantile 显示仍用过滤后的 floats（仅做表头/概览）
    items_demo = materialize_items_for_sweep(
        iter_signals(paths), ts_from=args.demo_from_ts, ts_to=args.demo_to_ts
    )
    demo_raw = extract_raw_au04(items_demo)
    print(
        f"\n## demo session ({args.demo_from_ts} → {args.demo_to_ts}) "
        f"face_detected 帧数: {len(demo_raw)} (raw items: {len(items_demo)})"
    )
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
                    # G1: 喂完整 items，含 no-face/NaN 状态重置
                    n_demo = count_brow_burst_fires_from_items(
                        items_demo, a, thr_hi, thr_lo,
                        brow_window=args.min_samples,
                        cooldown_frames=args.cooldown_frames,
                    )
                    n_full = count_brow_burst_fires_from_items(
                        all_items_full, a, thr_hi, thr_lo,
                        brow_window=args.min_samples,
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

    # F2 (P1) ADR-010 codex review: 跨 session 稳健性分析
    # 旧推荐基于单 demo session（638 帧）过拟合 → 升级为按 file=session 切分，
    # 报告每个候选 (α, q, gap) 的跨 session 触发分布 mean/std/cv，并据此推荐。
    if args.per_session_stats and len(paths) >= 2:
        # G1 + G3 (P0/P1) ADR-010 V0.3.8 codex 二审修复：
        #   - session_items: 完整 items（喂 trigger 用，保留 no-face/NaN 状态重置）
        #   - session_floats: 过滤后的 floats（quantile 计算用）
        #   - total_files vs valid_files 双口径报告，避免静默剔除空 session 抬高 touched_ratio
        print(f"\n## F2: 跨 session 切分（按 file，共 {len(paths)} sessions）")
        session_items: dict[str, list[dict]] = {}
        session_floats: dict[str, list[float]] = {}
        for path in paths:
            its = materialize_items_for_sweep(iter_signals([path]))
            vals = extract_raw_au04(its)
            if its:
                session_items[path.name] = its
            if vals:
                session_floats[path.name] = vals
            print(f"   {path.name}: {len(vals)} face_detected 帧 / {len(its)} raw items")
        total_files = len(paths)
        valid_files = len(session_floats)
        if valid_files >= 2:
            print(
                "\n## F2: 跨 session 触发分布 + leave-one-session-out 验证 "
                f"(cooldown={args.cooldown_frames}, brow_window={args.min_samples}, "
                f"valid_files={valid_files}, total_files={total_files})"
            )
            print(
                f"{'α':>6} {'P':>4} {'gap':>5} {'thr_hi':>8} {'thr_lo':>8} | "
                f"{'mean':>6} {'std':>6} {'cv':>6} {'min':>4} {'max':>4} "
                f"{'cov_v':>5} {'cov_t':>5} | {'loso_hi':>8}"
            )
            print("-" * 110)
            f2_candidates: list[dict] = []
            for a in alphas:
                # union EMA quantile（注意：union path 保留旧语义做对照；
                # LOSO 走 G2 修复后的 per-session ema）
                emas_union = ema_series(all_raw, a)
                emas_sorted = sorted(emas_union)
                # G2: LOSO 内部已改 per-session EMA
                loso = leave_one_session_out_quantiles(session_floats, a, quantiles_list)
                for q in quantiles_list:
                    thr_hi_union = quantile(emas_sorted, q)
                    for g in gap_ratios:
                        thr_lo_union = thr_hi_union * g
                        # G1: per_session_fires_stats 自动探测 items_mode（dict → items 路径）
                        stats = per_session_fires_stats(
                            session_items, a, thr_hi_union, thr_lo_union,
                            brow_window=args.min_samples,
                            cooldown_frames=args.cooldown_frames,
                        )
                        # LOSO: 每个 held-out session 用其他 union 的 quantile 当 thr_hi
                        loso_fires: list[int] = []
                        for held_out in sorted(session_items.keys()):
                            thr_hi_loso = loso[held_out][q]
                            thr_lo_loso = thr_hi_loso * g
                            loso_fires.append(count_brow_burst_fires_from_items(
                                session_items[held_out], a, thr_hi_loso, thr_lo_loso,
                                brow_window=args.min_samples,
                                cooldown_frames=args.cooldown_frames,
                            ))
                        loso_mean = sum(loso_fires) / len(loso_fires) if loso_fires else 0.0
                        # G3 (P1): 双口径
                        touched_ratio_valid = (
                            stats["sessions_touched"] / stats["n_sessions"]
                            if stats["n_sessions"] > 0 else 0.0
                        )
                        touched_ratio_total = (
                            stats["sessions_touched"] / total_files
                            if total_files > 0 else 0.0
                        )
                        cv_disp = "inf" if math.isinf(stats["cv"]) else f"{stats['cv']:.2f}"
                        print(
                            f"{a:>6.2f} {int(q*100):>4d} {g:>5.2f} {thr_hi_union:>8.4f} "
                            f"{thr_lo_union:>8.4f} | "
                            f"{stats['mean']:>6.2f} {stats['std']:>6.2f} {cv_disp:>6} "
                            f"{stats['min']:>4d} {stats['max']:>4d} "
                            f"{touched_ratio_valid*100:>4.0f}% {touched_ratio_total*100:>4.0f}% | "
                            f"{loso_mean:>8.2f}"
                        )
                        f2_candidates.append({
                            "alpha": a, "q": q, "gap": g,
                            "thr_hi": thr_hi_union, "thr_lo": thr_lo_union,
                            "stats": stats,
                            "touched_ratio_valid": touched_ratio_valid,
                            "touched_ratio_total": touched_ratio_total,
                            "loso_mean": loso_mean,
                        })

            # F2 推荐：跨 session mean ∈ [mean_min, mean_max] + cv ≤ cv_max + touched_total ≥ ratio
            # G3: 推荐准则用 touched_ratio_total（按 total_files 分母）避免静默抬高
            print(
                f"\n## F2: 跨 session 稳健推荐"
                f"（mean ∈ [{args.recommend_mean_min}, {args.recommend_mean_max}], "
                f"cv ≤ {args.recommend_cv_max}, "
                f"touched_total ≥ {args.recommend_min_touched_ratio*100:.0f}%）"
            )
            robust = [
                c for c in f2_candidates
                if (
                    args.recommend_mean_min <= c["stats"]["mean"] <= args.recommend_mean_max
                    and (not math.isinf(c["stats"]["cv"]))
                    and c["stats"]["cv"] <= args.recommend_cv_max
                    and c["touched_ratio_total"] >= args.recommend_min_touched_ratio
                )
            ]
            if robust:
                # 优先：cv 最小 → loso_mean 离区间中点最近 → 触发到的 session 最多
                target_mean = (args.recommend_mean_min + args.recommend_mean_max) / 2
                robust.sort(key=lambda c: (
                    c["stats"]["cv"],
                    abs(c["loso_mean"] - target_mean),
                    -c["touched_ratio_total"],
                ))
                for c in robust[:10]:
                    s = c["stats"]
                    cv_disp = "inf" if math.isinf(s["cv"]) else f"{s['cv']:.2f}"
                    print(
                        f"   α={c['alpha']:.2f}  P={int(c['q']*100):02d}  gap={c['gap']:.2f}  "
                        f"thr_high={c['thr_hi']:.4f}  thr_low={c['thr_lo']:.4f}  "
                        f"mean={s['mean']:.2f}  cv={cv_disp}  "
                        f"touched(valid/total)={c['touched_ratio_valid']*100:.0f}%/{c['touched_ratio_total']*100:.0f}%  "
                        f"loso_mean={c['loso_mean']:.2f}"
                    )
            else:
                print(
                    "   (无满足跨 session 稳健性的组合；考虑放宽 --recommend-* 阈值，"
                    "或采集更多 session 后重跑)"
                )


if __name__ == "__main__":
    main()
