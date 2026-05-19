#!/usr/bin/env python3
"""离线 replay 定位 brow_burst 不触发根因（ADR-005 V0.3.4 + 2026-05-16 川哥实测发现）

背景：
  demo 模式 raw 路径下，川哥皱眉 1144 帧、raw_AU04≥0.40 共 163 帧、最长连续 9 帧，
  按 demo brow_threshold=0.40 + window=5 应该触发 brow_burst 多次，但实际 0 次，
  9 次 trigger 全是 pose_held。

本脚本：
  纯 python 重跑 ProactiveQueryTrigger.observe（不依赖 daemon），逐帧记录决策路径，
  统计 face_detected 抖动 / brow_history 满窗次数 / 各通道 cooldown 阻挡分布，定位根因。

用法：
  PYTHONPATH=. ./.venv-poc-fresh/bin/python scripts/replay_trigger_diagnose.py \\
      --signals ~/.lnmem/data/signals_2026-05-16.jsonl \\
      --from-ts 2026-05-16T14:14:46
"""
from __future__ import annotations
import argparse
import json
from collections import Counter
from pathlib import Path

from poc.src.realtime.feedback.trigger import (
    ProactiveQueryTrigger,
    resolve_runtime_config,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--signals", required=True, type=Path)
    p.add_argument("--from-ts", required=True, help="ISO ts lower bound, e.g. 2026-05-16T14:14:46")
    p.add_argument("--to-ts", default="9999", help="ISO ts upper bound (exclusive)")
    p.add_argument("--demo", action="store_true",
                   help="强制 demo 配置（即使当前 trigger_mode=work 也跑 demo 阈值）")
    p.add_argument("--schmitt-ema", action="store_true",
                   help="TODO-1 (ADR-010): 启用 Schmitt+EMA 触发模式（α=0.40 + thr_high/low 数据驱动）")
    p.add_argument("--ema-alpha", type=float, default=None,
                   help="EMA α，默认走当前 mode profile（demo=0.40 / work=0.40）")
    p.add_argument("--thr-high", type=float, default=None,
                   help="Schmitt 进入阈，默认 demo=0.42 / work=0.55")
    p.add_argument("--thr-low", type=float, default=None,
                   help="Schmitt 退出阈，默认 demo=0.22 / work=0.30")
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    # G5 (P2) ADR-010 V0.3.9 codex 二审：改用 resolve_runtime_config() 取实时配置，
    # 不再读 module-level import-time 快照常量。这样脚本运行时与 daemon 看到的
    # mode/signal_path 永远一致，排障时不会因"脚本读 work、daemon 跑 demo"造成误判。
    runtime_cfg = resolve_runtime_config()
    print(
        f"# replay_trigger_diagnose — runtime config (resolve_runtime_config): "
        f"mode={runtime_cfg['mode']} signal_path={runtime_cfg['signal_path']}"
    )

    if args.demo:
        # 强制 demo 配置：直接取 demo profile（而非硬编码值，单一事实来源）
        from poc.src.realtime.feedback.trigger import _MODE_PROFILES
        demo_cfg = _MODE_PROFILES["demo"]
        brow_thr = demo_cfg["brow_burst_threshold_raw"]
        brow_window = demo_cfg["brow_burst_window"]
        pose_window = demo_cfg["pose_held_window"]
        pose_std = demo_cfg["pose_held_stddev_threshold_deg"]
        cd_brow = demo_cfg["cooldown_frames_brow"]
        cd_pose = demo_cfg["cooldown_frames_pose_held"]
        cd_vlm = demo_cfg["cooldown_frames_low_vlm_conf"]
        default_ema_alpha = demo_cfg["brow_ema_alpha_raw"]
        default_thr_hi = demo_cfg["brow_thr_high_raw"]
        default_thr_lo = demo_cfg["brow_thr_low_raw"]
        print(
            f"# FORCED DEMO config: brow_thr={brow_thr} window={brow_window} "
            f"pose_window={pose_window} cooldown_brow={cd_brow}"
        )
    else:
        # 跟随 runtime_cfg（resolve_runtime_config 已按当前 mode 选好 profile）
        brow_thr = runtime_cfg["brow_burst_threshold_raw"]
        brow_window = runtime_cfg["brow_burst_window"]
        pose_window = runtime_cfg["pose_held_window"]
        pose_std = runtime_cfg["pose_held_stddev_threshold_deg"]
        cd_brow = runtime_cfg["cooldown_frames_brow"]
        cd_pose = runtime_cfg["cooldown_frames_pose_held"]
        cd_vlm = runtime_cfg["cooldown_frames_low_vlm_conf"]
        default_ema_alpha = runtime_cfg["brow_ema_alpha_raw"]
        default_thr_hi = runtime_cfg["brow_thr_high_raw"]
        default_thr_lo = runtime_cfg["brow_thr_low_raw"]

    # TODO-1 (ADR-010): 可选 Schmitt+EMA 模式
    schmitt_kwargs: dict = {}
    if args.schmitt_ema:
        thr_hi = args.thr_high if args.thr_high is not None else default_thr_hi
        thr_lo = args.thr_low if args.thr_low is not None else default_thr_lo
        ema_alpha = args.ema_alpha if args.ema_alpha is not None else default_ema_alpha
        schmitt_kwargs = dict(
            use_schmitt_ema=True,
            brow_ema_alpha=ema_alpha,
            brow_thr_high=thr_hi,
            brow_thr_low=thr_lo,
        )
        print(f"# Schmitt+EMA ON: α={ema_alpha} thr_high={thr_hi} thr_low={thr_lo}")

    trig = ProactiveQueryTrigger(
        brow_threshold=brow_thr,
        brow_window=brow_window,
        pose_held_window=pose_window,
        pose_held_stddev_threshold_deg=pose_std,
        cooldown_frames_brow=cd_brow,
        cooldown_frames_pose_held=cd_pose,
        cooldown_frames_low_vlm_conf=cd_vlm,
        signal_path="raw",
        **schmitt_kwargs,
    )

    # 计数器
    stats: Counter = Counter()
    fires = []  # (ts, frame_id, reason)
    # brow 满窗候选追踪
    brow_full_window_count = 0      # _brow_history 长度等于 window 的帧数
    brow_all_above_count = 0        # 满窗且 all > thr（应触发，未必能 fire）
    brow_all_above_blocked_cd = 0   # 满窗 all > thr 但 in_cooldown
    # raw_AU04 时序的连续 > thr 串长度分布
    streak_now = 0
    streak_lens: list[int] = []
    streak_peaks: list[float] = []
    cur_streak_peak = 0.0
    # 跨 trigger.observe 后立刻问 _brow_history 状态
    last_brow_history_len = 0

    with args.signals.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            ts = item.get("ts", "")
            if ts < args.from_ts or ts >= args.to_ts:
                continue
            stats["total_frames"] += 1

            face = bool(item.get("face_detected"))
            if not face:
                stats["face_undetected"] += 1
            else:
                stats["face_detected"] += 1

            raw_au04 = item.get("raw_au", {}).get("AU04", 0.0)
            if raw_au04 >= brow_thr:
                stats[f"raw_AU04_ge_{brow_thr}"] += 1
            # 连续 > thr 串（用 strict > 对齐 trigger 判定）
            if raw_au04 > brow_thr:
                streak_now += 1
                cur_streak_peak = max(cur_streak_peak, raw_au04)
            else:
                if streak_now > 0:
                    streak_lens.append(streak_now)
                    streak_peaks.append(cur_streak_peak)
                streak_now = 0
                cur_streak_peak = 0.0

            # 调 observe 前看下 history 长度（debug）
            before_len = len(trig._brow_history)
            before_brow_cd_last = trig._last_trigger_frame.get("brow_burst")
            ev = trig.observe(item)
            after_len = len(trig._brow_history)
            last_brow_history_len = after_len

            # 调用后 brow_history 是否已满 + 全都过阈
            if after_len == brow_window and all(b > brow_thr for b in trig._brow_history):
                brow_all_above_count += 1
                # 如果没 fire brow_burst，必然是 cooldown 拦
                if ev is None or ev.reason != "brow_burst":
                    brow_all_above_blocked_cd += 1
            if after_len == brow_window:
                brow_full_window_count += 1

            # face_detected=False 让 brow_history 清空时，统计
            if not face and before_len > 0 and after_len == 0:
                stats["brow_history_cleared_by_face_undetected"] += 1

            if ev is not None:
                stats[f"fired_{ev.reason}"] += 1
                fires.append((ts[-13:-1], int(item.get("frame_id", -1)), ev.reason))

    # 报告
    print("\n## 帧统计")
    print(f"  total: {stats['total_frames']}  face_detected: {stats['face_detected']}  undetected: {stats['face_undetected']}")
    pct = (stats['face_undetected'] / max(stats['total_frames'], 1) * 100)
    print(f"  face_detected 抖动率: {pct:.1f}%")
    print(f"  raw_AU04>={brow_thr} 的帧: {stats[f'raw_AU04_ge_{brow_thr}']}")
    print(f"  brow_history 清窗（因 face_undetected）次数: {stats.get('brow_history_cleared_by_face_undetected', 0)}")

    # 末尾未结束的 streak 也算
    if streak_now > 0:
        streak_lens.append(streak_now)
        streak_peaks.append(cur_streak_peak)

    print("\n## raw_AU04 连续 strict>thr 串分布（窗口需 5 帧才能触发）")
    print(f"  总串数: {len(streak_lens)}")
    if streak_lens:
        from collections import Counter as _C
        bucket = _C()
        for ln in streak_lens:
            if ln >= 10: bucket[">=10"] += 1
            elif ln >= 5: bucket["5-9"] += 1
            elif ln >= 3: bucket["3-4"] += 1
            elif ln >= 2: bucket["2"] += 1
            else: bucket["1"] += 1
        for k in ["1", "2", "3-4", "5-9", ">=10"]:
            print(f"  长度 {k}: {bucket.get(k, 0)} 串")
        print(f"  最长串: {max(streak_lens)} 帧  peak: {max(streak_peaks):.3f}")
        # 列前 5 个最长串
        top_idx = sorted(range(len(streak_lens)), key=lambda i: -streak_lens[i])[:5]
        print("  最长 5 串详情:")
        for i in top_idx:
            print(f"    len={streak_lens[i]:<3} peak={streak_peaks[i]:.3f}")

    print("\n## brow_burst 触发链路诊断")
    print(f"  brow_history 满窗（len={brow_window}）次数: {brow_full_window_count}")
    print(f"  满窗 + all>{brow_thr}（理论触发条件）次数: {brow_all_above_count}")
    print(f"  其中被 cooldown 阻挡: {brow_all_above_blocked_cd}")
    print(f"  实际 fire brow_burst 次数: {stats.get('fired_brow_burst', 0)}")
    diff = brow_all_above_count - stats.get('fired_brow_burst', 0) - brow_all_above_blocked_cd
    print(f"  无法解释的差值: {diff} （若 >0 = trigger 逻辑还有未知 path）")

    print("\n## fire 序列（前 %d）" % args.top)
    for ts, fid, reason in fires[: args.top]:
        print(f"  {ts}  frame={fid:<6}  reason={reason}")

    print("\n## 触发类型分布")
    for k in sorted(stats.keys()):
        if k.startswith("fired_"):
            print(f"  {k}: {stats[k]}")


if __name__ == "__main__":
    main()
