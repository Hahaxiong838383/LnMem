#!/usr/bin/env python3
"""Trace Schmitt state machine per-frame state transitions（ADR-010 debug）

输出每一帧的：
  ts / raw / ema / above / samples_since_reset / cooldown_active / fire
专门定位"为什么 EMA 跌到 thr_low 解锁后再上升却不再 fire"的 bug。
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--signals", required=True, type=Path)
    p.add_argument("--from-ts", required=True)
    p.add_argument("--to-ts", default="9999")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--max-rows", type=int, default=80, help="只打印 above 变化或 fire 附近的帧")
    args = p.parse_args()

    if args.demo:
        cfg = dict(brow_threshold=0.40, brow_window=5,
                   pose_held_window=20, pose_held_stddev_threshold_deg=3.5,
                   cooldown_frames_brow=30, cooldown_frames_pose_held=30,
                   cooldown_frames_low_vlm_conf=30,
                   use_schmitt_ema=True, brow_ema_alpha=0.40,
                   brow_thr_high=0.42, brow_thr_low=0.22,
                   signal_path="raw")

    trig = ProactiveQueryTrigger(**cfg)
    rows = []
    last_above = False
    last_fire_fid = None
    cooldown_n = trig._cooldown_frames["brow_burst"]

    with args.signals.open() as f:
        for line in f:
            try:
                item = json.loads(line)
            except Exception:
                continue
            ts = item.get("ts", "")
            if ts < args.from_ts or ts >= args.to_ts:
                continue
            fid = int(item.get("frame_id", -1))
            raw = item.get("raw_au", {}).get("AU04", 0.0)
            face = bool(item.get("face_detected"))

            before_above = trig._brow_above_high
            before_samples = trig._brow_samples_since_reset
            in_cd_before = (last_fire_fid is not None and fid - last_fire_fid < cooldown_n)

            ev = trig.observe(item)

            after_above = trig._brow_above_high
            after_samples = trig._brow_samples_since_reset
            ema = trig._brow_ema
            fired = (ev is not None and ev.reason == "brow_burst")
            if fired:
                last_fire_fid = fid

            # 标注重要事件：above 翻转 / fire / cooldown 边界
            interest = (before_above != after_above) or fired or (ev is not None)
            rows.append((ts[-13:-1], fid, raw, ema, after_above, after_samples, in_cd_before, fired, ev.reason if ev else None, interest))

    # 只打印 interest 帧 + 第一次 latch 后 N 帧 trace
    print(f"trace 总帧数: {len(rows)}  cooldown={cooldown_n}")
    print(f"  ts          fid     raw    ema    above samples in_cd fire reason")
    printed = 0
    first_latch_fid = None
    for r in rows:
        if r[-1] or (first_latch_fid is not None and r[1] - first_latch_fid < 35):
            if r[7] and first_latch_fid is None:
                first_latch_fid = r[1]
            ema_str = f"{r[3]:.3f}" if r[3] is not None else "None "
            mark = "🔥" if r[7] else ("⬆" if r[4] and not rows[max(0,rows.index(r)-1)][4] else "")
            print(f"  {r[0]}  {r[1]:<6} {r[2]:.3f}  {ema_str}  {str(r[4]):<5} {r[5]:<7} {str(r[6]):<5} {str(r[7]):<5} {r[8] or '':<10} {mark}")
            printed += 1
            if printed >= args.max_rows: break


if __name__ == "__main__":
    main()
