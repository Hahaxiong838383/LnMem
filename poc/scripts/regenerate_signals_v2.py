#!/usr/bin/env python3
"""Day 6.5 重新生成 18-dim 信号（LTC 表情演化版）。

读取 poc/data/synthetic/llm_v2/script_*.json 剧本（900 个），
对每个剧本：
  1. 加载对应 persona 的 ExpressionProfile
  2. 用 ExpressionDynamics LTC 演化器替代静态 hint
  3. 落盘到 poc/data/synthetic/llm_v2_ltc/signals_<persona>_day<N>.jsonl

P-1 LNN-FIRST 验证点：
- 信号文件包含的 6 维表情维度是 ODE 演化输出，不是静态采样
- 不同 persona 的同一天剧本，会产生不同的表情轨迹

用法：
  PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/regenerate_signals_ltc.py
"""
from __future__ import annotations

import time
from pathlib import Path

from poc.src.script_to_signals import script_file_to_signals_file


def main() -> None:
    poc_root = Path(__file__).resolve().parents[1]
    script_dir = poc_root / "data/synthetic/llm_v2"
    profile_dir = poc_root / "data/personas/expression_profiles"
    out_dir = poc_root / "data/synthetic/llm_v2_ltc_v2"

    out_dir.mkdir(parents=True, exist_ok=True)

    scripts = sorted(script_dir.glob("script_*.json"))
    print(f"[regen-ltc] {len(scripts)} scripts to process")
    print(f"[regen-ltc] profile_dir: {profile_dir}")
    print(f"[regen-ltc] out_dir:     {out_dir}")

    n_done = 0
    n_with_profile = 0
    t0 = time.time()
    for i, sp in enumerate(scripts, 1):
        # script_vboss_007_day13.json → signals_vboss_007_day13.jsonl
        out_name = sp.stem.replace("script_", "signals_") + ".jsonl"
        out_path = out_dir / out_name

        # check if profile exists
        # parse "vboss_007" from "script_vboss_007_day13"
        stem_parts = sp.stem.split("_")
        persona_id = "_".join(stem_parts[1:3])  # "vboss_007"
        profile_path = profile_dir / f"{persona_id}.json"
        has_profile = profile_path.exists()
        if has_profile:
            n_with_profile += 1

        try:
            n = script_file_to_signals_file(
                script_path=sp,
                out_path=out_path,
                frame_rate_hz=1.0,
                noise_sigma=0.05,
                seed=hash(sp.stem) & 0xFFFFFFFF,
                profile_dir=profile_dir,
            )
            n_done += 1
            if i % 100 == 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                eta = (len(scripts) - i) / rate
                print(f"  [{i}/{len(scripts)}] {sp.stem} → {n} frames "
                      f"(profile: {'✓' if has_profile else '✗'}) "
                      f"[{rate:.1f}/s, ETA {eta:.0f}s]")
        except Exception as e:
            print(f"  [{i}/{len(scripts)}] {sp.stem}: ❌ {e}")

    elapsed = time.time() - t0
    print(f"\n[regen-ltc] done: {n_done}/{len(scripts)} success, "
          f"{n_with_profile} with LTC profile, {elapsed:.0f}s")


if __name__ == "__main__":
    main()
