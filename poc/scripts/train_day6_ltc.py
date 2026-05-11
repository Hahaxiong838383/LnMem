#!/usr/bin/env python3
"""Day 6.5 重训：用 LTC 演化版 18-dim 信号训练。

P-1 LNN-FIRST 准则下，验证 LTC 个性化数据是否让 LNN 学到 per-persona 表情风格。

期望：val_acc 仍 >= 70%（不退化），LNN 学到真正的 per-persona 表情差异。
关键 live_verify v3 验证：vboss_007 微笑→positive，vboss_001 微笑→neutral。
"""
from __future__ import annotations

from pathlib import Path

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.train import train


def main() -> None:
    cfg = LNNTrainConfig(
        feature_dim=18,
        sequence_length=60,
        sequence_stride=30,
        batch_size=32,
        lr=1e-3,
        epochs=20,
        device="auto",
        signals_dir=Path("poc/data/synthetic/llm_v2_ltc"),
        out_dir=Path("poc/data/checkpoints/day6_ltc"),
    )
    print(f"[train-day6] LTC-evolved signals dir: {cfg.signals_dir}")
    print(f"[train-day6] checkpoint out: {cfg.out_dir}")
    print(f"[train-day6] feature_dim={cfg.feature_dim} epochs={cfg.epochs}")

    checkpoint = train(cfg)
    print(f"\n[train-day6] DONE. final_state_acc={checkpoint.final_state_acc:.3f} "
          f"final_boundary_f1={checkpoint.final_boundary_f1:.3f}")


if __name__ == "__main__":
    main()
