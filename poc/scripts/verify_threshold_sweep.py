#!/usr/bin/env python3
"""Task D: Boundary threshold sweep — 找最优 sigmoid 阈值提升 F1.

Usage:
    cd /tmp/lnmem-fix/LnMem
    PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/verify_threshold_sweep.py
"""
from __future__ import annotations

import torch
import numpy as np

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import JsonlSignalsLoader
from poc.src.lnn.model import DefaultLNNFactory


def main() -> None:
    # 加载 checkpoint
    ckpt = torch.load(
        "poc/data/checkpoints/day5/checkpoint_day5.pt",
        weights_only=False,
    )
    cfg: LNNTrainConfig = ckpt["config"]
    print(f"[D] checkpoint loaded, {len(ckpt['persona_to_idx'])} personas")

    # 重建 val DataLoader
    loader = JsonlSignalsLoader()
    _, val_dl, _ = loader.build_loaders(cfg)
    print(f"[D] val DataLoader: {len(val_dl)} batches")

    # 重建 boundary model + 加载权重
    factory = DefaultLNNFactory()
    bm = factory.create_boundary_model(cfg)
    bm.load_state_dict(ckpt["boundary_model"])
    bm.eval()

    # 收集所有 raw logits 和 ground truth
    all_logits: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in val_dl:
            x, _state, pidx, boundary = batch
            b_logits = bm(x, pidx)  # (B, T, 1)
            all_logits.append(b_logits.squeeze(-1).cpu())  # (B, T)
            all_targets.append(boundary.cpu())  # (B, T)

    logits = torch.cat([l.flatten() for l in all_logits])
    targets = torch.cat([t.flatten() for t in all_targets])
    probs = torch.sigmoid(logits)

    n_pos = (targets == 1).sum().item()
    n_neg = (targets == 0).sum().item()
    print(f"[D] total frames={len(targets):,}, positive={n_pos:,} ({100*n_pos/len(targets):.1f}%), negative={n_neg:,}")

    # 阈值扫描
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0.0
    best_thr = 0.5

    print(f"\n{'threshold':>10} {'precision':>10} {'recall':>8} {'F1':>8} {'TP':>8} {'FP':>8} {'FN':>8}")
    print("-" * 72)

    for thr in thresholds:
        preds = (probs >= thr).float()
        tp = ((preds == 1) & (targets == 1)).sum().item()
        fp = ((preds == 1) & (targets == 0)).sum().item()
        fn = ((preds == 0) & (targets == 1)).sum().item()

        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)

        marker = " ★" if f1 > best_f1 else ""
        print(f"{thr:10.2f} {precision:10.4f} {recall:8.4f} {f1:8.4f} {tp:8.0f} {fp:8.0f} {fn:8.0f}{marker}")

        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print("-" * 72)
    print(f"\n[D] ★ 最优阈值: {best_thr:.2f} → F1={best_f1:.4f}")
    print(f"[D] 对比默认阈值 0.50: 见表中 threshold=0.50 行")
    print("[SWEEP DONE]")


if __name__ == "__main__":
    main()
