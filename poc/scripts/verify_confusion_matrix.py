#!/usr/bin/env python3
"""Verify Day 5 checkpoint with 8-class confusion matrix on val set."""
from __future__ import annotations

from pathlib import Path

import torch

from poc.src.lnn.dataset import JsonlSignalsLoader, _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory


def _safe_load_checkpoint(path: Path) -> dict:
    """Load checkpoint dict with compatibility for different torch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _resolve_runtime_device(device_name: str) -> str:
    """Resolve runtime device with graceful fallback when target is unavailable."""
    if device_name == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    if device_name == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return device_name


def main() -> None:
    """Load Day 5 checkpoint and print val confusion matrix + per-class metrics."""
    ckpt_path = Path("poc/data/checkpoints/day5/checkpoint_day5.pt")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    # 1) 加载 checkpoint
    checkpoint = _safe_load_checkpoint(ckpt_path)
    config = checkpoint["config"]

    # 2) 重建 val DataLoader（沿用 checkpoint config）
    loader = JsonlSignalsLoader()
    _, val_dl, _ = loader.build_loaders(config)

    # 3) 重建模型并在 val 上推理，收集 (pred, true)
    runtime_device = _resolve_runtime_device(config.resolved_device())
    device = torch.device(runtime_device)
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(config)
    state_model.load_state_dict(checkpoint["state_model"])
    state_model.to(device)
    state_model.eval()

    num_classes = len(_STATE_LABELS)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.long)

    with torch.no_grad():
        for x, state, pidx, _boundary in val_dl:
            x = x.to(device)
            pidx = pidx.to(device)
            logits = state_model(x, pidx)  # (B, T, 8)
            pred = logits.argmax(dim=-1).cpu().reshape(-1)
            true = state.reshape(-1).cpu().to(torch.long)

            for t, p in zip(true, pred):
                confusion[t.item(), p.item()] += 1

    # 4) 打印 8x8 混淆矩阵（行=真实，列=预测）
    print("# 8-class Confusion Matrix (rows=true, cols=pred)")
    header = "true\\pred".ljust(24) + " ".join(f"{i:>8d}" for i in range(num_classes))
    print(header)
    for i, label in enumerate(_STATE_LABELS):
        row_vals = " ".join(f"{int(v):>8d}" for v in confusion[i].tolist())
        print(f"{label[:22].ljust(24)}{row_vals}")

    # 5) 打印 per-class precision / recall / F1
    print("\n# Per-class Metrics")
    print(f"{'class':<24}{'precision':>12}{'recall':>12}{'f1':>12}{'support':>12}")

    conf_f = confusion.to(torch.float64)
    col_sum = conf_f.sum(dim=0)
    row_sum = conf_f.sum(dim=1)
    diag = conf_f.diag()

    for i, label in enumerate(_STATE_LABELS):
        tp = diag[i].item()
        fp = (col_sum[i] - diag[i]).item()
        fn = (row_sum[i] - diag[i]).item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        support = int(row_sum[i].item())
        print(f"{label:<24}{precision:>12.4f}{recall:>12.4f}{f1:>12.4f}{support:>12d}")

    # 6) 结尾 summary 标记
    total = int(confusion.sum().item())
    correct = int(confusion.diag().sum().item())
    acc = (correct / total) if total > 0 else 0.0
    print("\n## Summary")
    print(f"device={runtime_device}, val_frames={total}, correct={correct}, accuracy={acc:.4f}")


if __name__ == "__main__":
    main()
