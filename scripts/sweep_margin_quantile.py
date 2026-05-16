#!/usr/bin/env python3
"""Phase 4.1: 离线 margin / entropy 分位数 sweep（响应 codex F-04）

用最新 raw ckpt + 历史所有 signals 数据，跑 LNN forward，统计：
  - margin (top1 - top2) 分位数分布
  - entropy 分位数分布
  - 模拟不同 margin 阈值下每天 uncertainty trigger 次数

输出：数据驱动的 margin/entropy 阈值候选，替代 codex 否决的"拍脑袋 0.15"。
"""
from __future__ import annotations
import argparse
import json
import math
import sys
from collections import deque
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.online_adapter import OnlineLoRAAdapter
from poc.src.lnn.online_contracts import OnlineAdapterConfig
from poc.src.lnn.signals_loader import _signals_row_to_frame_dict
from poc.src.fusion import MultimodalFusion


def load_lnn(ckpt_path: Path):
    """与 spike 同套加载流程（F-01/F-02/F-03 修复）"""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    factory = DefaultLNNFactory()
    sm = factory.create_state_model(ckpt["config"])
    sm.load_state_dict(ckpt["state_model"])
    sm.eval()
    if ckpt.get("online_adapter_schema") == "phase2_v1" and "online_tau_drift" in ckpt:
        base = getattr(sm, "personalization", None)
        if base is not None:
            online = OnlineLoRAAdapter(base_adapter=base, online_config=OnlineAdapterConfig())
            online.tau_drift.data.copy_(ckpt["online_tau_drift"])
            sm.personalization = online
    return sm


def forward_window(sm, fusion_vecs: list[list[float]], persona_idx: int = 30):
    if len(fusion_vecs) < 60:
        return None
    x = torch.tensor(fusion_vecs[-60:], dtype=torch.float32).unsqueeze(0)
    pid = torch.tensor([persona_idx], dtype=torch.long)
    with torch.no_grad():
        logits = sm(x, pid)
    last = logits[0, -1, :]
    probs = torch.softmax(last, dim=-1).tolist()
    sp = sorted(probs, reverse=True)
    entropy = -sum(p * math.log2(max(p, 1e-12)) for p in probs)
    margin = sp[0] - sp[1]
    return {
        "margin": margin,
        "entropy": entropy,
        "top_prob": sp[0],
        "top_idx": int(last.argmax()),
    }


def pct(xs, p):
    return sorted(xs)[min(int(len(xs) * p / 100), len(xs) - 1)] if xs else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path,
                        default=REPO_ROOT / "poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_20260516T103643Z.pt")
    parser.add_argument("--days", nargs="+",
                        default=["2026-05-12", "2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"])
    parser.add_argument("--stride", type=int, default=60,
                        help="每 N 帧采样一次（默认 60s 一次，加速）")
    parser.add_argument("--out", type=Path,
                        default=Path("/tmp/lnmem-team-2026-05-16/margin_sweep_report.md"))
    args = parser.parse_args()

    print(f"[sweep] loading ckpt: {args.ckpt.name}", file=sys.stderr)
    sm = load_lnn(args.ckpt)
    fusion = MultimodalFusion()

    margins, entropies, top_probs = [], [], []
    top_idx_counts = {}
    window = deque(maxlen=60)
    total_frames = 0
    forwards = 0

    for day in args.days:
        f = Path.home() / f".lnmem/data/signals_{day}.jsonl"
        if not f.exists():
            continue
        print(f"[sweep] processing {day}...", file=sys.stderr)
        with f.open() as fh:
            for i, line in enumerate(fh):
                try:
                    row = json.loads(line)
                except: continue
                total_frames += 1
                try:
                    fd = _signals_row_to_frame_dict(row)
                    fused = fusion.fuse(fd)
                    if len(fused.vector) != 18:
                        continue
                    window.append(fused.vector)
                except Exception:
                    continue
                # 每 stride 帧 forward 一次（节省时间）
                if len(window) >= 60 and i % args.stride == 0:
                    result = forward_window(sm, list(window))
                    if result:
                        margins.append(result["margin"])
                        entropies.append(result["entropy"])
                        top_probs.append(result["top_prob"])
                        idx = result["top_idx"]
                        top_idx_counts[idx] = top_idx_counts.get(idx, 0) + 1
                        forwards += 1

    print(f"\n[sweep] total frames: {total_frames:,}, forwards: {forwards:,}", file=sys.stderr)

    # 分位数报告
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        f.write(f"# Phase 4.1 Margin / Entropy 分位数 Sweep 报告\n\n")
        f.write(f"**ckpt**: `{args.ckpt.name}`\n")
        f.write(f"**days**: {args.days}\n")
        f.write(f"**total frames**: {total_frames:,}\n")
        f.write(f"**forwards (stride={args.stride})**: {forwards:,}\n\n")

        f.write(f"## Margin 分布\n\n")
        f.write(f"| 分位数 | 值 |\n|---|---|\n")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            f.write(f"| P{p} | {pct(margins, p):.4f} |\n")
        f.write(f"| min | {min(margins):.4f} |\n| max | {max(margins):.4f} |\n")
        f.write(f"| mean | {sum(margins)/len(margins):.4f} |\n\n")

        f.write(f"## Entropy 分布\n\n")
        f.write(f"| 分位数 | 值 |\n|---|---|\n")
        for p in [5, 10, 25, 50, 75, 90, 95, 99]:
            f.write(f"| P{p} | {pct(entropies, p):.4f} |\n")
        f.write(f"| min | {min(entropies):.4f} |\n| max | {max(entropies):.4f} |\n")
        f.write(f"| mean | {sum(entropies)/len(entropies):.4f} |\n\n")

        # 模拟不同 margin 阈值的触发率
        f.write(f"## 阈值 sweep（margin < thr 视为不确定，可触发 uncertainty trigger）\n\n")
        days_n = len(args.days)
        f.write(f"| margin 阈值 | 命中 forwards | 命中率 | 推断/天 | 对应分位 |\n|---|---|---|---|---|\n")
        for thr in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
            hits = sum(1 for m in margins if m < thr)
            rate = hits / len(margins) * 100
            # 模拟每天触发：stride=60 → 每 forwards 个 ≈ 60s 一次，"hits / days" 估"每天 hits 个不确定点"
            per_day_est = hits / days_n
            # 对应分位
            mq = sum(1 for m in margins if m < thr) / len(margins) * 100
            f.write(f"| {thr:.2f} | {hits} | {rate:.1f}% | {per_day_est:.0f} | P{mq:.0f} |\n")

        # 模拟 entropy 阈值
        f.write(f"\n## Entropy 阈值 sweep（entropy > thr 视为不确定）\n\n")
        max_h = math.log2(8)
        f.write(f"| entropy 阈值 | 命中 forwards | 命中率 | 推断/天 | 对应分位 |\n|---|---|---|---|---|\n")
        for thr in [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 2.8]:
            hits = sum(1 for e in entropies if e > thr)
            rate = hits / len(entropies) * 100
            per_day_est = hits / days_n
            eq = sum(1 for e in entropies if e <= thr) / len(entropies) * 100
            f.write(f"| {thr:.2f} (max={max_h:.2f}) | {hits} | {rate:.1f}% | {per_day_est:.0f} | P{eq:.0f} |\n")

        # top_idx 分布
        f.write(f"\n## LNN 预测 state 分布（全 forwards）\n\n")
        f.write(f"| state | count | % |\n|---|---|---|\n")
        for idx, cnt in sorted(top_idx_counts.items(), key=lambda x: -x[1]):
            f.write(f"| {STATE_LABELS[idx]} | {cnt} | {cnt/forwards*100:.1f}% |\n")

        # 推荐阈值（数据驱动）
        f.write(f"\n## 推荐阈值（数据驱动 + work 节奏 ≤3/day）\n\n")
        # work 节奏：≤3 uncertainty trigger/day → margin 阈值选 P≤(3 * days / forwards * 100)
        target_per_day = 3
        target_pct = target_per_day * days_n / forwards * 100
        margin_thr_work = pct(margins, target_pct)
        entropy_thr_work = pct(entropies, 100 - target_pct)
        f.write(f"- **work 节奏目标**：≤ {target_per_day} uncertainty/天\n")
        f.write(f"- **推荐 margin 阈值**：{margin_thr_work:.4f} (P{target_pct:.1f})\n")
        f.write(f"- **推荐 entropy 阈值**：{entropy_thr_work:.4f}\n")
        # demo 节奏
        target_per_day_demo = 20
        target_pct_demo = target_per_day_demo * days_n / forwards * 100
        f.write(f"\n- **demo 节奏目标**：≤ {target_per_day_demo} uncertainty/天\n")
        f.write(f"- **推荐 margin 阈值（demo）**：{pct(margins, target_pct_demo):.4f}\n")
        f.write(f"- **推荐 entropy 阈值（demo）**：{pct(entropies, 100 - target_pct_demo):.4f}\n")

    # 终端摘要
    print(f"\n=========================", file=sys.stderr)
    print(f"📊 Margin / Entropy Sweep", file=sys.stderr)
    print(f"=========================", file=sys.stderr)
    print(f"forwards: {forwards:,}", file=sys.stderr)
    print(f"margin  P50={pct(margins,50):.4f} P25={pct(margins,25):.4f} P10={pct(margins,10):.4f}", file=sys.stderr)
    print(f"entropy P50={pct(entropies,50):.4f} P75={pct(entropies,75):.4f} P90={pct(entropies,90):.4f}", file=sys.stderr)
    target_pct_work = 3 * len(args.days) / forwards * 100
    print(f"\nwork 推荐 margin thr: {pct(margins, target_pct_work):.4f} (≤3 trigger/天)", file=sys.stderr)
    print(f"work 推荐 entropy thr: {pct(entropies, 100 - target_pct_work):.4f}", file=sys.stderr)
    print(f"\n完整报告: {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
