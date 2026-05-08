#!/usr/bin/env python3
"""人工验证 v2：专挑状态变化段落，对比模型预测 vs 真实标签。

重点看事件边界附近（状态切换处）模型的表现。

Usage:
    cd /tmp/lnmem-fix/LnMem
    PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/human_verify.py
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory

STATE_CN = {
    "low_energy": "低能量😴",
    "focused": "专注🎯",
    "distracted": "分心🫠",
    "fatigued": "疲劳😩",
    "emotional_positive": "积极😊",
    "emotional_negative": "消极😞",
    "neutral": "中性😐",
    "transitioning": "过渡🔄",
}


def find_transition_segments(
    path: Path, seq_len: int, max_search: int = 5000,
) -> list[int]:
    """找到包含状态变化的起始位置。"""
    states: list[str] = []
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i >= max_search:
                break
            row = json.loads(line.strip())
            states.append(row["ground_truth_state"])

    # 找状态切换点，然后取切换点前 10 帧作为窗口起始
    transitions = []
    for i in range(1, len(states)):
        if states[i] != states[i - 1] and i > 10 and i + seq_len - 10 <= len(states):
            transitions.append(i - 10)  # 让切换点在窗口中间偏前

    return transitions


def load_jsonl_segment(path: Path, start: int, length: int) -> list[dict]:
    rows = []
    with path.open("r") as f:
        for i, line in enumerate(f):
            if i < start:
                continue
            if i >= start + length:
                break
            rows.append(json.loads(line.strip()))
    return rows


def main() -> None:
    ckpt = torch.load(
        "poc/data/checkpoints/day5/checkpoint_day5.pt",
        weights_only=False,
    )
    cfg: LNNTrainConfig = ckpt["config"]
    p2i = ckpt["persona_to_idx"]

    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    state_model.load_state_dict(ckpt["state_model"])
    state_model.eval()

    boundary_model = factory.create_boundary_model(cfg)
    boundary_model.load_state_dict(ckpt["boundary_model"])
    boundary_model.eval()

    data_dir = cfg.signals_dir
    files = sorted(data_dir.glob("signals_*_day*.jsonl"))
    random.seed(123)
    random.shuffle(files)

    print("=" * 82)
    print("  LnMem Day 5 LNN — 人工验证 v2（专挑状态变化段落）")
    print("  重点看：切换处模型能否跟上节奏")
    print("=" * 82)

    shown = 0
    seg_len = 30  # 展示 30 帧，更紧凑

    for fpath in files:
        if shown >= 5:
            break

        name = fpath.stem
        parts = name.replace("signals_", "").rsplit("_day", 1)
        persona_id = parts[0]
        day_idx = int(parts[1]) if len(parts) > 1 else 0

        if persona_id not in p2i:
            continue

        # 找状态切换段
        transitions = find_transition_segments(fpath, cfg.sequence_length)
        if not transitions:
            continue

        start = random.choice(transitions)
        rows = load_jsonl_segment(fpath, start, cfg.sequence_length)
        if len(rows) < cfg.sequence_length:
            continue

        pidx = p2i[persona_id]
        vectors = torch.tensor(
            [r["vector"] for r in rows], dtype=torch.float32,
        ).unsqueeze(0)
        pidx_t = torch.tensor([pidx])

        with torch.no_grad():
            s_logits = state_model(vectors, pidx_t)
            b_logits = boundary_model(vectors, pidx_t)

        s_preds = s_logits.argmax(dim=-1).squeeze(0)
        b_probs = torch.sigmoid(b_logits.squeeze(-1).squeeze(0))

        # 只展示前 seg_len 帧（包含切换点）
        display_rows = rows[:seg_len]

        print(f"\n{'─' * 82}")
        print(f"  📂 {fpath.name}  👤 {persona_id}  📅 Day {day_idx}")

        # 找到这段里的活动类型
        activities = list(dict.fromkeys(r.get("activity", "?") for r in display_rows))
        print(f"  🏷️  活动: {' → '.join(activities)}")
        print(f"  {'帧':>5}  {'实际状态':<12} {'模型预测':<12} {'✓/✗':>3}  {'边界':>6}  活动")
        print(f"  {'─' * 72}")

        correct = 0
        prev_state = None
        for idx in range(len(display_rows)):
            row = display_rows[idx]
            true_state = row["ground_truth_state"]
            pred_idx = s_preds[idx].item()
            pred_state = _STATE_LABELS[pred_idx]
            match = "✓" if true_state == pred_state else "✗"
            if true_state == pred_state:
                correct += 1

            true_cn = STATE_CN.get(true_state, true_state)
            pred_cn = STATE_CN.get(pred_state, pred_state)

            # 标记状态切换
            transition_mark = ""
            if prev_state is not None and true_state != prev_state:
                transition_mark = "◀ 切换!"

            boundary_real = "🔸" if row.get("frame_idx", -1) == 0 else "  "
            boundary_pred = "🔹" if b_probs[idx].item() > 0.45 else "  "

            activity = row.get("activity", "")

            print(
                f"  {start + idx:>5}  {true_cn:<12} {pred_cn:<12} {match:>3}  "
                f"{boundary_real}{boundary_pred}  {activity:<12} {transition_mark}"
            )
            prev_state = true_state

        all_correct = (
            s_preds[:seg_len]
            == torch.tensor([
                list(_STATE_LABELS).index(r["ground_truth_state"])
                for r in display_rows
            ])
        ).sum().item()
        print(f"\n  📊 本段准确率: {all_correct}/{seg_len} = {100 * all_correct / seg_len:.1f}%")
        shown += 1

    print(f"\n{'=' * 82}")
    print("  图例: ✓=正确 ✗=错误 🔸=真实边界 🔹=预测边界 ◀切换!=状态变化处")
    print("  你来判断: 模型在状态切换处的反应速度和准确性是否合理？")
    print(f"{'=' * 82}")


if __name__ == "__main__":
    main()
