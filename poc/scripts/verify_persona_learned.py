#!/usr/bin/env python3
"""验证 LNN 是否真的学到了 per-persona 表情风格（Day 6.5 P0 关键验证）。

实验设计：
1. 构造一段相同的 18 维输入序列（模拟"持续微笑 5 秒"）
2. 用 vboss_007（外向）和 vboss_001（内敛）的 persona_idx 分别推理
3. 看模型输出的 state 概率分布是否不同

期望（LNN-FIRST 准则真正生效的标志）：
  vboss_007 在"看到笑"时，emotional_positive 概率应该高（外向爱笑 → 笑 = 真积极）
  vboss_001 在"看到笑"时，emotional_positive 概率应低（内敛 → 笑可能是社交礼貌而非积极）

如果两个 persona 给同样输入产生**显著不同**的状态分布，说明：
  ✅ LNN 通过 FiLM/τ-bias 学到了 per-persona 解码风格
  ✅ 不再是"全员一致的状态识别器"
  ✅ LNN 价值真正得到证明
"""
from __future__ import annotations

import os
from pathlib import Path

import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory


def load_model():
    candidates = [
        "poc/data/checkpoints/day6_ltc/checkpoint_day5.pt",
        "poc/data/checkpoints/day6_ltc/checkpoint_day6.pt",
        "poc/data/checkpoints/day5/checkpoint_day5.pt",
    ]
    ckpt_path = next((p for p in candidates if os.path.exists(p)), candidates[-1])
    print(f"  ⏳ 加载: {ckpt_path}")
    ckpt = torch.load(ckpt_path, weights_only=False)
    cfg: LNNTrainConfig = ckpt["config"]
    p2i = ckpt["persona_to_idx"]
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    state_model.load_state_dict(ckpt["state_model"])
    state_model.eval()
    return state_model, cfg, p2i


def make_smiling_sequence(seq_len: int = 60, smile: float = 0.8) -> torch.Tensor:
    """构造一段"持续微笑"输入：18 维全部归一到合理值。"""
    vec = torch.zeros(seq_len, 18)
    # face: eye_closure, pitch_norm, yaw_norm, gaze_stability
    vec[:, 0] = 0.15  # eye open
    vec[:, 1] = 0.05  # head pitch low
    vec[:, 2] = 0.05  # head yaw low
    vec[:, 3] = 0.85  # gaze stable
    # derived: fatigue, attention, pose, blink
    vec[:, 4] = 0.20  # low fatigue
    vec[:, 5] = 0.85  # high attention
    vec[:, 6] = 0.90  # pose steady
    vec[:, 7] = 0.0   # no blink burst
    # expression 6: smile, brow, cheek, lip_dep, jaw, mouth ← 这是关键！
    vec[:, 8] = smile        # smile_intensity
    vec[:, 9] = 0.05         # brow_furrow low
    vec[:, 10] = smile * 0.7 # cheek_raise (Duchenne 跟笑联动)
    vec[:, 11] = 0.05        # lip_corner_depress low
    vec[:, 12] = 0.10        # jaw_drop low
    vec[:, 13] = 0.20        # mouth_open_ratio low
    # audio: voice_present, utterance_density
    vec[:, 14] = 0.0   # 没说话（关键：测"安静的微笑"是否被识别为积极）
    vec[:, 15] = 0.0
    # context: known, focus
    vec[:, 16] = 1.0   # context known
    vec[:, 17] = 0.85  # focus stable
    return vec.unsqueeze(0)  # (1, T, 18)


def predict(state_model, vec, persona_idx):
    with torch.no_grad():
        logits = state_model(vec, torch.tensor([persona_idx]))
    last_logits = logits[0, -1, :]
    probs = torch.softmax(last_logits, dim=0)
    return probs


def show_distribution(label: str, probs: torch.Tensor) -> None:
    print(f"\n  {label}:")
    sorted_probs, sorted_idx = probs.sort(descending=True)
    for rank, (p, i) in enumerate(zip(sorted_probs[:5], sorted_idx[:5])):
        bar = "█" * int(p.item() * 30)
        print(f"    {rank+1}. {_STATE_LABELS[i.item()]:<22} {p.item():>6.1%} {bar}")


def main() -> None:
    print("\n" + "=" * 72)
    print("  Day 6.5 P0 验证：LNN 是否学到 per-persona 表情风格")
    print("=" * 72)

    state_model, cfg, p2i = load_model()
    print(f"  ✅ 模型加载完成（{len(p2i)} personas）")

    # 构造同一段"持续微笑 + 不说话"输入
    print("\n  📥 输入：持续微笑（smile=0.8）+ 不说话 + context_known")
    vec = make_smiling_sequence(seq_len=cfg.sequence_length, smile=0.8)

    # 测两个对比 persona
    test_personas = [
        ("vboss_001", "内敛/低 X / 高 N — 微笑可能是社交礼貌"),
        ("vboss_007", "外向/高 X / 低 N — 微笑 = 真积极"),
        ("vboss_006", "最克制 / τ_smile=18s — 笑得很慢很少"),
        ("vboss_028", "活泼 / τ_smile=1s — 反应最快"),
    ]

    results = []
    for pid, desc in test_personas:
        if pid not in p2i:
            print(f"  ⚠️  {pid} 不在 checkpoint，跳过")
            continue
        probs = predict(state_model, vec, p2i[pid])
        results.append((pid, desc, probs))

    # 对比展示
    for pid, desc, probs in results:
        show_distribution(f"{pid} — {desc}", probs)

    # 关键诊断：两个最对立的 persona 在"微笑+不说话"下 emotional_positive 的概率差
    if len(results) >= 2:
        print("\n  📊 关键对比（emotional_positive 概率）:")
        pos_idx = _STATE_LABELS.index("emotional_positive")
        for pid, desc, probs in results:
            print(f"    {pid:<14} P(positive) = {probs[pos_idx].item():>6.2%}")

        max_p = max(probs[pos_idx].item() for _, _, probs in results)
        min_p = min(probs[pos_idx].item() for _, _, probs in results)
        diff = max_p - min_p
        print(f"\n  ➡️  最大差异: {diff:.2%}")
        if diff > 0.10:
            print("  ✅ LNN 学到了 per-persona 解码风格（同输入不同输出 > 10pp）")
        elif diff > 0.03:
            print("  🟡 有差异但不显著（< 10pp）— LNN 可能未完全学透")
        else:
            print("  ❌ 差异极小（< 3pp）— LNN 没学到个性化，FiLM/τ-bias 失效")

    print("\n" + "=" * 72)


if __name__ == "__main__":
    main()
