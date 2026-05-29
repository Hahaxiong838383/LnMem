"""ADR-040 动作 3 诊断: lag ablation + window sweep

回答核心问题: 当前 online intent 路径的有效感受野多长? intent 任务依不依赖 >8s 历史?
不改架构, 用 daemon 实际 ckpt + 现有 intent_feedback 数据 (au_window).

实验 1 · lag ablation:
  对每条 au_window(18-dim AU 序列, 长度 T), 保留最近 k 帧, 把更早 prefix 替换为
  persistence(第 T-k 帧值). forward 对比 full vs ablated 的 state_probs / intent_probs
  L1 距离. 若 k 很小时 diff 已≈0 → 有效感受野 ≤ k 帧, 任务不依赖长历史.

实验 2 · window sweep:
  把序列截断到最近 w 帧 forward, 对比 full. 测 w=2/4/8/12/16 对预测的影响.
  (数据 au_window 最长 ~23 帧, 测不到 60, 但能回答 "8 帧够不够").

用法: .venv-poc-fresh/bin/python scripts/diag_horizon_ablation.py
输出: /tmp/horizon_ablation_result.md + stdout 表格
"""
from __future__ import annotations

import json
import glob
import os
from pathlib import Path

import torch
import torch.nn.functional as F

from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer, INTENT_LABELS

DAEMON_CKPT = Path(os.path.expanduser("~/.lnmem/ckpt/lnn_backbone_phase2_latest.pt"))
INTENT_CKPT = Path(os.path.expanduser("~/.lnmem/ckpt/intent_head_latest.pt"))
FEEDBACK_GLOB = os.path.expanduser("~/.lnmem/data/intent_feedback/*.jsonl")
PERSONA = 30
MIN_LEN = 8  # 只用长度 >= 8 的样本(保证能测 k 到 8)


def load_samples() -> list[list[list[float]]]:
    """读 au_window, 返回 [ [18-dim]*T, ... ] (只保留 T>=MIN_LEN)."""
    out = []
    for f in glob.glob(FEEDBACK_GLOB):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            aw = d.get("au_window") or []
            seq = []
            for fr in aw:
                au = fr.get("au")
                if isinstance(au, list) and len(au) == 18:
                    seq.append([float(x) for x in au])
            if len(seq) >= MIN_LEN:
                out.append(seq)
    return out


def forward_probs(inf: RealtimeLNNInferencer, seq: list[list[float]]):
    """forward 一条序列, 返回 (state_probs[8], intent_probs[3] or None)."""
    sm = inf._state_model
    x = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)  # (1,T,18)
    pid = torch.tensor([PERSONA], dtype=torch.long)
    with torch.no_grad():
        hidden = inf._forward_last_hidden_for_model(sm, x, pid)  # (1, hidden)
        state_logits = sm.readout(hidden)[0]
        state_probs = torch.softmax(state_logits, dim=-1)
        intent_probs = None
        if inf._intent_head is not None:
            intent_logits = inf._intent_head(hidden)[0]
            intent_probs = torch.softmax(intent_logits, dim=-1)
    return state_probs, intent_probs


def l1(a, b) -> float:
    return float((a - b).abs().sum().item())


def ablate_prefix(seq: list[list[float]], k: int) -> list[list[float]]:
    """保留最近 k 帧, 前 T-k 帧替换为第 (T-k) 帧值 (persistence prefix)."""
    T = len(seq)
    if k >= T:
        return [row[:] for row in seq]
    anchor = seq[T - k]
    return [anchor[:] for _ in range(T - k)] + [seq[i][:] for i in range(T - k, T)]


def main():
    print(f"daemon ckpt: {DAEMON_CKPT}")
    print(f"intent ckpt: {INTENT_CKPT}")
    inf = RealtimeLNNInferencer(
        ckpt_path=DAEMON_CKPT,
        intent_head_ckpt=INTENT_CKPT if INTENT_CKPT.exists() else None,
        persona_idx=PERSONA,
    )
    if not inf.is_enabled:
        print("ERROR: inferencer disabled, ckpt load failed")
        return
    samples = load_samples()
    print(f"样本数 (au_window len>={MIN_LEN}): {len(samples)}")
    print(f"intent_head: {'on' if inf._intent_head is not None else 'off'}")

    lines = ["# Horizon Ablation 诊断结果\n",
             f"- daemon ckpt sha8: {inf.ckpt_sha8}",
             f"- intent ckpt sha8: {inf._intent_head_ckpt_sha8}",
             f"- 样本数(len>={MIN_LEN}): {len(samples)}",
             f"- persona: {PERSONA}\n"]

    # ---- 实验 1: lag ablation ----
    print("\n=== 实验 1 · lag ablation (保留最近 k 帧, 早期 prefix→persistence) ===")
    print(f"{'k帧':>4} | {'state L1 mean':>14} | {'intent L1 mean':>15} | {'state max':>10} | {'intent max':>10}")
    lines.append("\n## 实验 1 · lag ablation\n")
    lines.append("| k 帧 | state L1 mean | intent L1 mean | state L1 max | intent L1 max | 解读 |")
    lines.append("|---|---|---|---|---|---|")
    for k in [1, 2, 4, 8]:
        sd, idd = [], []
        for seq in samples:
            ps_full, pi_full = forward_probs(inf, seq)
            ps_ab, pi_ab = forward_probs(inf, ablate_prefix(seq, k))
            sd.append(l1(ps_full, ps_ab))
            if pi_full is not None:
                idd.append(l1(pi_full, pi_ab))
        sm_mean = sum(sd) / len(sd) if sd else 0.0
        sm_max = max(sd) if sd else 0.0
        im_mean = sum(idd) / len(idd) if idd else 0.0
        im_max = max(idd) if idd else 0.0
        interp = "近0=感受野已饱和" if sm_mean < 0.01 else "有影响"
        print(f"{k:>4} | {sm_mean:>14.6f} | {im_mean:>15.6f} | {sm_max:>10.4f} | {im_max:>10.4f}")
        lines.append(f"| {k} | {sm_mean:.6f} | {im_mean:.6f} | {sm_max:.4f} | {im_max:.4f} | {interp} |")

    # ---- 实验 2: window sweep ----
    print("\n=== 实验 2 · window sweep (截断到最近 w 帧 vs 完整) ===")
    print(f"{'w帧':>4} | {'state L1 mean':>14} | {'intent L1 mean':>15} | {'适用样本':>8}")
    lines.append("\n## 实验 2 · window sweep (截断最近 w 帧 vs full)\n")
    lines.append("| w 帧 | state L1 mean | intent L1 mean | 适用样本数 | 解读 |")
    lines.append("|---|---|---|---|---|")
    for w in [2, 4, 8, 12, 16]:
        sd, idd, n = [], [], 0
        for seq in samples:
            if len(seq) <= w:
                continue  # 只测序列比 w 长的(否则截断无意义)
            n += 1
            ps_full, pi_full = forward_probs(inf, seq)
            ps_w, pi_w = forward_probs(inf, seq[-w:])
            sd.append(l1(ps_full, ps_w))
            if pi_full is not None:
                idd.append(l1(pi_full, pi_w))
        sm_mean = sum(sd) / len(sd) if sd else 0.0
        im_mean = sum(idd) / len(idd) if idd else 0.0
        interp = "近0=该w已含全部信息" if sm_mean < 0.01 else "截断有损"
        print(f"{w:>4} | {sm_mean:>14.6f} | {im_mean:>15.6f} | {n:>8}")
        lines.append(f"| {w} | {sm_mean:.6f} | {im_mean:.6f} | {n} | {interp} |")

    out = Path("/tmp/horizon_ablation_result.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n结果已写 {out}")


if __name__ == "__main__":
    main()
