"""train_intent_classifier.py — Studio 夜间训练入口（ADR-029 Sprint 2 钦定）

数据源：~/lnmem-daemon-sync/intent_feedback/*.jsonl
    每条带 {ts_ns, predicted, corrected, ui_session, note,
            hidden_state_window: [{ts_ns, h:[64]}, ...],
            au_window: [{ts_ns, au:[18]}, ...]}  (codex 二轮 retrofit H1 后)

输出：~/lnmem-train/ckpt/intent_head_YYYYMMDD-HHMMSS.pt + intent_head_latest.pt symlink

设计：
  - Phase 1 默认：head-only fine-tune（用 hidden_state_window 最后一帧）
    优点：训快，小样本可跑（M0 demo 50-100 条够）
    缺点：不利用序列动力学
  - Phase 2 (--bptt)：au_window 整段喂 LNN backbone 重新 forward + BPTT
    优点：真序列动力学训练（gemini H1 钦定精神）
    缺点：慢 + 需要 50+ 序列样本
    本 Sprint 2 默认 Phase 1，留 Phase 2 给 v0.2

gemini findings 整改：
  - H2 None 塌陷：Dynamic Focal Loss 替 cost-sensitive（codex 二轮 retrofit）
  - H1 序列动力学：本脚本读 hidden_state_window/au_window（codex H1 schema 已对齐）

CLI（与 scripts/studio_nightly_train.sh 配对）：
    python -m poc.training.train_intent_classifier \\
      --data-dir ~/lnmem-daemon-sync \\
      --out-ckpt ~/lnmem-train/ckpt/intent_head_xxx.pt \\
      --epochs 20 --batch-size 32 --device mps
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset


# ---- 常量 ----
INTENT_LABELS = ("positive", "negative", "none")
LABEL_TO_IDX = {label: idx for idx, label in enumerate(INTENT_LABELS)}
DEFAULT_HIDDEN_DIM = 64

# Pos/Neg 过采样倍数（gemini H2 整改：None 类太多时手动 oversample 少数类）
DEFAULT_OVERSAMPLE_MULTIPLIER = {"positive": 10, "negative": 10, "none": 1}


# ---- 数据加载 ----

class IntentFeedbackDataset(Dataset):
    """加载所有 intent_feedback jsonl 文件，提取训练样本。

    每个样本 = (hidden_state_64d, label_idx)。
    Phase 1: hidden_state 取 hidden_state_window 最后一帧（最接近 feedback 时刻的）
    Phase 2 (bptt=True): TODO 返回完整 au_window 序列，训练时跑 LNN backbone
    """

    def __init__(
        self,
        data_dir: Path,
        oversample: dict[str, int] | None = None,
        bptt_mode: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.bptt_mode = bptt_mode
        self.samples: list[tuple[list[float], int]] = []
        self._load_all_jsonl()
        if oversample:
            self._apply_oversample(oversample)
        self._summary()

    def _load_all_jsonl(self) -> None:
        feedback_dir = self.data_dir / "intent_feedback"
        if not feedback_dir.exists():
            print(f"[WARN] {feedback_dir} 不存在，无训练数据", file=sys.stderr)
            return
        jsonl_files = sorted(feedback_dir.glob("*.jsonl"))
        if not jsonl_files:
            print(f"[WARN] {feedback_dir} 下无 .jsonl 文件", file=sys.stderr)
            return

        for jf in jsonl_files:
            with open(jf, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sample = self._extract_sample(rec)
                    if sample is not None:
                        self.samples.append(sample)

    def _extract_sample(self, rec: dict) -> tuple[list[float], int] | None:
        # label = corrected (川哥纠正后的真值)
        label = rec.get("corrected") or rec.get("predicted")
        if label not in LABEL_TO_IDX:
            return None
        label_idx = LABEL_TO_IDX[label]

        # hidden state: 优先 hidden_state_window 最后一帧（codex 二轮 retrofit H1 后）
        hs_window = rec.get("hidden_state_window")
        if hs_window and isinstance(hs_window, list) and len(hs_window) > 0:
            last_h = hs_window[-1].get("h")
            if last_h and len(last_h) == DEFAULT_HIDDEN_DIM:
                return (last_h, label_idx)

        # 兼容：旧 schema 直接 hidden_state 字段
        direct_h = rec.get("hidden_state")
        if direct_h and len(direct_h) == DEFAULT_HIDDEN_DIM:
            return (direct_h, label_idx)

        return None  # 没有 hidden state，跳过

    def _apply_oversample(self, oversample: dict[str, int]) -> None:
        """少数类过采样（gemini H2 整改）"""
        before_count = Counter([INTENT_LABELS[s[1]] for s in self.samples])
        new_samples = []
        for sample in self.samples:
            label = INTENT_LABELS[sample[1]]
            mult = oversample.get(label, 1)
            new_samples.extend([sample] * mult)
        random.shuffle(new_samples)
        self.samples = new_samples
        after_count = Counter([INTENT_LABELS[s[1]] for s in self.samples])
        print(f"[oversample] 前: {dict(before_count)} → 后: {dict(after_count)}")

    def _summary(self) -> None:
        if not self.samples:
            print("[ERR] 0 训练样本！", file=sys.stderr)
            return
        labels = [INTENT_LABELS[s[1]] for s in self.samples]
        dist = Counter(labels)
        print(f"[dataset] {len(self.samples)} 样本，分布: {dict(dist)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        hidden, label_idx = self.samples[idx]
        return torch.tensor(hidden, dtype=torch.float32), torch.tensor(label_idx, dtype=torch.long)


# ---- 损失：Dynamic Focal Loss（gemini H2 整改） ----

def dynamic_focal_loss(
    logits: Tensor,
    targets: Tensor,
    gamma: float = 2.0,
    alpha: Tensor | None = None,
) -> Tensor:
    """Focal loss with optional class weight alpha.

    pt = softmax(logits)[target]
    focal = -alpha * (1-pt)^gamma * log(pt)

    alpha: shape [num_classes]，如 [1.0, 1.0, 0.5] 给 none 类降权
    None 时按 batch 内类别分布动态调权（少数类提权）
    """
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    targets_one_hot = F.one_hot(targets, num_classes=logits.size(-1)).float()
    pt = (probs * targets_one_hot).sum(dim=-1)
    log_pt = (log_probs * targets_one_hot).sum(dim=-1)
    focal_factor = (1 - pt) ** gamma

    if alpha is None:
        # 动态：按 batch 内频率倒数（防 batch 全 none）
        counts = torch.bincount(targets, minlength=logits.size(-1)).float()
        counts = torch.clamp(counts, min=1.0)
        alpha = 1.0 / counts
        alpha = alpha / alpha.sum() * logits.size(-1)  # 归一化让总权重 = num_classes
    alpha_t = alpha.to(logits.device)[targets]

    loss = -alpha_t * focal_factor * log_pt
    return loss.mean()


# ---- 训练循环 ----

def train_one_epoch(
    head: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    log_every: int = 10,
) -> dict[str, float]:
    head.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    class_correct = [0, 0, 0]
    class_n = [0, 0, 0]

    for step, (hidden, label) in enumerate(loader):
        hidden = hidden.to(device)
        label = label.to(device)

        logits = head(hidden)
        loss = dynamic_focal_loss(logits, label, gamma=2.0)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), max_norm=1.0)
        optimizer.step()

        pred = logits.argmax(dim=-1)
        total_loss += loss.item() * len(label)
        total_correct += (pred == label).sum().item()
        total_n += len(label)
        for i in range(3):
            class_correct[i] += ((pred == label) & (label == i)).sum().item()
            class_n[i] += (label == i).sum().item()

    per_class_acc = [
        class_correct[i] / max(class_n[i], 1) for i in range(3)
    ]
    print(
        f"[epoch {epoch:3d}] loss={total_loss/total_n:.4f} "
        f"acc={total_correct/total_n:.3f} "
        f"per-class={['{:.2f}'.format(a) for a in per_class_acc]}"
    )
    return {
        "loss": total_loss / total_n,
        "acc": total_correct / total_n,
        "per_class_acc": per_class_acc,
    }


# ---- 主入口 ----

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="包含 intent_feedback/*.jsonl 的目录（Studio 上 ~/lnmem-daemon-sync）")
    parser.add_argument("--out-ckpt", type=Path, required=True,
                        help="输出 ckpt 路径（建议 ~/lnmem-train/ckpt/intent_head_YYYYMMDD-HHMMSS.pt）")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="mps",
                        help="mps (Studio M2 Max) / cuda / cpu")
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--bptt", action="store_true",
                        help="Phase 2 端到端 BPTT 训练（暂未实现，留 v0.2）")
    parser.add_argument("--no-oversample", action="store_true",
                        help="禁用 Pos/Neg 10x 过采样（默认开）")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.bptt:
        print("[ERR] --bptt Phase 2 模式留 v0.2，本 Sprint 2 仅 head-only", file=sys.stderr)
        sys.exit(1)

    # device 选择
    if args.device == "mps" and torch.backends.mps.is_available():
        device = torch.device("mps")
    elif args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[device] {device}")

    # 加载数据
    oversample = None if args.no_oversample else DEFAULT_OVERSAMPLE_MULTIPLIER
    dataset = IntentFeedbackDataset(args.data_dir, oversample=oversample, bptt_mode=False)
    if len(dataset) == 0:
        print("[ERR] 0 训练样本，退出", file=sys.stderr)
        sys.exit(2)
    if len(dataset) < 10:
        print(f"[WARN] 仅 {len(dataset)} 样本，训练可能不稳", file=sys.stderr)

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0,  # macOS mps + fork 不稳，单线程稳
        drop_last=False,
    )

    # 加载 IntentClassifierHead（从 codex Sprint 2-B 一轮）
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from poc.src.lnn.intent_head import IntentClassifierHead

    head = IntentClassifierHead(hidden_dim=args.hidden_dim).to(device)
    print(f"[model] IntentClassifierHead params={sum(p.numel() for p in head.parameters())}")

    optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # 训练
    print(f"[train] {args.epochs} epochs, batch={args.batch_size}")
    final_metrics = {}
    for epoch in range(1, args.epochs + 1):
        final_metrics = train_one_epoch(head, loader, optimizer, device, epoch)
        scheduler.step()

    # 保存 ckpt
    args.out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": head.state_dict(),
        "hidden_dim": args.hidden_dim,
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "epochs": args.epochs,
        "dataset_size": len(dataset),
        "final_metrics": final_metrics,
        "seed": args.seed,
    }, args.out_ckpt)
    print(f"[saved] {args.out_ckpt}")

    # 维护 latest symlink
    latest = args.out_ckpt.parent / "intent_head_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    latest.symlink_to(args.out_ckpt.name)
    print(f"[symlink] {latest} → {args.out_ckpt.name}")

    print("[done]")


if __name__ == "__main__":
    main()
