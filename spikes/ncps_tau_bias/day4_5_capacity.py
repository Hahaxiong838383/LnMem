"""
Day 4-5 · 表达能力测试（48h）

目标：用 5 个 spike 内自包含的 prompt（synthetic_personas.py）训练，
      看 τ_offset 是否能区分 5 种行为模式。

验收：
- 红：classification 准确率 < 60% → 触发 Plan B
- 黄：60-80% → 容量边缘，需 M2-M3 进一步实验决策是否 4KB
- 绿：≥ 80% → 三阶段全绿，工程可行性破雷成功

执行：
  python day4_5_capacity.py --output day4_5_results.json
"""

import argparse
import json
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam

from day1_smoke import TauBiasedLTC, NCPS_AVAILABLE
from synthetic_personas import generate_all_personas_data, PERSONAS, INPUT_DIM


class PersonaClassifier(nn.Module):
    """
    用 LNN 隐状态作为序列特征，接一个分类头预测 user_id (0-4)。

    任务设计：模型不知道 user_id，只能通过 τ_offset 学到的隐状态差异来区分。
    """
    def __init__(self, input_size: int, units: int, num_users: int):
        super().__init__()
        self.lnn = TauBiasedLTC(input_size, units, num_users)
        # 分类头：最后一步 hidden → 5 类 logits
        self.head = nn.Linear(units, num_users)

    def forward(self, x, user_id):
        # x: [batch, seq_len, input_size]
        hx = None
        for t in range(x.shape[1]):
            out, hx = self.lnn(x[:, t, :], user_id=user_id, hx=hx)
        # out: [batch, units]
        return self.head(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="day4_5_results.json")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--units", type=int, default=16)
    args = parser.parse_args()

    if not NCPS_AVAILABLE:
        print("[RED] ncps 不可用 — Day 1 先解决")
        sys.exit(1)

    print("=" * 60)
    print("Day 4-5 · 表达能力测试（5 主人区分度）")
    print("=" * 60)

    # 加载 spike 内自包含的 5 主人合成数据
    X, y, user_ids = generate_all_personas_data()
    print(f"数据 shape: X={tuple(X.shape)}, y={tuple(y.shape)}")
    print(f"5 personas: {[p.name for p in PERSONAS]}")

    NUM_USERS = len(PERSONAS)

    # 80/20 split
    perm = torch.randperm(X.shape[0])
    n_train = int(X.shape[0] * 0.8)
    train_idx, test_idx = perm[:n_train], perm[n_train:]
    Xtr, ytr, utr = X[train_idx], y[train_idx], user_ids[train_idx]
    Xte, yte, ute = X[test_idx], y[test_idx], user_ids[test_idx]

    model = PersonaClassifier(input_size=INPUT_DIM, units=args.units, num_users=NUM_USERS)
    optim = Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()

    results = {
        "phase": "Day 4-5",
        "config": {"epochs": args.epochs, "batch": args.batch, "units": args.units},
        "epoch_metrics": [],
        "started_at": time.time(),
    }

    for epoch in range(args.epochs):
        # train
        model.train()
        train_loss = 0.0
        n_batches = 0
        for i in range(0, Xtr.shape[0], args.batch):
            xb = Xtr[i:i+args.batch]
            yb = ytr[i:i+args.batch]
            ub = utr[i:i+args.batch]

            # 因为 forward 一次只接受一个 user_id，简化处理：每个样本独立 forward
            logits_list = []
            for j in range(xb.shape[0]):
                logits = model(xb[j:j+1], user_id=ub[j].item())
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)

            loss = crit(logits, yb)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss += loss.item()
            n_batches += 1

        # eval
        model.eval()
        with torch.no_grad():
            logits_list = []
            for j in range(Xte.shape[0]):
                logits = model(Xte[j:j+1], user_id=ute[j].item())
                logits_list.append(logits)
            logits = torch.cat(logits_list, dim=0)
            pred = logits.argmax(dim=1)
            acc = (pred == yte).float().mean().item()

        avg_train_loss = train_loss / max(n_batches, 1)
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={avg_train_loss:.4f}  test_acc={acc*100:.2f}%")
        results["epoch_metrics"].append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "test_acc": acc,
        })

    final_acc = results["epoch_metrics"][-1]["test_acc"]

    if final_acc < 0.60:
        status = "RED"
        msg = f"区分度 {final_acc*100:.1f}% < 60% → 触发 Plan B (FiLM 单独使用 / BitFit+LoRA(2))"
    elif final_acc < 0.80:
        status = "YELLOW"
        msg = f"区分度 {final_acc*100:.1f}% 在 60-80% 黄区 → M2-M3 再做容量曲线 (1KB / 4KB / 16KB)"
    else:
        status = "GREEN"
        msg = f"区分度 {final_acc*100:.1f}% ≥ 80% → 三阶段全绿，工程可行性破雷成功"

    results["status"] = status
    results["final_test_acc"] = final_acc
    results["message"] = msg
    results["elapsed_sec"] = time.time() - results["started_at"]

    print("\n" + "=" * 60)
    print(f"Day 4-5 结论：[{status}] {msg}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已落盘: {args.output}")

    sys.exit(0 if status == "GREEN" else 1)


if __name__ == "__main__":
    main()
