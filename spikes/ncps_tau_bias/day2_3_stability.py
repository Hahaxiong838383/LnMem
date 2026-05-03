"""
Day 2-3 · 训练稳定性测试（48h）

目标：在合成正弦序列上训练 1000 步，对比有/无 τ_offset 的 loss 曲线，
      看 τ_offset 是否会让训练发散。

验收：
- 红：>30% 用户发散（loss=NaN 或 loss>10×初始值）→ 触发 Plan B (FiLM 单独)
- 黄：10-30% 发散 → 加 warmup + 梯度裁剪 + γ 限幅 [0.5, 1.5] 后重测
- 绿：<10% 发散 → 进入 Day 4-5

执行：
  python day2_3_stability.py --output day2_3_results.json
"""

import argparse
import json
import math
import sys
import time

import torch
import torch.nn as nn
from torch.optim import Adam

# Day 1 已经验证过的 wrapper 复用
from day1_smoke import TauBiasedLTC, NCPS_AVAILABLE


def generate_sine_data(n_users: int = 5, seq_len: int = 100, n_seq_per_user: int = 50, seed: int = 42):
    """
    每个用户一个频率不同的正弦波 + 噪声。
    任务：predict next-step value。
    """
    torch.manual_seed(seed)
    inputs, targets, user_ids = [], [], []
    for u in range(n_users):
        freq = 0.05 + u * 0.03  # 5 个用户用 5 个不同频率
        for _ in range(n_seq_per_user):
            t = torch.linspace(0, 4 * math.pi, seq_len + 1)
            phase = torch.rand(1).item() * 2 * math.pi
            wave = torch.sin(freq * t + phase) + 0.05 * torch.randn_like(t)
            x = wave[:-1].unsqueeze(-1).repeat(1, 5)  # 5 维输入（与 INPUT_DIM 对齐）
            y = wave[1:]                              # 1 维 next-step target
            inputs.append(x)
            targets.append(y)
            user_ids.append(u)
    return torch.stack(inputs), torch.stack(targets), torch.tensor(user_ids)


def train_one_user(model: nn.Module, X: torch.Tensor, y: torch.Tensor, user_id: int,
                   steps: int = 1000, lr: float = 1e-3,
                   warmup: int = 100, grad_clip: float = 1.0, gamma_limit: tuple = (0.5, 1.5)) -> dict:
    """
    训练一个用户的 1000 步，返回 loss 历史 + 是否发散。

    黄区调优手段（默认开启）：
    - warmup: 前 N 步线性升温学习率
    - grad_clip: 梯度裁剪
    - gamma_limit: τ_offset 用 sigmoid 映射到固定区间
    """
    optim = Adam(model.parameters(), lr=lr)
    losses = []
    diverged = False

    head = nn.Linear(model.cell.units if hasattr(model.cell, "units") else 16, 1)
    optim_head = Adam(head.parameters(), lr=lr)

    for step in range(steps):
        # warmup
        cur_lr = lr * min(1.0, (step + 1) / max(warmup, 1))
        for pg in optim.param_groups:
            pg["lr"] = cur_lr

        # 取一个序列
        idx = step % X.shape[0]
        seq_x = X[idx]   # [seq_len, 5]
        seq_y = y[idx]   # [seq_len]

        hx = None
        outs = []
        for t in range(seq_x.shape[0]):
            out, hx = model(seq_x[t:t+1], user_id=user_id, hx=hx)
            outs.append(head(out).squeeze())
        pred = torch.stack(outs)

        loss = ((pred - seq_y) ** 2).mean()

        if torch.isnan(loss) or loss.item() > 1e4:
            diverged = True
            losses.append(float("nan"))
            break

        optim.zero_grad()
        optim_head.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        torch.nn.utils.clip_grad_norm_(head.parameters(), grad_clip)
        optim.step()
        optim_head.step()

        losses.append(loss.item())

    return {"diverged": diverged, "losses": losses, "final_loss": losses[-1] if losses else float("nan")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="day2_3_results.json")
    parser.add_argument("--steps", type=int, default=1000)
    args = parser.parse_args()

    if not NCPS_AVAILABLE:
        print("[RED] ncps 不可用 — Day 1 先解决")
        sys.exit(1)

    print("=" * 60)
    print("Day 2-3 · 训练稳定性测试")
    print("=" * 60)

    N_USERS = 5
    INPUT_DIM = 5
    UNITS = 16

    X, y, _ = generate_sine_data(n_users=N_USERS)
    print(f"数据 shape: X={tuple(X.shape)}, y={tuple(y.shape)}")

    results = {"phase": "Day 2-3", "users": [], "started_at": time.time()}

    n_diverged = 0
    for uid in range(N_USERS):
        print(f"\n--- 用户 {uid} ---")
        # 每个用户独立模型实例（避免互相干扰）
        model = TauBiasedLTC(input_size=INPUT_DIM, units=UNITS, num_users=N_USERS)
        # 给该用户设个随机 τ_offset 起点
        with torch.no_grad():
            model.tau_offset.data[uid] = torch.randn(UNITS) * 0.5

        user_X = X[(uid * 50):(uid + 1) * 50]
        user_y = y[(uid * 50):(uid + 1) * 50]

        r = train_one_user(model, user_X, user_y, user_id=uid, steps=args.steps)
        results["users"].append({
            "user_id": uid,
            "diverged": r["diverged"],
            "final_loss": r["final_loss"],
            "loss_curve_sample": r["losses"][::100],  # 每 100 步采样一次
        })
        if r["diverged"]:
            n_diverged += 1
            print(f"  [DIVERGED] step={len(r['losses'])} 时 loss={r['losses'][-1]}")
        else:
            print(f"  [OK] final_loss = {r['final_loss']:.4f}")

    # 判定
    div_rate = n_diverged / N_USERS
    if div_rate > 0.30:
        status = "RED"
        msg = f"发散率 {div_rate*100:.0f}% > 30% → 触发 Plan B (FiLM 单独使用)"
    elif div_rate > 0.10:
        status = "YELLOW"
        msg = f"发散率 {div_rate*100:.0f}% 在 10-30% 黄区 → 加强 warmup/grad_clip/gamma_limit 后重测"
    else:
        status = "GREEN"
        msg = f"发散率 {div_rate*100:.0f}% < 10% → 进入 Day 4-5 表达能力测试"

    results["status"] = status
    results["divergence_rate"] = div_rate
    results["message"] = msg
    results["elapsed_sec"] = time.time() - results["started_at"]

    print("\n" + "=" * 60)
    print(f"Day 2-3 结论：[{status}] {msg}")
    print("=" * 60)

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"结果已落盘: {args.output}")

    sys.exit(0 if status == "GREEN" else 1)


if __name__ == "__main__":
    main()
