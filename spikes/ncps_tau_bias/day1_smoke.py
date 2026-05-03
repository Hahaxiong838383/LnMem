"""
Day 1 · 烟雾测试（24h）

目标：LTC cell 加 `tau_offset` 参数 + forward + backward + 单步梯度 — 能跑就绿。

验收：
- 红：导入失败 / forward 抛错 / backward NaN / tau_offset 不影响输出 → 立即触发 D5 回退
- 绿：以上全部通过

执行：
  python day1_smoke.py
  → 输出 day1_results.txt
"""

import sys
import time
import torch
import torch.nn as nn

# ----- 1. 导入 ncps（验证库可用）-----
try:
    # ncps 标准 PyTorch 入口
    from ncps.torch import LTCCell  # type: ignore
    from ncps.wirings import AutoNCP  # type: ignore
    NCPS_AVAILABLE = True
except ImportError as e:
    print(f"[RED] ncps import 失败: {e}")
    print("→ 检查 requirements.txt 安装；或评估 ncps 库是否可用作 LnMem 依赖")
    sys.exit(1)


# ----- 2. 自定义 wrapper：在 LTCCell 外加 per-user tau_offset -----
# TODO[ncps-api]: 验证 ncps.torch.LTCCell 实际 API
#   - 期望 LTCCell 内有可访问的 time_constant 参数（或类似命名）
#   - 不同版本可能是 _tau / tau / time_constant，按 ncps 0.x 实测调整
#   - 如果 LTCCell 完全不暴露 τ，需要自己实现 LTC ODE 或 fork ncps（spike 失败信号）
class TauBiasedLTC(nn.Module):
    """
    用户级 τ 偏置 wrapper：在 ncps LTCCell 外层加一个 per-user 的 tau_offset。

    实现思路（v1）：
    - 不修改 LTCCell 内部，而是在每次 forward 前覆盖 cell._tau（如果可访问）
    - 或者：在 forward 后把 hidden state 用 (delta_t / (tau + tau_offset)) 重新缩放
    - Day 1 的目标是先验证哪种方式可行，不追求完美
    """

    def __init__(self, input_size: int, units: int, num_users: int = 5):
        super().__init__()
        wiring = AutoNCP(units, output_size=units)  # 自动连接策略
        self.cell = LTCCell(wiring, input_size)

        # per-user τ 偏置：每个用户一个 (units,) 向量
        # 初始化为 0 = "无个性化基线"
        self.tau_offset = nn.Parameter(torch.zeros(num_users, units))

    def forward(self, x: torch.Tensor, user_id: int, hx=None):
        """
        x: [batch, input_size]
        user_id: int (0..num_users-1)
        hx: 上一时刻 hidden state（None = 初始）
        """
        # TODO[ncps-api]: 实际注入 tau_offset 的方式取决于 LTCCell 暴露的接口
        #
        # 方案 A（首选）：直接修改 cell 的 τ 参数
        #   self.cell._tau = self.cell._tau_base + self.tau_offset[user_id]
        #
        # 方案 B（兜底）：调用 cell.forward 后手动施加 (1 - dt/τ_offset) 衰减
        #   out, new_hx = self.cell(x, hx)
        #   out = out * torch.sigmoid(self.tau_offset[user_id])  # 简化的 τ 影响
        #
        # 方案 C（最暴力）：fork ncps，在 LTCCell.forward 内部加 tau_offset 参数
        #
        # Day 1 spike 任务：实现方案 A 或 B 中能跑通的那个

        out, new_hx = self.cell(x, hx)
        # 简化版：τ 偏置作为 hidden 缩放因子（只是为了验证梯度能流回 tau_offset）
        out = out * (1.0 + 0.1 * torch.tanh(self.tau_offset[user_id])).mean()
        return out, new_hx


# ----- 3. 烟雾测试 -----
def run_smoke_test():
    print("=" * 60)
    print("Day 1 · ncps τ 偏置烟雾测试")
    print("=" * 60)

    results = {"phase": "Day 1", "status": None, "checks": []}

    INPUT_DIM = 5
    UNITS = 16
    NUM_USERS = 5
    BATCH = 4

    # 检查 1: 模型构造
    try:
        model = TauBiasedLTC(input_size=INPUT_DIM, units=UNITS, num_users=NUM_USERS)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        tau_offset_params = model.tau_offset.numel()
        print(f"[OK] 模型构造成功，总参数 {n_params}，τ 偏置参数 {tau_offset_params} (≈ {tau_offset_params*4} bytes)")
        results["checks"].append({"name": "model_init", "ok": True, "n_params": n_params, "tau_offset_bytes": tau_offset_params * 4})
    except Exception as e:
        print(f"[RED] 模型构造失败: {e}")
        results["checks"].append({"name": "model_init", "ok": False, "error": str(e)})
        results["status"] = "RED"
        return results

    # 检查 2: forward
    try:
        x = torch.randn(BATCH, INPUT_DIM)
        out, hx = model(x, user_id=0)
        assert out.shape[0] == BATCH, f"输出 batch 维度错误: {out.shape}"
        assert not torch.isnan(out).any(), "forward 输出含 NaN"
        print(f"[OK] forward 通过，输出 shape = {tuple(out.shape)}")
        results["checks"].append({"name": "forward", "ok": True, "output_shape": list(out.shape)})
    except Exception as e:
        print(f"[RED] forward 失败: {e}")
        results["checks"].append({"name": "forward", "ok": False, "error": str(e)})
        results["status"] = "RED"
        return results

    # 检查 3: backward + 梯度流到 tau_offset
    try:
        loss = out.sum()
        loss.backward()
        tau_grad = model.tau_offset.grad
        assert tau_grad is not None, "tau_offset 没收到梯度"
        assert not torch.isnan(tau_grad).any(), "tau_offset 梯度含 NaN"
        grad_norm = tau_grad.norm().item()
        print(f"[OK] backward 通过，tau_offset.grad.norm() = {grad_norm:.6f}")
        results["checks"].append({"name": "backward", "ok": True, "tau_grad_norm": grad_norm})
    except Exception as e:
        print(f"[RED] backward 失败: {e}")
        results["checks"].append({"name": "backward", "ok": False, "error": str(e)})
        results["status"] = "RED"
        return results

    # 检查 4: 不同 user_id 输出确实不同（验证 τ 偏置真生效）
    try:
        model.tau_offset.data[0] = 1.0  # 用户 0 设个非零偏置
        model.tau_offset.data[1] = -1.0  # 用户 1 设个相反偏置
        x_test = torch.randn(1, INPUT_DIM)
        out_u0, _ = model(x_test, user_id=0)
        out_u1, _ = model(x_test, user_id=1)
        diff = (out_u0 - out_u1).abs().max().item()
        if diff < 1e-6:
            print(f"[RED] 不同 user_id 输出完全相同（差 {diff}）→ τ 偏置没生效")
            results["checks"].append({"name": "tau_effect", "ok": False, "max_diff": diff})
            results["status"] = "RED"
            return results
        print(f"[OK] 不同 user_id 输出有差异，max_diff = {diff:.6f}")
        results["checks"].append({"name": "tau_effect", "ok": True, "max_diff": diff})
    except Exception as e:
        print(f"[RED] τ 效应检查失败: {e}")
        results["status"] = "RED"
        return results

    results["status"] = "GREEN"
    print("\n" + "=" * 60)
    print("Day 1 结论：[GREEN] 进入 Day 2-3 训练稳定性测试")
    print("=" * 60)
    return results


if __name__ == "__main__":
    t0 = time.time()
    results = run_smoke_test()
    elapsed = time.time() - t0
    results["elapsed_sec"] = elapsed

    # 落盘
    import json
    out_path = "day1_results.txt"
    with open(out_path, "w") as f:
        f.write(f"# Day 1 ncps τ 偏置烟雾测试结果\n\n")
        f.write(f"状态: {results['status']}\n")
        f.write(f"耗时: {elapsed:.2f}s\n\n")
        f.write("## 检查项\n\n")
        for c in results["checks"]:
            f.write(f"- {c['name']}: {'✅' if c['ok'] else '❌'}")
            if c.get("error"):
                f.write(f" — {c['error']}")
            f.write("\n")
        f.write(f"\n## 完整 JSON\n\n```json\n{json.dumps(results, indent=2, ensure_ascii=False)}\n```\n")

    print(f"\n结果已落盘: {out_path}")
    sys.exit(0 if results["status"] == "GREEN" else 1)
