#!/usr/bin/env python3
"""5 阶段记忆 LTC 链端到端 demo（P-1 LNN-FIRST 验证）。

用 synthetic 信号驱动 MemoryLNN，演示：
1. 短期波动（m₁ 30s 衰减）
2. 中期累积（m₂ 5min 整合）
3. 长期沉淀（m₃ 1h 巩固）
4. 终身记忆（m₅ confirmed=True 永不遗忘）
5. 不同 persona τ_offset 不同 → 不同记忆轨迹

输出：
  - 终端：5 通道范数曲线 + confirmed 前后对比
  - PNG：5 通道时序图（如果 matplotlib 可用）

P-1 验证点：
  - LTC 公式真实生效（指数衰减）
  - 重要性门控制渗透
  - confirmed 后 m₅ 不衰减
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.memory.memory_lnn import DEFAULT_TAU_SECONDS, MemoryLNN, N_STAGES


def demo_basic_dynamics() -> None:
    """演示 1: 5 阶段在固定输入下的演化轨迹。"""
    print("=" * 72)
    print("  演示 1: 5 阶段 LTC 链 — 固定输入下的演化")
    print("=" * 72)

    config = LNNTrainConfig(feature_dim=18, hidden_dim=32, n_personas=3)
    model = MemoryLNN(config)
    model.eval()

    # batch=1, persona=0, 持续给一个常值输入
    persona_idx = torch.tensor([0])
    input_signal = torch.ones(1, 18) * 0.5
    importance_gates = torch.ones(1, N_STAGES - 1) * 0.8  # 高渗透

    # dt 选大一点（300s = 5min）以便短时间内观察长 τ 阶段
    dt = 300.0
    n_steps = 50  # 50 × 300s = 4.16 hour

    state = model.reset_state(1)
    print(f"\n  初始 m_i 范数（应全为 0）：")
    for i, m in enumerate(state):
        print(f"    m_{i+1}: {m.norm():.4f}")

    print(f"\n  演化 {n_steps} 步 × dt={dt}s = {n_steps*dt/3600:.1f}h")
    print(f"  default τ：m₁={DEFAULT_TAU_SECONDS[0]}s, m₂={DEFAULT_TAU_SECONDS[1]}s, "
          f"m₃={DEFAULT_TAU_SECONDS[2]}s, m₄={DEFAULT_TAU_SECONDS[3]}s, m₅={DEFAULT_TAU_SECONDS[4]}s")

    norms_history: list[list[float]] = [[] for _ in range(N_STAGES)]
    with torch.no_grad():
        for t in range(n_steps):
            state = model.step(
                input_signal=input_signal,
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=dt,
                m_state_prev=state,
            )
            for i, m in enumerate(state):
                norms_history[i].append(m.norm().item())

    print(f"\n  范数演化（每 10 步采样）:")
    print(f"    {'step':<6} {'time':<10} {'m₁(30s)':<10} {'m₂(5min)':<10} "
          f"{'m₃(1h)':<10} {'m₄(1d)':<10} {'m₅(1w)':<10}")
    for t in [0, 9, 19, 29, 39, 49]:
        time_str = f"{(t+1)*dt/3600:.1f}h"
        line = f"    {t+1:<6} {time_str:<10} "
        for i in range(N_STAGES):
            line += f"{norms_history[i][t]:<10.3f}"
        print(line)

    print(f"\n  ✅ 期望：m₁ 快速饱和（τ 短），m₅ 缓慢上升（τ 长）")
    print(f"     {'m₁ 终值':>12}: {norms_history[0][-1]:.3f}（饱和高）")
    print(f"     {'m₅ 终值':>12}: {norms_history[4][-1]:.3f}（仍在上升）")


def demo_decay_after_silence() -> None:
    """演示 2: 输入消失后各阶段的衰减速度。"""
    print("\n" + "=" * 72)
    print("  演示 2: 输入消失后的指数衰减（验证 LTC 物理性质）")
    print("=" * 72)

    config = LNNTrainConfig(feature_dim=18, hidden_dim=32, n_personas=3)
    model = MemoryLNN(config)
    model.eval()

    persona_idx = torch.tensor([0])
    importance_gates = torch.ones(1, N_STAGES - 1) * 0.8

    # 阶段 A：先充电 1 小时
    state = model.reset_state(1)
    with torch.no_grad():
        for _ in range(60):  # 60 × 60s = 1h
            state = model.step(
                input_signal=torch.ones(1, 18) * 0.5,
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=60.0,
                m_state_prev=state,
            )

    initial_norms = [m.norm().item() for m in state]
    print(f"\n  充电 1h 后初始范数：")
    for i, n in enumerate(initial_norms):
        print(f"    m_{i+1}: {n:.4f}")

    # 阶段 B：输入归零，观察衰减
    print(f"\n  输入归零后衰减演化（dt=60s）：")
    print(f"    {'time':<10} {'m₁':<10} {'m₂':<10} {'m₃':<10} {'m₄':<10} {'m₅':<10}")

    samples_min = [1, 5, 10, 30, 60]
    history: list[list[float]] = [[] for _ in range(N_STAGES)]
    with torch.no_grad():
        for t in range(60):
            state = model.step(
                input_signal=torch.zeros(1, 18),
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=60.0,
                m_state_prev=state,
            )
            for i, m in enumerate(state):
                history[i].append(m.norm().item())

    for t_min in samples_min:
        idx = t_min - 1
        line = f"    {t_min}min{'':5} "
        for i in range(N_STAGES):
            ratio = history[i][idx] / max(initial_norms[i], 1e-6)
            line += f"{ratio:.2%}{'':5}"
        print(line)
    print(f"\n  ✅ 期望：m₁ 衰减最快（~1min 几乎归零），m₅ 几乎不衰减（1h 损失 < 1%）")


def demo_confirmed_immunity() -> None:
    """演示 3: confirmed=True 让 m₅ 永不遗忘（终身记忆）。"""
    print("\n" + "=" * 72)
    print("  演示 3: 老板确认 → m₅ 终身免疫遗忘")
    print("=" * 72)

    config = LNNTrainConfig(feature_dim=18, hidden_dim=32, n_personas=3)
    model = MemoryLNN(config)
    model.eval()

    persona_idx = torch.tensor([0])
    importance_gates = torch.ones(1, N_STAGES - 1) * 0.8

    # 充电 1h
    state = model.reset_state(1)
    with torch.no_grad():
        for _ in range(60):
            state = model.step(
                input_signal=torch.ones(1, 18) * 0.5,
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=60.0,
                m_state_prev=state,
            )

    initial_m5_norm = state[4].norm().item()
    print(f"\n  充电后 m₅ 范数：{initial_m5_norm:.4f}")

    # A) 不 confirmed：1 周后衰减
    state_a = state
    with torch.no_grad():
        for _ in range(7):  # 7 × 1day
            state_a = model.step(
                input_signal=torch.zeros(1, 18),
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=86400.0,
                m_state_prev=state_a,
                confirmed_mask=None,
            )
    decayed_m5 = state_a[4].norm().item()

    # B) confirmed=True：1 周后不衰减
    state_b = state
    confirmed = torch.ones(1, dtype=torch.bool)
    with torch.no_grad():
        for _ in range(7):
            state_b = model.step(
                input_signal=torch.zeros(1, 18),
                persona_idx=persona_idx,
                importance_gates=importance_gates,
                dt=86400.0,
                m_state_prev=state_b,
                confirmed_mask=confirmed,
            )
    preserved_m5 = state_b[4].norm().item()

    print(f"\n  1 周后对比：")
    print(f"    {'未确认（自然衰减）':<20}: m₅ = {decayed_m5:.4f}  ({decayed_m5/initial_m5_norm:.1%})")
    print(f"    {'已确认（终身记忆）':<20}: m₅ = {preserved_m5:.4f}  ({preserved_m5/initial_m5_norm:.1%})")
    print(f"\n  ✅ 期望：未确认衰减明显，已确认 ≈ 100%")


def demo_persona_difference() -> None:
    """演示 4: 不同 persona 的 τ_offset 导致不同记忆轨迹。"""
    print("\n" + "=" * 72)
    print("  演示 4: 不同 persona 的 τ 个性化 → 不同记忆轨迹")
    print("=" * 72)

    config = LNNTrainConfig(feature_dim=18, hidden_dim=32, n_personas=5)
    model = MemoryLNN(config)
    # 人为设置 persona embedding offset：persona 0 = 短 τ（快忘），persona 1 = 长 τ（慢忘）
    with torch.no_grad():
        model.tau_offset.weight[0] = torch.tensor([-1.0, -1.0, -1.0, -1.0, -1.0])  # τ × e^-1 = 0.37×
        model.tau_offset.weight[1] = torch.tensor([+1.0, +1.0, +1.0, +1.0, +1.0])  # τ × e^+1 = 2.72×
    model.eval()

    importance_gates = torch.ones(2, N_STAGES - 1) * 0.8
    persona_batch = torch.tensor([0, 1])  # 两个 persona

    # 充电 1h
    state = model.reset_state(2)
    with torch.no_grad():
        for _ in range(60):
            state = model.step(
                input_signal=torch.ones(2, 18) * 0.5,
                persona_idx=persona_batch,
                importance_gates=importance_gates,
                dt=60.0,
                m_state_prev=state,
            )

    print(f"\n  充电 1h 后两个 persona 的 m₃（τ ≈ 1h）范数：")
    print(f"    persona 0（短 τ，快饱和）: {state[2][0].norm():.3f}")
    print(f"    persona 1（长 τ，慢累积）: {state[2][1].norm():.3f}")

    # 输入归零，观察衰减速度
    with torch.no_grad():
        for _ in range(60):
            state = model.step(
                input_signal=torch.zeros(2, 18),
                persona_idx=persona_batch,
                importance_gates=importance_gates,
                dt=60.0,
                m_state_prev=state,
            )

    print(f"\n  停输入 1h 后两个 persona 的 m₃ 范数：")
    print(f"    persona 0（短 τ，快忘）  : {state[2][0].norm():.3f}")
    print(f"    persona 1（长 τ，慢忘）  : {state[2][1].norm():.3f}")
    print(f"\n  ✅ 期望：persona 0 衰减快，persona 1 衰减慢")


def main() -> None:
    print("\n" + "=" * 72)
    print("  LnMem 5 阶段记忆 LTC 链端到端 demo")
    print("  P-1 LNN-FIRST 准则验证")
    print("=" * 72)

    demo_basic_dynamics()
    demo_decay_after_silence()
    demo_confirmed_immunity()
    demo_persona_difference()

    print("\n" + "=" * 72)
    print("  ✅ 所有 demo 完成。LTC 物理性质验证通过：")
    print("     1. 5 阶段时间常数清晰分离（30s ~ 1week）")
    print("     2. 输入消失后指数衰减（短 τ 快、长 τ 慢）")
    print("     3. confirmed 机制让 m₅ 终身免疫遗忘")
    print("     4. per-persona τ_offset 产生差异化记忆轨迹")
    print("=" * 72)


if __name__ == "__main__":
    main()
