---
adr: ADR-022
title: τ-bias 梯度黑洞病理修复 · 双率优化器 + 动态梯度平衡
status: accepted（2026-05-22 04:30 cc 实测 + gemini 4 轮评审钦定）
date: 2026-05-22
deciders: cc 主线 + gemini 3.5 high γ-3 spike4 review REVISE
related:
  - 021-ncps-tau-bias-route-c-handwritten-cell.md
  - ../specs/base-v3-training-spec.md §2.1 + §9 #13
  - ../reports/gemini-gamma3-spike4-review.md
parent: ../STYLE.md
体裁: L2（决策记录）
---

# ADR-022 · τ-bias 梯度黑洞病理修复

## 1. Context

**spec v0.2.8 §9 #13 原担心方向（错）**：「τ-bias 进了 ODE 但被 f_net shortcut，
gemini F-4.1 估计 bias_grad/f_net_grad < 1e-3 是 LNN-FIRST 假落地风险」

**γ-3 真训练实测真相（5/22 03:55）**：4 跑 spike (InfoNCE × 2 seed + VICReg × 2 seed)：
- bias_grad_ratio = **6.4 × 10⁸**（反向 8 个数量级异常）
- 改 persona_lr_scale 5x → 1x 后仍 4.29e8（基本不变）
- M1/M1-V ≈ 0.0-0.17（远低于 10 视频 Top-5 随机基线 50%）→ 表征 collapse

## 2. Decision

**root cause（gemini 评审 §B.1 钦定）**：「梯度黑洞 / 反向吞噬」

```
ODE: dh/dt = (f_net(x) + τ) ⊙ h
- τ 路径: 静态参数，反向梯度路径 O(1) 直达
- f_net 路径: 多层 MLP + CfC 时序展开，梯度路径长 + 指数衰减
- 结果: 99.9999% 优化器更新量给 τ，f_net 实际被冻结
- 模型退化: 与输入无关的查表机（Lookup Table）
```

**修复方案（gemini ADR-022 §B.2 钦定 + cc 调优）**：

### 2.1 双率优化器（Dual-Rate Optimizer）

```python
optimizer = AdamW([
    {"params": fusion + readout_heads, "lr": 1e-3},         # 主干
    {"params": f_net_params, "lr": 1e-3, "weight_decay": 1e-3},
    {"params": persona_tau_bias_params, "lr": **1e-6**},    # τ 极低，差 1000x
], lr=1e-3)
```

理由：差 10⁸ 量级的梯度黑洞，仅 LR 调整需要 10⁸ 量级压制。1e-3 / 1e-6 = 10³ 是合理起点，
配合 DGB 进一步收紧到 [0.1, 50] 安全区间。

### 2.2 动态梯度平衡（Dynamic Gradient Balancing, DGB）

每个 backward step 后 / optimizer step 前，做：
```python
g_tau_norm = persona_tau_bias.grad.norm()
g_fnet_norm = sum(p.grad.norm() for p in f_net.parameters())
target_ratio = 10.0  # γ ∈ [10, 100]
scale = g_fnet_norm / (g_tau_norm * target_ratio + 1e-9)
persona_tau_bias.grad.data *= min(scale, 1.0)  # 只缩小不放大
```

效果：强制 bias_grad/f_net_grad 比例 ≤ 1/10，τ 永远是 f_net 的辅助调制而非主导。

### 2.3 失活监测（Health Gate）

每 100 step 触发：若 bias_grad_ratio ∉ [0.1, 50]，alarm + dump 状态：
- < 0.1: 退化为 shortcut learning（gemini F-4.1 原版风险）
- \> 50: 梯度黑洞（本 ADR 修复目标）

## 3. 实施

### 3.1 修改 `poc/training/trainer.py`

1. AdamW param groups 加 `persona_tau_bias: lr=1e-6`（不依赖 `persona_lr_scale`）
2. 加 `_apply_dgb()` 方法（loss.backward 后 / optimizer.step 前调用）
3. health gate 警报阈值 [0.1, 50]，超过 dump checkpoint

### 3.2 修改 `TrainConfig`

```python
tau_lr: float = 1e-6           # 双率优化器 τ 极低 LR
dgb_target_ratio: float = 10.0  # DGB 强制 ratio ≤ 1/10
dgb_enabled: bool = True
```

### 3.3 删除/降级

- `persona_lr_scale` 字段标 deprecated（保留兼容性但不使用）

## 4. 验收

在 10 视频 spike 重测：
- ✅ bias_grad_ratio 稳定在 [0.1, 50] 范围（gemini 评审 §D.2 红线 #1）
- ✅ 双 seed (42/7) collapse var 均 ≥ 0.010（gemini §D.2 红线 #2）
- ✅ Shuffle test（gemini §D.2 红线 #3）：人工 shuffle 帧顺序后 M1 显著下降，
  证明模型确实用时序 ODE 而非静态 bias
- ✅ M1 Top-5 ≥ 25%（远低于 50% 视频级随机基线但显著高于 collapse 水平 0-17%）

只有上述 4 项全通过，才允许放行 Phase γ Step 1（5/27）。

## 5. 关联 spec 修订

- spec v0.2.8 §9 #13 风险描述需重写为「梯度黑洞 / 反向吞噬」
- §9 加新风险 #14: 「InfoNCE batch_size < 32 数学 broken」（gemini §C 钦定）
- §5 训练计划加 Warmup 阶段：前 10% epoch 用 VICReg 撑开表征空间，再切 InfoNCE
- §11.1 preflight #12 验收门改为本 ADR §4 的 4 项红线

## 6. 实测预期

- 首跑 ADR-022 实施后：bias_grad_ratio 应 ≤ 50（vs 修复前 6.4e8）
- collapse var 应稳定 > 0.01（vs 修复前 0.0075-0.0116 边缘）
- M1 Top-5 spike 阶段 ≥ 0.25（vs 修复前 0-0.17）

若实测仍 fail → 暴露更深 ODE 公式问题，重新立 ADR

---

## 7. v3 改判（2026-05-22 gemini 第 7 轮前置评审 F-3 HIGH）

### 7.1 背景
PLAN v1 一度把 `tau_lr=1e-5`（接续 gemini 第 6 轮 "保留学习能力但受控" 思路）。第 7 轮前置评审抓出该参数其实是「**双重锁死 = 伪学习**」：

- DGB 已经把 τ.grad 缩到 main_grad / 10
- 再叠 1e-5 LR → 实际 τ 更新量 ~$10^{-6} \sim 10^{-7}$
- τ 初值 0.1，整训练期累积位移 < 1%
- 这是为了通过 PC-1 `grad_norm > 1e-9` 验收的「表面合规」 —— 指标合规但功能死亡

### 7.2 v3 改判
```python
# trainer.py::TrainConfig
tau_lr: float = 1e-3            # v3: 释放到正常 LR 量级
dgb_target_ratio: float = 10.0  # DGB 单项控制（safety net + 主控）
dgb_enabled: bool = True
```

### 7.3 v3 验收（codex v2 整改后单测）
- τ.grad_norm ≥ 1e-5 量级（非 underflow）
- τ.item 在 epoch 内可测位移（mean diff > 1e-5）
- DGB scale 仍然生效（post-DGB ratio ≤ 0.1）

### 7.4 改判依据
gemini 第 7 轮原文：
> 这是一种为了通过 PC-1 grad_norm > 1e-9 验收而妥协的"表面合规"，τ 实际上已经失去了对时序动态的自适应调节能力，等同于死参数。

CLAUDE.md §3.1 钦定：cc/codex 与 gemini 对抗审给相反建议时，**默认采纳对抗审**。本次 cc 100% 采纳 F-3。

### 7.5 ADR 版本轨迹

| 版本 | 时间 | tau_lr | 决策依据 |
|---|---|---|---|
| v1 | 5/21 04:00 | 1e-6 | gemini 第 4 轮：双率优化器（首次 DGB） |
| v2 | 5/21 04:55 | 0.0 | gemini 第 5 轮：1e-6 后 underflow，直接 LR=0 |
| **v3** | **5/22 凌晨** | **1e-3** | **gemini 第 7 轮：1e-5 双重锁死 → 释放，让 DGB 单项控制** |

---

**📝 维护者**：cc 主线 + gemini 3.5 high 第 7 轮终审 | **签发**：2026-05-22 凌晨
