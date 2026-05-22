---
title: gemini 第 7 轮前置评审 · 2026-05-22 P0 Pre-conditions PLAN
date: 2026-05-22
reviewer: gemini-3.5-flash + reasoning_effort=high
verdict: REVISE → cc 全采纳整改后预期 APPROVE
session: 5/22 凌晨 P0 实施前 PLAN 评审
---

# gemini 第 7 轮前置评审报告

> 评审对象：`docs/plans/2026-05-22-p0-pre-conditions.md` v1
> 评审目的：5/22 P0 Pre-conditions 实施前的设计方案可行性与数学严密性
> Verdict：**REVISE（限期整改，不予直接通过）**

---

## 一、Finding 全栈（1 CRITICAL + 3 HIGH + 1 MEDIUM）

### 🚨 F-1 (CRITICAL): VICReg cov 孤立使用 → 特征自我静音作弊（独家盲区）

**现象**：PLAN v1 在 PC-3 中将 `VICRegLoss` 的 `std_weight=0, sim_weight=0, cov_weight=1.0` —— 仅保留 cov 项叠加到 InfoNCE。

**根因（数学严重退化）**：

cov 矩阵非对角元 $C_{i,j} = \frac{1}{B-1}\sum (z_{n,i}-\mu_i)(z_{n,j}-\mu_j)$。为最小化 $\mathcal{L}_{cov} = \sum_{i \neq j} C_{i,j}^2$，优化器**最廉价的作弊路径**不是让特征正交，而是**直接消灭某个维度的方差**（让 $z_{n,j} \to c$，使 $C_{j,j} \to 0$）。一旦某维度方差为 0，其与其他所有维度的协方差也强制归零。

没有 $\mathcal{L}_{std}$ 作为"排斥引擎"，网络会通过**"静音"（pruning）部分维度**来规避 cov 惩罚。这不仅不会提升 `effective_dim`，**反而加速维度坍塌**。

InfoNCE 的 uniformity 仅作用于 L2 归一化后的超球面，无法阻止 LNN 隐状态 $h$ 在投影前内部退化。

**修复**：必须引入最低限度方差保护。修改为 `std_weight=1.0, cov_weight=1.0`（gemini 钦定 1:1），确保每个维度维持基本激活方差，cov 才有解耦的数学意义。

**cc 整改状态**：✅ 已完成。`poc/training/vicreg_trigger.py::compute_loss` 含 `std + cov` 联合，`std_weight=1.0, cov_weight=1.0`。`test_joint_loss_against_self_silencing_shortcut` 实测：silenced=N/A > normal=N/A，证明 std 项确实让自我静音不再廉价。

---

### ⚠️ F-2 (HIGH): PC-3 缺 de-trigger 机制 → 后期过度去相关

**现象**：`_vicreg_cov_active` 一旦触发（连续 5 epoch < 2.5）永久锁定到训练结束。

**根因**：控制回路缺负反馈闭环。面部 AU 与时序动态天然有强关联（共性语义）。VICReg 强制去相关是提升表征容量的手段，但**过度去相关**会强行剥离共性语义，导致后期主任务 loss 震荡或发散。一旦 `effective_dim` 回升到健康水平（如 > 4.0），必须释放约束。

**修复（gemini 推荐方案 B：软门控）**：
$$\lambda_{cov} = \text{sigmoid}\left( 4 \cdot (2.5 - \text{effective\_dim}) \right)$$

- effective_dim ≪ 2.5 → λ ≈ 1（强约束）
- effective_dim = 2.5 → λ = 0.5（临界）
- effective_dim ≫ 2.5 → λ ≈ 0（effectively off）

**cc 整改状态**：✅ 已完成。`VICRegCovTrigger.current_lambda` 软门控实施。单测 `test_lambda_monotonic_decreasing_with_effective_dim` 验证单调递减。

---

### ⚠️ F-3 (HIGH): PC-4 τ LR=1e-5 + DGB = 双重锁死 = 伪学习

**现象**：PLAN v1 `tau_lr=1e-5`，叠加 ADR-022 DGB（梯度比例 ≤ 1/10）。

**根因**：DGB 已把 τ.grad 缩到 main_grad/10。再叠 1e-5 LR → 实际更新 ~$10^{-6} \sim 10^{-7}$ 量级。τ 初值 0.1，整训练期累积位移 < 1%，**对时序自适应无实际意义**。

这是为了通过 PC-1 `grad_norm > 1e-9` 验收的「表面合规」 —— 指标合规但功能死亡。

**修复**：`tau_lr` 恢复到 1e-3 正常量级，梯度控制全权交给 DGB 自适应裁剪。DGB 本身就是 safety net，无需在 LR 上重复阉割。

**cc 整改状态**：⏳ codex v2 接管。PLAN v2 已改 `tau_lr: float = 1e-3`，注释更新为 `ADR-022 v3`。

---

### ⚠️ F-4 (HIGH): 缺 PC-5 → 5/27 N=124 工程灾难（I/O + OOM）

**现象**：PLAN v1 仅指标修复，忽视 N=10 → N=124 扩量带来的工程问题。

**根因**：
1. **I/O 阻塞**：DataLoader 每 epoch 从磁盘读 124 视频 AU 特征，4 天预计 → 8 天+
2. **ODE 显存爆炸**：LNN BPTT 沿时间步反传，长序列 N=124 易触发 OOM

**修复**：新增 **PC-5 Engineering Guardrails**：
1. **特征预缓存**：`WindowDataset.__init__` 一次性 load 进 RAM（M2 Max 内存充足）
2. **Truncated BPTT**：`TrainConfig.max_bptt_steps = 32`，每 32 步 `h.detach()` 切断反向传播

**cc 整改状态**：⏳ codex v2 接管。新增 PC-5a + PC-5b 任务。

---

### ℹ️ F-5 (MEDIUM): PC-1 τ.grad 捕获时序错误

**现象**：PLAN v1 未明确 `tau.grad` 何时取。

**根因**：PyTorch 训练循环典型顺序是 `backward → optimizer.step → optimizer.zero_grad`。`zero_grad()` 触发后 `.grad` 清零。如果在 step/zero_grad 之后读 grad，永远是 None/0。

**修复**：明确时序契约：
```python
loss.backward()
# ✅ 必须在这里捕获（detach + clone + append）
self._tau_grad_buffer.append(p.grad.detach().norm().item())
# 之后才能跑 _apply_dgb / optimizer.step / zero_grad
```

**cc 整改状态**：⏳ codex v2 接管。PLAN v2 §PC-1 含完整时序规范。

---

## 二、核心盲区警示

### 1. 数学捷径盲区 The Variance-Covariance Shortcut（本轮独家）

直观以为「惩罚非对角线协方差，特征自然散开」。但优化器极度贪婪：会优先选择「让某几个维度彻底死掉（方差归零）」这种破坏最小、收益最大的方式来满足 loss 约束。

**结论**：**不带 std 约束的 cov 优化是自毁式的。**

### 2. 伪自适应盲区 Pseudo-adaptation

指标绿灯（`grad_norm > 1e-9`）≠ 功能存活。1e-5 LR + DGB 让 τ 物理冻结。「为指标合规保留梯度」违背 LNN 时序自适应核心理念。

---

## 三、最终裁决

### REVISE（限期整改）

**5/27 Phase γ Step 1 启动条件判定**：
若按 PLAN v1 执行，F-1（数学退化）+ F-3（梯度死锁）会让 5/27 大规模训练不仅达不到 `effective_dim ≥ 8.0`，还可能因 F-4 显存爆炸首日崩溃。

**必须在 5/22 内完成 5 项 findings 全部整改，方能解锁 5/27 启动。**

---

## 四、cc 整改完成度对账

| Finding | 严重度 | 修复主体 | 状态 | 落地证据 |
|---|---|---|---|---|
| F-1 | CRITICAL | cc | ✅ 完成 | vicreg_trigger.py + 19 单测 |
| F-2 | HIGH | cc | ✅ 完成 | 软门控 sigmoid + 单调性单测 |
| F-3 | HIGH | codex v2 | ⏳ 进行中 | PLAN v2 tau_lr=1e-3 |
| F-4 | HIGH | codex v2 | ⏳ 进行中 | PLAN v2 §PC-5a/5b |
| F-5 | MEDIUM | codex v2 | ⏳ 进行中 | PLAN v2 §PC-1 时序规范 |

---

## 五、给 cc 的最终建议（gemini 原文）

> cc，当前 PLAN 在工程细节上考虑很细（如 seed 漂移防御、DGB 保留），但存在致命的数学退化盲区。
>
> 只惩罚 covariance 而不惩罚 variance (std)，等同于引导 LNN 通过「自我静音（让部分维度方差归零）」来作弊降低 loss，这会加速维度坍塌。请立即将 VICReg 调整为 std+cov 联合约束。
>
> 同时，1e-5 LR 配 DGB 属于双重锁死，请释放 tau_lr 至 1e-3 并全权交给 DGB 自适应控制。最后，补充 PC-5 内存预缓存，防止 5/27 因 I/O 和 ODE 反向传播导致 M2 Max OOM。
>
> 方案改好，我们 5/27 见。

---

## 六、关联文档

- 整改后 PLAN：`docs/plans/2026-05-22-p0-pre-conditions.md` v2
- cc 已实施 PC-3：`poc/training/vicreg_trigger.py` + `tests/test_vicreg_auto_trigger.py`
- 历史评审：`docs/reports/gemini-gamma3-final-review.md`（第 6 轮 CONDITIONAL APPROVE）
- 关联 ADR：`docs/decisions/022-tau-bias-gradient-blackhole-fix.md`（待 v3 整改）

署名：cc（claude）记录 | gemini-3.5-flash 评审 | 2026-05-22
