# ADR-023: 范式级转换 · InfoNCE+L2-norm → ODE 动力学 Loss

> **决策者**：川哥钦定 / cc 主理 + codex 第 9 轮 + gemini 第 9 轮多轮证据收敛
> **日期**：2026-05-22
> **状态**：accepted
> **关联**：P-1 LNN-FIRST · ADR-016 base v3 · ADR-022 v3 (τ-bias) · 不可妥协 §10.1

---

## 1. 触发事件

5/22 5 组 sweep 实测：**effective_dim 全部 3.9515 ± 0.003**（threshold/cov_w/warmup/steepness 全无影响）。λ_cov 从 0.003 拉到 0.728（240x）也没动 effective_dim 一根毛。

## 2. 三层根因递进盘点

| 层 | 发现轮 | 诊断 | 关闭方式 |
|---|---|---|---|
| L1 | gemini 8 轮 | "表征冷凝" 误以为 std_weight=0 | 修订 → 8b 解耦 |
| L2 | D 阶段双引擎收敛 | std+cov 被 λ 整体压死 | v3 解耦修法（commit 00ee249）|
| **L3** | **codex 9 轮实测打破** | **effective_dim=sum(s)/s[0] 对全局缩放不变；std_target 改值只改常数项；cov_loss 不直接惩罚低秩** | **本 ADR 决策** |

**真根因**：不是参数，是**整个 InfoNCE+L2-norm 自监督对比学习框架**对 LNN 项目结构性不适配：
1. InfoNCE 用 cosine 相似度 → 只关心方向，不约束表征结构维度
2. L2-norm 强制 ‖z‖=1 → 把 LNN 想要的线性动力学压到非线性球面（gemini 9 论据）
3. cov_loss 惩罚的是当前坐标基协方差，不是低秩本身 → 模型走「低秩 + 近对角」绕开
4. effective_dim = sum(s)/s[0] 对幅值不变 → 即使 std 把幅值拉起来 metric 仍不动

## 3. 范式错位历史回溯

`spec/base-v3-training-spec.md` 当初选 InfoNCE 是借鉴 **SimCLR / MoCo SOTA 自监督**。但：
- SimCLR 是**静态图像**对比学习，**没有时序动力学概念**
- LnMem 是**时序 ODE 模型**（CfC 微分方程内核）
- 用 InfoNCE + L2-norm + AU augment 当主 loss = 把图像 SOTA 范式套到 ODE 项目

正如 mycc memory `feedback_user_mental_model_over_industry_pattern.md` 5/20 钦定：「把方案抽象成行业范式会丢失用户真实意图」。这是 cc 设计阶段就埋下的盲区。

## 4. 决策

**主损失从 InfoNCE+L2-norm 切到 ODE 动力学 loss 组合**。

### 4.1 数学结构

LNN ODE：`dh/dt = (f_net(x_t) + τ) ⊙ h`

主 loss = **多步 rollout 预测 + 重构自编码 + ODE 一致性**：

```
L_total = w_pred · L_predict + w_recon · L_recon + w_ode · L_ode + w_infonce · L_infonce_aux
```

#### L_predict（主目标：能预测未来）
给前 N 帧，让 LNN forward 预测后 K 帧：
```
L_predict = (1/K) · Σ_{k=1..K} MSE(h_pred[N+k], h_actual[N+k])
```
where `h_actual[N+k]` 是 backbone 直接 encode 真实输入得到的 hidden。

#### L_recon（强信息约束）
encoder + decoder，要求 latent 能解码回输入：
```
L_recon = MSE(decoder(h[t]), x[t])
```
强制 hidden 必须保留 input 的所有信息维度 → 自然反对维度坍塌。

#### L_ode（轻量自洽）
离散化跟一阶 Euler 自洽：
```
L_ode = || h[t+1] - (h[t] + dt · (f_net(x_t) + τ) ⊙ h[t]) ||²
```
权重轻（0.1），避免把网络锁死在严格 Euler 解。

#### L_infonce_aux（辅助保留时序对齐）
保留 InfoNCE 一小权重（0.1）保证同视频片段表征相似性。**不再是主损失**。

### 4.2 推荐权重初始值（codex / gemini 第 10 轮会再审）

```python
w_predict = 1.0
w_recon = 1.0
w_ode = 0.1
w_infonce = 0.1
```

### 4.3 网络结构改动

- **新增 Decoder**：64-d hidden → 52-d input space（简单 Linear 或 2-layer MLP）
- **CfC backbone 不动**（保留 ADR-022 v3 τ-bias 释放）
- **L2-norm 移除**（仅 L_infonce_aux 内部保留）
- **VICReg trigger 弃用**（重构 loss 自然撑大 effective_dim，不需要 cov 软门控）

### 4.4 数据流改动

- **不再需要 contrastive pairs**（anchor/positive/negatives 三元组废弃）
- 直接 (B, T, 52) 时序输入
- 切窗口 → 在窗口内做 prefix-predict 切分（seq_len=30 → prefix=15, predict=15）
- **AU augmentation 移除**（重构 loss 不需要 augment 防 shortcut，shortcut 自然被重构约束捕获）

## 5. 5/27 Phase γ Step 1 启动条件重新对账

| 原 PC | 新 PC（基于 ADR-023） |
|---|---|
| PC-1 τ-Monitor | ✅ 保留（ADR-022 v3 仍有效） |
| PC-2 DataLoader shuffle | ✅ 保留 |
| PC-3 VICReg std+cov | ❌ 弃用（被 ODE-loss 替代） |
| PC-4 τ LR=1e-3 + DGB | ✅ 保留 |
| PC-5a memory cache | ✅ 保留 |
| PC-5b truncated BPTT | ✅ 保留 |
| **新 PC-6** Decoder + L_recon | ⏳ ADR-023 新增 |
| **新 PC-7** L_predict 多步 rollout | ⏳ ADR-023 新增 |
| **新 PC-8** L_ode 一致性 | ⏳ ADR-023 新增 |

## 6. 4 项硬门重对账

| 硬门 | 原 InfoNCE 框架 | ODE-loss 框架 |
|---|---|---|
| #1 ratio ∈ [0.1, 50] | ADR-022 监控 | ✅ 不变 |
| #2 Shuffle Test M1 drop > 0.05 | M1 retrieval drop | ⏳ 改为 L_predict drop（重构/预测在 shuffle 后必崩）|
| #3 spatial aug + M5 < 0.15 | AU augment + 检索 acc | ❌ 弃用（aug 不用了，M5 改用 reconstruct error vs video_id 分类）|
| #4 M1-V N/A spike | 文本 disabled spike | ✅ 不变 |

## 7. 预测（基于 ADR-023 实施后）

50 epoch tiny N=10：
- **effective_dim**：8-12（重构 loss 强信息约束自然撑大）
- **Shuffle drop（用 L_predict）**：≥ 0.20（时序打乱后预测必崩）
- **L_recon final**：0.05-0.10（input 范围 [0,1] 下接近无损重构）
- **L_predict final**：0.20-0.40（短期预测良好，长期渐失）
- **总训练时间**：tiny 50 epoch ~15 分钟（比 InfoNCE 略慢，因为加 decoder forward）

## 8. 5/27 RED LIGHT 暂缓判定

**🟥 RED LIGHT** N=124 启动暂缓，必须先在 tiny N=10 验过 ADR-023 实施：
- effective_dim ≥ 8.0
- Shuffle drop ≥ 0.05（用新 metric）
- L_recon < 0.15

通过后才放行 Phase γ Step 1。

## 9. 风险登记

| 风险 | 应对 |
|---|---|
| Decoder 容量过大 → trivial 重构 | Decoder 必须 < CfC 1/4 参数量（Linear 或 1-layer MLP）|
| L_predict 长程不稳定 | K 步从 5 开始，分阶段扩到 15 |
| 完全弃用 InfoNCE 失去时序对比 | w_infonce_aux=0.1 保留 |
| ODE 自洽 loss 网络死锁 | w_ode=0.1 轻量 |
| 重构能力过强主导 latent | w_recon 可调，先 1.0 实测 |
| 这是范式级改动可能引入新盲区 | 派 gemini 第 10 轮 + codex 第 10 轮**并行**评审 |

## 10. LNN-FIRST 复对账

按 P-1 最高准则逐项核对：

| §10.1 不可妥协项 | ADR-023 影响 | 通过 |
|---|---|---|
| #1 LNN-FIRST | ✅ loss 直接训 ODE 动力学，是**强化**而非削弱 LNN |
| #2 per-user τ-bias | ✅ ADR-022 v3 保留 |
| #3 AU 多模态 | ✅ EventTensor schema 不变 |
| #4 离线优先 | ✅ 重构 loss 完全自监督 |
| #5 ODE 可解释 | ✅ 增强（loss 形式跟 LNN 公式直接对应）|
| #6 时序记忆 | ✅ 增强（多步 rollout 直接训时序）|

**结论**：ADR-023 全部 6 项合规且**多项强化**。

## 11. 实施时间线

| 阶段 | 时间 | 内容 |
|---|---|---|
| 5/22 上午（now）| 1 小时 | 本 ADR 落盘 + 派 codex 实施 + gemini 评审 spec |
| 5/22 下午 | 2 小时 | 整合 + tiny dry-run + 反向对抗 |
| 5/22 晚 | 1 小时 | Studio tiny 50 epoch 真训练 |
| 5/23-5/24 | 2 天 | 验收 + 调权重 + 沉淀第二版 spec |
| 5/25-5/26 | 2 天 | Phase γ Step 1 启动条件复核 |
| 5/27 | 启动 | 基于 ADR-023 跑 N=124（4 天）|

## 12.5 v1 → v2 整改（gemini 第 10 轮 BLOCK 钦定 2026-05-22）

cc 写的 spec v1 被 gemini 第 10 轮抓出两个**致命数学错误**：

### F-1 FATAL: L_predict 恒等死局
v1 设计 teacher-forcing rollout：
```
h_pred[N+k] = backbone_step(x[N+k], h_pred[N+k-1])
```
因 `h_pred[N-1] = h_seq[N-1]`，数学归纳得 `h_pred[N+k] ≡ h_seq[N+k]` 对所有 k，L_predict ≡ 0 完全无梯度。

### F-2 CRITICAL: L_ode Euler vs CfC 闭式解
CfC 用 `e^{-(τ+f)·dt}` 指数门控半隐式解析解，**显式拒绝 Euler**。强加 Euler 自洽 = 把 CfC 退化成 ResNet，废掉 ADR-022 τ-bias 释放。

### v2 改造：Latent World Model Rollout（Dreamer 风格）
**核心**：rollout 阶段 LNN **拿不到真实 x**，用 decoder 自回归预测的 x_pred 喂回。L_world_model 数学上必非零且无法塌陷。

详细 spec：`docs/specs/ode-dynamics-loss-v2.md`

新权重（gemini 钦定）：
- w_world_model = 1.0
- w_recon_prefix = 0.5
- w_infonce_aux = **0.0**（完全弃用，置信度 99%）
- w_ode = **0.0**（完全弃用，与 CfC 冲突）

5/27 verdict 第 4 次更新：仍 🟥 RED，但放行条件锁定 v2 实施。

---

## 12. 关联

- ADR-022 v3 τ-bias 释放 + DGB
- 旧 spec `docs/specs/base-v3-training-spec.md` 需大改（本 ADR 通过后下一步）
- 旧 PLAN `docs/plans/2026-05-22-p0-pre-conditions.md` 中 PC-3 标弃用
- mycc memory `feedback_user_mental_model_over_industry_pattern.md` 范式教训复用

---

**📝 维护者**：cc 主理 + codex 第 9 轮 + gemini 第 9 轮多轮收敛 | **签发**：2026-05-22
