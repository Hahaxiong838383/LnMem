---
adr-id: 013
title: base v2 川哥真实数据 A/B 实测失败根因 + base v2.1 微调方案
status: accepted（2026-05-21，cc 主线钦定）
date: 2026-05-21
deciders: 川哥 + cc + codex (5.4 high 实施 + 实测) + gemini (3.5-flash 方法论评审)
supersedes: —
related: ADR-007 Phase 1.5 / ADR-012 V-A→state / ADR-005 dual signal_path
---

## 上下文

ADR-007 Phase 1.4 train_base_v2.py 昨晚 23:24 完成训练（详见 [`docs/retrospectives/2026-05-20-day1-team-mode-retro.md`](../retrospectives/2026-05-20-day1-team-mode-retro.md)）：
- VEATIC 12400 帧训练，9-state Russell circumplex
- val_state_acc 0.48 / val_ccc_v 0.09 / val_ccc_a 0.05
- learned_sigma 0.349 → 0.326（σ 端到端可学验证成功）
- 6 ckpts 落 Studio + best 226KB rsync 到 Air

今早（2026-05-21 09:00）派 codex 5.4 high 跑 A/B replay（`scripts/ab_replay_ckpt_v1_v2.py`），结果详见 [`docs/reports/2026-05-21-base-v2-ab-replay.md`](../reports/2026-05-21-base-v2-ab-replay.md)：

**🚨 base v2 在川哥真人数据上完全失败**

## 关键实测数据

### Replay 范围
- 12 天 queries（5/9-5/20）共 428 events
- 筛选 `reason=brow_burst AND user_response.true_state IS NOT NULL` → **67 evaluable**
- 跳 3 个 missing window → **64 replayed**
- response_label_counts: `{mismatch: 64}` — **全是用户已纠正的 mismatch 样本**
- true_state_counts: `{distracted: 42, focused: 22}`

### 结果
| metric | base v1 | base v2 | delta |
|--------|---------|---------|-------|
| mismatch_rate | 100.00% | 100.00% | 0.00% |
| v2 top-1 = neutral | — | **64/64** | — |
| v2 次高 logit fallback | — | 主要 fatigued | — |

### Confusion Matrix（base v1 真人数据 64 mismatch 样本）
- focused (22 真值) → emotional_negative 11 / transitioning 11（**全错**）
- distracted (42 真值) → emotional_negative 19 / transitioning 20 / emotional_positive 3（**全错**）

## 两个根本问题（cc 自承）

### 🔴 P1: cc 验收 spec 设计错误（口径错配）
- ADR-007 plan 写的 "brow_burst mismatch 30.7%" 是 **trigger-level**（203 brow_burst 中用户 mark mismatch 比例）
- cc 让 codex 算的 spec 是 **state-level**（已知 mismatch 样本上预测对不对）
- 两个数字 by definition 不同，不能直接对账
- 实际 evaluable filter 后 64/64 全是 mismatch 样本 → mismatch_rate 锁死 100%

### 🔴 P2: base v2 模型坍缩到 neutral
- 64/64 全 top-1 neutral state=0
- VEATIC val 0.48 acc 不代表川哥真人场景
- 暴露 **domain shift + class collapse** 双重问题

## 根因分析（5 个候选）

| 候选 | 描述 | cc 当前估算概率 |
|------|------|---------------|
| A | 1 persona vs 31 persona — partial load 时 persona embedding 从 31 行均值压成 1 行，丢失 vboss_self 个性化 | 30% |
| B | VEATIC 西方人 + 视频 vs 川哥亚洲人 + daemon 实时 — 分布偏移 | 25% |
| C | class-balanced sampling 让 neutral 占 59% 但 8 扇区极不均（state 2/7 全空），readout 学到 "全 neutral 安全" | 20% |
| D | σ 端到端学 → collapse 软标签，所有样本接近 neutral | 15% |
| E | 9-class readout partial load 后新初始化，early_stop 18 太早 | 10% |

实际可能是 A+B+C 三者叠加。gemini 3.5-flash 评审待入。

## 决策

### 短期（今天）：**不上线 base v2**

base v1 + ADR-010/011 guard 继续做 daemon 主模型（已稳定跑 1 周）。base v2 失败原因写本 ADR 备案，不阻塞 demo 主线。

### 中期（本周 5/22-5/26）：**用川哥真人数据 fine-tune base v2 → base v2.1**

策略待 gemini 评审 + cc 决策。候选：
1. **冻结 CfC backbone + 仅 fine-tune readout**（保 VEATIC 学到的 V-A 表征，让 readout 适应川哥）
2. **distillation**：base v1 当 teacher 教 v2 学川哥 daemon signal 分布
3. **直接 supervised fine-tune**：64 mismatch label + soft assignment + class-weighted CE

数据资源：
- 428 queries（其中 67 含 true_state）
- ~/.lnmem/data/signals_2026-05-*.jsonl 数十万帧 unlabeled signal
- vboss_self persona_idx=30（base v1 中已训练）

### 长期（M1+）：**per-user persona 三件套**

详见 v0.3.x 产品定义 §5.2 L3：
- τ-bias / FiLM / 槽位偏好向量
- 3KB/user CPU 可推理
- 用川哥真实 daemon signals 持续在线漂移（fit_orchestrator 已就位）

## 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| 1 | 64 标注样本太少做 supervised fine-tune（可能过拟合）| gemini 评审吸收 + 加 data augmentation / 自监督预训练 |
| 2 | base v2.1 微调后仍坍缩 | dropout / class weight / margin loss / 早停 patience 更宽 |
| 3 | 重训 base v3 from scratch 时间成本 | 仅在 v2.1 仍失败时启动，先尝试低成本 fine-tune |
| 4 | 当前 daemon 跑 base v1，切换 v2.1 时如果效果反降需要快速回滚 | A/B 验收门：v2.1 vs v1 在 trigger-level mismatch 改善 ≥ 30% 才上线 |

## 给后续 cc 的纪律（spec 设计教训）

**验收口径错配是 cc 主线 spec 设计 P0 错误**。教训：

1. ADR 写 "mismatch 30.7%" 时必须**明确口径**（trigger-level vs state-level vs frame-level）
2. 验收脚本 spec 必须**复现 ADR 原口径**，不能"换个角度算"
3. 多类口径并存时**全报**，不能只挑一个
4. **codex 实测**是 cc spec 错误的最快暴露途径，要尊重实测数据，不能强行解读

## Gemini 3.5-flash 评审吸收（已入）

**根因 Top 1**：9-class readout 重初始化 + 极度欠拟合 → CfC 学到 **shortcut**（捷径坍缩到 V-A 原点 neutral）
**根因 Top 2**：Persona Conditioning 失效（31→1 均值压缩破坏 FiLM 动态轨迹）

### DFAT 微调方案（Decoupled FiLM-Adapter Tuning）

4 阶段实施：
1. **锁骨干**：CfC backbone + input_projection 全 requires_grad=False
2. **注入 vboss_self FiLM**：从 base v1 ckpt[idx=30] 加载 persona embedding
3. **暖场自监督**：用数十万帧 unlabeled signal 跑 Temporal Neighborhood Contrast（InfoNCE），让 FiLM γ/β 适应川哥真实信号动力学
4. **少样本有监督**：67 真实 label + Russell 距离惩罚 + SupCon (Khosla 2020) + 代价敏感矩阵（focused→negative 给 5x 梯度惩罚）

### 3 项强制要求（gemini 钦定）

| # | 要求 | 实施验证 |
|---|------|---------|
| 1 | 禁止全量微调 | trainable params < 10% 必断言 |
| 2 | 强制 persona_idx=30 vboss_self | 加载后 embedding 数值 == base v1[30] |
| 3 | 评估输出 confusion / Russell 距离 / 预测熵 | 熵 ≈ 0 时报警 collapse |

**Gemini 置信度评估**：今天下午跑出明显改善 **85%**。

理由：100% neutral 坍缩非 backbone 无能力，是 readout 重初始化 + 极少 epoch 的**捷径解**。冻骨干 + 恢复 vboss_self FiLM + 小样本强惩罚微调 → 表征空间分布快速拉回。

## 关联产出

- `scripts/ab_replay_ckpt_v1_v2.py`（493 行 + 3 测试）— codex 实施
- `docs/reports/2026-05-21-base-v2-ab-replay.md` — codex 真实 replay 报告
- `/tmp/lnmem-team-2026-05-20/gemini_base_v2_failure.md` — gemini 评审完整文本（116 行）
- `/tmp/lnmem-team-2026-05-20/codex_finetune_v2_1_spec.md` — cc spec（吸收 gemini）
- `scripts/finetune_base_v2_1.py`（codex 实施中，PID 50424）— DFAT 微调脚本

## Summary

base v2 在川哥真人数据上 64/64 坍缩到 neutral，**不上线**。
两个根本问题：(1) cc 验收 spec 口径错配（trigger vs state），(2) 模型 domain shift + class collapse。
中期方案 base v2.1 fine-tune 用川哥真人数据，策略待 gemini 评审 + cc 决策。
长期回到产品定义 v0.3.x 的 per-user persona 三件套（M1+）。

---

**模板版本：v0.1 · 2026-05-03**
