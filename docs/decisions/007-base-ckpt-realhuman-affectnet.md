# ADR-007: base ckpt 真人化（AffectNet V-A 替代合成 LTC 数据）

**状态**：accepted（M2 启动前必做，今日先锁路线）
**日期**：2026-05-16
**决策者**：川哥（最终） + cc（设计） + gemini（数据集推荐）+ codex（对抗审查）
**触发**：5/16 spike LNN realtime infer 在川哥真人脸上 state 准确率 = 0%
**Supersedes**: 无（与 ADR-003/005 互补，不冲突）

---

## 背景

5/16 三个证据链：
1. **lnn-first.md §4.1 早警告**："合成数据特征'饱和'是 PoC 真人验证最大杀手 — 合成 LTC 数据 emotional_positive 时 smile=0.98（饱和），真人 py-feat 输出通常只到 0.4-0.6 → train-inference 分布错位"
2. **5/16 spike 实证**：LNN forward 在川哥 15 events 上 state 预测准确率 = 0%
   - 川哥告知 focused 5 次 → LNN 预测 emotional_negative 3 + neutral 1 + transitioning 1
   - tau_drift 只 force fit 1 次（batch=5）远不够纠正
3. **gemini 调研**：AffectNet 优于 FER2013 — "Valence-Arousal 连续变量空间完美契合 LTC 微分方程的平滑演化需求，远优于 FER2013 的离散阶跃分类"

## 决策

**用 AffectNet（真人面部 V-A 连续标注）+ 早期用户真实 raw_au 数据，重训 base ckpt v2**。

替换合成 LTC 数据作为 base，让训练分布 ≈ 真人分布，新用户冷启动从 1-2 周 → 1-2 天。

## 选项与对比

| 选项 | 数据 | 优点 | 缺点 | 推荐度 |
|---|---|---|---|---|
| A. FER2013（7 类离散） | 35K 图片 | 公开 + 简单 | 离散标注与 LTC 连续演化不匹配 | ⚠️ 备用 |
| **B. AffectNet（V-A 连续）** | 1M 图片 | **连续 V-A 契合 LTC** | 仅学术 license（商用风险） | ✅ 选 |
| C. RAF-DB（混合标注） | 30K | 中等 | 标注质量较散 | 备用 |
| D. EmoNet（pre-trained） | - | 直接用 | 黑盒，可解释性差 | 备用 |
| E. 自采真人数据集 | - | 完全合规 | 工程量极大 + 数据偏小 | 不可行 |

## gemini 风险警告

> "多数面部集（含 AffectNet/RAF-DB）仅限学术，**强商用存在极大版权风险**"

**应对**：
- M2 阶段（早期用户群 10-50）仍 PoC + 学术研究身份 → AffectNet 可用
- M3+ 商用化前必须：
  - (a) 联系 AffectNet 团队商谈商用 license（pay-as-you-go 模式可能）
  - (b) 或自采替代数据集（用早期付费用户 opt-in 数据 + 公开 Creative Commons 数据集如 DISFA）
  - (c) 或转 EmoNet 等 commercial-OK 数据集

## 实施分阶段

| 阶段 | 工作 | 工时 | 时间窗口 |
|---|---|---|---|
| ADR-007 落盘（本文档） | 锁定方案 + 风险 | 30min ✅ | 今日 |
| 数据准备 | 下载 AffectNet + 跑 py-feat 提 AU 分布 | 1 天 | M0 W3 末 |
| schema 映射设计 | V-A 连续 → 8 state 离散映射规则 | 半天 | M0 W3 末 |
| 重训 base ckpt v2 | 训 + 评估 + A/B 对比 | 2-3 天 | M1 初 |
| 商用 license 解决 | 走法务流程 | - | M2 启动前 |

## V-A → 8 state 离散映射设计

| 8 state | V (valence) | A (arousal) |
|---|---|---|
| focused | 0 ~ 0.3 | 0.3 ~ 0.7（中等专注） |
| distracted | -0.2 ~ 0.2 | 0 ~ 0.3（低警觉） |
| fatigued | -0.3 ~ 0 | 0 ~ 0.2（低能量低唤醒） |
| emotional_positive | > 0.5 | > 0.3 |
| emotional_negative | < -0.3 | > 0.3 |
| low_energy | -0.2 ~ 0.2 | 0 ~ 0.2 |
| neutral | -0.1 ~ 0.1 | 0.1 ~ 0.4 |
| transitioning | （V/A 快速变化点）边界帧 |

具体阈值待数据驱动 sweep 调整。

## 后果

### 正面
- 新用户冷启动从 1-2 周 → 1-2 天（不再 0% OOD）
- 模型有"一般人类先验"，per-user fine-tune 只需 100+ events 而非 1000+
- 直接消除 ADR-005 v0.3 spike 暴露的根因

### 负面 / 风险
- license 风险（gemini 警告）
- 2-3 天工程量（M1 初优先级）
- V-A → 8 state 映射可能丢信息（需 ablation 验证）

### 触发的下游决策
- ADR-008 User-LoRA（共用 base ckpt v2，每用户学 LoRA 个性化）

## 反对意见 / 不同看法

- cc 原 ADR-006 编号（与 ADR-005 占位冲突）→ 改 ADR-007
- gemini 商用 license 警告 → 加显式应对方案

---

**末尾自检**：本 ADR 与 ADR-003/005 互补（不冲突）。ADR-003/005 解决"raw 信号通路"，本 ADR-007 解决"训练数据分布"。
