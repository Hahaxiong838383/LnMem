---
adr-id: 016
title: base v3 多模态 self-supervised LNN — 真 LNN-FIRST 训练范式（撤回 ADR-015 视觉冻结）
status: accepted（2026-05-21 13:55，川哥钦定 + cc 互校 gemini 后认知升级）
date: 2026-05-21
deciders: 川哥（产品愿景 + LNN-FIRST 守门员） + cc（执行）
supersedes: ADR-015（视觉冻结决定撤回） / ADR-013 v2.1 supervised 路径
related: ADR-007 / ADR-009 LNN-FIRST 拒绝 RAG / docs/product-definition.md §10.1 #2 多时间尺度 LNN
---

## 上下文

### 川哥关键洞察（2026-05-21 13:35）

cc 主线被 gemini "视觉权重 2/10" 误导差点写错 ADR-015 视觉冻结。川哥拉回：

> "视觉感知在 LnMem 中也重要，我在输入'昨晚 11 点和小李在公司吃了越南粉'的时候，AI 得知道我是什么表情，我是什么意图，我是哭着说还是笑着说？我是疑问句还是反问句？LnMem 使用 LNN 神经网络，就是为了让 AI 懂并且能预测用户在输入需求或者沟通时的心情状态及心理活动。不要用传统大模型的思路来干这件事。"

### cc 主线认知错误复盘

base v2 / v2.1 全是**传统 LLM-centric supervised 分类思路**：
- 把 LNN 当分类器训（9-state CE + Russell margin）
- 用硬标签（focused/distracted 等）
- 评估用 mismatch rate

**真 LnMem LNN-FIRST 不是这样**：
- 产品定义 §0：「**用液态神经网络（LNN）作为永远在线的心智核心**」
- §10.1 #2：多时间尺度 LNN 心智核心（不可妥协）
- ADR-009：拒绝 RAG 捷径坚守 LNN-FIRST（哲学级）

LnMem 视觉的真正用途：
- ❌ **不是**「识别川哥是 focused 还是 distracted 当 LLM context」
- ✅ **是**「川哥说话时的表情/语气/情绪作为 LNN 输入维度，让 hidden state 学到主人多模态连续表征」

### ADR-015 撤回理由

ADR-015 错在：
- 用 gemini supervised 评价指标（focused acc 0% → 失败）
- 没看到 LNN-FIRST 范式下 hidden state 表征学习的价值
- 把"分类错误"当"整个视觉路径失败"

ADR-016 撤回 ADR-015 视觉冻结决定，**视觉感知路径继续**，但**训练范式改**。

## 决策

### Base v3 核心范式

**Multi-modal Self-Supervised LNN with Weakly-supervised Readout**

```
[输入]
  文本（万能语法 20 槽 + LLM intent type）
  视觉（py-feat AU + 头部姿态 + 视觉情境）
  音频（语速/音调/沉默时长，M1+ 加）
  时间戳

  ↓ 多模态融合 → 32-64 dim

[LNN backbone (CfC) 自由学]
  - Persona conditioning: τ-bias + FiLM
  - 不分类，学连续 hidden state 演化
  - 多模态对齐 + 时序连续性 + persona 收敛

  ↓ 64-dim hidden state

[多 head readout (弱监督，附属)]
  Head 1: 8-state emotion (与 base v1 兼容)
  Head 2: V-A 连续回归 (CCC loss)
  Head 3: intent type (5 类：陈述/疑问/反问/感叹/请求)
  Head 4: confidence (uncertainty)
  Head 5: hidden_repr (64-dim 直接出给 KG 个性化用)
```

### 4 个训练 loss

**Loss 1：Temporal Contrast（主目标）**
- 同视频相邻 60-frame window → 应相似（h_t ~ h_{t+δ}）
- 远 window → 应远离
- InfoNCE 形式
- 数据：VEATIC 6GB 视频时序

**Loss 2：Cross-modal Contrast（M1+ 多模态加入后）**
- 同一时间窗口的视觉 + 文本 + 音频 hidden state → 应接近
- 这是 LnMem 真正的多模态融合

**Loss 3：Persona Contrast**
- 同 user 不同时间相似（slow drift）
- 不同 user 远离
- SupCon (Khosla 2020) 形式

**Loss 4：Weakly-supervised Readout（附属）**
- 只在标注样本上训 readout
- 不动 backbone
- 数据：67 川哥真实 label + EMOTIC + mst（多任务）

### 9.8GB 数据集分阶段定位（C 方案全留）

| 数据 | 用途 | 阶段 |
|------|------|------|
| **VEATIC 6GB** | 52w 帧视频时序 → self-supervised LNN 预训练 | M1 启动 Step 1 |
| **EMOTIC 3.4GB** | V-A-D 静态图 + 人物 bbox + 情境类别 → LNN 情境维度增强 | M1 中期 Step 2 |
| **mst 387MB** | 8 emotion class 静态图 → 多 head readout 弱监督训练 | M1 末 Step 3 |

跨域问题（西方人 → 亚洲人）用 self-supervised 鲁棒性缓解，最终用川哥真人 daemon 60w 帧微调收敛。

## Demo 升级路径（让川哥渐进看到效果）

### Phase α（今天 15:30-16:00，30min quick win）

**Demo 加 "📸 抓 webcam 表情" 按钮**：
- 输入文本时同步抓 1 帧 + py-feat 提 AU
- 显示当前 V-A + 8-state（用 base v1 + ADR-010/011 guard）
- 让川哥**视觉感受到多模态输入存在**

### Phase β（5/22-5/24，3 天）— 状态描述 schema

- LLM parser 加 `intent_type` 抽取（5 类：陈述/疑问/反问/感叹/请求）
- demo 输出**完整状态描述 JSON**：
  ```json
  {
    "text_raw": "昨晚 11 点和小李在公司吃了越南粉",
    "universal_grammar_20_slots": {...},
    "intent_type": "怀念分享",
    "emotional_state": {
      "v_a": [0.3, -0.1],
      "state_label": "relaxed_pleasant",
      "confidence": 0.72,
      "source": "webcam + text fusion"
    },
    "lnn_hidden_state": [...64 dim, given base v3],
    "follow_up_suggestions": [...]
  }
  ```
- 写入 graphiti KG 时包含完整状态描述

### Phase γ（5/27-6/10，2 周）— base v3 预训练实施

- VEATIC 全量 self-supervised 预训练（Loss 1）
- EMOTIC + mst 多任务 readout（Loss 4 主要训练数据）
- 输出 64-dim hidden state 给 demo 用

### Phase δ（6/11+，M1 起）— 川哥真人微调

- 累积 daemon 60w+ 帧 fine-tune base v3 backbone
- 启动 per-user τ-bias + FiLM + 槽位偏好向量
- daemon 切换 base v3，新 readout 输出多 head

## 与现有 ADR 关系

| ADR | 关系 |
|-----|------|
| ADR-007 base ckpt 真人化 | 部分超越：v3 取代 v2 supervised 路径，但 ADR-007 Phase 1.2 公共底座（extract_au_to_parquet）仍复用 |
| ADR-009 拒绝 RAG 坚守 LNN-FIRST | **完美贯彻**：v3 是真 LNN-FIRST（self-supervised hidden state 学习）|
| ADR-011 假脸 guard | 复用：v3 输入端仍需 guard |
| ADR-012 V-A→state Russell | 部分复用：v3 多 head readout 中的 V-A head 用 Russell 几何，但不是主目标 |
| ADR-013 v2 失败分析 | 完全承袭：v2 supervised 失败教训驱动 v3 范式转换 |
| ADR-014 决策检查点 | 承袭：commit 前回滚指南 |
| ADR-015 视觉冻结 | **撤回**：本 ADR 重启视觉路径 |

## 实施策略（不今天动）

**今天**：仅本 ADR 设计文档落地，不实施 base v3 代码。

**M1 启动条件**：
1. L4 主线 demo + CLS + UI 完成（W+ 砖 1/2/3）
2. 川哥真人 daemon 标注 events ≥ 200（数据量门槛）
3. cc 主线写出 base v3 实施 spec + gemini 评审

**先决条件**：v3 训练前需先：
- 升级 fusion vector 32-64 dim（加 intent + voice 维度）
- 写多 head readout 模型
- 写多 loss 训练框架
- 评估 self-supervised 表征质量的指标（hidden 聚类 / KG 召回质量）

## 不可妥协项检查

| 项 | 状态 |
|---|------|
| §10.1 #1 万能语法 20 槽 | ✅ 保留 + 增 intent_type |
| §10.1 #2 多时间尺度 LNN | ✅ **强化**（v3 才是真 LNN）|
| §10.1 #3 CLS | ⏳ 砖 2 今天做 |
| §10.1 #4 per-user 个性化三件套 | ✅ v3 backbone 准备好，M1 Phase δ 启动 |
| §10.1 #5 居家+车载双场景 | ⏳ M2+ |
| §10.1 #6 记忆 UI | ⏳ 砖 3 今天做 |

## 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| 1 | self-supervised 表征"看起来学到了"但下游任务不 work | 评估指标必含 KG 召回质量 + 跨样本 hidden 聚类清晰度 |
| 2 | 多模态融合 cross-modal contrast 数据缺音频/文本对齐时序 | M1+ 先单视觉模态，M2+ 加音频文本 |
| 3 | VEATIC 跨域到川哥真人仍存在 | self-supervised 比 supervised 鲁棒，且 Phase δ 用 daemon 数据微调 |
| 4 | 多 head readout 训练不稳定 | 每 head 独立 lr + gradient balancing |
| 5 | cc 又用 LLM 思路评价 base v3 | 评估必须看 hidden 表征质量，不是单 head acc |

## Gemini 评审吸收

Gemini 战略评审中**正确的部分**采纳：
- ✅ L4 主线优先（demo + CLS + UI）— 已在 W+ 执行
- ✅ UX 主动状态锚点（Cmd+Shift+F）— 留 Phase β 实施
- ✅ "用好解决的算法挑战逃避复杂系统集成" 警告 — cc 永久内化

Gemini 评审中**错误的部分**修订：
- ❌ "视觉权重 2/10" — gemini 用 LLM-centric 思路，没看到 LNN-FIRST 多模态价值
- ❌ "ADR-007 阶段性冻结" — 训练范式错不等于路径错，本 ADR 撤回
- ❌ "前台 App + 键盘频率替代视觉" — 工作类型 ≠ 情绪/意图

cc 主线纪律：gemini 评审是学术外脑，**不是最终决策者**；川哥从 LnMem 产品定义 + LNN-FIRST 准则层守门。

## Summary

撤回 ADR-015 视觉冻结决定。视觉感知路径继续，但训练范式从 supervised 分类改为 multimodal self-supervised LNN with weakly-supervised readout。
9.8GB 数据集全留（C 方案）：VEATIC (Step 1 主) + EMOTIC (Step 2 辅) + mst (Step 3 兜底)。
今天仅 ADR-016 设计文档落地，base v3 实施留 M1 启动（5/27+ Phase γ 预训练，6/11+ Phase δ 川哥真人微调）。
Phase α 今天 30min quick win：demo 加 webcam snapshot 按钮，让川哥视觉感受到多模态输入存在。
L4 主线（CLS + 记忆 UI）今天继续推进（不冲突）。

---

**模板版本：v0.1 · 2026-05-03**
