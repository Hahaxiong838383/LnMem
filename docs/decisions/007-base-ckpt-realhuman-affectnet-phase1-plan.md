---
adr-id: 007-phase1
title: ADR-007 base ckpt 真人化 Phase 1 实施方案（cc 采纳 gemini 调研后）
status: planned（2026-05-17 川哥钦定 P0；调研完成；待 W3 启动实施）
date: 2026-05-17
deciders: 川哥 + cc + gemini（gemini-3-pro-preview Phase 1 调研）
supersedes: —
related: ADR-007（主 ADR）/ ADR-005（dual signal_path）/ ADR-008（multi-user 后续依赖本 ADR）/ ADR-010（trigger 链路）/ ADR-011（数据质量 guard）
---

## 上下文

### 触发本 ADR 的数据证据
5/9-5/19 共 402 次真实反馈分析：
- **brow_burst mismatch 30.7%（55/179）** — base ckpt 在川哥真脸上 V-A 映射不准
- **user_triggered 13 次 100% mismatch** — 用户主动 cmd+e 想纠错时 LNN 永远判错
- LNN realtime margin=0.9994 但虚假自信（合成数据训练的 base 不认真人脸）

### 与产品定义 §10.1 的关系
- 技术 2（多时间尺度 LNN 心智核心）— base ckpt 是其基础
- 产品 4（per-user 个性化栈）— OnlineLoRAAdapter 训练在污染 base 上无意义

### 路径上的关键岩石
不修 base = 后续 ADR-008 multi-user / 多时间尺度 LNN / 用户记忆界面 都被 base 错误标签污染。

## 决策（采纳 gemini 调研 + cc 主线 3 处修订）

### 采纳要点

| 维度 | gemini 推荐 | cc 采纳？ |
|---|---|---|
| 数据集 | AffectNet Kaggle 精简版（M0 W3 快速跑通） | ✅ 采纳 |
| AU 提取 | py-feat CPU + batch=32 + Parquet 落盘 | ✅ 采纳 |
| V-A → state 映射 | Russell circumplex 启发 + focused 边界 sweep | ✅ 采纳 |
| Loss | CCC 回归（V-A）+ CE 分类（8 state）联合 | ✅ 采纳 |
| 训练资源 | M2 Max 纯 CPU + 1-2h / 100 epoch | ✅ 采纳 |
| **预训练时冻结 LNN 动力学 tau，仅训空间映射层** | gemini Phase 1 重要警告 | ✅ **核心采纳**（详见 cc 修订 1）|

### cc 主线 3 处修订

**修订 1（强化）**：gemini 风险 #1 "静态图破坏 LNN 动力学" 升级为**预训练阶段强制要求**：
- Phase 1 预训练**只更新空间编码器 + emotion/state 输出头**
- LTC 动力学参数（tau / leakage / 时间常数）保持初始值，**不参与 gradient**
- Phase 2 由现有 `fit_orchestrator.try_fit_step` 用川哥真实时序信号微调 tau

**修订 2（数据集）⚠️ V-A 丢失 finding（2026-05-20 00:15 实测）**：
- **原计划**：Kaggle 精简版做 V-A 连续回归 → 8 state 映射
- **实测发现**：Kaggle 16 个 affectnet + HuggingFace 12 个 community 数据集**全部只保留 emotion class，丢弃 V-A 连续标注**（实证：sarabhian/affectnet-helper schema `class_label/array/img_path`；Piro17/affectnethq HF schema `image, label`）
- **退化版采纳（川哥钦定）**：
  - 用 emotion class（8 类）替 V-A 连续，loss 从 CCC + CE 联合 → 单 CE
  - 修订 1 "冻结 tau" 仍是核心保护，LNN 动力学不被静态图破坏
  - V-A 动态特性损失由 Phase 2 fit_orchestrator 川哥真时序信号补回
- **后续 Aff-Wild2 切换**（M0 W4+）：争取官方申请，恢复 V-A 回归 + 真序列动力学训练

**修订 3（验收门）**：从 gemini 提议的"< 10% brow_burst mismatch"调整为：

**修订 3（验收门）**：从 gemini 提议的"< 10% brow_burst mismatch"调整为：
- **Verify**：川哥 402 次真实反馈 replay，base v2 vs base v1 混淆矩阵对比，brow_burst mismatch 从 30.7% → **< 15%**
- **Guard**：base v2 在 AffectNet 测试集（held-out）8-state 分类准确率 ≥ 60%（防止过拟合到川哥单人）
- **额外门**：base v2 加载到 daemon 后跑 24h 真实信号，user_triggered mismatch 率从 100% → **< 40%**

## 实施分解（5 个 phase × 2.5 天）

| Phase | 内容 | 工时 | 验收 |
|---|---|---|---|
| 1.1 数据 | Kaggle 精简版下载 + 解压 + 验证完整性 | 0.25d | 文件清单 + 抽样 100 张目视检查 |
| 1.2 AU 提取 | py-feat batch 跑全量 + Parquet 落盘 | 0.5d | parquet 文件 + NaN 率 < 5% |
| 1.3 V-A→state 映射 sweep | 9 组阈值组合扫 + 川哥真信号 baseline 评估 | 0.5d | 选定一组 → ADR 落盘 |
| 1.4 预训练 base v2 | CCC + CE 联合 loss + **冻结 tau** + 100 epoch | 1d | ckpt + train log + 验证集准确率 |
| 1.5 A/B 验收 | 川哥 402 次反馈 replay 对比 base v1 vs v2 | 0.25d | 混淆矩阵图 + ADR 验收报告 |

## AI 协作分工（cc 主线 + team 模式）

| Phase | 主实施 | 对抗审查 | cc 主线 |
|---|---|---|---|
| 1.1 数据 | cc Bash | — | 直接做 |
| 1.2 AU 提取 | codex（写 batch 脚本 + Parquet I/O 优化） | cc 审 | 验收 + 跑 |
| 1.3 V-A→state sweep | codex（写 sweep 脚本） | gemini（评 Russell circumplex 学术依据） | 决策 + 落盘 |
| 1.4 预训练 | codex（写 train_base_v2.py + 冻结 tau 实现） | gemini（评 CCC loss 合理性 + 检查冻结实现是否真冻住 tau） | 跑训练 + 验收 |
| 1.5 A/B 验收 | cc Bash + replay 脚本 | codex 审验收方法 | 写本 ADR 验收段 |

## 风险清单（gemini 5 项 + cc 加 2 项）

| # | 风险 | 缓解 |
|---|---|---|
| 1 | **动力学缺失（critical）**：静态图破坏 LNN tau 学习 | ✅ 强制冻结 tau（修订 1）+ M0 W4 切 Aff-Wild2 真序列 |
| 2 | 商用版权暴雷 | M3 阶段评估替换或购买授权（不阻塞 PoC）|
| 3 | 数据域偏移（欧美脸） | Aff-Wild2 含更多场景 + Phase 2 用川哥真信号微调 |
| 4 | V-A → 8 state 硬截断阶跃 | sweep 设计 + 后续可改 soft assignment |
| 5 | 低分辨率劣化 | Kaggle 精简版分辨率 96×96 起步，必要时切官方版 |
| 6（cc 加）| 冻结 tau 实现 bug（动力学参数偷偷更新） | gemini 审 + 加单测断言训练前后 tau bitwise 相等 |
| 7（cc 加）| Phase 1.5 验收数据 leakage | 川哥 402 次反馈中部分已用于 OnlineLoRAAdapter 训练，验收时必须区分"训练用过的 vs hold-out" |

## 配套产出

- 调研报告：`/tmp/lnmem-team-2026-05-17/gemini_adr007_plan.md`
- 调研 prompt：`/tmp/lnmem-team-2026-05-17/gemini_adr007_phase1_prompt.md`
- 主 ADR：`docs/decisions/007-base-ckpt-realhuman-affectnet.md`（status 升级到 implementing）

## 时间线

- 2026-05-16: ADR-007 立项（accepted, M1 启动前必做）
- 2026-05-17 23:00: 川哥宏观复盘后钦定 P0 + team 模式启动
- 23:10: gemini ADR-007 Phase 1 调研 prompt 发出
- 23:22: gemini 调研完成 5 章节 + 推荐路径
- 23:40: cc 主线采纳 + 3 处修订 + 本 Phase 1 plan 落盘
- 2026-05-20 00:10: Kaggle 凭据装好 + 启动 vfomenko 5.4GB 下载（30min 后发现 stuck）
- 00:15: 实测发现 Kaggle/HF 全社区版无 V-A → 退化版决策（修订 2）
- 00:35: A 轨切换 mstjebashazida/affectnet 340MB → 30K 图 + 8 emotion class，387MB unpacked
- 00:45: cc 调研 V-A 数据集全景 → 发现 EMOTIC 是即时可申请的 V-A-D 候选
- 00:55: 川哥提交 EMOTIC Google Form 申请（forms.gle/JxcNkFHfvVyGYjjc9）→ 等邮件审核
- **双轨齐备**：A 轨 fallback 数据 ready，B 轨等 EMOTIC 通过决定走真 V-A 版

## 双轨执行矩阵（明日 Phase 1.2 启动条件）

| 场景 | A 轨数据 (mst affectnet) | B 轨 EMOTIC 通过 | Phase 1.4 训练 loss |
|---|---|---|---|
| EMOTIC 未通过 / 慢 | ✅ 立即可用 30K 图 | ❌ | 退化版：单 CE emotion 分类 |
| EMOTIC 通过 | 备份 | ✅ 3.2GB 24K 图 + V-A-D 三维 | 完整版：CCC V-A 回归 + CE state 联合 |

Phase 1.2 (codex AU 提取) 不依赖 EMOTIC，可立即用 A 轨数据启动。
Phase 1.3 (V-A→state sweep) 仅在 EMOTIC 到位时启动。
Phase 1.4 训练时按上表分叉。
