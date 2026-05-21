---
title: "LnMem 项目状态总览（活文档）"
last-updated: 2026-05-21 21:30（Day 2 末 · Phase β 完工 + γ-spec v0.2.3 codex 4 轮 APPROVE）
audience: [AI 同事 cc/codex/gemini, 工程师, 投资人]
快速入口:
  - 新 AI 同事 → onboarding-ai.md（30 秒上手）
  - 技术全景 → architecture-overview.md（5 分钟懂全栈）
  - 项目目标原则 → product-definition.md（投资人友好）
  - 路线图 → ../roadmap/milestones.md
---

> **3 分钟读完掌握 LnMem 全局状态**。每次重大进展（commit / ADR / milestone）后必更。
>
> 详细路线见 [`ROADMAP.md`](../ROADMAP.md)，决策记录见 [`docs/decisions/`](decisions/)，里程碑见 [`roadmap/milestones.md`](../roadmap/milestones.md)。

---

## 1. 项目一句话定位

**LnMem = Liquid neural Memory** —— 一个能「陪你数年」的伴侣型 AI 记忆系统，居家+车载持续运行，用万能语法把你说的/看到的/做的事统一编码，**用液态神经网络（LNN）作为永远在线的心智核心**。

与传统 LLM+RAG 的本质区别：传统方案在对话时检索过去；LnMem 在永远在线的连续时间状态里持有当下，通过事件分割主动写入，通过夜间巩固主动学习。

## 2. 项目原则（不可妥协）

### P-1 LNN-FIRST（最高准则）

> 见 [`docs/principles/lnn-first.md`](principles/lnn-first.md)

凡涉及「**时间 / 演化 / 记忆 / 个性化**」，第一性原理必须是 LTC/CfC 微分方程。用 LLM/RNN/规则替代 LTC 必须有 ADR + 川哥签字。

**典型违反**（已发生过两次，已修正）：
- 把 LNN 当 supervised 分类器训（base v2/v2.1 → ADR-013 收口）
- 因 gemini「视觉权重 2/10」准备冻结视觉（ADR-015 → ADR-016 撤回）

### 不可妥协 6 项（产品定义 §10.1）

| # | 项 | 当前状态 |
|---|---|---------|
| 1 | 万能语法 20 槽 | ✅ schema_v1 封版 / parser v0 已上线 |
| 2 | 多时间尺度 LNN 心智核心 | ⏳ base v3 范式锁定（ADR-016），M1 启动 |
| 3 | CLS 夜间整合 | ✅ v0 已上线（cls_consolidate.py）|
| 4 | per-user 个性化三件套（τ-bias + FiLM + 槽偏好）| ⏳ base v3 backbone 准备好，M1 Phase δ |
| 5 | 居家+车载双场景 | ⏳ M2+ |
| 6 | 记忆 UI 可改可删 | ✅ v0.1 删除已上线，修改留 Phase β |

### 工程纪律（CLAUDE.md §3 三道门）

1. **接口先行**：先写 Protocol/interface + schema，再写实现
2. **测试先行**：单测定义验收，TDD（红→绿→重构）
3. **codex 对抗审查**：cc 写完立即 codex 挑刺，再 commit

cc/codex/gemini 意见冲突 → 默认采纳对抗意见 / 不同意必须明确分歧点 / 川哥最终决策 / 留 ADR 记录。

## 3. 全局核心概念速览

| 概念 | 中文 | 在 LnMem 里的角色 |
|------|------|------------------|
| **LNN** (Liquid Neural Network) | 液态神经网络 | 心智核心（LTC/CfC 微分方程） |
| **CfC** (Closed-form Continuous-time) | 闭式连续时间网络 | LNN 的快速近似实现 |
| **τ-bias** | 时间常数偏置 | per-user 个性化的"反应速度" |
| **FiLM** (Feature-wise Linear Modulation) | 特征级线性调制 | per-user 个性化的"看人下菜" |
| **万能语法 20 槽** | who/what/when/where/why/... | 任何事件的统一符号编码 |
| **Russell V-A** | Valence-Arousal 情感二维 | 9-state 情感投影（1 圆心 + 8 扇区）|
| **AU** (Action Unit) | 面部动作单元 | FACS 编码，py-feat 提取，**作为 LNN 输入信号**（不是分类） |
| **bi-temporal KG** | 双时态知识图谱 | 节点带 t_valid + t_invalid，永不物理删 |
| **graphiti** | — | KG 后端（Phase A InMemory，M1 接 graphiti-core） |
| **CLS** (Complementary Learning Systems) | 互补学习系统 | 每日 02:00 cron 把当天 events 浓缩为 3 条精华 |
| **Intent** | 意图类型（陈述/疑问/反问/感叹/请求）| Phase β 由 parser 抽取 |
| **EventTensor** | 事件张量 | schema_v1 封版的运行时数据载体 |
| **base v1/v2/v3** | LNN 训练范式版本 | v1=合成数据 / v2=supervised(失败) / v3=multimodal self-supervised |
| **VEATIC / EMOTIC / mst** | 训练数据集 | 9.8GB 总量（6GB+3.4GB+387MB），base v3 三阶段用 |
| **DFAT** (Decoupled FiLM-Adapter Tuning) | 冻 backbone 调 FiLM | base v2.1 微调方法（已证不足，待 v3 替代）|
| **vboss_self** | 川哥真人 persona | base v1 的 30 虚拟主人 + 1 真人画像中的真人 idx |

## 4. 当前阶段

**阶段**：M0 W3 Day 2 末（2026-05-21 21:30）—— **Phase β 完工 + γ-spec v0.2.3 codex 4 轮 APPROVE**
**主分支**：`poc/lnn-validation`（独立仓 [Hahaxiong838383/LnMem](https://github.com/Hahaxiong838383/LnMem)）
**最新 commit**：`265c8f4` (γ-spec v0.2.3 final · codex 4 轮 APPROVE)

### Day 2 完工清单（5/21 单日 15 commits / 4 ADR / ~5000 行）

#### L4 W+ 三砖（上午）
| 砖 | 内容 | 状态 |
|---|------|------|
| **砖 1** | M0 端到端 demo（parser→KG→query）| ✅ `scripts/e2e_demo.py` + `docs/reports/2026-05-21-e2e-demo.md` |
| **砖 2** | CLS v0 夜间整合 | ✅ `scripts/cls_consolidate.py` + `docs/reports/cls_2026-05-21.md` |
| **砖 3** | 记忆 UI v0.1 + Phase α webcam | ✅ `apps/demo_e2e/main.py` + `multimodal.py` + ADR-017 |

#### Phase α 多模态升级 + 3 bug 修复（下午）
| 项 | 内容 | 状态 |
|----|------|------|
| ADR-016 范式转换 | 撤回 ADR-015 视觉冻结，立 base v3 multimodal self-supervised LNN | ✅ |
| py-feat 3 bug 修复 | detect_image API / 列名 / 范围 [0,1] | ✅ commit fc69ccb |
| UI 全面友好化 | HELP 字典 11 术语 + 4 步引导 + tooltip + 友好错误 | ✅ |
| Demo 状态 | http://localhost:8501 川哥可真实体验 | ✅ |

#### Phase β 5 砖（晚上 + 修订）
| 砖 | 内容 | 状态 |
|----|------|------|
| **β-1** | EventTensor schema v1→v1.1（intent_type + multimodal_context）+ ADR-018 | ✅ cc 主线 |
| **β-2** | parser intent_type 5 类抽取（陈述/疑问/反问/感叹/请求）| ✅ codex 5.3 high |
| **β-3** | demo 输出完整状态描述 JSON（川哥钦定 6 维全景）| ✅ cc 主线 |
| **β-4** | graphiti SQLite 持久化（bi-temporal 保留）| ✅ codex 5.3 high |
| **β-5** | multimodal_context 持久化（不再 session_state 旁挂）| ✅ cc 主线 |
| **真 LLM 验证** | 5/5 = 100% intent_type 准确率（含反问/疑问边界）| ✅ |

#### γ-spec base v3 实施 spec（深夜，4 轮 codex 严审）
| 阶段 | 内容 | 状态 |
|------|------|------|
| **v0.1 初稿** | Agent 后台写 534 行 10 段齐备 | ✅ |
| **双评审** | codex 5.4 + gemini 3.5 high 学术 + 工程对抗审 | ✅ |
| **v0.2 修订** | 吸收双评审 5 核心建议 + ADR-019 决策记录 | ✅ |
| **codex 4 轮严审** | v0.2 / v0.2.1 / v0.2.2 / v0.2.3 共抓 14 处 gap 全修 | ✅ |
| **最终 APPROVE** | codex 第四轮 ✅ APPROVE，残留 0 阻塞 gap | ✅ commit 265c8f4 |

### Phase α 演示当前状态（可立刻跑）

```bash
cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc
GEMINI_PROXY_TOKEN=<token> PYTHONPATH=. \
  .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py
# → http://localhost:8501
```

**实测指标**（5/21 21:00 验证）：
- LLM parse: 4.91s/句（vs 修复前 45s 超时 → 9x ⚡）
- P0 槽位填充: 4/4（100%）
- 总槽位填充: 16/20
- intent_type 抽取: 5/5 = 100% 准确率（陈述/疑问/反问/感叹/请求）
- py-feat 真识别: source=`pyfeat+heuristic_v2` 工作（笑脸 V=+0.68 / 紧张脸 V=-0.35）
- 完整状态描述 JSON: 6 维全展开（含 multimodal_context + lnn_hidden_state placeholder）
- CLS 精华: 显示昨日 5 events → 3 highlights
- 全量测试: **56 passed / 1 skipped** ✅

### Phase α 演示当前状态（可立刻跑）

```bash
cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc
GEMINI_PROXY_TOKEN=<token> PYTHONPATH=. \
  .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py
# → http://localhost:8501
```

**实测指标**（5/21 16:00 修复后）：
- LLM parse: 4.91s/句（之前 45s 超时，9x 改进）
- P0 槽位填充: 4/4（100%）
- 总槽位填充: 16/20
- CLS 精华: 显示昨日 5 events → 3 highlights
- multimodal: mock 模式毫秒返回，pyfeat 模式真 py-feat 提取（首次 ~5s）

## 5. 进展时间线（M0 W3）

| 日期 | 关键产出 | commits |
|------|---------|---------|
| **5/20 Day 1** | Team 模式启动 / ADR-007 Phase 1.2-1.4 + ADR-012 + graphiti Phase A | 8 commits / 2710 行 |
| **5/21 Day 2** | base v2.1 DFAT 失败 + ADR-016 范式撤回 + L4 三砖 + Phase α + 修 3 bug | 7 commits / ~1400 行 / 3 ADR |

详见 [`docs/retrospectives/2026-05-21-day2-phase-alpha-retro.md`](retrospectives/2026-05-21-day2-phase-alpha-retro.md)。

## 6. TODO（按优先级）

### ✅ Phase β（5/22-5/24 原计划，5/21 提前完工）

- [x] ✅ ADR-018 EventTensor schema v1→v1.1
- [x] ✅ LLM parser 加 intent_type 5 类抽取
- [x] ✅ demo 输出完整状态描述 JSON（6 维）
- [x] ✅ graphiti KG SQLite 持久化（β-4）
- [x] ✅ multimodal_context 持久化到 EventTensor（β-5）
- [ ] **P1** L4-bonus UX Cmd+Shift+F 主动状态锚点（留 5/26 修订周）

### 🚀 修订周（5/22-5/26，5 天 · 明日启动）

- [ ] **γ-1** (5/22) 数据 pipeline + ncps spike：Studio ssh + ncps CfC API + tiny VEATIC AU 抽取 + Sampler ±60/>30s + VEATIC viewer_id 去重数
- [ ] **γ-2** (5/23-5/24) 模型架构 2 层 CfC 实测：参数量 + FiLM/τ-bias + 5 head readout + collapse 四联监控
- [ ] **γ-3** (5/25) InfoNCE vs VICReg 双路 spike：tiny 1 epoch + Linear Probe acc + 视频 ID Shortcut 反测 + V-A 标注口径 Bland-Altman
- [ ] **γ-preflight** (5/26) 10-video 11 项 checklist 川哥钦定 → 5/27 启动 Phase γ Step 1

### ⏳ Phase γ（5/27-6/10，14 天）

- [ ] Step 1 VEATIC 全量 self-supervised 预训练（4 天，Studio）
- [ ] Step 2 EMOTIC readout V-A 弱监督（3 天）
- [ ] Step 3 mst readout 8-state + intent_type（4 天）
- [ ] Step 4 评估 + 调参 + 导出 base_v3.pt（4 天）
- [ ] 评估指标三联：M1 KG 召回 Top-5 ≥40% / M3 时序 ratio ≥1.5 / M4 Linear Probe / M5 视频 ID 反测 <30%

### ⏳ Phase δ（6/11+，M1 起）

- [ ] 川哥真人 daemon ≥60w 帧累积
- [ ] base v3 backbone fine-tune
- [ ] per-user τ-bias + FiLM + 槽偏好向量启动（不可妥协 §10.1 #4 落地）
- [ ] TTA Tent 跨域 fail-safe（ADR-021 待立）

### Phase γ（5/27-6/10，2 周）

- [ ] base v3 self-supervised 预训练
  - Loss 1: Temporal Contrast (VEATIC 6GB 主)
  - Loss 4: Weakly-supervised Multi-head Readout (EMOTIC + mst)
- [ ] 输出 64-dim hidden state 给 demo 用

### Phase δ（6/11+，M1 起）

- [ ] 川哥真人 daemon ≥60w 帧累积
- [ ] base v3 backbone 微调
- [ ] per-user τ-bias + FiLM + 槽偏好向量启动

## 7. 关键文件入口

### 顶部入口
- [`README.md`](../README.md) - 项目首页
- [`ROADMAP.md`](../ROADMAP.md) - 路线总览
- [`CLAUDE.md`](../CLAUDE.md) - AI 同事工作纪律
- [`AGENTS.md`](../AGENTS.md) - codex 工作纪律
- [`GEMINI.md`](../GEMINI.md) - gemini 工作纪律

### 核心文档
- [`docs/product-definition.md`](product-definition.md) - v0.3.2 封版产品定义
- [`docs/principles/lnn-first.md`](principles/lnn-first.md) - LNN-FIRST P-1 准则
- [`docs/STYLE.md`](STYLE.md) - 文档写作规范
- [`docs/development-harness.md`](development-harness.md) - BRO-PDCA × 目标驱动验收
- [`docs/capability-specs/cognitive.md`](capability-specs/cognitive.md) - 认知能力规范

### 路线 + 决策 + 复盘
- [`roadmap/milestones.md`](../roadmap/milestones.md) - M0-M12 里程碑
- [`docs/decisions/`](decisions/) - 17 个 ADR（最新 017）
- [`docs/retrospectives/`](retrospectives/) - 日复盘（最新 2026-05-21）
- [`docs/reports/`](reports/) - 验证报告（最新 cls_2026-05-21.md）

### 代码主干
- [`apps/demo_e2e/`](../apps/demo_e2e/) - Streamlit Phase α demo
- [`scripts/`](../scripts/) - CLS 整合 / base v2 训练 / spike 实时推理
- [`universal_grammar_parser/`](../universal_grammar_parser/) - 万能语法 20 槽 parser
- [`graphiti_zh_adapter/`](../graphiti_zh_adapter/) - bi-temporal KG 适配器
- [`contracts/`](../contracts/) - EventTensor / Protocol（schema_v1 封版）
- [`poc/`](../poc/) - LNN 训练 + daemon + ckpts

## 8. 团队配置

| 引擎 | 角色 | 状态 |
|------|------|------|
| **川哥** | 架构 lead / LNN-FIRST 守门员 / 最终决策 | 主战 |
| **cc** (Claude Sonnet 4.6) | 主线实现 / 文档 / 终验 / commit | 主战 |
| **codex** (gpt-5.4 high / 5.3-codex high) | 对抗审查 / 大模块实现 | 配额限制时备用 |
| **gemini** (3.5-flash / 3-pro-preview) | 学术外脑 / 战略评审 | 评审优先，不下最终决策 |

## 9. 协作纪律快速锚点

| 场景 | 必做 |
|------|------|
| 写代码前 | 三道门（Protocol → 测试 → codex 审）|
| 改 EventTensor schema | 红线，写 ADR + 川哥签字 |
| 任何对 §10.1 不可妥协项的妥协建议 | 停下问川哥 |
| cc/codex/gemini 意见冲突 | 默认采纳对抗意见 / 不同意必须写 ADR |
| LLM 调用 timeout | 检查 `reasoning_effort=minimal` + `max_tokens=4096` |
| streamlit / 长服务路径 | 用 `Path(__file__).resolve().parents[N]` 不用相对 cwd |
| commit 前 | 跑 pytest（apps/ contracts/ graphiti/ parser/）+ 端到端真验证 |
| commit 后 | 立即 push origin/poc/lnn-validation |

---

**📝 维护者**：cc | **最近更新**：2026-05-21 16:30
**📂 复盘节奏**：每天日终 / 每个 milestone 完工
