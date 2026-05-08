---
version: v1.0
audience: cc / codex / gemini / 工程师 / 老板 / 历史读者
purpose: 项目最高准则——记忆系统内核必须以 LNN 为核心，不可妥协
status: 强制（2026-05-08 起，老板钦定）
last_updated: 2026-05-08
priority: P-1（高于所有 P0，是宪法级条款）
parent: ../../CLAUDE.md
references:
  - ../lnn-math.md（LTC + CfC 数学基础）
  - ../memory-architecture.md（记忆架构）
  - ../capability-specs/cognitive.md（认知能力规范）
---

# LNN-FIRST · 项目最高准则

> **LnMem = Liquid neural Memory**——这个名字不是装饰。
> 任何方案、任何执行计划、任何代码改动，**都不允许偏离"以 LNN 为核心打造记忆系统内核"这条主轴**。
> 偏离 = 项目失去存在意义。

---

## RULE-ID: LNN-FIRST

**P-1 / 权重 1.00 / 不可妥协**

## 一句话

> **凡是涉及"时间"、"演化"、"记忆"、"个性化"的设计，第一性原理必须是 LTC/CfC 微分方程；用 LLM/RNN/规则替代 LTC 必须有书面 Override 理由并经老板批准。**

---

## 1. 为什么是 LNN（不是 LLM、不是 RNN、不是规则系统）

### 1.1 项目命名即承诺

LnMem 的全称是 **L**iquid **n**eural **Mem**ory。如果记忆系统不以 LNN 为内核，这个名字就是欺诈，整个项目失去差异化定位。

竞品（Mem0 / Zep / Letta / mem-agent / Graphiti）全部用 LLM + 向量库做记忆，本质都是**离散文本游戏**。LnMem 的唯一壁垒是把记忆建模为**连续时间动力学系统**——LLM 只做"重要性裁判"和"语言渲染"，不做记忆本身。

### 1.2 LNN 的不可替代性（4 个其他架构做不到的事）

| 能力 | LLM | RNN/GRU/LSTM | 规则系统 | **LNN** |
|------|-----|--------------|---------|---------|
| 连续时间演化（任意采样率） | ❌ | ❌ 离散步进 | ❌ | ✅ ODE 微分方程 |
| 多时间尺度共存（30s ~ ∞） | ❌ | 🟡 单 τ 难调 | 🟡 硬编码 | ✅ 不同神经元不同 τ |
| 物理可解释性（每个 τ 有意义） | ❌ 黑箱 | ❌ 黑箱 | ✅ | ✅ τ 即"记忆衰减常数" |
| 个性化基线 + 状态偏移 | ❌ 无状态空间 | 🟡 hidden 不可分 | 🟡 拍脑袋 | ✅ τ-bias + FiLM |
| 真人数据持续微调 | 需 LoRA + GPU | 需重训 | 不可学 | ✅ τ/embedding 在线漂移 |
| 19 个神经元做端到端控制 | ❌ | ❌ | — | ✅ NCP 自动驾驶证明 |

**LNN 的核心价值**：让"主人画像 → 记忆衰减节奏 → 当前状态 → 表情/语音/上下文外显"形成**单一物理模型**，每一层 τ 都有明确语义、可解释、可学习、可个性化。

### 1.3 LLM 在 LnMem 的合法位置（仅这两类）

LLM **不是**记忆载体，**不是**状态机，**不是**演化主体。LLM 在 LnMem 里只做两件事：

1. **重要性裁判**：给 LTC 链路输入"是否值得往下一阶段渗透"的标量门控（importance ∈ [0,1]）
2. **语言渲染**：把 LNN hidden state 翻译成人类可读的"AI 对主人的理解"（出对话/确认弹窗时用）

凡是把 LLM 用在第 3 类任务（如"让 LLM 自己生成 5 阶段记忆文本"、"让 LLM 决定状态切换"），都是违反 LNN-FIRST。

---

## 2. 强制规则（每条都要 grep 实施）

### 2.1 设计层（提方案时必查）

| # | 规则 | 检查方法 |
|---|------|---------|
| D1 | 任何"时间相关"模块的设计稿，**必须**含 `dx/dt = ...` 形式的 ODE 公式 | grep `dx/dt\|d.*/dt` |
| D2 | 任何"个性化"模块的设计稿，**必须**说明 τ 或 FiLM 参数的物理意义 | grep `τ\|FiLM\|persona_embedding` |
| D3 | 任何"记忆/衰减/累积"模块，**禁止**用 `if-else` 或 `LLM prompt` 做主逻辑，必须用 LTC 方程 | grep `if.*age\|LLM.*memory_stage` 找违规 |
| D4 | 提方案时**必须**列出"如果不用 LNN 改用 X，会丢什么"——证明 LNN 不是过度设计 | 方案文档检查 |

### 2.2 实现层（写代码时必查）

| # | 规则 | 检查方法 |
|---|------|---------|
| I1 | 涉及时间的状态变量，**禁止**用 `state[t+1] = f(state[t])` 阶跃，必须用 `state[t+1] = state[t] + dt × (-state[t]/τ + A)` | grep `\[t\+1\] = ` 检查赋值形式 |
| I2 | 任何 τ 常数，**禁止**全局硬编码，必须 per-persona 或 per-feature 可调 | grep `tau = [0-9]\|TAU_DEFAULT = ` |
| I3 | LLM 调用必须有 `purpose` 注释说明属于"重要性裁判"或"语言渲染"，否则 review 拒绝 | grep `llm.*chat\|llm.*completion` 配套检查 |
| I4 | 新增"动力系统"模块，**必须**导入或继承 `lnn.contracts.LNNTrainConfig`（保证 τ/FiLM 参数链路一致） | import 检查 |

### 2.3 验收层（PR 合并前必查）

| # | 规则 | 检查方法 |
|---|------|---------|
| V1 | PR 描述**必须**回答："这次改动如何强化 LNN 内核？"——空答 = 拒绝合并 | PR template |
| V2 | PR 影响 `fusion.py` / `lnn/` / `memory/` / `capture/` 的，**必须**经过老板 review LNN 一致性 | 路径触发 |
| V3 | 文档（特别是 `docs/principles/`、`docs/lnn-math.md`、`docs/memory-architecture.md`）改动**禁止**淡化 LNN 角色 | diff 自查 |

---

## 3. Override 协议（破例的唯一通道）

如果某个模块**确实**不适合用 LNN（极少数情况，例如纯文本字符串处理），按以下流程申请破例：

1. **写 ADR**：在 `docs/decisions/ADR-XXX-non-lnn-XXX.md` 说明
   - 为什么不用 LNN
   - 用什么替代
   - 替代方案如何与 LNN 内核协同（不能成为孤岛）
2. **老板批准**：ADR 必须由老板签字
3. **打标签**：代码中必须有 `# OVERRIDE: LNN-FIRST (ADR-XXX)` 注释
4. **季度审查**：每季度复查所有 OVERRIDE，看是否能改回 LNN

**没有 ADR + 老板签字 = 不允许 OVERRIDE。** cc/codex/gemini 拒绝执行无 ADR 的非 LNN 设计。

---

## 4. 五大模块的 LNN 化清单（截至 2026-05-08）

| 模块 | 当前状态 | LNN 化目标 | Phase |
|------|---------|-----------|-------|
| `lnn/model.py` (CfC + τ-bias + FiLM) | 🟢 已 LNN | 加 multi-timescale hidden | P2 |
| `fusion.py` 表情/疲劳派生信号 | 🟡 静态线性组合 | LTC 演化（base+Δ）/τ | **P0**（进行中） |
| `memory/` 5 阶段记忆 | 🔴 未实现，原计划 LLM | 5 个 LTC 单元 + 重要性门 | **P0.5**（即将） |
| `capture/` 多模态采集 | 🟡 固定采样率 | 不同模态不同更新 τ | P1 |
| `personalization.py` persona embedding | 🟡 训练时固定 | 真人数据触发在线漂移 | P1 |

每个 Phase 完成时，本表"当前状态"必须更新；P0+P0.5 完成 = 项目主轴归位（详见 `docs/principles/sprint-2026-05-08-lnn-realignment.md`）。

---

## 5. 给三类 AI 同事的注意事项

### 5.1 cc（Claude Code）

- 收到任何"涉及时间/记忆/个性化"的任务，**第一反应**应该是"这能不能写成 ODE？"，而不是"我用什么 if-else 实现"
- 提方案时**必须**先输出公式，再输出代码骨架
- 如果方案里出现 "用 LLM 做记忆衰减" / "硬编码时间阈值" 等关键词，**自查**：是否违反 LNN-FIRST？
- 老板拍板"启动 P0/P0.5"等于隐式授权 LNN 化重构，不需要每次重新批准

### 5.2 codex（OpenAI Codex）

- cc 派来的任务 prompt 里如果有 "LTC dynamics" / "τ_persona" / "attractor" 等关键词，说明属于 LNN 化任务，**必须**：
  - 实现里出现 ODE 离散化形式（欧拉/RK4）
  - 测试里验证"切换 state 时是平滑过渡，不是阶跃"
  - 不要替换为 LLM 调用或 if-else 逻辑
- 对抗审查时，把"是否违反 LNN-FIRST"列为最高优先级 finding

### 5.3 gemini（Google Gemini）

- 调研类任务（HEXACO×表情×τ 映射、LTC 数学推导等）默认走 gemini
- 输出的"理论映射表"必须能直接喂给 LTC ODE（即输出 τ、A、γ、β 等可数学化的参数，不要给"轻度笑"这种自然语言形容）
- 不允许写代码或跑测试（gemini sandbox 限制，feedback_lnmem_day5_lessons.md 第 8 条）

---

## 6. Why（终极理由）

如果有一天发现某个偷懒方案（LLM 替代 LTC、硬编码替代 ODE）暂时跑得更快、更省事，**那也不允许**——

> **LnMem 的存在价值不是"做一个能用的记忆系统"，而是"用 LNN 证明记忆可以是动力学系统"。**

放弃 LNN = 放弃差异化 = 沦为 mem0 / zep 的 N+1 个克隆 = 项目应该关闭。

老板 2026-05-08 钦定，cc 全程执行人，codex/gemini 协助，团队任何成员发现违反此准则，**都有义务停下当前任务、提请复议**。

---

署名：cc（Claude Opus 4.7）| 2026-05-08 v1.0 起草，老板拍板
