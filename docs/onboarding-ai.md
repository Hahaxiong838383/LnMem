---
title: "AI 同事入门 30 秒（cc/codex/gemini 必读）"
audience: AI 同事第一次进 LnMem repo
purpose: 30 秒锚定项目目标 + 原则 + 当前阶段 + 协作纪律
last_updated: 2026-05-21
体裁: L2（高密度规则索引）
---

> **新 AI 同事进 LnMem repo 第一站。30 秒读完，开始协作。**

---

## 1. 项目一句话（必背）

**LnMem = Liquid neural Memory** —— 一个能「陪你数年」的伴侣型 AI 记忆系统。
**用 LNN 作为永远在线的心智核心**，让 AI 懂并能预测用户的心情状态及心理活动。

与传统「LLM + RAG」对立：
- 它们：对话时检索过去
- 我们：永远在线连续时间状态里持有当下，主动写入主动学习

## 2. 不可妥协 6 项（红线）

任何对以下 6 项的妥协建议，**停下问川哥**：

1. **万能语法 20 槽** 全系统共享 schema（核心 IP）
2. **多时间尺度 LNN 心智核心**（区分纯 LLM agent 的关键）— Phase γ 落地中
3. **CLS 双系统离线 consolidation**（"数年长期连续运行"工程基础）— v0 已上
4. **per-user 个性化栈**（τ-bias + FiLM + 槽偏好向量）— Phase δ 启动
5. **居家+车载双场景统一架构**（市场定位核心）— M6 起
6. **用户可见可编辑可治理的记忆界面**（行业首创）— v0.1 删除已上线

详细：[`product-definition.md`](product-definition.md) §10.1

## 3. P-1 最高准则：LNN-FIRST

> 凡涉及「**时间 / 演化 / 记忆 / 个性化**」，第一性原理必须是 **LTC/CfC 微分方程**。
>
> 用 LLM/RNN/规则替代 LTC 必须有 ADR + 川哥签字。

**已踩过的坑**（cc 主线认知错误案例）：
- ADR-013：把 LNN 当 supervised 9-state 分类器训 → 64/64 collapse
- ADR-015：cc 听 gemini「视觉权重 2/10」想冻视觉路径 → 川哥拉回 → ADR-016 撤回

详细：[`principles/lnn-first.md`](principles/lnn-first.md)

## 4. 当前阶段（2026-05-21 末）

- **主分支**：`poc/lnn-validation`
- **里程碑**：M0 W3 Day 2 末（Phase β 完工 / Phase γ ready）
- **最新 commit**：`265c8f4` (γ-spec v0.2.3 codex 4 轮 APPROVE)
- **下一步**：5/22 进入修订周 Day 1（γ-1 数据 pipeline spike）

详细状态：[`project-status.md`](project-status.md)

## 5. 30 秒文档地图

| 你想知道 | 读这个 |
|---------|--------|
| 项目状态 + TODO | [`project-status.md`](project-status.md) |
| 技术全栈架构 | [`architecture-overview.md`](architecture-overview.md) |
| 产品定义 + 不可妥协 | [`product-definition.md`](product-definition.md) |
| LNN-FIRST P-1 准则 | [`principles/lnn-first.md`](principles/lnn-first.md) |
| 路线图 M0-M12 | [`../roadmap/milestones.md`](../roadmap/milestones.md) |
| 所有决策 | [`decisions/`](decisions/) 共 19 个 ADR |
| base v3 训练 spec | [`specs/base-v3-training-spec.md`](specs/base-v3-training-spec.md) |
| 文档写作规范 | [`STYLE.md`](STYLE.md) |
| AI 协作 SOP | [`../CLAUDE.md`](../CLAUDE.md)（cc 视角）/ [`../AGENTS.md`](../AGENTS.md)（codex 视角）/ [`../GEMINI.md`](../GEMINI.md)（gemini 视角）|
| 日复盘 | [`retrospectives/`](retrospectives/) |
| 实验报告 | [`reports/`](reports/) |

## 6. 三道门（任何写代码前必读）

```
1. 接口先行：先写 Protocol/typing.Protocol + schema → 再写实现
2. 测试先行：单测定义验收（TDD 红→绿→重构）
3. codex 对抗审查：cc 写完立即派 codex 挑刺 → 才能 commit
```

**反模式**（禁止）：
- ❌ 直接写实现不写 Protocol
- ❌ 测试事后补
- ❌ cc 自己 review 自己

详细：[`../CLAUDE.md`](../CLAUDE.md) §3

## 7. cc/codex 意见冲突诊断流程（§3.1）

cc 和 codex 评审意见**相反**时：

1. **默认采纳 codex 对抗审查**（保守优先）
2. cc 不同意 → **必须明确分歧点 + 列实测证据**
3. **川哥作为最终决策者**，留 ADR 记录
4. 周复盘回看冲突案例

**已踩过的坑**（5/21 实证）：
- cc 主线写 spec v0.2 自评 5/5 全 A → codex 抓 6 处 gap → REVISE
- cc 第二轮自评 9/9 全修 → codex 又抓 3 处 → REVISE
- 4 轮严审共抓 14 处 gap，全部解决 → v0.2.3 APPROVE

**铁律**：派 codex 评审重要 spec / ADR 至少 2-3 轮，单轮 APPROVE 容易漏。

## 8. 多引擎 Team 模式分工

| 引擎 | 角色 | 何时调用 |
|------|------|---------|
| **川哥** | 架构 lead / LNN-FIRST 守门员 / 最终决策 | 重要决策 + 红线 |
| **cc**（Sonnet 4.6）| 主线实现 + 文档 + 终验 + commit | 主战全程 |
| **codex**（gpt-5.4 high / 5.3-codex high）| 对抗审查 + 实现复杂模块 | 评审 + 大代码任务 |
| **gemini**（3.5-flash high / 3-pro-preview）| 学术外脑 + 战略评审 | 学术评审，不下最终决策 |

## 9. 常用调用模板

### gemini proxy 调用（parser / 抽取场景必加 reasoning_effort=minimal）

```bash
curl -H "Authorization: Bearer $GEMINI_PROXY_TOKEN" \
  -H "Content-Type: application/json" \
  http://23.226.135.149:4000/v1/chat/completions \
  -d '{
    "model": "gemini-3-flash-preview",
    "messages": [...],
    "max_tokens": 4096,
    "reasoning_effort": "minimal"
  }'
```

**铁律**：max_tokens ≥ 4096，否则 thinking 吃光 message。

### codex 后台跑大任务

```bash
nohup codex exec \
  -C /Users/rayman.chen/mycc/2-Projects/LnMem-poc \
  --full-auto \
  -c model="gpt-5.3-codex" \
  -c model_reasoning_effort="high" \
  - < /tmp/codex_prompt.md \
  > /tmp/codex_log.md 2>&1 &
disown
```

## 10. demo 启动（30 秒跑起来）

```bash
cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc

# 必备：设环境变量
export GEMINI_PROXY_TOKEN=<token>

# 启动 streamlit demo
PYTHONPATH=. .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py
# → http://localhost:8501
```

体验顺序：1️⃣ 文本输入 → 2️⃣ 📸 抓帧（默认 pyfeat 真识别）→ 3️⃣ 🚀 解析 → 4️⃣ 🔎 查询 → 🌙 看 CLS 精华

## 11. 测试 + verify

```bash
# 全量测试
PYTHONPATH=. .venv-poc-fresh/bin/python -m pytest apps/ contracts/ graphiti_zh_adapter/ universal_grammar_parser/ -q

# 期望：56 passed / 1 skipped
```

## 12. 与父项目 mycc 关系

- LnMem 是**独立 GitHub 仓库** `Hahaxiong838383/LnMem`
- mycc 是川哥日常协作框架（不同仓）
- LnMem 工作在 mycc 主线追踪走 task-log + RECENT_EVENTS
- git config 已统一 `Rayman Chen <chenchuan.futurus@gmail.com>` 全局

## 13. 紧急联系

- 川哥不在 → 别擅自改 §10.1 不可妥协项
- 自己拿不准 → 找 cc/codex/gemini 三方对账
- LNN-FIRST 准则有疑问 → 读 [`principles/lnn-first.md`](principles/lnn-first.md) §3 Override 协议
- 任何对 §10.1 妥协建议 → CLAUDE.md §5 强制停下问老板

---

**📝 写给 AI 同事的话**：你不是责任主体，川哥才是。但你的产出会被 cc 主线终验 + 川哥签字才合并。所以：**宁可慢点修对，不要快速 commit 错的**。

**📝 维护者**：cc 主线 | **生效**：2026-05-21 21:30
