# LnMem Persona Assessment Tool

人格画像测评与虚拟画像生成工具。

v0.2 把原始 demo 改成 7 章结构化测评：

- 93 道题。
- 35 道情境题或叙事题。
- 本地评分。
- LLM 分析。
- 完整 JSON profile 导出。
- 批量虚拟画像生成与差异校验。

## 运行

推荐使用本地静态服务：

```bash
python3 -m http.server 8000
```

然后打开：

```text
http://127.0.0.1:8000/persona-assessment.html
```

也可以打开：

```text
http://127.0.0.1:8000/index.html
```

`persona-assessment.html` 保留旧入口，会加载同一套模块化应用。

## 文件结构

- `index.html`: 新入口。
- `persona-assessment.html`: 兼容旧入口。
- `data/*.json`: 题库。
- `data/chapters.json`: 7 章题目编排。
- `js/app.js`: 应用启动。
- `js/state.js`: 状态、题库加载、本地存储。
- `js/render.js`: 页面渲染与交互绑定。
- `js/scoring.js`: 本地评分。
- `js/charts.js`: 图表渲染。
- `js/export.js`: 导出、LLM 分析入口、批量生成入口。
- `js/llm-client.js`: OpenAI、Anthropic、DeepSeek 统一调用。
- `js/profile.js`: profile schema 合并与校验。
- `js/batch.js`: 批量生成与差异校验。
- `prompts/analysis-prompt.txt`: LLM 分析提示词。
- `prompts/generation-prompt.txt`: 批量生成提示词。
- `examples/sample-profile.json`: 完整 profile 示例。
- `scripts/validate-v02.mjs`: v0.2 验收脚本。

## 题库结构

v0.2 从当前实际 120 题收敛为 93 题。

| 模块 | 题数 | 说明 |
|---|---:|---|
| HEXACO | 36 | 每维 6 题，每维 1 道情境题 |
| ECR-R | 12 | 6 道叙事化 Likert |
| 动机 | 15 | GSE、SDT、领域自评和情境题 |
| PVQ | 10 | Schwartz 价值观 |
| CAPS | 20 | 具体生活和工作情境 |

情境题与叙事题总数为 35，道数超过 30%。

## 章节

测评被拆成 7 章：

1. 基本特质底盘。
2. 表达、协作与执行。
3. 规划、探索与动机。
4. 自主、胜任与关系动能。
5. 价值观优先级。
6. 压力、关系与决策情境。
7. 亲密、边界与 AI 互动。

每章由 `data/chapters.json` 的 `question_ids` 精确控制。

## 理论基础

本工具保留 v0.1 的理论选型：

- HEXACO: Lee & Ashton 六因素人格模型。
- ECR-R: 依恋焦虑与依恋回避。
- GSE: 一般自我效能。
- SDT: 自我决定理论中的自主、胜任、关系需求。
- Schwartz PVQ-10: 价值观排序。
- CAPS: Mischel & Shoda 的情境签名模型。

## 本地评分

`js/scoring.js` 负责本地评分：

- Likert 题按量表最大值归一到 0-1。
- 反向题会自动反转。
- 情境题读取选项 `signals`。
- ECR 输出 anxiety、avoidance 和依恋类型。
- PVQ 输出真实排序，不使用写死图表数据。

## LLM 分析

结果页提供 `调用 LLM 分析`。

用户填写 provider 和 API Key 后，应用会调用：

- OpenAI: `https://api.openai.com/v1/chat/completions`
- Anthropic: `https://api.anthropic.com/v1/messages`
- DeepSeek: `https://api.deepseek.com/chat/completions`

LLM 只负责解释与扩展，不改写本地分数。

## Profile Schema

导出的 profile 至少包含：

- `profile_id`
- `version`
- `created_date`
- `is_real_person`
- `trait_layer`
- `attachment_layer`
- `motivation_layer`
- `situational_signatures`
- `multimodal_signature`
- `ai_interaction_design`
- `narrative_layer`
- `raw_assessment_data`
- `metadata`

其中 v0.2 新增重点是：

- `situational_signatures`
- `multimodal_signature`
- `ai_interaction_design`
- `narrative_layer`

## 批量生成

`js/batch.js` 提供真实 LLM 批量生成：

- 输入参考 profile。
- 输入生成数量和用户提示词。
- 输出完整虚构人物 profile 数组。
- 校验 schema 完整性。
- 校验 HEXACO 分数范围。
- 校验 `if_then_patterns >= 5`。
- 校验两两 HEXACO cosine distance 大于 `0.2`。
- 输出分布 warning。

## 隐私说明

题目答案默认保存在浏览器 `localStorage`。

只有用户点击 LLM 分析或批量生成时，答案和 profile 才会发往所选 provider。

API Key 也保存在浏览器本地状态中。

生产化时建议改成后端代理或更严格的密钥管理。

## 验收

运行：

```bash
node scripts/validate-v02.mjs
```

脚本检查：

- 总题量是否为 93。
- 情境题或叙事题是否不少于 28。
- 每题是否有 `subtitle` 和 `_why_for_user`。
- 章节是否为 7 章。
- 章节题目总数是否为 93。
- 是否存在写死价值观图表。
- LLM 与批量生成模块是否存在。
- JS 模块数是否不少于 6。
- 源码行宽是否不超过 200。
- 示例 profile 是否超过 8KB。
- 示例 profile 是否包含完整 schema。
- README 是否不少于 80 行。

## 开发约束

- 不引入后端。
- 不做用户系统。
- 不做云同步。
- 不替换理论模型。
- 桌面优先。
- 所有核心逻辑保持浏览器可运行。

## 已知局限

- LLM 调用需要用户提供真实 provider API Key。
- 浏览器直连第三方 LLM 适合 demo，不适合生产密钥管理。
- 当前没有自动化端到端浏览器测试。
- 当前没有真实 t-SNE 图，仅有 batch validation 的距离数据。
- 情境题 signals 是产品化启发式向量，不是临床结论。
- 移动端可用，但不是 v0.2 的重点。
