# LnMem 术语字典

> 给 cc/codex 速查用。每条 ≤ 2 行解释 + 详细位置链接。

## 核心架构

| 术语 | 解释 | 详细 |
|------|------|------|
| **LNN** | Liquid Neural Network · 液态神经网络（LTC/CfC），连续时间动力学，τ 可学，本项目记忆系统的"心脏" | [`memory-architecture.md`](memory-architecture.md) §1 |
| **LTC** | Liquid Time-Constant Network，LNN 的一种实现 | [`source-research.md`](source-research.md) §1.1 |
| **CfC** | Closed-form Continuous-time，比 LTC 推理快 2-3× 的实现 | [`source-research.md`](source-research.md) §1.1 |
| **ncps** | mlech26l/ncps 库（PyTorch backend，2.3k stars），LTC/CfC 唯一成熟开源实现 | [`memory-architecture.md`](memory-architecture.md) §3.1 |
| **七层认知架构** | L1 感知 / L2 工作记忆 / L3 处境理解 / L4 自传记忆 / L5 意图共情 / L6 元认知 / L7 共生协议 | [`product-definition.md`](product-definition.md) §4.1 |

## 万能语法

| 术语 | 解释 | 详细 |
|------|------|------|
| **万能语法** | 20 槽固定 schema，横切 L2-L5 的事件张量统一格式 | [`product-definition.md`](product-definition.md) §4.3 |
| **EventTensor** | 万能语法 20 槽的张量数据结构，shape `[B, 20, 4]` | [`memory-architecture.md`](memory-architecture.md) §5.1 |
| **P0 / P1 / P2 / P3** | 槽位优先级档：P0 必填 / P1 检索主键 / P2 时态推理 / P3 仅向量化 | [`product-definition.md`](product-definition.md) §4.3 |
| **Overflow Capsule** | 槽位外溢通道——长尾叙事/价值判断填不下槽位时的自由文本补充 | [`product-definition.md`](product-definition.md) §5.3 |

## 三层记忆

| 术语 | 解释 | 详细 |
|------|------|------|
| **L1 工作记忆** | 秒-分钟，活在 LNN 隐状态里，承载"正在发生什么" | [`product-definition.md`](product-definition.md) §5.2 |
| **L2 情景记忆** | 小时-天，Graphiti KG bi-temporal，承载"何时何地与谁做了什么" | [`product-definition.md`](product-definition.md) §5.2 |
| **L3 语义/人格记忆** | 周-年，LoRA + 偏好向量，承载"主人是怎样的人" | [`product-definition.md`](product-definition.md) §5.2 |
| **CLS** | Complementary Learning Systems，海马-皮层双系统，对应"夜间巩固" | [`source-research.md`](source-research.md) §1.6 |
| **Consolidation** | CLS 离线巩固——L2 → L3 的"夜间学习"，用户可见仪式感 | [`product-definition.md`](product-definition.md) §5.4 |
| **bi-temporal** | KG 双时间索引：t_valid（事件何时发生）/ t_invalid（何时失效） | [`source-research.md`](source-research.md) §3.2 |
| **Graphiti** | getzep/graphiti 开源时间感知 KG，本项目 L2 的实现 | [`memory-architecture.md`](memory-architecture.md) §2 |

## 用户可见命名（产品侧）

| 工程代号（内部） | 用户界面命名（对外） | 内容 |
|----------------|------------------|------|
| 今日精华回顾 | **「今天」/「今日小结」** | 3 条以内："事实-情绪-未完成事项" |
| 主人画像微调 | **「我对你的新理解」** | 本周新增/减弱的偏好 + 证据来源 |
| 记忆变更日志 | **「记忆账本」** | 新增了什么、合并了什么、降权了什么 |

## 个性化栈

| 术语 | 解释 | 详细 |
|------|------|------|
| **LoRA** | Low-Rank Adaptation，rank=8，per-user adapter on LLM | [`memory-architecture.md`](memory-architecture.md) §7 |
| **偏好向量** | per-user 20 维槽位偏好，注入 LNN 输入侧 embedding | [`memory-architecture.md`](memory-architecture.md) §7 |
| **BitFit** | 仅训练 LTC cell bias 的极轻量个性化（研究探索路径） | [`memory-architecture.md`](memory-architecture.md) §7 |
| **State-offset Tuning** | ❌ 已确认在 LTC/CfC 上不可行，已替换为 LoRA + 偏好向量 + BitFit | [`source-research.md`](source-research.md) §1.4 |

## 工程方法论

| 术语 | 解释 | 详细 |
|------|------|------|
| **vibecoding** | cc + codex + 人三方协作工程模式，是 v0.3 时间线压缩 30-40% 的根本依据 | [`product-definition.md`](product-definition.md) §11.2, [`../CLAUDE.md`](../CLAUDE.md) |
| **三道门** | PR 必经：cc 实现 + codex 对抗审查 + 人签名 | [`../CLAUDE.md`](../CLAUDE.md) §3 |
| **三色判定** | 红/黄/绿 早期预警机制，M1/M2/M3 末三次渐进评估 | [`../roadmap/milestones.md`](../roadmap/milestones.md) §3 |
| **双门 + 四象限** | M4 MVP α 的用户行为门 + 能力门 + 4 种行动决策 | [`product-definition.md`](product-definition.md) §13.1 |

## 数据策略

| 术语 | 解释 | 详细 |
|------|------|------|
| **类型 1 数据** | 多 LLM 联合生成的合成对话语料（1 万条） | [`product-definition.md`](product-definition.md) §6.1 |
| **类型 2 数据** | LLM 生成的虚拟主人长期轨迹（50 主人 × 30 天） | [`product-definition.md`](product-definition.md) §6.1 |
| **类型 3 数据** | 团队真实交互（脱敏 + 老板亲自一审，占比 ≤ 20%） | [`product-definition.md`](product-definition.md) §6.1 |
| **三层脱敏** | L1 内容过滤 / L2 实体替换 / L3 老板一审 | [`product-definition.md`](product-definition.md) §6.1 |
| **假门测试** | M2-M3 给种子用户看 demo 视频问付费意愿——别等 M4 才知道 | [`product-definition.md`](product-definition.md) §9 |

## 风险与降级

| 术语 | 解释 | 详细 |
|------|------|------|
| **不可妥协 6 项** | 12 月内必做：万能语法 / LNN / CLS / 个性化 / 双场景 / 用户可治理记忆 | [`product-definition.md`](product-definition.md) §10.1 |
| **老板三重身份偏差** | 老板既是数据提供者+决策人+早期用户，需要 4 条缓解措施 | [`product-definition.md`](product-definition.md) §13.3 |
| **协作磨合期** | M0 第一周不开发，先建立 vibecoding SOP | [`product-definition.md`](product-definition.md) §7.4 |

---

**v0.3 封版同步 · 2026-05-03**
