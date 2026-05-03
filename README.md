# LnMem · 以 LNN 为心脏的记忆系统

> **如果你是 Claude Code / Codex 第一次到这里**：跳到 [给 AI 同事的 30 秒入口](#给-ai-同事的-30-秒入口)。
> **如果你是工程师**：读 [快速开始](#快速开始)。
> **如果你是投资人/老板**：读 [`docs/product-definition.md`](docs/product-definition.md) §0 §1。

---

## 它是什么

**一个能"陪你数年"的 AI 伙伴的记忆系统**——居家+车载持续运行，用万能语法把你说的、看到的、做的事统一编码，用液态神经网络（LNN）作为永远在线的心智核心。

**与传统"LLM + RAG"方案的本质区别**：传统方案在对话时检索过去；LnMem 在永远在线的连续时间状态里持有当下，通过事件分割主动写入，通过夜间巩固主动学习。

**项目阶段**：v0.3 产品定义封版 · M0 启动准备中（2026-05-03）

---

## 给 AI 同事的 30 秒入口

文档读取顺序（**按这个顺序读，不要跳序**）：

1. [`docs/product-definition.md`](docs/product-definition.md) §0–§1（30 秒愿景）
2. [`docs/memory-architecture.md`](docs/memory-architecture.md) §0–§2（30 秒技术骨架）
3. [`CLAUDE.md`](CLAUDE.md)（vibecoding 工作纪律，**必读**）
4. [`roadmap/milestones.md`](roadmap/milestones.md) §当前月份段（你现在该做什么）

**红线规则**：
- 任何对 [§10 不可妥协 6 项](docs/product-definition.md) 的妥协建议——**先停下问老板**
- 三色判定（[`roadmap/milestones.md`](roadmap/milestones.md)）连续两次进入红区——**先停下问老板**
- 类型 3 数据（团队真实交互）使用边界灰色地带——**先停下问老板**

**提交规则**：
- 所有 PR 必须经过 cc 实现 + codex 对抗审查 + 人类工程师签名
- AI 不是责任主体（[`CLAUDE.md`](CLAUDE.md) §AI 同事责任边界）

---

## 文档地图

```
LnMem/
├── README.md                    # 你正在读
├── CLAUDE.md                    # vibecoding SOP + AI 同事工作纪律
├── docs/
│   ├── product-definition.md    # 产品定义 v0.3（封版，716 行）
│   ├── memory-architecture.md   # LNN-centric 记忆系统技术架构 v0.1
│   ├── source-research.md       # 12/18 月可行性研究底稿
│   └── glossary.md              # 关键术语字典
└── roadmap/
    └── milestones.md            # M0-M12 里程碑 + 红黄绿三色 + M0 启动清单
```

---

## 核心架构（一图）

```
[多模态感知] → [万能语法 Parser] → [EventTensor 20 槽]
                                          ↓
                              [L1 LNN τ≈秒, 64 neurons]
                                          ↓
                              [L2 LNN τ≈分钟, 32 neurons]
                                          ↓
                              [事件分割门控] —boundary?→
                                          ↓
                                  [L2 情景记忆 · Graphiti KG]
                                          ↓ (夜间)
                                  [CLS Consolidation]
                                          ↓
                                  [L3 语义/人格记忆 · LoRA + 偏好向量]
                                          ↓
                                  [今天 / 我对你的新理解 / 记忆账本]
                                          ↓
                                  [用户可读/编/删]
```

详见 [`docs/memory-architecture.md`](docs/memory-architecture.md) §2。

---

## 核心承诺（v0.3 §1.3）

1. **它会记得**：能回答"上周三晚上和谁吃饭"等含时间锚点的查询
2. **它会变化**：用一个月后的它，比第一天更懂你
3. **它在场景间无缝**：早晨家里、下午车里、晚上回家——同一个它
4. **它知道边界**：你能看到、能改、能删、能带走
5. **它会让你看见**：黑箱学习 → 对账单（行业首创）

---

## 不可妥协的 6 项（v0.3 §10.1）

**技术**：
1. 万能语法 20 槽作为全系统共享 schema
2. 多时间尺度 LNN 心智核心（哪怕只有两层）
3. CLS 双系统离线 consolidation 流程

**产品**：
4. per-user 槽位偏好向量 + LoRA 个性化栈
5. 居家+车载双场景统一架构
6. 用户可见、可编辑、可治理的记忆界面

---

## 时间线（vibecoding 提速版，v0.3 §7.1）

| 月 | 关键里程碑 |
|----|-----------|
| M0–M3 | 核心闭环搭建（万能语法 + LNN + 事件分割 + 个性化 v0） |
| **M4** | **🎯 MVP α**：10 名种子用户开始日均使用 |
| M5–M6 | CLS 巩固 + 车载场景试点 |
| **M7-M8** | **🎯 MVP β**：扩到 25–50 用户 |
| M9–M11 | 主动推理 + 多设备同步 + 长期 KG 治理 |
| M12 | 18 月架构成型（提前 6 月达成） |

红黄绿三色判定见 [`roadmap/milestones.md`](roadmap/milestones.md)。

---

## 快速开始（M0 启动后填充）

```bash
# M0 第一周完成后这一节会更新
git clone https://github.com/Hahaxiong838383/LnMem.git
cd LnMem
# pip install -r requirements.txt   # 待 M0 第一周建立
# python -m synthetic_data_generator # 待 M0 第一周建立
```

详见 [`roadmap/milestones.md`](roadmap/milestones.md) M0 启动检查清单。

---

## 风险已知（v0.3 §9）

🔴 **高**：国产 LLM API 限流 / Graphiti 中文 entity 质量 / 用户付费意愿未知 / 合成数据污染 LNN 训练
🟡 **中**：LNN 多模态训练稳定性 / 事件分割长尾 / 车载集成成本

每个风险都有降级路径，详见 [`docs/product-definition.md`](docs/product-definition.md) §9。

---

## 协作链路

| 角色 | 职责 |
|------|------|
| **人** | 产品决策 / 用户访谈 / 最终架构判断 / AI 产出最终签名 |
| **Claude Code（cc）** | 架构设计 / 代码 / 文档 / 测试 |
| **Codex（gpt-5.4）** | 对抗审查 / 工程规范 / 边界条件 |

详见 [`CLAUDE.md`](CLAUDE.md)。

---

**v0.3 封版（M0 启动用版本） · 2026-05-03**
