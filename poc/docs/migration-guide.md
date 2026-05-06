---
name: PoC → 主分支工程化迁移建议
audience: 主分支工程化执行的 cc + codex（M1 启动时读）
purpose: Day 14 PoC 完成后必产文档——告诉主分支"什么可以借鉴 / 什么不能直接复制 / 工程纪律是什么"
status: v0.1 骨架（2026-05-05 Day 0 起草，Day 14 末完整版）
parent: ./poc-spec.md
---

# PoC → 主分支工程化迁移建议

> **目的**：PoC 验证有效后启动"PoC → 主分支工程化"流程的指南。**PoC 代码绝不直接 copy 到主分支**——主分支按工程标准重新实现，PoC 产出作为参考。

⚠️ **状态**：v0.1 骨架，Day 14 末完整版。

---

## 1. 迁移什么（值得借鉴的产出）

预期 Day 14 填实清单：

- **数据生成 pipeline 的 schema**（虚拟主人画像 + 剧本生成 + 信号转换）
- **LNN 模型架构的具体形态**（StateRecognitionLNN / BoundaryDetectionLNN 的层数 / 维度 / 损失设计）
- **训练 pipeline 的混合策略**（合成数据 + 真实数据混合训练比例）
- **5 阶段记忆形成流程的工程设计**（重要性标记规则 / 工作记忆缓冲区 / 事件分割触发条件 / LLM 精炼 prompt / Consolidation 4-Part 报告）
- **UI 设计的 5 个核心可视化模式**（AI 心智参数面板 / 今天报告 / 用户确认机制）

---

## 2. 不迁移什么（PoC-only，主分支用更专业方案）

| PoC 实现 | 主分支替换 |
|---------|----------|
| Streamlit 单页 UI | Web 主入口 + Android PWA + 车载 / 居家双场景 |
| SQLite 情景记忆 + JSON 语义记忆 | Graphiti（bi-temporal KG）+ Neo4j + Qdrant |
| 单文件 `app.py` 架构 | 四层同心圆架构（按 [memory-architecture.md](../../docs/memory-architecture.md)） |
| MacBook Air 本地训练 | 云端 + 联邦学习 + 边缘推理 |
| 启发式 importance score | LNN 学到的 importance gating |

---

## 3. 迁移工程纪律

- PoC 代码绝不直接 copy 到主分支
- 主分支按工程标准（[CLAUDE.md §2.1](../../CLAUDE.md) PR 三道门 + [development-harness.md](../../docs/development-harness.md) BRO-PDCA × AC + DoD）重新实现
- PoC 产出作为"参考实现"和"产品需求规范"
- 任何 PoC 中"凑合实现"的部分（启发式规则 / mock 数据 / 简化逻辑），主分支必须显式重新设计

---

## 4. 给 M1 启动的具体动作建议（Day 14 末填）

预期填实：
- 主分支应该新建什么 capability-spec
- 主分支应该立即启动哪 3-5 个工程任务
- 主分支应该把 PoC 哪些验证产物存到 `evals/` 下作为基线

---

## 5. PoC 验证产出之外的发现（Day 14 末填）

PoC 过程中可能产生的"意外发现"清单——比如：

- 某个 trait 在 LNN 训练中比预期重要 / 不重要
- 某个多模态信号的工程难度被低估 / 高估
- 某个评估指标的实际信噪比比预期好 / 差
- 老板自用过程中的认知盲区

这些"意外"对主分支工程化决策有价值，必须在 Day 14 完整版列清楚。

---

署名：cc（Claude Opus 4.7）| 2026-05-05 v0.1 骨架起草
