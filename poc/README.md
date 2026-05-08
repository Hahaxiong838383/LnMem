# LnMem PoC — LNN 验证项目

> 12-14 天 PoC + 7 天老板自用，证明 **LNN 真的能在多模态时序数据上学到主人特征**，且学习过程对老板和团队**可见、可解释、可治理**。

**分支**：`poc/lnn-validation`（长期分支，不直接合并到 main）
**状态**：v0.16 Day 6.5 完成（2026-05-08），P0 LTC 演化 + P0.5 5 阶段记忆 LNN 化
**演示对象**：内部团队（含老板）

---

## ⚠️ P-1 最高准则（PoC 也不例外）

> **[`../docs/principles/lnn-first.md`](../docs/principles/lnn-first.md) — LNN-FIRST**
>
> PoC 的所有模块（capture / fusion / lnn / memory）**必须**以 LNN（LTC/CfC 微分方程）为核心。
> 凡"时间 / 演化 / 记忆 / 个性化"用 LLM/规则替代 LTC，必须 ADR + 老板签字。
> 老板 2026-05-08 钦定。

---

## 30 秒入口（必读）

按以下顺序阅读 4 份文档即可上手：

1. [`docs/poc-spec.md`](docs/poc-spec.md) — **PoC 完整规范**（目标/AC/DoD/时间盒/失败兜底/资源/归因）
2. [`docs/data-generation-todo.md`](docs/data-generation-todo.md) — **老板 Day 0 立即用**（30-50 虚拟主人画像 schema + 范例 + 评估标准）
3. [`docs/limitations.md`](docs/limitations.md) — PoC 诚实声明（v0.1 骨架，Day 14 末完整版）
4. [`docs/migration-guide.md`](docs/migration-guide.md) — PoC → 主分支工程化迁移建议（v0.1 骨架，Day 14 末完整版）

辅助文档：
- [`docs/lnn-training-log.md`](docs/lnn-training-log.md) — LNN 训练实验日志 schema
- [`CHANGELOG.md`](CHANGELOG.md) — PoC 内部变更日志

---

## 与主分支的关系

PoC 严格遵守主分支：

- **万能语法 20 槽 schema**（不修改）
- **记忆架构定义**（不引入新概念）
- **意图预测三层定义**（[`../docs/capability-specs/cognitive.md`](../docs/capability-specs/cognitive.md)）
- **BRO-PDCA × 目标驱动验收**（[`../docs/development-harness.md`](../docs/development-harness.md)）

PoC **不修改主分支任何 .md 文档**——所有 PoC 相关文档放在 `poc/docs/` 下。

---

## 目录结构

```
poc/
├── README.md                # 入口（本文件）
├── CHANGELOG.md             # 内部变更日志
├── data/
│   ├── personas/            # 虚拟主人画像 JSON（老板 Day 0 写）
│   ├── synthetic/           # 合成多模态数据（Day 4 cc 批量生成）
│   └── real/                # 老板自用真实数据（Day 8-12 累积）
├── docs/
│   ├── poc-spec.md          # 完整规范（核心）
│   ├── data-generation-todo.md  # 画像设计 brief
│   ├── limitations.md       # 诚实声明
│   ├── migration-guide.md   # 主分支迁移建议
│   └── lnn-training-log.md  # 训练实验日志
├── src/                     # 见下方"已实现 vs 计划中"两表
└── demos/                   # smoke 真采脚本 + Day 14 录屏
```

### `poc/src/` 已实现 vs 计划中

> 文档真实性优先于"看起来完整"。下表反映 HEAD `a0adc55` 的真实落盘状态。

**已实现（Day 0-5）**：

| 路径 | Day | 用途 | 实测 |
|------|-----|------|------|
| `capture/` | 1 | 多模态采集骨架（face/audio/context/buffer） | smoke 4/4 PASS（25.5fps + 16kHz + osascript + 478 landmarks） |
| `fusion.py` | 2 | 12-dim FusionFrame 融合层（face4+derived4+audio2+context2） | 12/12 PASS |
| `validate_personas.py` | 0 | 画像校验（HEXACO/ECR/Schwartz/SDT 30 维门） | 217/217 门通过 |
| `data/event_schema.py` | 3 | ScriptEvent + Activity(8) + State(8) + 校验器 | 23 tests |
| `data/llm_clients.py` | 4 | MiniMax-M2.7 + Gemini Proxy + Fallback 三件套 | 28 tests |
| `lnn/contracts.py` | 5 | 接口冻结层: LNNTrainConfig + 4 Protocol | 7 tests |
| `lnn/model.py` | 5 | CfC step-by-step + τ-bias + FiLM (codex 写) | 8 tests |
| `lnn/dataset.py` | 5 | numpy 预处理 + FileGroupedSampler | 8 tests |
| `lnn/train.py` | 5 | 端到端训练: **val_acc=70.2%**, bnd_f1=0.295 | 20 epochs MPS |
| `capture/face_signals.py` AU 扩展 | 6 | 6 维 AU 表情特征(smile/brow/cheek/lip/jaw/mouth) | 6 tests PASS |
| `fusion.py` v1.0 | 6 | FEATURE_DIM 12→18 + `_expression_features()` | 10 tests PASS |
| `lnn/expression_dynamics.py` (P-1 LTC) | 6.5 | LTC 表情演化器 + ExpressionProfile schema | 11 tests PASS |
| `data/personas/expression_profiles/*.json` | 6.5 | LLM 推导 30 主人 × 18 LTC 参数 | 个性化生效 |
| `data/synthetic/llm_v2_ltc/*.jsonl` | 6.5 | 900 LTC 演化 18-dim 信号 | 100% LTC profile |
| `memory/memory_lnn.py` (P-1 P0.5) | 6.5 | 5 阶段 LTC 链 + persona τ + confirmed 终身免疫 | 10 tests PASS |
| `memory/importance_judge.py` | 6.5 | LLM 重要性裁判（仅输出 g 标量门） | — |

**验证脚本（Day 5）**：

| 脚本 | 用途 |
|------|------|
| `scripts/verify_confusion_matrix.py` | 8×8 混淆矩阵 + per-class F1 |
| `scripts/verify_threshold_sweep.py` | boundary sigmoid 阈值扫描 |
| `scripts/human_verify.py` | 合成数据：专挑状态切换段人工对比 |
| `scripts/live_verify.py` | **真人实时验证**：摄像头+麦克风+应用→模型预测 |

**计划中**：

| 路径 | 阶段 | 硬门槛 |
|------|------|--------|
| `lnn/` 扩展 | Day 6 | feature_dim 12→16~18（加 AU 表情特征），boundary 调参 |
| `memory/` | Day 7 | 5 阶段记忆形成 pipeline + F1 SlotValue 类型收紧 |
| `ui/` | Day 7-14 | Streamlit 主入口 + 5 可视化 + Day 14 录屏 |
| `multimodal/` | Day 8+ | 真实采集累积（在 capture/ 之上扩 wrapper，老板自用产数据） |


---

## 4 个不可妥协原则

1. **诚实优于惊艳** — LNN 30-40% / LLM 60-70% 的协作占比必须明说
2. **架构形态优于性能数字** — 完整流程跑通 + 学习趋势可见，不追求 90%+
3. **可解释优于黑箱** — 训练 / 个性化参数 / 长期记忆全部可视化
4. **用户参与优于自动化** — 所有"AI 对主人的新理解"必须老板确认才进长期记忆

---

## 时间盒一览（详见 [`docs/poc-spec.md`](docs/poc-spec.md) §2）

| Day | 阶段 | 硬门槛 | 状态 |
|-----|------|--------|------|
| 0 | 老板做画像 + cc 起草规范 | 30 画像通过校验 | ✅ |
| 1-2 | 多模态信号流入 | Day 1 末稳定 30fps + 16kHz + 上下文 | ✅ |
| 3-4 | 数据生成 | 30 画像 × 30 天 = 7,191h | ✅ |
| 5 | LNN 预训练 + 验证 | 状态识别 ≥30% → 实际 70.2% | ✅ |
| 6 | 特征扩展 + 去偏 | feature_dim 12→18 ✅ | ✅ |
| 6.5 | **P0 LTC 表情演化 + P0.5 5 阶段记忆 LTC** | LNN-FIRST 准则 + 137 测试 PASS | ✅ |
| 7 | 5 阶段记忆 visualize + UI confirm | 已大半提前到 6.5 | 🟡 |
| 8-12 | 老板自用累积真实数据 | Day 12 末真实数据 ≥ 10h | 🔲 |
| 13 | 真实数据微调 | 微调 vs 预训练对比 | 🔲 |
| 14 | 演示 + 录屏 | 5 个可视化全部出片 | 🔲 |

---

## 协作纪律

按 [`../CLAUDE.md`](../CLAUDE.md) §2.1 PR 三道门：

1. cc 实现（代码 + 测试 + 文档）
2. codex 对抗审查（每模块 1 次）
3. 老板 review + 决定是否进下一天

每日 retro 写入 [`CHANGELOG.md`](CHANGELOG.md)。

---

署名：cc（Claude Opus 4.7）| 2026-05-05
