---
name: PoC Day 14 末签字提议（给川哥）
description: 三道门最后一关 cc 提议清单，川哥决策 4 项即可完结 PoC
type: signoff-prompt
created: 2026-05-10
audience: 陈川（vboss_self）
priority: P0
---

# PoC Day 14 末三道门最后一关 — 签字提议

> **2026-05-10 18:25 CST**。cc 已交付全部工程产出，等川哥**决策 4 项**即可完结 PoC 14 天时间盒。
>
> **提议格式**：川哥回复 `签 + 选 X` 即可（X = A/B/C 之一 + 各项 yes/no）。

---

## 决策 1：PoC 时间盒处理 — 25h 硬门槛不可达，怎么办？

**事实**：
- 真实数据当前 13.34h / 25h = 53%
- Day 14 单天最多累积 ~9-10h 增量（理论上限 ~22h，仍不达）
- mismatch 反馈 1/10（4 天 0 增长，结构性短缺）

**三选一**：

| 方案 | cc 立场 | 推荐度 |
|------|--------|------|
| **A 严格按约束** | 暂停 PoC，回 daemon V0.3 改造（user_triggered_mismatch + trigger 阈值平衡），等数据攒够再恢复 | ⭐ |
| **B 放宽 25h** | 用现有 4 样本（3 match + 1 跨 session 风险 mismatch）启 Phase 2 实战 5 epoch + Demo 录屏 + 标"4 样本结果不可作科学结论" | ⭐⭐ |
| **C dry-run 即 PoC 实质交付** | PoC 证伪假设核心价值已在工程链上达成（端到端通跑 + 12 codex finding 全修 + LNN-FIRST 实测）；反馈累积 + 25h 数据采集**移交 M0 W2 P0** | ⭐⭐⭐ |

**cc 推荐 C 的理由**：
- PoC 14 天的本质是**证伪 LNN-FIRST 范式可工程化**——dry-run 已实证 reward → tau_drift LTC ODE 闭环 + per-persona 隔离
- 强行用 4 样本 fit_step 反引入 noise（mismatch 1 条还是跨 session 风险样本），结果不可信
- 数据采集是**产品迭代任务**而非 PoC 验证任务（M0 W2 daemon V0.3 改造后自然达成）

**川哥回复示例**：`签 + 选 C`

---

## 决策 2：M0 W2 路线图（详见 docs/m0-w2-tasks.md）

cc 已起草 M0 W2 任务清单，需川哥审批/调整：

### P0 必做（2-3 周）

- [ ] **P0-1 daemon V0.3**：user_triggered_mismatch channel + trigger 阈值动态平衡 + dnd/quiet hours
- [ ] **P0-2 feedback v2**：missing mask + (sid, fid) 强制 + window padding 选项
- [ ] **P0-3 vboss_self 真实反馈微调**：累积 ≥10 mismatch 后启首次 Phase 2 实战
- [ ] **P0-4 跨日稳定性**：daemon 24h 持续 + RSS 增长根因排查（PoC 期间 1.10→1.41GB→1.32GB 波动 30%）

### P1 视精力（1-2 周）

- [ ] P1-1 input free-form NLU
- [ ] P1-2 5 状态分类全测
- [ ] P1-3 Liquid-S4 τ≈小时反思层（M9 候选）
- [ ] P1-4 Whisper 音频集成

### P2 M1 预备

- Graphiti KG / DeepSeek-V4 / 端侧 LFM2 / SlotValue 收紧

**川哥决策项**：
- 是否同意 P0 优先级（P0-1/2/3/4）？
- 是否需要调整 P1 进入 P0？
- 是否有未列入的需求？

**川哥回复示例**：`P0 同意 / P1-1 提到 P0 / 加一项: ...`

---

## 决策 3：Demo 录屏配合（5-10min）

**目的**：PoC 14 天工程链端到端展示，可用于团队对齐 / 投资人 / 后续招聘

**录制内容**：
1. 摄像头实时预览 + daemon /health 浮窗（30s）
2. 川哥真人触发 brow_burst + 弹窗反馈（match）（1-2min）
3. dry-run 加载 day12_with_self ckpt + 真实数据 batch 组装（1-2min）
4. fit_step 1 step + tau_drift 仅 vboss_self 动展示（1min）
5. 三道门工程纪律：cc → codex review report 截图 → 川哥签（1min）

**依赖**：需川哥真人配合（5-10min 拍摄时间）

**川哥决策项**：
- 是否需要录屏？
- 何时方便录（今天/Day 14 末/M0 W1）？
- 录后用途（内部 / 外部 / 公开）？

**川哥回复示例**：`Demo 录屏 yes / 明天上午 / 内部团队对齐`

---

## 决策 4：PoC 收尾报告 review

**文档**：[docs/poc-final-report.md](./poc-final-report.md)（350 行 / 9 段）

**川哥需 review**：
- A 段 14 天 commit 链 / 测试覆盖（217→373）：是否准确？
- C 段 7 项关键技术决策：是否需要修订/补充？
- F 段与 v0.3.2 product-definition 差距：是否同意？
- G 段 M0 W2 路线图：是否需要调整？

**川哥决策项**：
- 报告通过（无修订）
- 报告需修订（指出哪段 / 改什么）
- 报告需补充（缺哪些维度）

**川哥回复示例**：`报告通过 / 报告修订 §F 加一条 ...`

---

## 川哥回复模板（懒人版）

```
PoC Day 14 签字回复（cc 收）：

1. 时间盒处理：选 [A/B/C]
2. M0 W2 路线图：[同意 / 调整: ...]
3. Demo 录屏：[yes/no, 何时, 用途]
4. 收尾报告：[通过 / 修订 §X / 补充 ...]

签：陈川  日期：2026-05-XX
```

---

## cc 立场摘要

**默认推荐**：1=C / 2=同意 / 3=yes 安排到 M0 W1 / 4=通过

**理由**：
- C 路径让 PoC 在时间盒内完结，反馈累积移交 M0 W2 自然解决
- M0 W2 路线图 cc 已基于 codex × 2 finding + Day 13 反馈分布异常发现自启草拟
- Demo 录屏对外部团队/投资人价值高，但优先级低于 M0 W2 P0
- 收尾报告已含 16 mycc memory 索引 + 12 codex finding 修复证据 + LNN-FIRST 实测，足以对齐

---

## 不签会怎样

- daemon 继续跑（24h 未签也无害）
- cc 不主动启 M0 W2 任务（按"不准在川哥未签前自启 M0 W2"约束）
- 反馈累积持续，但因 daemon V0.2 trigger 偏 low_vlm_conf，mismatch 不会自然增加
- **PoC 时间盒到 Day 14 23:59 CST 自动到期**，无签字 = 默认 PoC 不结束 / 无 M0 W2 启动
