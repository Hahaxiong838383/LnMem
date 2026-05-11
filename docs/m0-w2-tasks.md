---
name: LnMem M0 W2 任务清单
description: PoC 14 天后接续的 2-3 周开发任务，按 P0/P1/P2 分级
type: task-list
created: 2026-05-10
phase: M0 W2 (post-PoC)
prereq: PoC Day 0-13 工程链 + 三道门 codex × 2 通过
---

# LnMem M0 W2 任务清单

> PoC 14 天交付了端到端工程链 + dry-run 证据，**反馈累积与 25h 数据采集移交 M0 W2**。本文档是 PoC 收尾报告 §G 的展开。

---

## P0 任务（必做，预计 2-3 周）

### P0-1 daemon V0.3：反馈类型分布平衡

**根因**（PoC Day 13 §C 发现）：daemon V0.2 trigger 偏 low_vlm_conf（55% skip）不偏 user 主动纠正（mismatch 1/11 = 9%），Phase 2 实战触发条件 mismatch ≥ 10 实质不可能达。

**子任务**：
- [ ] **user_triggered_mismatch channel**：川哥可主动按快捷键 / CLI / 前端按钮报告"模型搞错了，true_state 是 X"
  - 实现路径：daemon HTTP 9876 加 `/feedback/manual` 端点，POST {true_state, note} → 写入 queries jsonl with reason="user_triggered"
- [ ] **trigger 阈值动态平衡**：基于过去 24h 反馈分布自调
  - skip 占比 > 40% → 收紧 low_vlm_conf 阈值（0.65 → 0.50）
  - mismatch 占比 < 10% → 弹窗时给 "press X to nudge" 提示鼓励纠正
- [ ] **dnd / quiet hours**：daemon 后台采集但弹窗仅在川哥 active 窗口（FocusedApp 检测 + 23:00-07:00 silent）
- [ ] 测试：feedback_engine 端到端 + 阈值平衡 sanity（注入 60% skip 历史 → 阈值自调收紧）

**验收 DoD**：
- daemon V0.3 上线 1 周后，mismatch 占比 ≥ 25%（vs PoC 9%）
- skip 占比 < 40%

### P0-2 feedback_to_batch v2

**根因**（PoC Day 13 codex F2-04 + F1-01）：missing 语义被默认 0/0.5 实值替代污染训练 / V0.1 fallback 仍允许跨 session 错配（虽 warn 但不 strict）。

**子任务**：
- [ ] **missing mask 显式化**：fusion vector 旁附 (T, 18) bool mask，下游 LNN 训练时把 missing 位置的 loss weight 设 0
- [ ] **(session_id, frame_id) 强制**：去 V0.1 fallback，feedback session_id=None 直接 reject + warn（要求 daemon V0.3 写齐）
- [ ] **window padding 选项**：可配置 `padding_strategy = "drop" | "missing_pad"`，默认 drop（PoC 行为），missing_pad 用 missing mask 标记
- [ ] 测试：missing mask 端到端 / strict session join / padding 两种 strategy

**验收 DoD**：
- F2-04 finding 标注 closed
- 100% feedback 都有 session_id（无 fallback warning）

### P0-3 vboss_self 真实反馈微调（首次 Phase 2 实战）

**前置**（必须满足）：
- 累积反馈 ≥ 30 条（理想，min 10）
- mismatch ≥ 10 条
- 反馈来自 ≥ 2 不同 session（避免单 session bias）

**子任务**：
- [ ] 加载 day12_with_self/checkpoint_day12.pt + OnlineLoRAAdapter wrap
- [ ] build_training_batch + fill_match_labels_with_argmax
- [ ] 跑 5+ epoch + 监控：
  - loss 曲线（应单调下降至收敛）
  - per-persona drift L2 累积（仅 vboss_self idx=30 应增长）
  - 同帧前后 lnn_score 对比（mismatch 反馈帧训练后预测 confidence 应下降）
- [ ] 存 ckpt → `m0w2_phase2_first_run/checkpoint.pt`
- [ ] 写 Phase 2 实战报告 `docs/m0-w2-phase2-first-run.md`

**验收 DoD**：
- mismatch 反馈帧训练后 lnn_score 严格下降（与 PoC dry-run sanity 一致）
- vboss_self 持续 10+ 反馈后 tau_drift L2 增长但 < clip_norm=0.5
- 训练后 ckpt 加载 + 新 forward 不破坏 base 30 personas

### P0-4 跨日稳定性

**PoC 观察**：daemon 单天最长 5.72h（Day 13 17:46 CST），RSS 1.10 → 1.41GB（**有内存增长**），未跨 24h 真实场景验证。

**子任务**：
- [ ] daemon 24h 持续运行验证（macOS launchd 后台守护 + log rotation）
- [ ] RSS 增长根因排查（推测 frame buffer 累积 / VLM async cache / Python GC）
- [ ] 健康度 watchdog：RSS > 2GB 自动重启 daemon（保留 session_id 持久化）
- [ ] 跨天 frame_id 重置事件正确触发新 session_id（V0.3 行为，与 PoC F1-01 修复对齐）

**验收 DoD**：
- daemon 连续运行 ≥ 72h 0 ERROR / 0 内存爆 / 0 数据丢失
- RSS 24h 后 < 1.5GB

---

## P1 任务（视精力，1-2 周）

### P1-1 input free-form NLU

**当前 PoC**：input label 川哥写 free text（"我不在，检测错了吧"），reward=0 跳过，**17% 反馈样本价值未释放**。

**子任务**：
- [ ] LLM 提取隐含 label：用 Gemini 3 / GPT-5.4 把 note 文本 → STATE_LABELS 之一 + confidence
- [ ] 阈值：LLM confidence ≥ 0.7 才采用，否则保 reward=0
- [ ] 标注 source="input_llm_extract"（区别于 mismatch_true_state）

### P1-2 5 状态分类全测

**当前**：仅 mismatch+distracted 1 条 / brow_burst+match 3 条。其他 5 状态（fatigued / idle / low_energy / emotional_negative / emotional_positive）daemon 触发与 user 反馈映射未实测。

**子任务**：
- [ ] 川哥手动做 5 状态 × 各 5min（共 25min）真实采集
- [ ] 看 daemon trigger 触发率 / VLM label 一致性 / 川哥反馈分布
- [ ] 弱信号状态（如 emotional_negative anger 区分 emotional_negative sad）专项设计

### P1-3 Liquid-S4 τ≈小时反思层（M9 候选）

**v0.3.2 决策**：Liquid-S4 LRA 87.32% 候选，M9 实测 vs LTC 决定主路径。

**子任务**：
- [ ] 集成 Liquid-S4 实现
- [ ] 与 LTC 在同一数据集上对比：推理速度 / 长时记忆容量 / 训练稳定性
- [ ] 决策：保留 LTC 主路径 / 切 Liquid-S4 / 双路径

### P1-4 Whisper 音频集成

**当前**：fusion audio_feats 全 missing_default=0.5，**audio 模态完全空载**。

**子任务**：
- [ ] Whisper.cpp 集成 daemon V0.3
- [ ] voice_present / utterance_density 真实计算
- [ ] 测试 daemon 跑 Whisper 时 RSS 增量 / 实时性

---

## P2 任务（M1 预备）

### P2-1 Graphiti KG bi-temporal
v0.3.2 定义的双时间维度知识图谱

### P2-2 DeepSeek-V4 主对接 + GLM-5.1 fallback
当前 PoC 用 Gemini 系列代理，正式版需切回 DeepSeek/GLM

### P2-3 端侧 LFM2
v0.3.2 探索清单：端侧推理优化

### P2-4 SlotValue 类型收紧（H 级，ADR 002）
PoC Day 11 后未做的 F1 finding

---

## 任务依赖图

```
P0-1 daemon V0.3 ────────────────┐
                                  ├──▶ P0-3 vboss_self 微调（前置满足后启）
P0-2 feedback v2 ────────────────┘

P0-4 跨日稳定性  ───── 独立，与 P0-1/2/3 并行

P1-1 input NLU   ───── 依赖 P0-2 v2（reward 信号扩展）
P1-2 5 状态全测  ───── 依赖 P0-1 V0.3（user_triggered + 阈值平衡）
P1-3 Liquid-S4   ───── 独立
P1-4 Whisper     ───── 独立

P2 任务统一进 M1 预备
```

---

## 时间盒预估

| 任务 | 工时（cc 主线，含 codex 对抗审）|
|------|--------------------------------|
| P0-1 daemon V0.3 | 5-7 天 |
| P0-2 feedback v2 | 3-4 天 |
| P0-3 vboss_self 微调（依赖反馈累积） | 2-3 天 + ≥1 周数据采集等待 |
| P0-4 跨日稳定性 | 3-4 天 |
| **P0 合计** | **2-3 周**（部分并行） |
| P1-1 NLU | 2-3 天 |
| P1-2 5 状态 | 2 天 |
| P1-3 Liquid-S4 | 5-7 天 |
| P1-4 Whisper | 3-4 天 |
| **P1 合计** | **1-2 周**（视精力） |

---

## 风险

1. **数据采集仍是主卡点**：P0-3 需要川哥每天用 daemon ≥ 4-6h，且 user_triggered_mismatch 需要川哥**主动**报错（非被动等弹窗）
2. **跨日稳定性内存泄漏未根治**：RSS 1.10→1.41GB 在 5.72h 内增长 30%，24h 可能爆 → P0-4 优先级建议提至 P0 第一
3. **Liquid-S4 ROI 不明**：M9 实测才知是否切主路径，资源投入要看 LTC 是否瓶颈
4. **三道门工程纪律延续到 M0 W2**：每个 P0 模块必须 codex 对抗审，预计 codex 5 + 7 + 5 ≈ 17 finding 修复时间

---

## 立即可做（Day 14 / 本周）

- [ ] 川哥审本任务清单 + 路线图调整
- [ ] 创建 GitHub Issue / Linear ticket 跟踪 P0 进度
- [ ] M0 W2 启动会（如有团队）：拉齐节奏 + 分工
