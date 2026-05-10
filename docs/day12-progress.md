---
name: LnMem PoC Day 12 进度
description: PoC 12-14 天阶段 / Day 12 起步 / Phase 2 在线学习工程骨架 + daemon 反馈累积观察期
type: progress
created: 2026-05-10
phase: PoC Day 12 (剩 3 天)
---

# LnMem PoC Day 12 — 反馈累积 + Phase 2 工程骨架

> Day 12 = PoC 12-14 天倒数第三天 / 反馈样本累积期（远未达 30 条 Phase 2 启动门槛）/ cc 主线先搭工程骨架 + 川哥继续真人使用攒数据。

---

## A. Day 12 起步快照（2026-05-10 12:55 CST）

### A.1 daemon V0.2 健康状态（HTTP /health）

```json
{
  "status": "ok",
  "uptime_s": 3211.964,
  "frames_processed": 2726,
  "last_frame_ts": "2026-05-10T04:56:45.684415Z",
  "feedback": {
    "observed": 2726,
    "triggered": 32,
    "rate_blocked": 29,
    "dnd_blocked": 0,
    "queried": 3,
    "responses": 3
  }
}
```

| 指标 | 值 | 评估 |
|------|-----|------|
| uptime | 53.5min | ✅ 持续运行 |
| frames_processed | 2726 | ≈51 fps × 53min，符合预期 |
| triggered/observed | 1.2% (32/2726) | ✅ 触发率合理 |
| queried/h | ≈3.4/h (3/53min×60) | ✅ 远低于 5/h 安全线（限速生效） |
| rate_blocked/triggered | 90.6% (29/32) | ✅ 限速主导，FB-09 vlm_no_face_skip 也在拦 |

**daemon PID**：15934（川哥常用终端启动，TCC 摄像头授权链 ✅）
**venv**：`.venv-poc-fresh`（Day 11 重建，79 包）

### A.2 反馈样本累积（5 条，远未达 Phase 2 启动门槛）

| ts | reason | label | true_state | 备注 |
|----|--------|-------|-----------|------|
| 5/9 12:56 | low_vlm_conf | mismatch | distracted | 光线极暗 VLM 弃权，川哥说"我不在，检测错了" |
| 5/9 13:16 | low_vlm_conf | input | — | 川哥 free-form："我不在，检测错了吧" |
| 5/10 04:05 | brow_burst | match | — | 单纯 brow_burst（无 VLM 标注），用户确认匹配 |
| 5/10 04:32 | low_vlm_conf | skip | — | VLM 自报 "no face" 但仍触发了一次（FB-09 修复前残留？需查） |
| 5/10 04:54 | brow_burst | match | — | brow_furrow=0.916 + VLM 标 AU4+AU7 疲惫专注，valence=-0.3 |

**关键观察**：
- ✅ FB-09 修复（05a2bb4 + e42c7ad）大量 `vlm_no_face_skip` 写入 skipped jsonl（5/10 04:52 起密集出现）
- ⚠️ 5/10 04:32 那条 low_vlm_conf 仍进入 queries — 可能是 e42c7ad 修复前的残留 trigger，或 confidence=0 但 daemon 当时用 gemini-3.1-pro-preview（不是 flash-lite）走了不同分支，需要查代码确认
- ✅ trigger 频率 3.4/h < 5/h，限速 90% 拦截过敏触发
- 真实数据时长累积：~2-3h（5/9 30min + 5/10 53min 当前次 + 之前几次）

### A.3 距离 PoC 硬门槛差距

| 门槛 | 当前 | 差距 |
|------|------|------|
| 真实数据 ≥ 25h（Day 12 末） | ~2-3h | **远未达** |
| 反馈样本 ≥ 30 条触发 Phase 2（理想） | 5 条 | -25 条 |
| 反馈样本 ≥ 10 条最低门槛 | 5 条 | -5 条 |
| mismatch ≥ 10 条触发首次 LoRA 微调 | 1 条 | -9 条 |

**结论**：Phase 2 启动**需要川哥继续真人使用 4-6h/天**（cc 不可代劳）。本 turn cc 主线最高 ROI = **预先搭好 Phase 2 工程骨架**（接口 + 测试），等反馈攒够无缝触发训练。

---

## B. Day 12 P0 任务规划

### P0-1 daemon 稳定性观察（持续，川哥用产品攒反馈）
- ✅ daemon PID 15934 运行 53min 无 ERROR
- ✅ trigger 频率 3.4/h < 5/h 安全线
- 🔄 cc 每 30min /health 拉快照写入本文 §C 时序日志

### P0-2 Phase 2 在线学习工程骨架（cc 主线，本 turn）
- 设计 `OnlineLoRAAdapter` API（包装 `DefaultPersonalizationAdapter` + frozen base + per-persona τ-drift delta）
- 写 TDD 测试骨架（验证 LNN-FIRST P-1 合规：delta 仅作用于 τ 维度，仍走 LTC ODE）
- 数据层 `feedback_dataset.py`（读 `~/.lnmem/feedback/queries_*.jsonl` → reward 信号）
- 实现 `fit_step()` 留 codex 接力（5.3-codex high）或 cc 主线写（依据 §D 决策）

### P0-3 vboss_self 真人画像入 ckpt
- personalt-test v0.2 自评已有（HEXACO E=0.75 / O=0.92 极高）
- vboss_001 距离 0.52 是关键校准点
- 加 1 slot 微调 5-10 epoch（独立任务，不依赖反馈累积）
- partial 字段 ⚠️ 必须显式标注（CAPS / 多模态 / Schwartz 全空）

---

## C. Daemon /health 时序日志

| 时间 | uptime | frames | triggered | queried | rate_blocked | 备注 |
|------|--------|--------|-----------|---------|--------------|------|
| 12:55 CST | 3212s | 2726 | 32 | 3 | 29 | Day 12 起步快照 |
| 13:10 CST | 3717s | 3090 | 38 | 3 | 35 | +8.4min / +364 帧 / +6 触发但 0 新弹窗（限速吃满）/ queried/h ≈ 2.9 ✅ |

## C.1 Phase 2 工程骨架（cc 主线 13:00-13:15 CST 完成）

**新增文件**：
- `poc/src/lnn/online_contracts.py`（156 行）frozen dataclass + REWARD_MAP + FeedbackEvent.reward()
- `poc/src/lnn/feedback_dataset.py`（136 行）jsonl loader + filter + count，零 PyTorch 依赖
- `poc/src/lnn/online_adapter.py`（131 行）`OnlineLoRAAdapter`：包装 `DefaultPersonalizationAdapter`，base 冻结 + `tau_drift` 增量
- `poc/tests/test_feedback_dataset.py`（14 tests）解析 / 过滤 / fail-soft
- `poc/tests/test_online_adapter.py`（15 tests）冻结 / 加法叠加 / 序列化 / **LNN-FIRST P-1 合规验证**

**Gate 1 verify**：29/29 PASS（0.91s）
**Gate 2 guard**：**275/275 全量回归 PASS**（8.7s，无原有测试退化）

**关键合规验证（LNN-FIRST P-1）**：
- ✅ `test_only_tau_drift_has_grad_after_loss`：替换到 `StateRecognitionLNN` 后端到端 loss.backward()，base 全部参数 grad=None / zero，仅 `tau_drift` 有 grad
- ✅ `test_replace_in_state_model_keeps_forward_signature`：模型 forward 仍走 LTC ODE（`self.backbone.rnn_cell` + softplus 时间常数路径不变）

## C.2 待 Phase 3（实现 fit_step）

| 模块 | 状态 |
|------|------|
| OnlineAdapterConfig（trigger 阈值 / lr / clip_norm） | ✅ 已定 |
| OnlineLoRAAdapter（forward 包装） | ✅ 已实现 |
| FeedbackDataset（jsonl → reward） | ✅ 已实现 |
| **fit_step(events, model, optim) → loss / drift_norm** | ⏳ 待 cc 主线实现 |
| trigger_if_ready(feedback_dir) → bool | ⏳ 待 cc 主线实现 |
| 端到端 mismatch reward 微调实测（mismatch 反馈下 lnn_score 下降） | ⏳ 待反馈累积 ≥ 10 条 mismatch 触发 |

---

## D. 实施路由决策

**route = cc-main（不派 codex）**：
- LNN-FIRST P-1 是 PoC 不可妥协 6 项之一，τ-bias 漂移涉及 LTC ODE 时间常数，cc 主线写更安全
- 工程量 ~300 行 + 测试，按 codex 5 档难度 = 普通档，cc 主线 1-2h 可完成
- 派 codex 反而引入"对抗审"流程拖慢，不如 cc 写完后让 codex 一次性对抗审

**Phase**：
1. cc 主线 Phase 2 骨架（本 turn）→ 接口 + TDD 测试 + 数据层
2. cc 主线 fit_step + ckpt save/load（next turn）
3. 反馈累积到 ≥ 10 条 mismatch → 首次实例化运行
4. codex 对抗审 1 次（按三道门工程纪律）
5. 川哥签

---

## E. 不可妥协约束（违反 = PoC 失败）

1. LNN-FIRST P-1：τ-bias 漂移必须用 LTC ODE，不准 LLM/规则替代
2. 评估同构（cognitive.md §5）：禁问卷弹窗评分，多模态信号自动聚合
3. BRO-PDCA：O 阶段写 AC+DoD，C 阶段附证据
4. 三道门：cc 实现 → codex 对抗审 → 川哥签
5. 真实数据量 < 15h → 不进 Day 13，必须延长川哥自用
6. Phase 2 启动 = 累积反馈 ≥ 10 条最低门槛（不可妥协到 5 条触发）

---

## F. 待办（Day 12 末验收 DoD）

- [ ] daemon 稳定运行 > 6h，无 ERROR/Traceback
- [ ] 反馈样本累积 ≥ 10 条（最低）/ ≥ 30 条（理想）
- [ ] Phase 2 工程骨架就位（OnlineLoRAAdapter + TDD + feedback_dataset）
- [ ] vboss_self 入 ckpt（5-10 epoch 微调完成）
- [ ] task-log + RECENT_EVENTS 追加
- [ ] daemon /health stats 时序写入本文 §C
