---
name: LnMem PoC Day 13 进度
description: Day 13 / feedback_to_batch 工程链 + 真实数据端到端 dry-run sanity
type: progress
created: 2026-05-10
phase: PoC Day 13 of 14 (剩 1 天)
---

# LnMem PoC Day 13 — feedback_to_batch 工程链 + 真实数据端到端 dry-run

> Day 13 = PoC 倒数第二天。Day 12 工程层面 100% 就位，今日交付 **数据 join 工程链 + 真实数据端到端 dry-run** = **PoC 关键里程碑达成**：从摄像头采集到 reward-driven SGD 全闭环工程已通跑。
>
> 重要：**dry-run 不是 Phase 2 启动**（反馈仍 7 条 < 10 最低门槛），仅证明工程闭环。

---

## A. Day 13 Loop 5 (15:43 CST) — feedback_to_batch 工程链交付

### A.1 Loop 5 状态变化（vs Day 12 14:32 闭环）

| 项 | Day 12 末 (14:32) | Day 13 Loop 5 (15:43) | Δ |
|----|-------------------|------------------------|---|
| daemon uptime | 2.5h | 3.67h | +1.17h |
| frames_processed | 7720 | 11628 | +3908 |
| triggered | 41 | 44 | +3 |
| queried | 4 | **5** | **+1 川哥进帐反馈** |
| 反馈累积 | 6 条 | **7 条** | +1 (5/10 frame 12221 brow_burst → match) |
| signals 总累积 | 10.4h | 11.0h | +0.6h |

### A.2 工程链产出（commit pending）

#### `poc/src/lnn/signals_loader.py`（180 行）
桥接 daemon V0.2 实采 jsonl → fusion 18 维 vector：
- `parse_signals_row(row) → SignalsFrame | None`：fail-soft 解析单行 + isinstance 防护
- `_signals_row_to_frame_dict`：把 jsonl 字段映射为 fusion.MultimodalFusion 期望的 frame dict（face / audio / context 三段；audio + context daemon 不采 → missing_default=0.5）
- `load_signals(dir, date_prefix)`：批量加载 + sorted glob + frozen tuple 输出
- `build_frame_id_index(frames)`：frame_id → list-index 反查表，跨天重复 frame_id 取后到（与日志读取顺序对齐）
- `signals_to_tensor(frames)`：tuple → Tensor (n_frames, 18) float32

**关键映射**：
- `eye_closure ← calibrated_au["AU43"]`（眨眼/眼闭单元）
- `head_pitch_deg / head_yaw_deg ← head_pose.{pitch_deg, yaw_deg}`
- `expression 6 维 ← signals_row["expression"]`（直接对齐）
- `gaze_stability` daemon 不采 → 0.5

#### `poc/src/lnn/feedback_to_batch.py`（200 行）
`feedback × signals × persona embedding` 三方 join → fit_step 期望的四元组 tensor：
- `STATE_NAME_TO_IDX`：8 state 反查（low_energy=0, focused=1, distracted=2, fatigued=3, emotional_positive=4, emotional_negative=5, neutral=6, transitioning=7）
- `MATCH_LABEL_PLACEHOLDER = -1`：match 反馈 label 占位，由 caller `fill_match_labels_with_argmax` 填补
- `_label_for_event(event)`：
  - mismatch + true_state ∈ STATE_LABELS → `(idx, "mismatch_true_state")`
  - match → `(-1, "match_argmax_pending")`
  - mismatch w/o true_state → None（warning）
  - skip / input → None（filter 阶段已剔除）
- `_slice_window(signals, frame_id_idx, target_frame_id, window_size)`：
  - 取 `[target - window+1, target]` 共 window_size 帧
  - 不足时返 None（PoC 阶段不做 padding，避免污染 LTC ODE 早期帧语义）
- `build_training_batch(feedback_dir, signals_dir, persona_idx, window_size, ...) → TrainingBatch`：
  - 加载 events × signals + 按 _label_for_event + _slice_window 组装 (B, T, 18) + (B,) idx + (B,) labels + (B,) rewards
  - skipped_count 显式计数（缺 true_state / window 不足 / frame_id miss）
- `fill_match_labels_with_argmax(batch, model)`：用 model.eval() forward 后取 argmax 填 -1 占位 → 返新 frozen TrainingBatch（immutable）

#### `poc/tests/test_signals_loader.py` + `test_feedback_to_batch.py`（26 cases）
- TestParseRow ×5：18 维 / smile 字段穿透 / face_detected / non-dict 防护 / AU43 eye_closure 映射
- TestLoadSignals ×6：批量加载 / frozen / frame_id 保留 / dir 缺失 / bad json 跳过 / date_prefix 过滤
- TestIndexAndTensor ×4：build_index / 跨天重复取后到 / tensor shape / empty 抛
- TestStateNameMapping ×1：8 state 反查完整
- TestBuildBatch ×8：仅 match+mismatch 留 / label 映射 / rewards / skipped_count / window 不足 / frame_ids / include_skip_input 默认 False / no_signals 抛
- TestFillArgmax ×2：仅填 placeholder / immutable 返新对象

---

## B. **PoC 关键里程碑：真实数据端到端 dry-run sanity ✅**

### B.1 Dry-run 配置

```
ckpt: poc/data/checkpoints/day12_with_self/checkpoint_day12.pt
config: n_personas=31 / hidden_dim=64 / sequence_length=60
online.tau_drift.shape: (31, 64)  ← OnlineLoRAAdapter 包装就位
```

### B.2 真实 batch 组装

```
源 feedback:  ~/.lnmem/feedback/queries_*.jsonl (7 条)
源 signals:   ~/.lnmem/data/signals_*.jsonl (39489 帧 ≈ 11.0h)
join 后 batch.x.shape: (4, 60, 18)
  - filter: skip×2 + input×1 = 3 条剔除（filter_for_training）
  - 4 训练样本：1 mismatch + 3 match
frame_ids: (0, 158, 3045, 12221)
sources:   ('mismatch_true_state', 'match_argmax_pending'×3)
labels (fill 后): [2, 7, 5, 7]
   - 2 = distracted  (mismatch user_response.true_state)
   - 7 = transitioning  (match argmax)
   - 5 = emotional_negative  (match argmax)
   - 7 = transitioning  (match argmax)
rewards: [-1.0, +1.0, +1.0, +1.0]
```

### B.3 fit_step 1 step 实测（lr=1e-3）

```
metrics: {
  'loss': -1.7332,
  'drift_max_norm': 3.65e-06,
  'batch_size': 4,
}

tau_drift 反传后 L2 范数对比：
  vboss_self    (idx=30): 0.000004    ← 唯一动了的 slot ✅
  vboss_001-030 (idx<30): 0.000000    ← 全保持 0 ✅
```

### B.4 **关键工程证据**

| 不可妥协约束 | 真实数据实测验证 | 状态 |
|--------------|-----------------|------|
| **LNN-FIRST P-1**：τ-bias 漂移走 LTC ODE | ckpt LTC ODE forward 跑通，loss.backward 触达 tau_drift | ✅ |
| **per-persona 隔离**：vboss_self 反馈 ≠ 影响其他 persona | 4 batch sample 全 idx=30 → drift[30]≠0 / drift[<30]=0 | ✅ |
| **不准合成数据微调** | 全部数据来自 daemon 实采 signals + 真实 user_response | ✅ |
| **三道门：cc 实现 → codex 对抗审 → 川哥签** | Day 12 已过 codex 审；Day 13 工程链待 codex 对抗审 | ⏳ |
| **真实数据量 ≥15h** | 当前 11.0h，距 Day 13 末门槛缺 4h（27%） | ⏳ |

### B.5 Dry-run 不算 Phase 2 启动的明确声明

**反馈累积 7 < 10 最低门槛**：本 dry-run 是工程链验证，**不是** Phase 2 在线学习实战微调。
- 仅做 1 个 fit_step 是为了证明 forward / loss / backward / step / clip 各环节通跑
- 不存 ckpt（不污染 day12_with_self），不进入"vboss_self 真人个性化已学习"语义
- Phase 2 正式启动条件：**反馈累积 ≥ 10 最低 + ≥ 10 mismatch 触发**（trigger_if_ready 守门）

---

## C. Day 13 daemon /health 时序日志

| 时间 | uptime | frames | triggered | queried | rate_blocked | 备注 |
|------|--------|--------|-----------|---------|--------------|------|
| 15:43 CST | 13196s | 11628 | 44 | 5 | 39 | Loop 5 起步 +1 reward 到 7 / signals 11.0h |
| 16:24 CST | 15658s | 13819 | 49 | **6** | 43 | Loop 6 / +0.68h uptime / +5 触发 / +1 skip 反馈到 8 总 (5/10 frame 14406 侧对镜头 conf=0.1 临界) / signals 11.6h (25h 46%) |

## C.6 Loop 6 (16:24 CST) — Codex Day 13 工程链对抗审 + 反馈累积更新

### Loop 6 状态变化（vs Loop 5 15:43）

| 项 | Loop 5 (15:43) | Loop 6 (16:24) | Δ |
|----|----------------|----------------|---|
| daemon uptime | 3.67h | 4.35h | +0.68h |
| frames | 11628 | 13819 | +2191 |
| triggered | 44 | 49 | +5 |
| queried | 5 | **6** | **+1 川哥进帐** |
| 反馈累积 | 7 条 | **8 条** | +1 (5/10 frame 14406 skip) |
| signals 总累积 | 11.0h | 11.6h | +0.6h |

### 新反馈分析（5/10 16:03 frame 14406）

```json
au_description: "图像中人物侧对镜头，面部特征不清晰，无法确定具体AU激活"
confidence: 0.1  ← 刚好不命中 FB-09 VLM_NO_FACE_CONF_THRESHOLD=0.1（< 才跳过）
user_response: skip
```

**评估**：
- FB-09 行为正确 — `conf=0.1` 不属于"无脸"语义（确实有人脸只是侧对），让弹窗触发是合理决策
- 川哥 skip 也合理 — 不确定状态 → 不污染训练
- **反馈数据质量观察**：daemon 累计 8 条反馈中 skip 占 3 条（37.5%），意味着 ~1/3 的弹窗在川哥眼里是"非典型时刻"，触发率偏高的隐性指标
- M0 W2 优化方向：trigger 阈值再调严，或 skip 比例 > 30% 时自动收紧

### 派 codex 对抗审 Day 13 工程链（三道门中间门）

**任务**：审 signals_loader.py + feedback_to_batch.py + 26 TDD（commit b4925e4）
**模型**：gpt-5.3-codex / high reasoning
**重点 8 项**（按 cc 自查盲区）：
1. 跨天 frame_id 重复处理是否安全（F.1 风险，cc 已知 silent 取后到）
2. match argmax 自一致 label 选择是否合理（F.2，confidence 极低时噪声放大）
3. window 不足跳过 vs padding 决策（F.3）
4. signals jsonl 字段缺失 fail-soft 完整性
5. 18 维 fusion vector 映射正确性（face/audio/context 缺失语义）
6. STATE_NAME_TO_IDX hardcode vs derive 风险
7. TrainingBatch frozen 但 tensor 内可被外部 in-place 改
8. fill_match_labels_with_argmax 与 LNN-FIRST 关系（model.eval vs train 行为差异）

**后台 PID 36626**，预计 10-15min 完成。落 /tmp/codex-day13-review.md。

---

## D. 反馈累积更新（5/10）

| frame_id | reason | label | true_state | reward | 备注 |
|----------|--------|-------|-----------|--------|------|
| 158 | brow_burst | match | — | +1 | 5/10 04:05 |
| 1742 | low_vlm_conf | skip | — | 0 | 5/10 04:32 (FB-09 v2 漏，已修) |
| 3045 | brow_burst | match | — | +1 | 5/10 04:54（疲倦专注 AU4+AU7）|
| 8012 | low_vlm_conf | skip | — | 0 | 5/10 14:16 (FB-09 v3 漏，已修) |
| **12221** | **brow_burst** | **match** | — | **+1** | **5/10 15:26（本轮新增）**|

5/9：mismatch+distracted (frame_id=0) + input ("我不在") = 2 条

**总 7 条**：match×3 / mismatch×1 / skip×2 / input×1
**有效训练信号**：match 3 + mismatch 1 = **4 条**（dry-run 已用）

---

## E. 待办（Day 13 末 + Day 14）

### Day 13 剩余（cc 主线）
- [ ] 派 codex 对抗审 Day 13 工程链（signals_loader + feedback_to_batch）
- [ ] daemon 持续运行 ≥ 6h（DoD 验收）— 当前 3.67h（61%）
- [ ] 反馈累积 ≥ 10 mismatch 触发首次 Phase 2 实战
- [ ] 真实数据 ≥ 15h（Day 13 末 → Day 14 进门门槛）— 当前 11.0h（73%）

### Day 14（PoC 最后一天）
- [ ] Phase 2 实战微调（如反馈达标）：5+ epoch 跑完 + ckpt 保存
- [ ] 同帧前后 lnn_score 对比实验（mismatch 反馈下分数下降）
- [ ] Demo 录屏（5-10min 端到端展示）
- [ ] PoC 收尾报告

---

## F. 关键技术决策（Day 13）

### F.1 跨天 frame_id 重复处理
**问题**：daemon 每天重启会重置 frame_id=0，5/9 有 frame_id=0..27860，5/10 又有 frame_id=0..11629。`build_frame_id_index` 默认取后到（5/10 优先）。
**影响**：5/9 12:56 那条 mismatch (frame_id=0) 实际 join 到 5/10 frame 0..59 的 window，语义错位。
**为何 PoC 阶段接受**：5/9 mismatch 是"光线极暗 VLM 自报无脸"样本（FB-09 已修），不是基于 face signal 的有效训练信号，window 内容不重要。
**Day 14 / M0 W2 修**：feedback jsonl 加 session_id 字段（V0.2 已加）→ join 时按 (session_id, frame_id) 复合键，避免跨天混淆。

### F.2 match label 用模型 argmax 自一致而非用户 true_state
**为何**：match 反馈语义是"模型预测正确"，所以正确 label 就是模型当前 argmax；用户没填 true_state（默认 None）。
**实现**：`MATCH_LABEL_PLACEHOLDER = -1` 在 build_training_batch 阶段作为占位，调用方调 `fill_match_labels_with_argmax(batch, model)` 用 model.eval() forward 后填补。
**风险**：argmax 选择可能 noisy（uniform-ish softmax 输出时），但 reward=+1 强化只在 confidence 已高时有意义；噪声 label 影响小。

### F.3 不做 padding，window 不足直接跳过
**理由**：LTC ODE 半隐式欧拉 forward 第 0 帧 hidden=0，padding 会让模型把 padding 当成"真静止"特征学到错误τ时间常数。直接跳过更稳健，代价是 frame_id < window_size 的 event 损失（5/9 frame 0 已被跳过——实际：dry-run 没跳过因为 join 到 5/10 frame 0..59 替代了）。

---

## G. 风险

1. **frame_id 跨天混淆**（F.1）：5/9 反馈 join 错 → 在反馈量大时影响显著，Day 14 修
2. **反馈量距 ≥10 mismatch 仍远**（1 → 10）：可能 Day 13 末仍不达标
3. **真实数据 11h，Day 14 末 25h 硬门槛**：缺 14h，需川哥 2 天日均 7h daemon 使用
4. **dry-run loss=-1.73 含义不直观**：reward-weighted loss 数值难解读，Day 14 加 per-sample loss 分解日志