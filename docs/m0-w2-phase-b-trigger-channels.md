---
name: M0 W2 Phase B (P0-1.5) — trigger 通道多样化
description: 解决"都是皱眉 1.0"UX 痛点 / 数据驱动定阈值 / 新增 pose_held + per-channel cooldown
type: design-doc
phase: M0 W2
created: 2026-05-11
status: implemented (cc 主线) → codex 对抗审 → 川哥签字
related:
  - poc/src/realtime/feedback/trigger.py
  - poc/tests/test_feedback_trigger.py
  - scripts/replay_triggers.py
  - docs/m0-w2-tasks.md
---

# Phase B — trigger 通道多样化设计

## 0. TL;DR

| 项 | V0.2 | V0.3 Phase B |
|---|---|---|
| 通道数 | 2（brow_burst / low_vlm_conf） | **3**（+ `pose_held`） |
| brow 阈值 | 0.85 + 10 帧 | **0.95 + 15 帧**（数据驱动收紧） |
| cooldown | 无 | **per-channel**（brow/pose 300帧 / vlm 120帧） |
| 总触发次数（21.8h 实测） | 355 | **147（-58.6%）** |
| brow 占比 | **71.8%（主导）** | **18.4%** |
| pose_held 占比 | 0 | **53.7%（接管）** |

**核心成果**：弹窗 reason 不再单一，"都是皱眉 1.0" UX 痛点消除。

---

## 1. 痛点

川哥 2026-05-11 反馈：daemon 每次弹窗的 trigger snapshot 显示 `brow_furrow=1.0`，提示内容雷同。

定位：
- V0.2 只有 2 个 trigger 通道（`brow_burst` / `low_vlm_conf`）
- brow_burst 占比 71.8%（主导）→ 弹窗 reason 基本是它
- brow_furrow 饱和到 1.0 是常态（P75=0.936）

---

## 2. 数据驱动决策（2026-05-09~11 共 78480 帧 = 21.8h）

### 2.1 Expression 字段分布（仅 `face_detected=true`）

| 字段 | P50 | P75 | P90 | P95 | P99 | 饱和率（=1.0） |
|---|---|---|---|---|---|---|
| brow_furrow | 0.000 | 0.936 | 1.000 | 1.000 | 1.000 | 23.7% |
| smile_intensity | 0.510 | 1.000 | 1.000 | 1.000 | 1.000 | **49.2%** |
| cheek_raise | 0.521 | 1.000 | 1.000 | 1.000 | 1.000 | **49.2%** |
| lip_corner_depress | 0.000 | 0.126 | 0.254 | 0.320 | 0.419 | 0% |
| jaw_drop | 0.000 | 0.066 | 0.236 | 0.342 | 0.524 | 0% |

**关键发现**：smile / cheek 信号 49% 帧饱和到 1.0 → absolute threshold 失效（实测 0.6 / 0.8 / 0.95 触发次数完全一样，均 ~7490 次）→ **放弃 smile_burst / cheek_burst 通道**。

### 2.2 brow_burst 触发频率（不同阈值 × 窗口大小）

| 阈值 | window=5 | window=10 | window=15 |
|---|---|---|---|
| 0.85 (V0.2) | 0.7min/次 | 2.5min/次 | 6.3min/次 |
| **0.95 (V0.3)** | 0.8min/次 | 3.1min/次 | **8.0min/次** |
| 0.99 | 0.8min/次 | 3.4min/次 | 8.8min/次 |

**决策**：选 `0.95 + 15 帧`（**~8min/次**），比 V0.2 0.85+10 帧的 2.5min/次降 3 倍，符合"低频高质"原则。

### 2.3 head_pose stddev 滑窗（pose_held 通道阈值起点）

| window | P10 | P25 | P50 | P75 | std<2.5° 帧占比 |
|---|---|---|---|---|---|
| 20 | 0.00° | 0.00° | 3.81° | 8.95° | ~25% |
| 30 | 0.00° | 0.00° | 4.48° | 9.63° | ~25% |
| **60** | 0.00° | 0.00° | 5.66° | 10.59° | ~22% |

**决策**：选 `window=60 / stddev<2.5°` → 触发频率每 2.3min/次（原始），加 5min cooldown 后实测 ~5min/次（合理）。

### 2.4 VLM conf 分布

- `<0.1`: **48.2%**（自报无脸，已有 FB-09 过滤）
- `<0.65`: 49.8%（仅多 1.6pp）→ 两极分布
- 触发占比：V0.3 中 27.9%（与 V0.2 持平，未动）

---

## 3. 实现

### 3.1 trigger.py 三大改动

#### a. brow 阈值收紧
```python
_BROW_BURST_WINDOW = 15       # V0.2: 10
_BROW_BURST_THRESHOLD = 0.95  # V0.2: 0.85
```

#### b. 新增 pose_held 通道
- 滑窗 60 帧 max(pitch_std, yaw_std, roll_std) < 2.5°
- **outlier 整窗清空**：任一角 > 60°（py-feat 偶发错检）整窗作废
- **缺 head_pose 字段**：清窗（不连续不能算"持续不动"）
- **face_detected=False**：清窗

#### c. per-channel cooldown
```python
_COOLDOWN_FRAMES_BROW = 300         # 5 min
_COOLDOWN_FRAMES_POSE_HELD = 300    # 5 min
_COOLDOWN_FRAMES_LOW_VLM_CONF = 120 # 2 min
```

按 `frame_id` 计（1fps 即秒）。每通道独立，互不阻塞。

### 3.2 API 兼容性

```python
ProactiveQueryTrigger(
    # V0.2 兼容参数（默认值改）
    brow_threshold=0.95,       # was 0.85
    brow_window=15,            # was 10
    vlm_conf_threshold=0.65,   # 未变
    # V0.3 新增参数（全部带默认值）
    pose_held_window=60,
    pose_held_stddev_threshold_deg=2.5,
    pose_outlier_abs_threshold_deg=60.0,
    cooldown_frames_brow=300,
    cooldown_frames_pose_held=300,
    cooldown_frames_low_vlm_conf=120,
    home=None,
)
```

调用方依赖（grep 全仓检查）：
- `daemon_main.py:378` — 仅传 `vlm_conf_threshold=fb.query_confidence_threshold`，其余默认 ✅
- `engine.py:79` — `trigger or ProactiveQueryTrigger()` 全默认 ✅
- 无 yaml/toml 硬编码阈值 ✅

**结论**：V0.2→V0.3 兼容，调用方零改动。

---

## 4. 离线回放验证（21.9h 真实数据，含 codex 一审 6 finding 修复后）

脚本：`scripts/replay_triggers.py`
输出：`/tmp/phase_b_replay_v2_report.md`

| 维度 | V0.2 | V0.3 Phase B |
|---|---|---|
| 总触发次数 | 636 | **240（-62.3%）** |
| 触发频率 | 1/124 帧 ≈ **2.1min** | 1/329 帧 ≈ **5.5min** |
| 通道多样性 | 2 | **3** |
| 中位间隔 | 31 帧 | 300 帧（与 cooldown 一致）|

**通道分布**：

| 通道 | V0.2 | V0.3 |
|---|---|---|
| brow_burst | **81.8%（主导）** | **22.9%** |
| low_vlm_conf | 18.2% | 22.5% |
| pose_held | 0 | **54.6%（接管）** |

**每日分布**（V0.3）：
- 2026-05-09: 88 次 (brow=30, low_vlm_conf=17, pose_held=41)
- 2026-05-10: 123 次 (brow=11, low_vlm_conf=32, pose_held=80)
- 2026-05-11: 29 次 (brow=14, low_vlm_conf=5, pose_held=10)

> **注**：codex 一审 F1-01 揭示 frame_id 跨天回退导致 cooldown 永真 → V0.2 一审基线被低估 281 次（355 vs 636）。二审基线修复后结论更准。

---

## 5. 测试覆盖

`poc/tests/test_feedback_trigger.py` 33 个测试（V0.2: 21 + V0.3 新增: 12）：

- `TestPoseHeldChannel`（5）: 触发 / 不触发 / outlier 清窗 / 缺字段清窗 / face=False 清窗
- `TestPerChannelCooldown`（4）: brow cooldown / 通道独立 / low_vlm_conf cooldown / pose cooldown
- `TestV03DefaultThresholds`（3）: brow 0.95/15 / pose 60/2.5/60 / cooldown 300/300/120

**全量回归**：403/403 ✅

---

## 6. 放弃的设计（and Why）

### 6.1 smile_burst / cheek_burst（信号病态）

实测 smile_intensity / cheek_raise 49% 帧饱和到 1.0。改阈值（0.6 vs 0.95）触发次数完全相同（~7490）→ 阈值无效，本质 binary 信号。

加入会引入大量伪触发，**坚决放弃**。

### 6.2 smile_rate（窗口内 smile>0.5 占比）

实测 47% 帧满足"过去 30 帧内 50% 时间 smile>0.5"→ 川哥日常脸本身就是"半笑容"基线，这维度不能区分情绪。

### 6.3 brow_delta（突增）

可探索方向（未实施），但当前 0.95+15 帧 + cooldown 已解决核心痛点，留 M0 W2 P1 处理。

---

## 7. LNN-FIRST 合规性

P-1 准则审视：trigger 是 hand-crafted 启发式 sampler，**不是** LNN 核心。但其作用域是"是否调起 LNN 推理 + 弹窗"，属 LNN 前端 sampler，与 LNN 内核（LTC ODE / CfC）解耦。

未来路径：trigger 输出的 `TriggerEvent.signal_snapshot` 已含 pose 字段，可作为 LNN 输入 feature 的一部分（M0 W2 P1）。

**不违反 LNN-FIRST**。

---

## 8. 后续路径

### 阶段对接

- **M0 W1 daemon V0.3 (已上线)**：trigger 在 daemon 内自动生效，无需改 daemon 代码
- **M0 W2 P0-3 vboss_self 微调**：trigger 触发的 mismatch 反馈喂给 LoRA fit_step
- **M0 W2 P1**：brow_delta 突增检测 / smile_rate 修正基线 / pose 配合 LNN feature

### 需观察

- **pose_held 53.7% 是否过强**：若实际弹窗后川哥经常说"我没专注啊"，需调严 stddev 阈值
- **brow_burst 18.4% 是否过弱**：若强烈情绪漏报，可放宽到 0.93 + 12 帧重试
- **触发间隔分布**：V0.3 中位 5min 是否符合川哥心理预期，需 1 周实测

---

## 9. 三道门状态

- ✅ **第一道：cc 实现**（trigger.py + 12 新测试 + replay 脚本 + 报告）
- ✅ **第二道一审：codex BLOCK**（7 finding：F1×1+F2×3+F3×3），cc 修 6 个（F3-07 延后独立 commit）
  - F1-01 frame_id 跨天回退 → 加 `_detect_and_handle_frame_rollback`
  - F2-02 no-face 只屏蔽 vlm 通道 → 改 if/elif 结构，brow/pose 仍评估
  - F2-03 pose 字段防御性 parse → 新增 `_safe_float`（排除 bool + isfinite）
  - F2-04 enable_pose_held 开关 → V0.2 replay 真实禁用语义
  - F3-05 roll 可选 → 缺 roll 不入 deque，max 只看 pitch+yaw
  - F3-06 brow cooldown 内不清窗 → 解禁后下一帧立即触发，不需重攒
- ✅ **第二道二审：codex BLOCK**（2 新 finding：F1-01×F3-06 交互 + F3-05 corner case）
  - NF-01: rollback 仅清 cooldown 不清 evidence → 跨 session 旧 brow 窗口污染新 session 首帧
  - NF-02: roll "先满后缺" 时 deque 内容滞后 pitch/yaw 窗口（时序错位）
  - cc 修：rollback 同时清 evidence；缺 roll 时 clear roll deque（不仅"不 append"）
- ✅ **第二道三审：codex PASS**（NF-01 + NF-02 均 FIXED，无新 bug；`/tmp/codex-phase-b-review3.md`）
- ⏳ **第三道：川哥签字**（待川哥实测 daemon 重启后多通道触发）
- 📌 **F3-07 daemon config 暴露**（延后到独立 commit，不阻塞本轮）

---

## 10. 改动文件清单（含 6 finding 修复后）

| 文件 | 改动 | 行数 |
|---|---|---|
| `poc/src/realtime/feedback/trigger.py` | 加 pose_held + cooldown + 阈值收紧 + 6 finding fix | +210 / -20 |
| `poc/tests/test_feedback_trigger.py` | 12 个 V0.3 测试 + 6 个 finding 修复测试 = 18 新增 | +220 |
| `poc/tests/test_feedback_v02_boundary.py` | 更新到 V0.3 阈值 | ±15 |
| `poc/tests/test_feedback_engine.py` | 更新 brow 帧数+cooldown 兼容 | ±10 |
| `scripts/replay_triggers.py` | 离线回放脚本（新建）+ V0.2 用 enable_pose_held=False | +233 |
| `docs/m0-w2-phase-b-trigger-channels.md` | 本设计文档（新建） | +220 |

**测试覆盖**：trigger 21 (V0.2) + 12 (V0.3 新通道+cooldown) + 6 (一审 fix) + 2 (二审 NF-01/NF-02 fix) + 1 (F3-05 满前缺补充) = **42 个 trigger 测试**；全量回归 **412/412 ✅**。
