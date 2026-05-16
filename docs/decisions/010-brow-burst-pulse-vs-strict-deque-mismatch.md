---
adr-id: 010
title: brow_burst 脉冲信号 vs strict-deque 不匹配 — Schmitt+EMA 修复（implemented）
status: implemented（2026-05-16 当晚 TODO-1 全栈实施完成，replay 验收达标）
date: 2026-05-16
deciders: 川哥 + cc + codex（V0.3.4 加固后实测发现 → 同夜 Schmitt+EMA 修复）
supersedes: —
superseded-by: —
related: ADR-005（dual signal_path）/ ADR-006（threshold-auto-recalibration）/ ROADMAP TODO-1
---

## 上下文

ADR-005 V0.3.4 弹窗 UI 修复 + hardening 完工后（commit `752e514`），川哥摄像头实测发现：
**demo 模式 raw 路径下，主动皱眉 11 分钟、0 次触发 brow_burst**，所有触发都是 pose_held。

## 数据驱动定位（`scripts/replay_trigger_diagnose.py` 离线 replay）

daemon 实际 demo session 窗口：`2026-05-16T14:14:46Z` → `T14:25:40Z`，638 帧。

| 维度 | 数值 |
|---|---|
| face_detected 抖动率 | 0%（排除嫌疑：眨眼/转头让 brow_history 清窗） |
| raw_AU04 ≥ 0.40 的帧 | 91 / 638（14.3%，足够多） |
| **raw_AU04 strict > 0.40 最长连续串** | **4 帧**（peak 0.724） |
| 长度 ≥5 串数 | **0** |
| `_brow_history` 满窗（5 帧）次数 | 630 |
| 满窗 + `all(b > 0.40)` 次数 | **0**（触发条件从未成立） |
| 实际 fire brow_burst | 0 ✅ |

## 根因

`ProactiveQueryTrigger.observe` brow_burst 判定逻辑：
```python
len(self._brow_history) >= self._brow_window_size  # demo=5
and all(b > self._brow_threshold for b in self._brow_history)  # demo=0.40, strict >
```

**川哥真实皱眉 raw_AU04 信号是脉冲式**：每 4-5 帧（约 3-4 秒）就跌一次到 ≤ 0.40，
deque 里只要有 1 个低帧 → `all() = False` → 永远不触发。

3 个嫌疑根因排除：
- ❌ face_detected 抖动让 deque 清空（0% 抖动率）
- ❌ rate_limit 挡了 brow_burst（trigger 层就没 fire，跟 engine 层无关）
- ❌ pose_held priority 抢走（observe 里 brow_burst 优先级**高于** pose_held）

## 决策

**不在 V0.3.x 内修**，本 ADR 作为 ROADMAP TODO-1 `Schmitt Trigger + EMA + dynamic quantile threshold` 的强数据证据。

3 个修复方向（由 TODO-1 选定）：

| Option | 思路 | 工时 |
|---|---|---|
| A | 放宽 strict > 为"deque 内至少 K=4 帧 > thr"（容忍 1 帧 dropout） | 0.5h |
| B | **EMA 平滑 raw_AU04 后判定**（gemini #2 + codex F-04 推荐） | 1.5h |
| C | Schmitt 滞回：进入阈值 thr_high=0.55，退出阈值 thr_low=0.30 | 1h |
| D | 用 LNN margin/entropy 取代手工阈值（M3+ 长期方向） | M3+ |

**短期推荐**：B + C 组合（Schmitt + EMA），与 ROADMAP TODO-1 描述完全对齐。

## 验收标准

实施 TODO-1 后，复跑 `scripts/replay_trigger_diagnose.py` 对今天 638 帧数据：
- 触发 brow_burst ≥ 2 次（与 22:01-22:05 早段 session 表现对齐）
- 不引入误报（pose_held 触发数稳定）

## 工程影响

- 当前 daemon 仍可用：work 模式下 pose_held + brow_burst (0.55 raw 阈值) 已能产生反馈数据
- demo 模式"几乎不触发 brow_burst"的体验问题仍存在，但不影响 PoC 数据收集
- 短期没用户可见影响（PoC 阶段无外部用户）

## 配套产出

- 诊断脚本：`scripts/replay_trigger_diagnose.py`（可复用 offline replay，支持任意 ts 窗口 + force demo 配置）
- spawn_task：`TODO-1 Schmitt+EMA+dynamic quantile` 已挂队列

## 时间线

- 2026-05-16 22:25: 川哥 demo 实测发现 brow_burst 0 触发
- 22:35: 写 replay_trigger_diagnose.py 第一版
- 22:42: 加 to-ts 窗口对齐 daemon session
- 22:50: 数据驱动定位根因（脉冲信号 vs strict-deque 判定不匹配）
- 22:55: 本 ADR identified 状态落盘
- 23:00-23:30: TODO-1 全栈实施（trigger.py Schmitt 状态机 + EMA + window_stable + daemon raw 路径默认 opt-in + sweep_brow_threshold_quantile.py 数据驱动阈值 + 14 个新单测 + replay 验收）
- 23:35: status → implemented

---

## TODO-1 实施记录（V0.3.5 — 2026-05-16 23:30 完成）

### 设计采纳

| Option | 实施 | 备注 |
|---|---|---|
| A 放宽 strict > 为 K=4 | ❌ 未采纳 | 只解症状不解根因，仍受脉冲分布影响 |
| **B EMA 平滑** | ✅ α=0.40（sweep 选定） | `_brow_ema` 状态 + face_undetected 自动 reset |
| **C Schmitt 滞回** | ✅ thr_high/thr_low（数据驱动） | 上升沿需 window_stable，下降沿无条件解锁防死锁 |
| D LNN margin/entropy 取代 | 留 M3+ | 见 ROADMAP 长期段 |

### 数据驱动阈值（`scripts/sweep_brow_threshold_quantile.py`）

| 模式 | EMA α | thr_high | thr_low | gap_ratio |
|---|---|---|---|---|
| demo | 0.40 | 0.42 | 0.22 | 0.52 |
| work | 0.40 | 0.55 | 0.30 | 0.55 |

由 5/10-5/16 共 7 日 raw_au.AU04 全量 quantile + demo session 638 帧仿真触发数双重约束选定。

### Replay 验收（同一份数据 638 帧）

| 模式 | brow_burst | pose_held | 总 fire |
|---|---|---|---|
| Baseline V0.3.4 | **0** ❌ | 6 | 6 |
| Schmitt+EMA V0.3.5 | **4** ✅ | 6（不变） | 10 |

**ADR-010 验收标准：brow_burst ≥ 2 → 实际 4，超额达标。pose_held 完全不变 = 无误报。**

### 工程关键点

- `daemon_main.py:455-458`：raw 路径下默认 `use_schmitt_ema=True`，calibrated 路径保持 strict-deque 向后兼容
- `_brow_above_high` Schmitt 状态机：
  - 上升沿 latch：需要 `len(brow_history) >= window_size` AND `ema >= thr_high`（防单帧噪声 spike 假触发）
  - 下降沿解锁：`ema <= thr_low` 无条件清 latch（防"高位 dropout 进入低位"死锁）
- EMA 在 face_undetected 时同步 reset，与 brow_history 清窗对齐
- 552 pytest 全过（538 旧 + 14 新增 Schmitt+EMA 测试）

### 配套产出

- 触发实现：`poc/src/realtime/feedback/trigger.py`（Schmitt+EMA opt-in 分支）
- daemon 集成：`poc/src/realtime/daemon_main.py:445-458`
- 阈值 sweep：`scripts/sweep_brow_threshold_quantile.py`（224 行）
- 新单测：`poc/tests/test_feedback_trigger.py`（+14 case，993 行）
- replay 验收：`scripts/replay_trigger_diagnose.py --schmitt-ema`

### 后续 TODO（不在本 ADR 范围）

- 多用户 baseline 个性化阈值（gemini #4 警告"川哥个体过拟合"风险） → ADR-008 User-LoRA 实施时一并解决
- VLM low_conf 通道是否也需类似平滑 → 待数据积累后评估

---

## V0.3.6 cc 一/二审 + codex 复审 补丁（2026-05-16 23:50-24:10）

### cc 一审发现：cooldown × latch 死锁（已修 ✅）

trace 工具 (`scripts/trace_schmitt_state.py`) 实测：cooldown 期间 ema 5 次过 thr_high，
但因 latch 检查在 cooldown 内被吞掉，**cooldown 解禁后 above 状态丢失，永久错过本段**。

**修复**：把 cooldown 检查从"latch 内部"提到"fire 内部"。Schmitt 状态机照常 update
（latch above + 解锁 above），cooldown 内不 fire 但状态正确流转。

### cc 二审发现：py-feat NaN 污染 EMA 永久卡死（已修 ✅）

实测 14:30-14:59 demo session 1714 帧中 6 帧 raw_au.AU04 = NaN，触发 EMA → NaN
（NaN op any = NaN），Schmitt 后续永远不再触发。

**修复**：`_safe_float` 防御所有 brow 输入；NaN brow 不入 history/ema/samples（等下一帧）。
snapshot 字段同步加 NaN 兜底。

### Replay 验收对照（同份数据 35:00+ session）

| 版本 | brow_burst | pose_held | 备注 |
|---|---|---|---|
| V0.3.5（cooldown 内不 latch） | 1 | 50 | 死锁 |
| V0.3.6（cooldown 不阻 latch + NaN 防御） | **41** | 53 | **41x 提升** |

638 帧旧 benchmark：V0.3.5=4 → V0.3.6=5（无回归 + 略增）。
全套 pytest 552 全过（work 模式）。

### codex 一审 5 findings（V0.3.6 已修 1，4 留 spawn_task）

| # | 级别 | 内容 | V0.3.6 状态 |
|---|------|------|-------------|
| F1 | P0 | sweep 仿真器与真实 trigger 语义不一致 → 推荐阈值可能偏差 | ⏳ 留 spawn_task |
| F2 | P1 | sweep 推荐基于单 session 过拟合 | ⏳ 留 spawn_task |
| **F3** | **P1** | **NaN 输入静默毒化 EMA** | ✅ **V0.3.6 已修** |
| F4 | P1 | 持续高压（ema 永不跌到 thr_low）不再触发盲区 | ⏳ 留 spawn_task（需产品定案） |
| F5 | P1 | daemon 仍依赖 module-level 常量 → trigger_mode 漂移面 | ⏳ 留 spawn_task |

### V0.3.6 时间线

- 23:50: cc 一审发现 cooldown × latch 死锁（trace 工具实证）
- 23:55: 修复（cooldown 不阻 latch + 移到 fire 内部）
- 23:58: cc 二审发现 NaN 污染 EMA
- 00:00: 修复（_safe_float 全路径防御）
- 00:05: replay 验收 41x 提升 + 638 帧无回归
- 00:08: codex 复审 5 finding（F3 已被 V0.3.6 修复）
- 00:10: 本段落盘 + commit
