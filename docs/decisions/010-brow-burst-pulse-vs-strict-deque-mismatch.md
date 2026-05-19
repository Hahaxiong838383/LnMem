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

---

## V0.3.7 川哥端到端实测 + codex F5 修复（2026-05-17 00:30 完成）

### 川哥实测黄金证据

V0.3.6 daemon 清洁重启后川哥真坐摄像头前皱眉，5min 内：

| 时间 | 类型 | raw_AU04 | ema | above_high | 川哥响应 |
|---|---|---|---|---|---|
| 11:16 | pose_held | 0.459 | 0.456 | True | match ✅ |
| 12:39 | **brow_burst** | 0.655 | 0.437 | True | match ✅ |
| 13:56 | **brow_burst** | 0.547 | 0.431 | True | mismatch |
| 15:10 | **brow_burst** | 0.877 | 0.560 | True | mismatch |
| 18:00 | **brow_burst** | 0.496 | 0.431 | True | mismatch |

**brow_burst 5min 内 4 次反复触发** ✅ V0.3.6 cooldown × latch 死锁修复 + NaN 防御彻底见效。

### 川哥 3 mismatch 的双重价值

1. **e2e 链路黄金证据**：trigger → 弹窗 → 用户响应 → 落盘 queries.jsonl 全链路通
2. **ADR-007 必要性活样本**：raw_AU04 0.50-0.87 触发但川哥说"不是我"，
   = 合成数据训出的 base ckpt 在川哥脸上 V-A 映射不准 → ADR-007 AffectNet 真人化训练的强需求证据
3. **这 3 个 mismatch 是 LNN 训练黄金数据**：直接喂入 fit_orchestrator 优化 OnlineLoRAAdapter
   tau_drift，让模型学到"raw_AU04 高但情绪不负面"的川哥个体 baseline

### codex F5 修复（同夜并行处理）

trigger.py 加 `_MODE_PROFILES` 字典 + `resolve_runtime_config()` 函数：daemon 每次构建
trigger 都重读 env / ~/.lnmem/{trigger_mode,signal_path}，从根上消除 codex F5 警告的
"daemon module-level 配置漂移面"。Backward-compat module-level 常量保留（旧测试 0 影响）。

### codex 5 findings 闭环最新进展

| # | 级别 | V0.3.5 状态 | V0.3.6 状态 | V0.3.7 状态 |
|---|------|------------|-------------|-------------|
| F1 | P0 | ⏳ | ⏳ | ⏳ spawn_task |
| F2 | P1 | ⏳ | ⏳ | ⏳ spawn_task |
| F3 | P1 | ⏳ | ✅ 已修 | ✅ |
| F4 | P1 | ⏳ | ⏳ | ⏳ spawn_task（需产品定案）|
| **F5** | **P1** | ⏳ | ⏳ | ✅ **已修** |

剩 F1/F2/F4 进 spawn_task 明日处理。

### V0.3.7 pytest

557 全过（V0.3.6 552 + F5 新增 5 个 resolve_runtime_config 测试），0 回归。

### 配套发现 — 待治：py-feat 假阳性脸（spawn_task 已挂）

实测 16:00-16:30 共 1800 帧川哥离开电脑期间：face_detected=True 100% + raw_AU04
全 NaN + head_pose pitch=3.075° std=0°（py-feat 缓存/冻结值）→ pose_held 每 30s
误触发 75 次纯噪声。spawn_task 已挂队列。

### V0.3.7 时间线

- 00:15: 川哥回电脑前实测，4 brow_burst + 1 pose_held + 5min 内验证 e2e 链路
- 00:20: 发现 py-feat 假阳性脸 1.5h 75 次 pose_held 噪声 → spawn_task 挂队列
- 00:25: codex F5 实施（_MODE_PROFILES + resolve_runtime_config）
- 00:32: pytest 557 全过验证 F5 修复无回归
- 00:35: 本段落盘 + commit + push

---

## V0.3.8 codex F1/F2/F4 闭环 + 二审 G1-G4 修复（2026-05-17 凌晨）

### 本轮工作目标

V0.3.7 后剩 F1（P0）+ F2（P1）+ F4（P1）三个 finding 未闭环（F3 V0.3.6 修，F5 V0.3.7 修）。
cc 本轮闭环 + codex 二审找到 4 个新风险 G1-G4 一并修。

### 一审 F1/F2/F4 修复

| Finding | 修复路径 | 测试 |
|---------|---------|------|
| F1 (P0) sweep 仿真不等价 | 删 `simulate_schmitt`，加 `count_brow_burst_fires` 调真 `ProactiveQueryTrigger.observe()` | `TestSweepConsistency` ×5 |
| F2 (P1) sweep 过拟合 | 加 `per_session_fires_stats` + `leave_one_session_out_quantiles` + 新推荐准则 mean/cv/touched | `TestSweepCrossSessionStats` ×5 |
| F4 (P1) 持续高压盲区 | 产品定案 = **方案 A**（严格 Schmitt，必须低位回落才再触发；持续高压另由 pose_held 通道覆盖）。**未改实现**，加 lock-in 测试 | `TestSustainedHighOneShotSemantics` ×2 |
| F5 漏网点 | `fit_orchestrator.py:309` ckpt 元数据 signal_path 改用 `resolve_runtime_config()` | （V0.3.7 测试已覆盖 resolve 函数）|

V0.3.8 一审后 pytest = 574 全过。

### codex 二审 4 P0/P1 G-finding + 1 P2

| # | 级别 | 证据 | 修复 |
|---|------|------|------|
| **G1** | **P0** | sweep `count_brow_burst_fires(values)` 把 no-face/NaN 帧过滤后再喂 trigger → 抹掉了生产链路的 `_brow_ema/above/samples` 状态重置 & frame_id 真实跨度。codex 最小复现：`truth=0 sweep=1` | 加 `count_brow_burst_fires_from_items` + `materialize_items_for_sweep`，main 改喂完整 items |
| **G2** | **P1** | LOSO `train_vals.extend(...)` + 一次 `ema_series()` 无 session 边界 reset；`names=list(session_values.keys())` 顺序敏感 | LOSO 内部改 **per-session ema_series** + `names=sorted(...)` |
| **G3** | **P1** | `touched_ratio` 分母只用 `valid_files`（vals 非空），静默剔除空 session 抬高覆盖率 | 报告 **valid/total 双口径**，推荐准则用 `touched_ratio_total` |
| **G4** | **P1** | F4 lock-in 测试只覆盖纯净序列，未覆盖 high → NaN 假阳性 → high 交互（ADR-011 清状态后是否 re-fire？） | 产品定案 = **NaN 假阳性 = session 重启 = 允许 re-fire**，加 `test_high_then_nan_false_positive_then_high_refires` lock-in |
| G5 | P2 | `scripts/replay_trigger_diagnose.py` 仍 import module-level 常量 | 跳过（codex 同意非运行时漂移面），留作下轮一致性收口 |

### 测试覆盖

| 测试类 | # | 覆盖 |
|--------|---|------|
| `TestSweepConsistency` | 5 | F1：sweep 与 trigger.observe 1:1 同构 |
| `TestSweepCrossSessionStats` | 5 | F2：per_session_fires_stats / LOSO 函数行为 |
| `TestRuntimeConfigResolve` | 3 | F5：env 反映 / fallback / dict 独立 |
| `TestSustainedHighOneShotSemantics` | 3 | F4+G4：one-shot / 回落 re-fire / NaN 切断 re-fire |

V0.3.8 二审后 pytest = **578 全过**，0 回归。

### codex 5 findings + 二审 G-findings 闭环最新进展

| # | 级别 | V0.3.6 | V0.3.7 | V0.3.8 | V0.3.9 |
|---|------|--------|--------|--------|--------|
| F1 | P0 | ⏳ | ⏳ | ✅ + G1 (P0) 二审修 | ✅ |
| F2 | P1 | ⏳ | ⏳ | ✅ + G2/G3 (P1) 二审修 | ✅ |
| F3 | P1 | ✅ | ✅ | ✅ | ✅ |
| F4 | P1 | ⏳ | ⏳ | ✅ + G4 (P1) 二审修 | ✅ |
| F5 | P1 | ⏳ | ✅ | ✅ + fit_orch 漏网点 | ✅ |
| G5 | P2 | — | — | ⏳ 下轮 | ✅ V0.3.9 修 |

### 产品语义定案（含 F4 + G4）

- **F4 A 语义** lock-in：持续高压 ema 不跌至 thr_low 时只 fire 1 次（严格 Schmitt 教科书；
  持续高压另走 pose_held 通道而非 brow_burst 重复触发）。
- **G4 NaN 假阳性 = session 重启** lock-in：ADR-011 清状态后允许 re-fire（防错过真实皱眉）。
  这是 F4 A 语义在"假阳性切断"场景下的明确例外，不矛盾。

---

## V0.3.9 G5 replay 脚本一致性收口（2026-05-17 凌晨）

### 修复

`scripts/replay_trigger_diagnose.py` 改用 `resolve_runtime_config()` + `_MODE_PROFILES`，
移除对 module-level 常量 `_TRIGGER_MODE / _SIGNAL_PATH / _BROW_BURST_THRESHOLD_RAW /
_BROW_BURST_WINDOW / _POSE_HELD_WINDOW / _POSE_HELD_STDDEV_THRESHOLD_DEG /
_COOLDOWN_FRAMES_BROW / _COOLDOWN_FRAMES_POSE_HELD / _COOLDOWN_FRAMES_LOW_VLM_CONF`
的直接 import。

### 影响

- 排障场景下"脚本看到的 mode/signal_path"与"daemon 运行时的 mode/signal_path"必然一致
  （之前脚本读 import-time 快照，可能落后于 daemon 实际运行的配置）。
- `--demo` 强制路径改用 `_MODE_PROFILES["demo"]` 字典直读（而非硬编码 0.40/5/20/3.5/30），
  单一事实来源 = 与 trigger.py mode 切换逻辑同源。
- `--ema-alpha / --thr-high / --thr-low` 缺省值改成 None → 自动取当前 profile 对应值
  （之前硬编码 0.40 / 0.42-0.55 / 0.22-0.30）。

### 端到端验证

```
PYTHONPATH=. ./.venv-poc-fresh/bin/python scripts/replay_trigger_diagnose.py \
  --signals ~/.lnmem/data/signals_2026-05-16.jsonl \
  --from-ts 2026-05-16T14:14:46 --demo --schmitt-ema
# → fire 序列正常输出：brow_burst 84 + pose_held 85（与 V0.3.7 川哥实测同量级）
```

pytest 仍 **578 全过**（resolve_runtime_config 测试已锁住 G5 修复的"运行时一致性"）。

### V0.3.9 时间线

- 02:00: 川哥转 G5 任务给 cc 直接闭环（spawn_task 升级为本轮）
- 02:05: replay 脚本 refactor，imports 改用 resolve_runtime_config + _MODE_PROFILES
- 02:08: 手动验证 replay 跑通（demo + schmitt + 实际 jsonl）
- 02:10: pytest 578 全过验证无回归
- 02:15: codex 对抗审查（gpt-5.3-codex + medium）
- 02:20: 本段落盘 + commit
