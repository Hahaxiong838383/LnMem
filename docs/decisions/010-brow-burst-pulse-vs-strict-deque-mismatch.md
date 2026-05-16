---
adr-id: 010
title: brow_burst 触发条件与 raw_AU04 脉冲信号不匹配（TODO-1 Schmitt+EMA 强证据）
status: identified（待 ADR-006/TODO-1 实施修复）
date: 2026-05-16
deciders: 川哥 + cc + codex（V0.3.4 加固后实测发现）
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
- 22:55: 本 ADR 落盘
