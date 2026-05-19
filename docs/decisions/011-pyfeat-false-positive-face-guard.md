---
adr-id: 011
title: py-feat 假阳性脸 guard — raw_au 全 NaN + pose 冻结 → 视为 face_undetected
status: implemented（V0.3.8，2026-05-17 trigger.py + signal_pipeline.py 双层落地）
date: 2026-05-17
deciders: 川哥 + cc
supersedes: —
superseded-by: —
related: ADR-010（V0.3.6 Schmitt+EMA NaN 防御为本 ADR 揭示 pose_held 残留噪声）/ ADR-005（dual signal_path）
---

## 上下文

ADR-010 V0.3.6 上线后（Schmitt+EMA + NaN 污染 EMA 防御），川哥 2026-05-16 下午
离开电脑 1.5h 期间 daemon 仍触发 186 次（queried 75 + rate_blocked 111），
**全部 reason=pose_held，0 brow_burst**。

### 实测数据揭穿（`~/.lnmem/data/signals_2026-05-16.jsonl` 16:00-16:30 1800 帧）

| 维度 | 数值 |
|---|---|
| `face_detected=True` 占比 | **100%**（1800/1800） |
| `raw_au.AU04` finite 值占比 | **0%**（全 NaN） |
| `head_pose.pitch_deg` | 恒定 3.075°（std=0°） |
| `head_pose.yaw_deg` | 恒定 0° |
| pose_held 触发次数 | **75 次**（每 cooldown=30s 触发一次） |
| 用户视角 | 75 条 macOS 弹窗 timeout |

### 根因

`poc/src/vision/au_extractor.py:114` 把 py-feat 输出无脑套 `float()`，未跳过 NaN：

```python
au_dict = {au: float(row[au]) for au in AU_NAMES if au in fex.columns}
```

py-feat `img2pose` 在没真人脸时会返"face-like rows"但 AU/pose 全 NaN，又因
extractor 缓存 `self._last_pose`（line 109/133），返回的 pose 是上一次有真脸
时的快照。下游：

1. `signal_pipeline.py:78` `if raw_au:` —— `{'AU01': nan, ...}` 是 truthy →
   `face_detected=True` 被错误置位；
2. `trigger.py` V0.3.6 NaN 防御已让 brow 通道不入 EMA → brow_burst 不再误触；
3. **但 pose_held 通道**：`_update_pose_window` 用 `_safe_float` 过 NaN 没问题，
   但 head_pose 是冻结的有限值（3.075, 0, 0），完全合法 → deque 填满 →
   `std=0 < thr=2.5°` → 每 cooldown 后立刻触发。

### 后果

- queries.jsonl 落 75 条 `pose_held=skip` 噪声 → **污染 LNN 训练数据**；
- LNN inferencer 拿假阳性帧做 forward → margin/entropy 落盘失真；
- 川哥回来 75 条弹窗 timeout 堆积 → 体验灾难。

## 决策

**双层 NaN-AU guard：上游净化 + 下游兜底**。

1. **`signal_pipeline.py` 治本**：`if raw_au and _has_finite_value(raw_au):`
   —— py-feat 全 NaN 输出直接被丢，`face_detected` 保持 False，
   下游 LNN + queries.jsonl + trigger 全部受益（不仅 trigger）。

2. **`trigger.py` 兜底**：`observe()` 入口 `_is_false_positive_face(item)`
   与 `not face_detected` 走同一清窗 + return None 分支。即使有别的代码路径
   绕过 signal_pipeline（如 replay 脚本直读旧 jsonl），trigger 仍不会被骗。

3. **不动 `au_extractor.py`**：au_extractor 缓存 `self._last_pose` 的行为
   有其他正常路径依赖（短暂遮挡时的姿态平滑），独立 ADR 后续处理。

### 判定规则（保守 — 宁可漏报不可误杀真脸）

| `raw_au` 状态 | 判定 |
|---|---|
| 字段缺失 / 不是 dict | 不视为假阳性（schema 兼容） |
| 空 dict `{}` | 不视为假阳性（上游"无 AU 数据"语义） |
| 至少 1 个值是有限数 | **真脸**（容忍 py-feat 偶发 1-2 个 AU NaN） |
| 全部值非有限（NaN/inf/非数） | **假阳性 → 视为 face_undetected** |

## 选项与对比

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| A · trigger.py 单点 | 改动最小，blast radius 仅 trigger | 仍污染 queries.jsonl 和 LNN 训练数据 | ✗ |
| B · signal_pipeline.py 单点 | 治本，所有下游受益 | replay 脚本读旧 jsonl 时不再有 guard | ✗ |
| **A+B 双层（已选）** | 治本+兜底，覆盖 daemon + replay 双路径 | 两处都要测 | ✅ |
| D · au_extractor.py 改 | 最深治本 | 改动 py-feat 包装层，blast radius 大；缓存 pose 有其他用途 | 留给后续 ADR |
| E · 连续 N 帧 NaN 才算 | 抗短暂抖动 | 实测 100% NaN，N=1 已足够；引入 N 帧 buffer 复杂度↑ | ✗ |

## 为什么选 A+B 双层

1. **覆盖盲点**：B 治本但 trigger 兜底 A 能挡 replay 脚本 / 旧 jsonl 重放路径；
2. **判定逻辑可复用**：`_has_finite_value` (signal_pipeline) 和 `_is_false_positive_face`
   (trigger) 共享 "至少 1 个有限值就放行" 规则，未来挪到共用 utils 容易；
3. **单帧判定够用**：实测 1800/1800 全 NaN，N-帧 buffer 是工程过度；
4. **不破老语义**：缺失/空 `raw_au` 不触发 guard → 老 schema、calibrated-only 路径全兼容；
5. **测试可证伪**：7 个新单元测试覆盖 single/window/clears-evidence/partial-NaN/missing-field/empty-dict/brow-evidence。

## 后果（影响 / 副作用）

**正面**：
- 离屏期间 0 pose_held 噪声触发（vs V0.3.6 1.5h → 75 次）
- queries.jsonl / LNN 训练数据不再被假阳性帧污染
- 用户回到电脑前不再有 macOS 弹窗 timeout 堆积

**负面 / 风险**：
- 如果 py-feat 真有 N 个 AU 全部偶发 NaN 但脸其实是真的 → 该帧被丢一次（下一帧通常恢复，window 重新累积；只是延迟触发 N 帧，可接受）
- `_is_false_positive_face` 每帧调用 = N_AU 次 `isinstance + isfinite`，30Hz 下成本可忽略

**触发的下游决策**：
- au_extractor.py 的 `self._last_au/_last_pose` 缓存策略待独立 ADR 评估（是否在缓存上也加 NaN 净化）
- ADR-006 threshold-auto-recalibration 设计时需排除假阳性帧（已被本 ADR guard 在源头滤掉）

## 反对意见 / 不同看法

- **cc 自审**：曾考虑只在 trigger 层修（方案 A），代价是 queries.jsonl 仍污染。
  最终选 A+B：B 是治本，A 是 daemon 外路径（replay/旧文件）的兜底。

- **codex 一审（2026-05-17）** 找出 2 P0 + 2 P1 + 1 P2：
  - **P0-1（已采纳）**：`isinstance(v, (int, float))` 对 numpy.float32 返 False
    → 真脸 raw_au 用 numpy 类型会被误杀为假阳性。codex 实测脚本 5 秒内重现。
    修：抽 `_is_finite_number()` helper 用 `numbers.Real` + `float()` 转换。
  - **P0-2（已采纳）**：`raw_au` 是 list 时 guard 返 False，下游 `raw_au.get(...)`
    崩 `AttributeError`。codex 实测 5 秒重现。修：guard 把 "非 dict 且非 None"
    视为假阳性提前 return；observe() 入口加 `isinstance(... dict) else {}` 兜底。
  - **P1-1（已采纳）**：`test_nan_au_clears_brow_evidence` 没开 `use_schmitt_ema=True`，
    没真正覆盖 EMA 分支。修：拆为 `_legacy` + 新 `_schmitt_ema_state` 两个测试，
    后者显式断言 `_brow_ema/_brow_above_high/_brow_samples_since_reset` 三态重置。
  - **P1-2（部分接受）**：signal_pipeline 缺单测。本 ADR 未补（pipeline 测试需
    mock extractor/caliper/vlm/lnn 四层依赖，工作量超出本 fix 范围）。已通过
    trigger 层 10 测试 + 共享 numpy 子类断言间接验证 `_has_finite_value` 语义。
  - **P2（已记入"已知技术债"）**：两层 guard 重复实现，留待后续抽 common util。

## 验证 / 复盘

**Verify（目标验证）**：
- 10 个新单元测试 `TestFalsePositiveFaceGuard` 全过（基线 7 + codex 一审补 3：numpy 子类 / list 脏数据 / Schmitt EMA 状态重置）；
- 用 `signals_2026-05-16.jsonl` 16:00-16:30 1800 帧做离线 replay 待跑（建议命令 `python scripts/replay_trigger_diagnose.py --signals signals_2026-05-16.jsonl --window 16:00-16:30`），期望从修复前 75 次 pose_held 降到 0。**实测证据待 V0.3.8 实际部署后补到本 ADR 后续修订**。

**Guard（回归防护）**：
- pytest 全套 562 → 577 pass（旧 562 + 新 10 + 并行 V0.3.7 commit 增加的 5 个）；
- 0 回归。

**回退路径**（V0.3.8 双层 guard，回退必须两层都拆）：
- 治本层 `signal_pipeline.py:_process` —— 把 `if raw_au and _has_finite_value(raw_au):` 改回 `if raw_au:`；
- 兜底层 `trigger.py:observe` —— 把 `or self._is_false_positive_face(item)` 删掉；
- 测试层 —— 删 `TestFalsePositiveFaceGuard` 类；
- 单删 trigger 层不能恢复旧行为（pipeline 层仍会过滤 NaN）。

**已知技术债（codex 一审 P2，留待后续 ADR）**：
- `_has_finite_value` (signal_pipeline) 与 `_is_finite_number` (trigger) 是孪生函数，应抽到 `poc/src/common/numeric_utils.py` 共享 + 参数化一致性单测。本 ADR 范围内未做以减小 blast radius。

**关联实测**：等待川哥下一次"离开电脑 1h+ 回来" 端到端验证 daemon 弹窗数量。

---

**模板版本：v0.1 · 2026-05-03**
