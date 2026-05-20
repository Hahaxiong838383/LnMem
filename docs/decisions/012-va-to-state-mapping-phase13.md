---
adr-id: 012
title: Phase 1.3 V-A → state 离散化 — Russell circumplex 8 扇区 + neutral disk（共 9 state），r_neutral=0.35 / theta_offset=0°
status: accepted（2026-05-20，cc 主线钦定，等 Phase 1.4 全量重测验证）
date: 2026-05-20
deciders: 川哥 + cc + codex 5.4 high (实施) + gemini 3-pro (学术评审)
supersedes: —
related: ADR-007 Phase 1 plan / ADR-011 假脸 guard
---

## 上下文

ADR-007 Phase 1.3 要把连续 V-A 离散化到 LNN 兼容 state 做 CE 分类 loss（与 Phase 1.4 train_base_v2 联合）。
cc 主线设计 Russell circumplex (1980) 二维 V-A + 8 扇区 + neutral disk 基线。
Team 模式（cc 主线 spec / codex 5.4 high 实施 / gemini 3-pro 学术评审 / cc 跑实测 + 决策）。

数据：VEATIC 12 视频 × 均匀 30 帧抽样 = 360 帧 V-A 真值（rating_averaged/*.csv）+ py-feat AU（[veatic_sweep_360frames.parquet](../../../tmp/lnmem-team-2026-05-20/veatic_sweep_360frames.parquet)）。

## 决策（cc 主线最终）

### 1. state 离散化方案

**Russell-LnMem 9 state**（cc 主线 variant）：

| state_id | 标签 | 几何区间 |
|----------|------|---------|
| 0 | neutral | `r < r_neutral` 圆盘 |
| 1-8 | 8 扇区 | `r ≥ r_neutral`，每扇区 45°（标准 Russell）|

**总 9 标签**，与产品定义 §10.1「8-state」语义对齐方式：在 Phase 1.4 训练 CE loss 时把 neutral state 作为「第 9 类」（不与 base v1 8 emotion class 强行对齐 — base v1 是 emotion class，base v2 是 V-A state，本就是不同 schema）。

**注**：cc 主线 spec 第 79 行 "state_id 0..7 共 8 state" 是笔误（漏 state 8），更正为 0..8 共 9 state。

### 2. 阈值参数（基于 sweep + gemini 评审）

| 参数 | 选定值 | 来源 |
|------|--------|------|
| `r_neutral` | **0.35** | sweep boundary_stability 最低（0.1028） + gemini 文献基线 ≥ 0.3 |
| `theta_offset_deg` | **0°** | sweep 同 r_neutral 下 0° 优于 22.5°（boundary 0.1028 vs 0.1194） |
| `n_states` (扇区数) | 8 | Russell 标准 45° 扇区 |

### 3. Sweep 实测数据（10 组候选）

| scheme | coverage | boundary | va_mse | entropy | 推荐 |
|--------|----------|----------|--------|---------|------|
| rn0.15_θ0 | 61.5 | 0.1944 | 0.0690 | — | N（coverage 失） |
| rn0.15_θ22.5 | 22.75 | 0.1972 | 0.0689 | — | N |
| rn0.2_θ0 | 60.0 | 0.1944 | 0.0676 | — | N |
| rn0.2_θ22.5 | 29.0 | 0.2056 | 0.0675 | — | N |
| rn0.25_θ0 | 51.5 | 0.1889 | 0.0626 | — | N |
| rn0.25_θ22.5 | inf | 0.2000 | 0.0621 | — | N |
| rn0.3_θ0 | inf | 0.1528 | 0.0562 | — | N |
| rn0.3_θ22.5 | inf | 0.1750 | 0.0559 | — | N |
| **rn0.35_θ0** | inf | **0.1028** | 0.0570 | 1.769 | **✅ 选** |
| rn0.35_θ22.5 | inf | 0.1194 | 0.0565 | — | N |

**实测 recommendation 表**：[sweep_report_v2.json](../../../tmp/lnmem-team-2026-05-20/sweep_report_v2.json)

### 4. 推荐阈值实测分布（360 帧）

```
state_id: [0,    1,   2,  3,   4,   5,   6,   7,   8 ]
counts:   [212,  34,  0,  11,  77,  16,  7,   0,   3 ]
比例:     [59%, 9.4%, 0%, 3%, 21%, 4%, 2%,   0%, <1%]
state_entropy = 1.769 bits（满分 log2(9)=3.17）
```

⚠️ **state 2 / state 7 完全空**：抽样 360 帧太小 + VEATIC 跨文化偏差导致某些扇区饥荒。

## Gemini 学术评审吸收

✅ **吸收**：
1. `r_neutral` 候选去 0.1 加 0.35（gemini: IEMOCAP/SEMAINE 文献基线 r=0.3，0.1 仅 <1% 面积过小）
2. 加 `state_entropy` 第 4 指标（gemini: 信息熵补全 coverage_balance 局限）
3. 决策树增加「entropy 偏好」tie-break（差 ≥0.3 时偏 entropy 高的）

⚠️ **记录为 Phase 1.4 风险（不在 Phase 1.3 实施）**：
1. **Soft Assignment / Label Smoothing**：CE 硬切 + CCC 回归联合训练在扇区边界有梯度冲突 — Phase 1.4 train_base_v2 必须实施（角距离 Gaussian 软标签 或 focal loss + margin）
2. **跨文化偏差**：中国用户情绪表达克制，高唤醒扇区有「标签饥荒」风险 — Phase 1.4 后用川哥真信号微调时需注意
3. **Quantile Calibration 自适应阈值**：固定 r_neutral 在不同用户群可能失效 — Phase 2（多用户 / ADR-008）才考虑

## 风险与缓解

| # | 风险 | 缓解 |
|---|------|------|
| 1 | **360 帧抽样不通过 coverage_balance ≤5** | Phase 1.4 全量提取 124 视频后必须重跑 sweep 验证 r_neutral=0.35 仍最优 |
| 2 | **state 2/7 完全空** | Phase 1.4 训练用 class-balanced sampling 或忽略空 state（动态调整 n_states）|
| 3 | **CE 硬切边界梯度冲突**（gemini） | Phase 1.4 train_base_v2 强制 soft assignment（角距离 Gaussian σ=0.2 候选）|
| 4 | **跨文化偏差** | Phase 1.4 用川哥真实信号微调 tau 后由 LTC 动力学自适应 |

## 验收门

**Verify**（本 ADR）：
- ✅ sweep 4 测试全过（含 `state_entropy_and_decision_priority`）
- ✅ 360 帧真实 sweep 推荐 `russell8_rn0.35_theta0`（boundary 最低 0.1028）
- ✅ recommendation reason 含「fallback prefers boundary+entropy」

**Guard**：
- ✅ poc/tests 573 + 2 skipped 不破

**Phase 1.4 trigger**：
- 全量 VEATIC（~250K 帧）重跑 sweep 必须确认推荐仍为 r_neutral∈[0.3, 0.4]，否则需重新 sweep 并升 ADR

## 配套产出

- 脚本：[scripts/sweep_va_to_state.py](../../scripts/sweep_va_to_state.py)（225 行，cc 主线 P3 技术债：拆 helper ≤200 行）
- 测试：[tests/test_sweep_va_to_state.py](../../tests/test_sweep_va_to_state.py)（4 测试）
- cc spec：`/tmp/lnmem-team-2026-05-20/cc_b_spec_va_to_state.md`
- codex 一轮 findings：`/tmp/lnmem-team-2026-05-20/codex_b_sweep_findings.md`
- codex 二轮 findings：`/tmp/lnmem-team-2026-05-20/codex_b_sweep_v2_findings.md`
- gemini 评审：`/tmp/lnmem-team-2026-05-20/gemini_b_review_raw.md`
- sweep report v2：`/tmp/lnmem-team-2026-05-20/sweep_report_v2.json`

## 时间线

- 2026-05-20 10:30: cc 主线写 spec
- 10:50: codex 5.4 high 一轮实施（225 行脚本 + 3 测试）
- 11:00: gemini 3-pro 学术评审完成（5 项建议）
- 11:05: cc 跑真实 360 帧 sweep — 10 组全不过 coverage，推荐 fallback rn0.3
- 11:08: cc 派 codex 5.3 medium 二轮吸收 gemini 0.35 + entropy
- 11:15: codex 二轮完成 4 测试 + cc fix entropy bug（counts[:8] → counts[:n_states+1]）
- 11:18: cc 跑 v2 实测 — 推荐 **`russell8_rn0.35_theta0`**（boundary 0.1028 改善 33%）
- 11:20: 本 ADR 落盘

## 反对意见 / 不同看法

- **cc 自审**：360 帧抽样不通过 coverage_balance 是硬伤。理论上应等 Phase 1.4 数据后才能定阈值。
  采纳现决策的理由：sweep 相对排序（不同 r_neutral 的 boundary 对比）在小数据上仍有意义，且 0.35 与 gemini 文献基线一致。
  Phase 1.4 重跑作为 trigger 写入风险清单。
- **gemini 强建议**：CE 硬切 + CCC 联合训练 high risk — 本 ADR 不展开（Phase 1.4 范围）但 prominent 记录
- **codex 一轮 finding**：state 编号 0..7 vs 0..8 spec 矛盾 — cc 主线承认 spec 笔误并在本 ADR 修正

---

**模板版本：v0.1 · 2026-05-03**
