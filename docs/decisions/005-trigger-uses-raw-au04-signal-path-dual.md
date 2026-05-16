# ADR-005: trigger 应用层用 raw_au.AU04 + dual signal_path（最小改动方案）

**状态**：accepted
**日期**：2026-05-16
**决策者**：川哥（最终） + cc（设计 v0.3） + codex（v0.1/v0.2 对抗审查纠偏） + gemini（threshold 调研）
**触发**：codex 对抗审查 7 finding 推翻 cc 初版 v0.1（CRITICAL bug：维度错算 + 监督标签缺失 + 切换变量虚构）+ lnn-first.md §4.1 明确视觉层不受 P-1 约束，逼 cc 设计 v0.3 最小改动方案

**Supersedes**: ADR-003

---

## 背景

5/16 D-1 数据分析发现：`expression.brow_furrow`（calibrated）在川哥脸上 polarized 双峰（51% 在 [0, 0.1] + 30% 在 [0.9, 1.0]，熵 61%）。Calibration 函数 gamma 拉伸把 raw > 0.19 全 clamp 到 1.0。川哥 raw_au.AU04 baseline mean=0.286，永远被错算成"满分皱眉"，daemon 一直追假信号。

cc 初版 v0.1 提议"完全绕过 calibration + LNN input 10→23 + 重训 base ckpt"，被 codex 对抗审查揪出 3 个 CRITICAL bug：
- F-01: input 是 18 维不是 10 维，全链路 contract 会断裂
- F-02: 历史 235K signals 没 ground_truth_state，监督训练无法闭环
- F-06: LNMEM_SIGNAL_PATH 切换变量代码里不存在，cc 凭空编造

并提示 F-03：lnn-first.md §4.1 明确视觉感知层**不受 P-1 约束**，"违反宪法"定性错误。

gemini 调研第 4 点关键启发：**"LNN-FIRST 准则下，漂移应在微分方程的集成项（τ 驱动）中自然吸收，在应用层维持固定阈值更有利于系统行为的确定性调试"**。

这正好回答：**应用层（trigger）用固定数据驱动阈值就好，演化层（LNN）让 tau_drift 自然学**。

## 决策

**采纳 v0.3 最小改动方案**：

1. **trigger 应用层**改用 `raw_au.AU04`（py-feat 原始皱眉强度），数据驱动阈值 **0.55 (P91)**（4 天 235K 帧 sweep 模拟每天 ~5-6 次触发，与 work 节奏一致）
2. **新增 `LNMEM_SIGNAL_PATH = raw / calibrated` 双源切换**（env 变量 > `~/.lnmem/signal_path` 文件 > "calibrated" 默认），与 LNMEM_TRIGGER_MODE 同范式
3. **calibrated 模式作为向后兼容默认**，零破坏
4. **不动 LNN 18 维 fusion contract**（保留 eye_closure / fatigue / attention / pose_steadiness / blink_burst 派生语义）
5. **不重训 base ckpt**（fit_orchestrator 继续走 OnlineLoRAAdapter 训 tau_drift）
6. **signal_snapshot 同时记 raw + calibrated 双值**（便于 A/B 离线对照）

## 决策依据（codex F-01/F-02/F-03/F-04/F-05 + gemini 第 4 点合并）

| 维度 | v0.1 cc 提议 | codex finding | v0.3 决策 |
|---|---|---|---|
| 改动半径 | 全链路重写 | F-01 维度断裂 | **仅 trigger.py 1 处** |
| 训练 | 重训 base ckpt | F-02 无监督标签 | **不重训，沿用现有 base** |
| 哲学定性 | "违反 P-1" | F-03 视觉层不受 P-1 约束 | **改口径："AU04 calibration 信息退化"** |
| 输入维度 | 23 纯 raw | F-04 丢产品 KPI 派生特征 | **18 维不变** |
| 阈值选取 | 0.95 沿用 | F-05 不能盲沿用 | **数据驱动 P91=0.55** |
| 切换机制 | LNMEM_SIGNAL_PATH（虚构） | F-06 代码不存在 | **真做这个变量** |
| A/B 对比 | 路径对比 cold start | F-07 confounding | **离线 replay + 长期在线** |

## 阈值数据驱动（4 天 235K 帧 sweep 结果）

```
川哥 raw_au.AU04 分布：mean=0.286, std=0.173
P50=0.250  P75=0.395  P90=0.539  P95=0.624  P99=0.757

阈值 sweep（连续 15 帧 + 5min cooldown）：
thr=0.50  → 14 次/4天 (~9/天)   P87
thr=0.55  → 9 次/4天 (~5.7/天)  P91  ✅ 选这个
thr=0.60  → 5 次/4天 (~3.2/天)  P94
thr=0.70  → 1 次/4天             P98
```

work 节奏目标：每小时 ≤ 3 询问 + 每天 8-10 小时使用 = 24-30 询问/天上限。0.55 阈值给约 5-6 brow + 其他通道 = 合理。

## 选项与对比

| 选项 | 改动半径 | 工时 | 风险 | 哲学一致性 | 选 |
|---|---|---|---|---|---|
| v0.1 全链路替换 | 极大（4 层）| 5-7h | 3 CRITICAL | 表面纯粹有 bug | ❌ codex 否决 |
| v0.2 raw 旁路 LoRA | 大（双输入头）| 8h | 中 | 旁路绕 calibration | ⚠️ 不必要 |
| **v0.3 trigger 用 raw** | **小（1 处）** | **2h** | **低** | **gemini 建议+lnn-first §4.1 允许** | ✅ |
| v0.4 修 calibration 算法 | 小 | 1h | 低但违反川哥哲学 | 仍是规则系统 | ❌ |

## 实施改动清单（v0.3 最终）

```
poc/src/realtime/feedback/trigger.py
  - _detect_setting() 通用 env/file 双源读取 helper（与 trigger_mode 同范式）
  - _SIGNAL_PATH 模块级变量（raw / calibrated）
  - _BROW_BURST_THRESHOLD_CAL = 0.95  /  _BROW_BURST_THRESHOLD_RAW = 0.55 双阈值
  - observe() 按 _SIGNAL_PATH 选 raw_au.AU04 或 expression.brow_furrow
  - signal_snapshot 同时记 raw + calibrated 双值（A/B 离线对照）

scripts/signal-path.sh
  - helper：./scripts/signal-path.sh raw / calibrated / status
  - 写 ~/.lnmem/signal_path 文件 + RESTART_NEEDED 让 supervisor 重启 daemon

测试：
poc/tests/test_feedback_trigger.py
  - TestSignalPathSwitching：3 case 验证 raw / calibrated / fallback
```

## 后果

### 正面
- 川哥体感问题立刻解决（不再追假皱眉信号）
- 哲学一致（gemini 第 4 点 + lnn-first §4.1 + LNN-FIRST 不动核心）
- 2h 工程完成（vs v0.2 的 8h）
- A/B 对比仍可做（signal_snapshot 记双值）
- 5 个产品 KPI 派生特征保留

### 负面 / 风险
- raw_au.AU04 仍是 py-feat 输出，如果未来 py-feat 升级导致分布漂移，0.55 阈值也会失效（同样问题任何固定阈值都有）
- signal_path 切换需 daemon 重启（与 trigger_mode 同设计），不能运行时切

### 触发的下游决策
- ADR-006（未来）：threshold 是否需要 long-term auto-recalibration（gemini 第 4 点建议让 LNN tau_drift 吸收 baseline drift，应用层固定）
- M0 W3 探索：是否进一步把 trigger 决策权交给 LNN（"LNN 自学'何时该询问主人'"，不再用规则阈值）

## A/B 实验设计（响应川哥要求 "证明 LNN 思想正面效果"）

| 维度 | A 路径（calibrated） | B 路径（raw） |
|---|---|---|
| 历史 | 5 天 138 events | 切换后新数据 |
| brow_burst trigger | expression.brow_furrow > 0.95 | raw_au.AU04 > 0.55 |
| match 率历史 | 70% (但川哥说是虚假"懂"，因模型一直追 polarized 假信号) | 待观察 |
| 监控期 | 历史已记录 | 7-14 天 |

**判断指标**：
1. **B 路径 brow_burst match 率（绝对值）**：> 70% → 强化"懂你"信号
2. **B 路径 mismatch 时 true_state 分布**：distracted 比例若下降 → LNN 学到了真信号下的正确语义
3. **触发频率 / 节奏感**：5-6 次/天 vs A 的 mismatch 集中爆发 → 用户体验

## 反对意见 / 不同看法

- **cc v0.1 自信"5-7h 重训"** → 被 codex F-02 击碎（无监督标签），cc 撤回
- **cc 把 calibration 定性为"违反 P-1"** → 被 codex F-03 + lnn-first.md §4.1 击碎，cc 修 ADR-003 口径
- **gemini 推荐 P99=0.78 阈值**（更严格） → cc 选 P91=0.55（work 节奏每天 5-6 次，更符合 PoC 实际使用频率）

## 标签：A/B 对照锚点

```yaml
# v0.3.2 启用后 fit_step_history 新 metadata
signal_path: raw  # 或 calibrated
brow_burst_threshold: 0.55  # 或 0.95
brow_burst_source_field: raw_au.AU04  # 或 expression.brow_furrow
adr_label: post-ADR-005 (v0.3 dual signal_path)
```

---

**末尾自检**：本 ADR 严格符合 [CLAUDE.md §3.1 cc/codex 意见冲突诊断流程]：
- cc v0.1 错（论证不准 + bug 多），codex 全部驳回 → cc 默认采纳 + 修 ADR-003 + 提 v0.3 → 川哥批 v0.3
- 决策依据：哲学（lnn-first §4.1 不约束视觉层）+ 数据（quantile sweep）+ 工程（最小改动半径）+ team 协同（gemini 4 点 + codex 7 finding）
- 不同意见保留：cc v0.1/v0.2 错误路径完整记录在 ADR-003 / 本 ADR，回看可总结"cc 容易过度工程化"
