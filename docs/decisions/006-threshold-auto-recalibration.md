# ADR-006: trigger threshold long-term auto-recalibration（占位）

**状态**：placeholder（ADR-005 §107 占位，待 M0 W3+ 实施）
**日期**：2026-05-16（占位）
**触发**：ADR-005 §107 "下一轮 PR 计划升级到 HTTP endpoint 运行时动态切换"

---

## 背景

ADR-005 trigger 阈值（raw_au.AU04 > 0.55）是数据驱动的**固定值**。
gemini 调研第 4 点：长期会有 baseline drift，需"动态分位数 + EMA + Schmitt Trigger"。

## 决策（待定）

placeholder。M0 W3 真做时填充。

## 候选方案

1. 滚动 30 天 raw_au.AU04 P91 重新计算阈值
2. EMA 风格：threshold = α * P91_30d + (1-α) * threshold_prev
3. Schmitt Trigger 防震荡：高阈值进入触发，低阈值退出，避免边缘抖动
4. 所有阈值变更必须落盘 ADR 记录决策（auditability）

## 何时启动

满足以下任一即启动 ADR-006 实施：
- 累积 1 万+ 真信号 events
- 川哥反馈"阈值不准"（持续 1 周）
- M0 W3 P1+ 正式排期

---

**待补完**。本文件占位避免 ADR 编号冲突（ADR-007/008/009 等已占用后续编号）。
