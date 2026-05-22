# ADR-026: 罕见 AU 监控 V0.3 D'（双数据集校准 + round 2 对抗审改良）

> **决策者**：cc 主理 + codex 集成 + gemini round 2 对抗审
> **状态**：accepted
> **日期**：2026-05-22
> **关联**：ADR-025 · ROADMAP 5/27 启动门 · LNN-FIRST

---

## 1. 触发

ADR-025 (V0.2) 默认 `fixed_bottom=(18, 19)`（AU28/AU43）被双数据集实测证伪：

| AU | 川哥 14 天（378k 帧）freq | VEATIC 10 vid freq | 结论 |
|---|---:|---:|---|
| AU28 | 0.589（第 2 高频） | 0.544（第 14） | 非罕见，不能固定 bottom |
| AU43 | 0.394（第 10） | 0.584（第 12） | 非底部长尾 |
| AU07 | 0.087（最罕见） | 0.114（最罕见） | 跨数据集共识罕见 |
| AU10 | 0.112（罕见） | 0.235（罕见） | 跨数据集共识罕见 |

结论：V0.2 的固定项不再成立，必须改为跨数据集共识罕见 AU，并引入动态剔除机制。

## 2. gemini round 2 finding 处置

| Finding | 结论 | 理由 |
|---|---|---|
| 1 fixed 数量过多，压缩动态自由度 | **采纳** | fixed 从 4 缩到 2，控制在 `bottom_k` 的 30% 内 |
| 2 AU12 属性格敏感，不是生理罕见 | **采纳** | fixed 候选剔除 AU12，仅保留生理低频共识项 |
| 3 fixed 可能变成高频，需 evict | **采纳** | 新增 `evicted_fixed` + top-K 检查后动态补位 |
| 4 N=124 前需全量再校准 | **延后到启动前执行** | 依赖 N=124 完整 AU cache 就绪 |
| 5 方差归一化/稳健统计增强 | **延后** | 先保持 V0.3 D' 最小变更，待 N=124 baseline 后决策 |

## 3. V0.3 D' 变更清单

1. `DEFAULT_FIXED_BOTTOM_INDICES`：`(18, 19)` → `(5, 7)`（AU07/AU10）
2. `partition_au_by_frequency`：新增 fixed-topK evict 逻辑
3. `AuBuckets`：新增 `evicted_fixed: tuple[int, ...] = ()`
4. `verdict_to_dict`：新增 `rare_au_evicted_fixed`
5. 单测：更新 fixed 行为断言，新增 evict/keep/daemon-cache 端到端覆盖

## 4. N=124-校准（启动前必做）

**截止时间**：2026-05-27 Phase γ Step 1 启动前。

执行步骤：
1. 用 N=124 全量 AU cache 重新统计 20 维激活频率。
2. 运行 fixed auto-select 候选（以跨数据集一致性 + 生理罕见优先）。
3. 复跑 `partition_au_by_frequency` dry-run，检查 `evicted_fixed` 是否长期为空或仅偶发。
4. 若 fixed 出现进入 top-K 的稳定模式，更新 fixed 列表并回写 ADR。

产物：
- 一份 `freq_rank` 对账表（20 AU 全列）
- 一次 `rare_au_verdict` 基线记录（含 `rare_au_evicted_fixed`）

## 5. LNN-FIRST 六项对账

| 项 | 对账 |
|---|---|
| #1 时间/演化建模不可退化 | ✅ 仅监控与分桶策略变更，不引入 if-else 替代 ODE 动力学 |
| #2 per-user τ 个性化 | ✅ 无改动 |
| #3 多模态输入契约 | ✅ AU 列序仍锁定 `event_tensor[:, :20]` |
| #4 离线优先 | ✅ 无改动 |
| #5 ODE 可解释路径 | ✅ 仅增强评估可解释性（evicted 可追因） |
| #6 时序记忆闭环 | ✅ 无改动 |

## 6. 验证

```bash
$ pytest poc/training/tests/test_rare_au_monitor.py -v
22 passed in 2.98s

$ pytest poc/training/tests/ -q
80 passed, 5 warnings in 4.82s
```

daemon cache dry-run（`/tmp/daemon_au_cache.npz`）：
- `evicted_fixed=()`
- `bottom=(3, 5, 6, 7, 9, 12, 13)`
- `freq[5]=0.086587`, `freq[7]=0.112083`（AU07/AU10 仍处低频段）

---

**维护备注**：ADR-026 是 ADR-025 的数据证伪后修订；N=124 全量数据到位后必须再校准一次 fixed_bottom。
