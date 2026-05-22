# ADR-025: 罕见 AU 重建监控 V0.2（gemini round 14 N=124 长尾警告落地）

> **决策者**：cc 主理 + codex 集成 + gemini D3 对抗审 4 finding 采纳（2026-05-22 21:45）
> **状态**：accepted
> **关联**：ADR-024 数据 effective_dim 物理上限 · ROADMAP P0-2 · gemini round 14 sign-off "最后盲区警告"

---

## 1. 触发事件

5/22 gemini round 14 sign-off Phase γ Step 1 GREEN LIGHT，**唯一遗留警告**：

> N=124 时数据将呈现极强的长尾稀疏性（罕见微表情如 AU43/AU28 出现频率极低）.
> 不能仅看全局 effective_dim. 必须监控"罕见 AU 重建误差"，
> 防止模型为了刷高主成分指标而彻底牺牲对边缘、低频关键微表情的重建能力.

ROADMAP P0-2 钦定 5/27 启动前必须落地此监控，否则 N=124 4 天训练可能拿到"全局指标好看但长尾死透"的伪 PASS。

## 2. 设计

### 2.1 数据契约
- 输入：VEATIC dataset + LNN 模型 rollout pred/target (B, T_roll, 52)
- AU 段：`event_tensor[:, :20]`（py-feat 实际输出 20 AU，`[20:24]` 是 head pose 零 padding）
- 分桶：top-7 高频 / median-6 中频 / bottom-7 罕见

### 2.2 metric 三件套
1. **per-bucket world model MSE**：`L_world_X = MSE(pred[:, :, bucket_X], target[:, :, bucket_X])`
2. **per-bucket persistence baseline**：prefix 最后一帧反复 expand_as → 与 target 做 MSE
3. **smoothed ratio**：`ratio_X = L_world_X / (pers_X + 1e-3)`（< 1 ⇔ 模型比 trivial 强）
4. **spread**：`ratio_bottom - ratio_top`（> 0 ⇔ 长尾改善差于高频）

### 2.3 verdict 阈值（V0.2）
| 触发条件 | verdict |
|---|---|
| `ratio_bottom ≥ 1.0`（罕见 AU 比 trivial 还差）| **red** |
| `ratio_bottom ≥ 0.5` AND `spread ≥ 0.5` | **red** |
| `ratio_bottom ≥ 0.5` AND `spread ≥ 0.3` | **yellow** |
| `ratio_bottom < 0.5`（罕见 AU 已强于 trivial 2x+） | **green**（spread 检查跳过） |
| 其他 | **green** |

## 3. gemini D3 对抗审 5 findings 应对

| Finding | 风险 | 决定 | 落地 |
|---|---|---|---|
| 1 分母 max(pers, 1e-9) 让恒 0 AU 列 ratio 爆表 | High | **采纳** | 改 `(pers + 1e-3)` 平滑（`RATIO_DENOM_SMOOTHING`） |
| 2 spread 在 ratio_bottom 已强时仍触发 yellow（"优秀惩罚"） | Medium | **采纳** | 加 `SPREAD_GATE_RATIO_BOTTOM=0.5` gate |
| 3 阈值 0.5 把 AU43 等低强度 AU 误算 0 频率 | High | **采纳** | `AU_ACTIVATION_THRESHOLD` 0.5 → 0.1 |
| 4 N=10→N=124 频率分布漂移导致桶定义不稳 | Medium | **采纳** | 加 `DEFAULT_FIXED_BOTTOM_INDICES=(18, 19)` 死守 AU28/AU43 |
| 5 MSE 桶内方差霸权（噪声列吞并真长尾信号） | Medium | **延后** | 记 issue，N=124 baseline 后视情况加方差归一化 |

dry-run 实证 V0.1 → V0.2：未训练模型在 V0.1 触发 false red（ratio_bottom=1.015 因恒 0 AU 误算），V0.2 修正后 green（spread 检查 gate 跳过+smoothing 拉回 ratio）。

## 4. 实施

### 4.1 新增文件
- `poc/training/rare_au_monitor.py`（200 行，纯函数 + AuBuckets/RareAuVerdict dataclass）
- `poc/training/tests/test_rare_au_monitor.py`（19 单测全过）

### 4.2 改动文件
- `poc/training/evaluate.py`（codex 集成）：
  - ODE-loss 分支开头算 `activation_freq` + `buckets`
  - rollout 循环用 `et_strided` 顺手修了 stride 对齐 bug（原代码 naive/pers baseline 用未 stride 的 et）
  - 累积 per-window pred/target → 循环外 `compute_per_bucket_mse` → `evaluate_rare_au_drop` → `verdict_to_dict` merge 进 result

### 4.3 result dict 新增 key
- `L_world_top` / `L_world_median` / `L_world_bottom`
- `ratio_top` / `ratio_median` / `ratio_bottom` / `ratio_spread`
- `rare_au_verdict` / `rare_au_reason`
- `rare_au_buckets` / `rare_au_activation_freq` / `rare_au_n_samples`

## 5. 5/27 启动门补充

ADR-024 4 道全 5 启动门基础上**新增第 6 门**：

```
6. rare_au_verdict ∈ {green, yellow}（red 必须先排查长尾原因，不允许进入 N=124 4 天训练）
```

5/27 启动 N=124 前必须先在 v3.0 ckpt 上跑这套监控并出基线，否则阈值（spread 0.3/0.5）仍是 V0.2 起步无数据支撑（同 ADR-024 §8 教训）。

## 6. LNN-FIRST 6 项对账

| 项 | 影响 |
|---|---|
| #1 LNN-FIRST | ✅ 长尾监控强化 LNN 内核"全维度建模"哲学 |
| #2 per-user τ-bias | ✅ 不变 |
| #3 AU 多模态 | ✅ 显式列序契约（AU28=18, AU43=19）锁死 |
| #4 离线优先 | ✅ 不变 |
| #5 ODE 可解释 | ✅ 不变（只读 rollout pred） |
| #6 时序记忆 | ✅ 不变 |

## 7. 验证

```
$ pytest poc/training/tests/test_rare_au_monitor.py -v
19 passed in 1.08s

$ pytest poc/training/tests/ -q
73 passed in 3.50s
```

dry-run（1 epoch mock training）：
- AU28(18), AU43(19) 死守 bottom ✅
- ratio_bottom = 0.975 < red 1.0 → green
- 旧 V0.1 在同场景 false red

## 8. 待办

- 5/23 在 v3.0 200ep ckpt + 真 VEATIC 10 视频上跑一次拿 baseline（验 V0.2 阈值是否合理）
- 5/27 启动前看是否需要按 baseline 调阈值
- Finding 5 方差归一化记 issue（N=124 baseline 后视情况决定）
- 把"rare_au_verdict 第 6 启动门"写进 ROADMAP 5/27 启动 checklist

---

**📝 维护者**：cc + codex 集成 + gemini D3 4 finding 采纳 | **签发**：2026-05-22 21:45
