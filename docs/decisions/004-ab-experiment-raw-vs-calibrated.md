# ADR-004: LNN 自学习有效性 A/B 实验框架 — raw 路径 vs calibrated 路径

**状态**：proposed
**日期**：2026-05-16
**决策者**：川哥（实验目标） + cc（设计） + 待 codex 二审
**触发**：ADR-003 决定走 raw 直通后，川哥要求"完成之后和旧数据/效果对比，作为佐证说明 LNN 思想确实起到正面效果"

---

## 背景

川哥的核心命题：
> "重要的是，路径 1 完成之后，我们和之前的旧数据和效果，进行对比，作为佐证，来说明 LNN 的思想，确实能起到正面效果。"

这不是工程问题，是**科学实验**：用 A/B 对比证明 LNN 自学习哲学的有效性，否则 P-1 LNN-FIRST 只是信仰不是事实。

## 决策

设立 **LnMem A/B 实验框架**，记录两条独立训练 pipeline 的预测准确率长期对比：

- **A 路径（对照组 / 历史包袱）**：raw_au → calibration 函数 → expression.brow_furrow → daemon trigger → LNN（5/12-5/16 累积 ckpt 链）
- **B 路径（实验组 / LNN-FIRST 纯版）**：raw_au → daemon trigger 直读 raw → LNN（新 ckpt 链，从 ADR-003 之后起）

## 实验设计

### 数据基线

| 维度 | 数据 |
|---|---|
| 历史 events | 138 queries (5/12-5/15) + 6 demo queries (5/16) = **144** |
| 历史 signals 帧 | 235,299 帧 (5/12-5/16) |
| 路径 A 当前 best ckpt | `checkpoint_phase2_20260516T081409Z.pt` |
| 路径 B 起点 | 待实施 ADR-003 后生成 |

### 三个核心指标（按重要性排序）

#### 1. **同 events 双路预测分歧率**（最直接）

对 144 个历史 events，跑两个 ckpt 各做一次 forward，对比预测的 state：
- agreement_rate = 两路预测相同的比例
- 如果 < 50% → 两路是不同模型；如果 > 90% → calibration 实际没大影响

#### 2. **真实用户反馈 match 率长期对比**（最有产品意义）

部署 raw 路径后，**未来 7-14 天**真实使用中：
- 路径 A 历史 match 率：brow 70%, pose_held 17%, low_vlm_conf 7% (5/12-5/15 数据)
- 路径 B 新数据 match 率：待观察

预期：
- ✅ 若 B 路径 brow match > 70% 且 pose match > 30% → **LNN 自学习有效**（P-1 哲学得证）
- ⚠️ 若 B 持平 → 信号源没问题但建模能力到顶
- ❌ 若 B 反而下降 → **重要 finding**，可能 raw 直通需配套 FiLM 等结构（不是 calibration 错，是 LNN 架构不足）

#### 3. **mismatch 时 true_state 的分布变化**（看 LNN 学会了什么）

路径 A 历史 mismatch 时川哥常说"我其实是 distracted"（19/29）。如果路径 B 上线后 distracted 类 mismatch 比例下降，说明 LNN 学到了"被 calibration 误标皱眉时川哥实际处于 distracted"。

### 测试脚本框架

需新增：
- `scripts/ab_replay_predict.py` — 加载两个 ckpt + 历史 events，输出双路预测对比表
- `scripts/ab_match_rate_compare.py` — 按时间窗口比对 A/B 路径的 match 率趋势
- `~/.lnmem/ab_metrics.jsonl` — 每次 fit 后追加双路 metric 一行

### Pipeline 改造（与 ADR-003 同步）

```
现状（路径 A）：
  capture → py-feat → raw_au → calibration → expression → daemon(trigger) → LNN
                                              ↓
                              brow_furrow / smile_intensity 等 polarized 字段

新增（路径 B）：
  capture → py-feat → raw_au → daemon(trigger 用 raw) → LNN（input 20 维 AU）
                       ↑
              直接消费，不经 calibration
```

`signal_path` 字段加入 fit_step_history.jsonl 区分两路 ckpt。

## 选项与对比（实验范式选择）

| 选项 | 优点 | 缺点 | 推荐 |
|---|---|---|---|
| A. 离线 replay 一次性对比 | 快速；可量化 | 缺少长期使用反馈 | 必做（短期） |
| B. 长期 A/B 部署对比 | 真实使用证据 | 慢（≥1 周） | ✅ 必做（长期） |
| C. 仅看新路径 match 率 | 简单 | 没对比就没说服力 | ❌ 不够 |
| D. A + B 组合 | 短期 + 长期双佐证 | 工作量大 | ✅ **采纳** |

## 出口标准（如何宣布 LNN 思想有效）

**短期（路径 1 实施后 + 离线 replay 完成）**：
- 双路对 144 历史 events 预测分歧率 > 30% → 证明 calibration 是真噪声
- 配合 ADR-003 的数据熵证据（83% vs 61%）

**长期（部署 7-14 天后）**：
- 路径 B 真实 match 率比 A 同期数据高 ≥ 10% → **LNN 自学习有效性硬证据**
- 或：路径 B match 率持平但 brow_furrow 分布更接近 raw_au 分布（说明 LNN 学到了正确的 baseline 感知）

## 后果

### 正面
- 项目第一次用科学实验佐证 P-1 LNN-FIRST 哲学
- 失败/成功都有价值（如果失败说明 LNN 架构还需调整，不是哲学错）
- 沉淀的方法论可复用到未来其他通道（VLM / pose / EEG…）

### 负面 / 风险
- 7-14 天观察期间川哥需要继续使用（不能停 daemon）
- 如果 B 路径短期 match 率显著下降，川哥体验会变差（但这是必要的科学代价）

## 反对意见 / 不同看法

- **codex 暂无意见**（待审）
- **可能的反驳**：A/B 对比的 confounding 是 ckpt 训练数据量不同（A 有 5 天数据，B 是 cold start）。**应对**：把 B 也跑 5 天后再做横向对比，不能开头几天就下结论

---

**末尾自检**：本 ADR 与 ADR-003 配套。ADR-003 是"做什么"（绕过 calibration），本 ADR 是"如何证明做对了"。两者缺一不可。
