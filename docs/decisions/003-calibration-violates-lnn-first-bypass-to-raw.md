# ADR-003: AU04 calibration 算法在川哥脸上信息退化（已 superseded by ADR-005）

> ⚠️ **2026-05-16 修订**：codex 对抗审查 F-03 指出 lnn-first.md §4.1 明确视觉感知层**不受 P-1 约束**，原标题"违反 LNN-FIRST"的论证不准确。修订要点：
> - calibration 层本身**不违反 P-1**（lnn-first.md §4.1 明确允许 "PoC 阶段允许用'特征校准层'作为权宜之计"）
> - 真正问题是 **AU04 的 gamma 拉伸算法** `cal=clamp01(((raw-0.04)/0.15)^0.8)` 在川哥脸上把 raw>0.19 全 clamp 到 1.0，造成 22% 信息熵损失
> - **本 ADR 被 ADR-005 superseded**（不再做"完全绕过 calibration"，改做"trigger 应用层用 raw_au.AU04 + LNN 内核演化层不动"的最小改动）
> - 保留本 ADR 作为踩坑过程记录（cc 论证不准 → codex 纠偏 → ADR-005 修正方案）

**状态**：superseded by ADR-005
**日期**：2026-05-16
**决策者**：川哥（最终决策） + cc（数据 finding） + 路径 1 实施待 codex 二审
**触发**：M0 W2 Phase B 5/12-5/16 真摄像头数据验证 + 川哥 5/16 17:00 体验验证发现 "大多数情况都判定皱眉但其实没皱"

---

## 背景

LnMem PoC M0 W2 期间，川哥反馈"daemon 大多数情况判定我皱眉但其实没皱"。cc 跑 D-1 信号熵分析发现决定性数据：

**raw 信号（py-feat 原始 AU04 皱眉强度）健康**：
- mean=0.286, std=0.173
- 自然钟形分布（62% 集中 [0, 0.3]，川哥日常 baseline ~0.2）
- 信息熵 2.76/3.32 = **83%**

**calibrated_au.AU04 / expression.brow_furrow（daemon 实际看到的）极度 polarized**：
- mean=0.396, std=0.448（双峰）
- 51.3% 在 [0, 0.1]（"完全没皱"）+ 30.3% 在 [0.9, 1.0]（"满分皱"），中间 18%
- 信息熵 2.02/3.32 = 61%（**比 raw 损失 22% 信息**）

**关键证据 — raw → calibrated 配对统计（5000 样本）**：
```
raw=0.1 → calibrated mean=0.576 (std=0.414)  ← std 巨大
raw=0.3 → calibrated mean=0.592 (std=0.416)
raw=0.5 → calibrated mean=0.581 (std=0.420)
raw=0.7 → calibrated mean=0.591 (std=0.406)
```

raw 在 [0.1, 0.7] 范围内 calibrated mean 都 ≈ 0.58 但 std=0.41 → calibration **把同一 raw 值随机映到 0 或 1**（双峰极端），中间过渡几乎被吃掉。这是个**隐式规则二值化器**。

## 决策

**绕过 calibration 函数，让 daemon / LNN 直接消费 raw_au（20 维 AU 向量），让 LTC 输入相关 τ + FiLM + tau_drift 自学 per-user "raw → state" mapping。**

5/12 + 5/16 累积的 fit ckpt（基于 calibrated 路径）**标记为对照组**，作为 A/B 实验的"旧路径"基线。

## 选项与对比

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| **A. 删 calibration 走 raw 直通** | 符合 P-1 LNN-FIRST；信号熵 +22%；让 LNN 真自学 | 重训 base ckpt；calibrated fit 历史作废 | ✅ 选 |
| B. 修 calibration 函数让中间过渡平滑 | 不动架构；向后兼容 | 仍是规则系统，信息退化（非违反 P-1）；川哥已否决 | ❌ 哲学上拒绝 |
| C. 加 per-user calibration baseline table | 立竿见影；川哥脸 baseline 校准 | **赤裸信息退化（非违反 P-1）**（"用规则替代 LTC"）；cc 已撤回 | ❌ 撤回 |
| D. 保留 calibration + raw 双路并行 | 渐进；可对比 | 架构变复杂；仍走错路径 | ⚠️ 不选（不彻底） |

## 为什么选 A（核心论据）

### 1. P-1 LNN-FIRST 是宪法级条款，不能让步

P-1 原话："凡是涉及…个性化…的设计，第一性原理必须是 LTC/CfC 微分方程；用 LLM/RNN/规则替代 LTC 必须有书面 Override 理由并经老板批准"

calibration 是个**规则函数**（按某种 baseline 做映射），是 LLM 监督学习范式的残留。让 LNN 自学 raw 才是项目存在意义。

### 2. 神经科学 + LTC 论文双重支撑 LNN 自学能力

- LTC (Hasani et al., 2020)：τ 是输入相关的，神经元面对不同输入自动展现不同时间动力学
- FiLM (Perez et al., 2017)：γ/β 学到 per-style normalization，等价 per-user 信号校准
- Synaptic scaling：生物神经元长期暴露某 firing rate 会自适应阈值
- Predictive coding：大脑用 prediction error 学，不用显式标签

四个机制都说明 LNN 三件套（tau_drift + FiLM + 槽位偏好）**理论上足以处理 per-user raw signal 校准**，不需要外部 calibration。

### 3. 川哥的产品哲学 = 项目差异化

> "我为什么选择 LNN 的架构，就是希望摆脱这种死板的打标签校准的方式，而是使用更自然无感的、在日常使用过程中的自动校准和学习，这才是这个项目的核心差异之一。"

如果做 calibration 表，LnMem 跟 Mem0 + calibration 表没差异，**项目失去差异化定位**。

### 4. 数据熵证据

raw 熵 83% > calibrated 熵 61%，删 calibration 直接 +22% 信息给 LNN 用。这是工程上的**直接收益**，与哲学方向一致。

### 5. PoC 阶段最有价值的 ADR

这是 LnMem PoC 第一次用真实数据 + 真实用户体感**反向揭穿自己架构的"伪 LNN"陷阱**。比单纯做 fit cron 更有项目价值。

## 后果（影响 / 副作用）

### 正面

- 架构回归 LNN-FIRST 哲学根基
- 信号熵 +22%，模型理论可学到更细粒度信号
- A/B 对比将提供 **LNN 自学习有效性的硬证据**（项目核心假设的真正检验）

### 负面 / 风险

- **5/12 + 5/16 累积的 calibrated 路径 ckpt 作废**（保留为对照组）
- LNN base 模型需要重训（输入维度可能从 N 变 20）
- 4-6h 工程量 + 重训
- 短期 daemon 预测准确率可能**下降**（新模型 cold start）— 这是 A/B 实验的"代价"
- raw_au 20 维直接喂 LNN 可能引入**噪声维度**（如 AU43 闭眼对状态信号不强），需要 FiLM 学权重

### 触发的下游决策

- **ADR-004**：FiLM γ/β 学习策略（per-user hypernetwork vs 直接 fit）
- **ADR-005**：A/B 对比实验设计（同 events 两路预测 + 长期 match 率追踪）
- **roadmap M0 W3 调整**：raw 直通工程项排进 W3 优先级
- 标记：所有 5/12-5/16 fit_step_history 标 `signal_path=calibrated`，新 fit 标 `signal_path=raw`

## 反对意见 / 不同看法

- **cc 初提 D-2**："加 raw signal per-user baseline 校准层" — 被川哥识破是 LLM 范式残留，cc 已撤回并自我检讨
- **codex 暂无意见** — 待 raw 路径设计完成后做对抗审查
- **gemini 暂无意见** — 待问 LNN 处理 raw multi-dim 输入的工程最佳实践

## 实施分阶段

| 阶段 | 内容 | 工时 | 出口标准 |
|---|---|---|---|
| ADR-003 落盘（本文档） | 决策记录 + 踩坑标签 | 30min ✅ | 已完成 |
| 现状摸清 | calibration 在哪 / LNN 输入维度 / trigger 用哪个字段 | 30min | 出实施方案 |
| 实施 raw 直通 | feature pipeline / trigger / fit_orchestrator | 2-3h | 510 现有测试 PASS + 新 raw 路径单测 |
| LNN base 重训 | 用历史 raw_au 数据 cold start | 1-2h | 新 ckpt 健康 |
| A/B 对比实验 | 旧 calibrated ckpt vs 新 raw ckpt 同 events 对比 | 持续 | 报告 + 长期 match 率追踪 |

## 标签：踩坑经验回溯锚点

```yaml
# 5/12 + 5/16 fit ckpt 性质
signal_path: calibrated  # 历史包袱，A/B 对照组
discovery_ts: 2026-05-16T17:30:00+08:00
discoverer: 川哥（体感 finding）+ cc（数据 D-1 验证）
problem: calibration 函数把 raw 0.21 → 0.94（polarized 二值化）
fix: ADR-003 走 raw 直通路径
```

---

**末尾自检**：本文档严格符合 [CLAUDE.md §3.1 cc/codex 意见冲突诊断流程] —
- cc 主线提了 D-2（错），川哥反对（对），cc 撤回 + 写本 ADR 记录分歧根因
- 决策依据：哲学（P-1）+ 数据（熵 83% vs 61%）+ 论文（LTC/FiLM）+ 神经科学（synaptic scaling）
- 不同意见保留：cc 的 D-2 错误路径完整记录，3 个月后回看可总结"为何 cc 容易掉入 LLM 范式陷阱"
