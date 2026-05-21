---
title: γ-1 数据 pipeline + ncps spike 报告
date: 2026-05-21（夜间 23:00-23:40 提前完工）
audience: cc / codex / gemini / 川哥
purpose: γ-1 完整产出锚定 + γ-2 启动 ready 证据链
体裁: L3（实验报告）
related:
  - ../specs/base-v3-training-spec.md §3.1 / §4.3 / §6.1
  - ../decisions/021-ncps-tau-bias-route-c-handwritten-cell.md
  - ../decisions/016-base-v3-multimodal-self-supervised-lnn.md
---

# γ-1 完工报告（5/21 夜间提前 1 天）

> **执行**：team 模式 · cc 主线 + codex 5.3 high 并行
> **结论**：γ-1 三大输出全部就绪，γ-2 可立即启动 + ADR-021 落地 Path C 路径

## 1. 三大产出

### 1.1 VEATIC 数据 pipeline 探查 ✅

| 维度 | 实测 |
|---|---|
| 视频文件数 | **124 mp4**（spec 假设 ~120，符合）|
| Annotation 类型 | `rating_averaged/{video_id}_{valence|arousal}.csv` |
| CSV schema | per-frame `frame_id, value` 浮点 |
| 视频帧数分布 | 1267 - 4471 frames（3.5x 差异）|
| 总帧数估算 | ~22w 帧 × 124 视频 |
| **viewer_id raw 数据** | **❌ 不存在**（公开版预先平均，zip 无 raw per-viewer rating）|

**关键决定**：
- `Loss 3 Persona Contrast` **永久关定案**（spec v0.2.4 §3.3 weight=0 完全正确）
- VEATIC viewer_id 担心彻底打消（gemini A.1 + codex F-1）
- Phase γ Step 2 V-A readout 弱监督走 rating_averaged 即可

### 1.2 ncps τ-bias 三路径 spike ✅

**实测结果**：

| Path | verdict | 详情 |
|---|---|---|
| A · timespans 外部 dt | **ABANDONED** | ncps 1.0.1 broadcast bug 实测确认 |
| B · hook time_a.weight | **WEAK** | p<0.05 显著但 persona sep 1.003x（hidden 扰动但无分化）|
| C · 手写 CfCCell | **γ-2 启动** | +1 天工时，spec §8 已注明 |

**详见**：`docs/decisions/021-ncps-tau-bias-route-c-handwritten-cell.md`

**LNN-FIRST 评估**：spec §9 #13 风险（τ 没真进 ODE）→ Path C 化解 ✓

### 1.3 codex Sampler + collapse monitor 实现 ✅

**产出**：
- `poc/data/temporal_sampler.py` (188 行) - VEATIC Temporal Contrast Sampler
  - positive: 同视频 ±60 frames (2s @ 30fps)
  - negative: 跨视频 或 同视频 >30s gap
  - 视频频次均衡：1/sqrt(frames) 抑制长视频
  - 每 batch 混合 ≥ 4 视频（防视频 ID shortcut）
  - 可学 σ（soft assignment LNN-FIRST 一性原理）
- `poc/monitoring/collapse_monitor.py` (124 行) - collapse 四联监控
  - var(h_t) < 0.01 → CRITICAL
  - pairwise dist < 0.1 → HIGH
  - linear_probe_ratio (待 γ-2 接真 probe head)
  - kg_retrieval_acc (待接 graphiti SQLite)
- 测试 7 passed + 全量 63 passed ✅

**5 个翻车点（codex 自审）**：
1. video_id int 编码（γ-2 加 string 映射）
2. NaN 序列化（γ-2 接 wandb 前归一化）
3. emit_warning print 占位（γ-2 接 structured logger）
4. 单视频兜底退化（VEATIC 124 视频不触发）
5. 无状态随机采样（γ-2 加 epoch 级调度）

## 2. γ-2 (5/23-5/24) 启动 ready 证据链

| 前置 | 状态 |
|---|---|
| ncps 1.0.1 装好 Air+Studio | ✅ |
| CfC 2 层 stack 97K params forward OK | ✅（5/21 22:55 Studio 实测）|
| Loss 3 永久关定案（无 viewer_id 需求）| ✅（5/21 23:25 VEATIC 实测）|
| τ-bias Path C 实施路径锚定 | ✅（ADR-021 §2.3 伪代码）|
| Temporal Contrast Sampler | ✅（codex 实现 + 测试）|
| collapse 四联监控 | ✅（codex 占位实现 + 测试）|
| **可立刻进 γ-2** | ✅✅✅ |

## 3. spec v0.2.4 → v0.2.5 微调（不修订主体）

仅在 changelog 加 v0.2.5 段：
- §8 γ-2 工时 1.5d → 2.5d（Path C 手写 cell +1 天，引 ADR-021）
- §9 #13 风险实测确认（Path A/B WEAK，γ-2 启用 Path C）
- §11.1 #12 验收门槛收紧（sep > 1.5 要求，比初版 > 1.1 严格）

## 4. Phase γ 排期调整

原 5/27-6/10 14d → 调整后 5/27-6/10 14d **整体工期不变**：
- γ-2 +1 天（Path C 手写 cell）
- Step 4 调参压缩 4d → 3d（评估自动化脚本已部分就绪）
- 净 ±0

## 5. 关键洞察

1. **gemini C.1 部分对部分错**：vanilla ncps 不是完全黑盒（暴露 time_a/timespans），但 vanilla hook 不够 → 手写 cell 必经
2. **VEATIC viewer_id 担心**：公开版数据预先平均，担心从根上不成立 → Loss 3 永久关
3. **Phase γ 14d 工期保住**：通过 Step 4 调参压缩吸收 γ-2 +1 天
4. **codex team 协作收益再次验证**：cc 写 spike + codex 写 Sampler/monitor 完全并行 + 0 冲突 + 7 测试一次过

---

**📝 维护者**：cc 主线 | **完工**：2026-05-21 23:40 | **下一步**：5/22 早 git pull + β-self-test
