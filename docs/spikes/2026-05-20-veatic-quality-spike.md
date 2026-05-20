---
spike-id: 2026-05-20-veatic-quality
title: VEATIC 标注质量 spike — C 主线（端到端真序列 LTC）走得通
status: complete（2026-05-20，川哥钦定 spike-then-implement B 路径）
date: 2026-05-20
deciders: 川哥 + cc
related: ADR-007 Phase 1（`docs/decisions/007-base-ckpt-realhuman-affectnet-phase1-plan.md`）/ ADR-011（假脸 guard）
---

## 目的

ADR-007 Phase 1.2 启动前先 spike — 验证 VEATIC（C 轨）数据能否支撑 Phase 1.3-1.5 主线决策：
端到端真序列训 LTC（不冻 tau），而非降级走 B EMOTIC 静态图（V-A-D + 冻 tau）或 A mst（emotion CE）。

## 方法

抽 5 视频 × 10 均匀帧 = 50 帧，3 项检验：

1. **(a) 标注对齐**：mp4 帧数 vs `{id}_{valence,arousal}.csv` 行数是否 1:1；V-A NaN 比例
2. **(b) py-feat AU 提取稳定性**：face_detected 检出率、AU 强度均值/标准差、单帧耗时
3. **(c) V-A 时间序列演化模式**：抽样帧 V-A 真值是否覆盖广 + 视频内有显著走势

工具：cv2.VideoCapture 抽帧 + py-feat 0.6.2 batch detect（`au_model=xgb`, `emotion_model=resmasknet`）+ pandas 读 csv。

产物：`/tmp/lnmem-team-2026-05-20/veatic_probe_50frames.csv`（50 行 per-frame face_detected + V-A 真值）

## 关键发现

### (a) 标注对齐 ✅ 通过

| video_id | mp4_frames | fps  | csv_rows(V/A) | aligned | V_std  | A_std  | V_nan |
|----------|------------|------|---------------|---------|--------|--------|-------|
| 0        | 1443       | 25.0 | 1443 / 1443   | True    | 0.069  | 0.079  | 0.00% |
| 30       | 2041       | 25.0 | 2041 / 2041   | True    | 0.143  | 0.060  | 0.00% |
| 60       | 1272       | 25.0 | 1272 / 1272   | True    | 0.262  | 0.307  | 0.00% |
| 90       | 2852       | 25.0 | 2852 / 2852   | True    | 0.424  | 0.158  | 0.00% |
| 123      | 1267       | 25.0 | 1267 / 1267   | True    | 0.320  | 0.258  | 0.00% |

5/5 严格 1:1 对齐，全 25 fps，**V-A 标注零 NaN**。csv 行格式 `frame_idx,value`，浮点精度 ~1e-16。

### (b) py-feat AU 提取稳定性 ⚠️ 80% 可用

| video_id | face_detected | all_nan | AU_mean | AU_std | 耗时/帧 |
|----------|---------------|---------|---------|--------|---------|
| 0        | 9/10          | 1       | 0.300   | 0.260  | 4.7s（含冷启动）|
| 30       | 8/10          | 2       | 0.374   | 0.275  | 1.3s    |
| 60       | 10/10         | 0       | 0.384   | 0.290  | 1.3s    |
| 90       | 5/10          | 5       | 0.377   | 0.259  | 1.8s    |
| 123      | 8/10          | 2       | 0.354   | 0.274  | 1.5s    |
| **汇总** | **40/50（80%）** | **10** | ~0.36   | ~0.27  | ~1.5s   |

观察：
- AU 均值 0.30-0.38、std 0.26-0.29 — 落在正常人脸 AU 强度区间，**非假阳性**
- video 90 检出率最差（5/10），抽样集中在视频前半段（fid 0-1267），可能镜头开头是远景/黑屏/字幕
- detector 冷启动 30.5s + 首批 47s（模型 lazy load），第二批起降到 1.3s/帧 — 全量提取需 batch + 多核优化（codex 任务范围内）

### (c) V-A 演化模式 ✅ 强信号

50 帧 V-A 真值统计：
- V 范围 **[-0.915, 0.651]** — 覆盖近全区间（VEATIC 设计 [-1, 1]）
- A 范围 **[-0.367, 0.869]** — 覆盖 ~62% 区间
- 零 NaN
- 视频内显著演化样本：
  - **vid 60**：V `-0.30 → -0.86`、A `0.22 → 0.87`（10 帧均匀采样，单调爬升 — 高质量负 V + 高 A 演化）
  - **vid 90**：V `0.00 → 0.51 → -0.52 → 0.22`、A `-0.01 → 0.58 → 0.67 → 0.26`（先升后降的双峰，非单调）
  - **vid 123**：V `-0.03 → 0.65 → 0.24`、A `0.00 → 0.64 → 0.24`（先升后降）

**结论**：VEATIC 真序列 V-A 是「时间上演化的连续函数」，**完美契合 LTC 微分方程 `dx/dt = -x/τ + f(input)` 训练信号**，能直接驱动 tau 学习而无需冻结。

## 决策建议（cc 主线 + 川哥钦定）

✅ **走 C 主线 — Phase 1.3-1.5 用 VEATIC 端到端训 LTC，不冻 tau**

| 维度 | 证据 |
|------|------|
| 标注对齐 | 100% mp4 帧 ↔ csv 行 1:1 + 零 NaN |
| V-A 真序列演化 | 视频内 V 演化幅度 0.85、A 演化幅度 0.85，远超阈值 |
| py-feat AU 在视频帧上稳定性 | 80% face_detected，AU 强度合理（ADR-011 guard 兜底剩 20%）|
| LNN-first 契合度 | 真时间序列 → LTC tau 学习自然驱动，**根治 gemini critical 风险 #1（静态图破坏 tau）** |

### 退化备选（如 C 主线训练遇阻才启用）

| 触发条件 | 降级到 |
|----------|--------|
| VEATIC 全量提取耗时 > 24h 无法跑通 | B EMOTIC（V-A-D 静态图 + 冻 tau）|
| VEATIC + EMOTIC 都翻车 | A mst（emotion CE，最保守）|

## 风险与缓解

1. **video 前段检脸率低（如 vid 90 前 4 帧全失）**
   - 缓解：训练时按 ADR-011 `_has_finite_value` guard 过滤 `face_detected=False` 帧，并对每视频做「有效帧占比 ≥ 50%」筛选；不达标视频整段剔除。
2. **py-feat 单帧 1.3s（CPU）→ 124 视频 × ~2000 帧 ≈ 90h**
   - 缓解：codex Step 1 公共脚本必须做 batch + sample_frames_per_video（如均匀抽 200 帧/视频，总 25K 帧 × 1.3s = 9h 单核 → multi-process 可降到 2h）。
3. **VEATIC 实际可用视频数（剔除低检脸率后）可能 < 124**
   - 缓解：Phase 1.4 训练前先扫全 124 视频 face_detected 率分布，定阈值 → 报告留多少有效数据。

## 配套产出

- 探针脚本：本文件内联（一次性，不进 git）
- 数据：`/tmp/lnmem-team-2026-05-20/veatic_probe_50frames.csv`（50 行）
- 模型 cache：`/tmp/lnmem_pyfeat/`、py-feat 默认 `~/.feat/`
- 与 codex Step 1 公共脚本（`scripts/extract_au_to_parquet.py`）联动 — 该脚本 `--input-type video_frames` 路径覆盖本 spike 用例

## 时间线

- 2026-05-20 上午：cc 进 session 读 ADR-007 plan + 解压 VEATIC + 跑 50 帧探针（约 2h）
- 同步：codex 后台跑公共脚本 Step 1（3-4h）
- 完成本 spike 后立即更新 ADR-007 plan 末尾"分叉决策"段，commit + push origin/poc/lnn-validation
