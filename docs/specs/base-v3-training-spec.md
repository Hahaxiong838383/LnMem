---
version: v0.1
audience: cc / codex / 工程师 / 川哥
purpose: ADR-016 base v3 multimodal self-supervised LNN 的工程实施细化 spec
status: 草稿（2026-05-21 起草，等川哥过目 + codex 对抗审查）
last_updated: 2026-05-21
related:
  - ../decisions/016-base-v3-multimodal-self-supervised-lnn.md
  - ../decisions/013-base-v2-failure-and-finetune-plan.md
  - ../decisions/018-eventtensor-schema-v1_1-intent-type-multimodal-context.md
  - ../principles/lnn-first.md
  - ../product-definition.md
parent: ../STYLE.md
体裁: L3（半结构化 spec）
---

# Base v3 训练 Spec（多模态 self-supervised LNN）

> **本 spec 是 ADR-016 的工程实施细化，不是替代。** ADR-016 定的是范式骨架（为什么 + 大致怎么），本 spec 定的是 Phase γ（5/27-6/10）「实际怎么写 + 怎么跑 + 怎么验」。
>
> **P-1 LNN-FIRST 守则贯穿全文**：base v3 不是 LLM 分类器，是用 LNN 学连续 hidden state 演化。评估主指标必须是 **KG 召回质量 + hidden 聚类清晰度 + 时序一致性**，不是 accuracy。任何把 v3 当 supervised 多分类模型来评的做法，都触发 LNN-FIRST 红线。

---

## 1. 范围与目标

### 1.1 这份 spec 是什么

ADR-016 的工程实施细化版。ADR-016 解决「v3 为什么不是 v2 + 范式怎么改」的认知问题；本 spec 解决「Phase γ 实际跑训练时的代码、数据、超参、评估、风险」的工程问题。两份文档配对使用：先读 ADR-016 拿到「为什么」，再读本 spec 拿到「怎么做」。

### 1.2 目标（Phase γ：5/27-6/10）

跑出可用的 base v3 ckpt，满足：

1. 在 VEATIC 视频时序上学到**连续 hidden state 演化**（Loss 1 收敛，相邻帧 hidden 距离 < 远帧）
2. EMOTIC + mst 上 readout 弱监督训练**不破坏 backbone 表征**（Loss 4 收敛，但 backbone 冻结）
3. hidden state 64 维**对 graphiti KG 可用**（语义检索 Top-K 命中率优于 base v1 V-A 2 维直接当 query）
4. 跨样本 hidden 聚类清晰度**优于 base v2**（silhouette score 提升，t-SNE 视觉分离）
5. ckpt ≤ 5MB，单帧推理延迟 ≤ 15ms（Studio M2 Max CPU 路径，daemon 兼容）

### 1.3 非目标（明确不做）

- ❌ **不继续 base v1/v2 supervised 分类路径**——v2 已 64/64 坍缩 neutral，supervised 9-state CE 在川哥真人数据上证伪
- ❌ **不以 accuracy 为主指标**——违反 LNN-FIRST 准则 V1/V3，等于把 v3 退化成 v2
- ❌ **不引入 LLM 在 backbone 训练 loop 里**——LLM 仅作为 ADR-018 intent_type 标签源（offline 预标注），不进 backbone forward
- ❌ **不实施 Loss 2 cross-modal**——本阶段音频/文本对齐数据不足，留 M2+
- ❌ **不在 Phase γ 做川哥真人微调**——那是 Phase δ（6/11+）的事，本 spec 只覆盖 Phase γ

### 1.4 关联里程碑

- **Phase α**（已完成 5/21 quick win）：demo 加 webcam snapshot 按钮
- **Phase β**（5/22-5/24）：状态描述 schema + intent_type 抽取（ADR-018）
- **Phase γ（本 spec 覆盖范围，5/27-6/10）**：base v3 预训练 + readout 弱监督
- **Phase δ**（6/11+，M1 起）：川哥真人 daemon 数据微调

---

## 2. 模型架构

### 2.1 总览

```
[Input] (per frame, dt = 1/30 s)
  视觉 AU + head pose      (24 dim) — py-feat 已实施
  万能语法 20 槽 one-hot   (20 dim) — Phase β intent_type 加 5 dim
  音频特征占位             ( 6 dim) — M1+ 加，本阶段填 0
  时间戳归一化             ( 2 dim) — [sin(2πt/T_day), cos(2πt/T_day)]
                          ───────
                          总 52 dim raw input

  ↓ Input Projection (Linear 52 → 32-64)

[Multi-modal Fusion Vector] (32-64 dim)

  ↓ Persona Conditioning
    - FiLM: γ, β = MLP(persona_embedding)
    - τ-bias: τ_i = τ_base * exp(persona_bias_i)

[CfC Backbone] (ncps 库的 CfC，4 层多时间尺度)
  Layer 1: τ ∈ [0.1s, 1s]   — 快速表情/微动
  Layer 2: τ ∈ [1s, 10s]    — 情绪窗口
  Layer 3: τ ∈ [10s, 60s]   — 短期 mood
  Layer 4: τ ∈ [60s, 600s]  — 长时段落级

  ↓ 64-dim hidden state h_t (主输出)

[Readout Heads] (附属，弱监督，backbone 冻结时才训)
  Head 1 (state_8)  : Linear 64 → 8, softmax     — 8-state emotion
  Head 2 (va_cont)  : Linear 64 → 2, tanh        — V-A 连续回归
  Head 3 (intent)   : Linear 64 → 5, softmax     — intent type
  Head 4 (conf)     : Linear 64 → 1, sigmoid     — confidence
  Head 5 (hidden_repr): Identity 64 → 64         — 直出给 KG
```

### 2.2 各层细节

| 层 | 模块 | input dim | output dim | 参数量预估 |
|----|------|-----------|------------|-----------|
| Input projection | nn.Linear + LayerNorm + GELU | 52 | 32（α 试跑）/64（β 全量） | 52×64+64 ≈ 3.4K |
| Persona embedding | nn.Embedding(num_personas=64, dim=16) | — | 16 | 1K |
| FiLM head | MLP 16 → 2×fusion_dim | 16 | 128 | 4K |
| CfC Layer 1-4 | ncps.torch.CfC | 64 | 64 | 4 × ~5K ≈ 20K |
| Head 1 state_8 | Linear | 64 | 8 | 0.5K |
| Head 2 va_cont | Linear | 64 | 2 | 0.13K |
| Head 3 intent | Linear | 64 | 5 | 0.33K |
| Head 4 conf | Linear | 64 | 1 | 0.07K |
| Head 5 hidden_repr | Identity | 64 | 64 | 0 |

**总参数量预估：≈ 30K**（远小于 base v2 的 ~80K）。ckpt 序列化估 5MB 内（FP32 权重 + persona embedding 表 + 元信息）。

### 2.3 Studio M2 Max 推理延迟预估

- CPU 路径（daemon 兼容）：单帧 forward 估 8-12ms（CfC 4 层 + 5 head readout）
- MPS 路径（训练）：batch=64 单步 forward 估 25-40ms
- 训练吞吐目标：≥ 500 frames/sec（VEATIC 52w 帧约 17 min/epoch）

**Why（参数量压缩比 v2 小）**：base v2 用 9-state Russell margin 训练，readout 重，引入大量冗余。v3 backbone 主线学表征，readout 是附属轻量 head，单 head 0.07-0.5K 参数。

---

## 3. 4 个 Loss 函数

### 3.1 Loss 1 — Temporal Contrast（主目标，weight=1.0）

**目标**：同一视频内时间相近的 hidden state 互相接近，时间远离的远离。让 backbone 学到「时序连续性」这个第一性原理。

**形式**：InfoNCE

```
给定 anchor h_t（视频 V 在时刻 t 的 hidden），
正样本 h_{t+δ}（δ ∈ [-2s, +2s] 同视频随机采）
负样本 h_{t'} K=128 个（同 batch 其他视频或同视频 |Δt| > 30s 段）

L_temporal = -log [ exp(sim(h_t, h_{t+δ}) / τ_nce) /
                    Σ_k exp(sim(h_t, h_k) / τ_nce) ]

sim = cosine similarity
τ_nce = 0.1（InfoNCE temperature，固定）
```

**输入张量形状**：
- `h_anchor`: (B, D=64)
- `h_pos`: (B, D=64)
- `h_neg`: (B, K=128, D=64)

**关键超参**：
- positive window δ ∈ [-60 帧, +60 帧]（30 fps，2 秒窗）
- negative pool size K = 128
- negative sampling：50% 同 batch 跨视频 + 50% 同视频 |Δt| > 30s 段
- τ_nce = 0.1

**启用条件**：Step 1 全程（5/27-5/30）

**失败模式**：
- Mode collapse：所有 hidden 都收敛到同一点 → 监控 `std(h)`，<0.05 报警停训
- 学到 ID 而非语义：t-SNE 按视频 ID 分簇而非按语义分簇 → 加 batch-level video shuffle + 混合多视频 batch

### 3.2 Loss 2 — Cross-modal Contrast（M1+，weight=0 本阶段）

**目标**：同一时间窗的视觉/文本/音频 hidden 互相接近。

**本阶段不实施**：Phase γ 没有同步的多模态对齐数据（VEATIC 是纯视频 + ratings，没文本不没音频片段对齐）。Phase γ weight=0，留接口和注释占位。

### 3.3 Loss 3 — Persona Contrast（weight=0.3）

**目标**：同 user 不同时间的 hidden 互相接近（slow drift），不同 user 远离。

**形式**：SupCon (Khosla et al. 2020)

```
L_persona = -Σ_i (1/|P(i)|) Σ_p∈P(i) log [
  exp(sim(h_i, h_p) / τ_supcon) /
  Σ_a≠i exp(sim(h_i, h_a) / τ_supcon)
]

P(i) = batch 内与 anchor i 同 persona_idx 的样本集
τ_supcon = 0.07
```

**输入张量形状**：
- `h`: (B, D=64)
- `persona_idx`: (B,) 整数 ID

**关键超参**：
- τ_supcon = 0.07
- 同 batch 内 persona 多样性约束：≥ 8 个不同 persona_idx（DataLoader 自定义 sampler 保证）

**启用条件**：Step 1（VEATIC 视频有 viewer_id，可用作 persona）

**失败模式**：
- Persona 数太少 → SupCon degenerate 到 pair contrast，弱化为正则项
- VEATIC viewer 与 EMOTIC/mst 不重叠 → Step 2/3 用 mock persona 或冻结此 loss

### 3.4 Loss 4 — Weakly-supervised Readout（weight=0.5）

**目标**：用少量标注样本训 readout head，不动 backbone。

**形式**：多任务加权 sum

```
L_readout =
  α_1 * CE(head_1(h), state_8_label) +
  α_2 * CCC(head_2(h), va_label) +     // V-A concordance correlation coefficient loss
  α_3 * CE(head_3(h), intent_label) +
  α_4 * BCE(head_4(h), correct_flag)

α_1 = 1.0, α_2 = 1.0, α_3 = 0.5, α_4 = 0.3
```

**输入张量形状**：
- `h`: (B, D=64)，**带 `detach()`**（关键：backbone 不接收 readout 梯度）
- 各 head 输入是 `h.detach()`，输出对应 label

**关键超参**：
- backbone `requires_grad=False`（Step 2 起强制）
- readout 单独 lr = 1e-3（比 backbone 高一个量级）
- early-stop on readout val CCC < 0.05 提前停（防 readout 学 shortcut）

**启用条件**：Step 2 起（EMOTIC V-A）+ Step 3（mst 8 class）

**失败模式**：
- Readout 学到 shortcut（如全预测 majority class）→ 监控 pred 熵，熵 < log(2) 报警
- backbone 意外被更新（漏 detach）→ 训练循环加 `assert backbone params unchanged` 哨兵

### 3.5 三 loss 联合训练流程

```python
# Pseudo-code（实际实施见 §5 schedule）
total_loss = (
  1.0 * L_temporal +
  0.0 * L_crossmodal +    # M1+
  0.3 * L_persona +
  0.5 * L_readout         # Step 2/3 才启用
)
```

**Why（loss weight 选这组）**：Loss 1 是主目标必须 dominant，Loss 3 防 backbone 忽略 persona 信息（v2 失败教训之一），Loss 4 只用 0.5 因为它走 detach 路径不能反传 backbone，但权重不能太低否则 readout 收敛慢。

---

## 4. 数据 Pipeline

### 4.1 三数据集状态与路径

| 数据 | 大小 | Studio 路径 | 当前预处理状态 | Phase γ 用途 |
|------|------|------------|--------------|-------------|
| VEATIC | 6 GB | `/Users/fu-121/lnmem-data/veatic/` | 部分帧已抽 AU parquet（ADR-007 Phase 1.2），剩余约 30% 未抽 | Step 1 主：自监督预训练（Loss 1 + Loss 3） |
| EMOTIC | 3.4 GB | `/Users/fu-121/lnmem-data/emotic/` | 图未抽 AU；标注是 V-A-D + bbox + 情境 | Step 2 辅：readout Head 2 (V-A) 弱监督 |
| AffectNet-mst | 387 MB | `/Users/fu-121/lnmem-data/affectnet-mst/` | 已抽 AU（base v1 复用过）| Step 3 兜底：readout Head 1 (8-state) 弱监督 |

### 4.2 数据增强策略

**视觉（per frame）**：
- AU 抖动：高斯噪声 σ=0.02（参考 base v2 已用）
- AU dropout：随机 mask 20% AU 通道，强制模型不依赖单一 AU
- temporal masking：30% 概率随机 mask 5-10 帧（仿真遮挡）

**文本**（Step 2/3）：
- 万能语法 20 槽：5% 概率随机置零槽位（仿真槽位提取失败）

**时间戳**：
- 训练时随机偏移 ±30 min（弱化对绝对时间的依赖）

### 4.3 Batch Sampler 设计

**Step 1（Loss 1 + Loss 3 联合）必须保证**：

1. 每个 batch 内 **≥ 8 个不同 persona_idx**（Loss 3 多样性）
2. 每个 batch 内 **≥ 16 个不同 video_id**（Loss 1 跨视频 negative）
3. 同 video 内连续 60 帧序列做 anchor + positive window（Loss 1 时序）

实施：自定义 `VeaticSeqSampler`，batch_size=64 时切成 16 video × 4 frames × 1 persona_group。

**Step 2/3**：标准 random batch（弱监督 readout 不需要时序结构）。

### 4.4 跨域风险与缓解

| 风险 | 现象 | 缓解 |
|------|------|------|
| 西方人 → 亚洲人 AU 分布偏移 | VEATIC/EMOTIC 西方人为主，川哥亚洲人 baseline 差异（py-feat 实测 AU01 偏高 0.85 / AU04 衰减）| **AUCaliper 校准层** 用 base v1 训练时方案（lnn-first.md §4.1）—— 推理时也用，Step 3 加入校准的合成 calibration 样本 |
| VEATIC viewer ratings 标注稀疏 | 不是每帧都有 V-A label | Loss 1 不需要 label（self-supervised），Loss 3 用 viewer_id 即可，不影响 |
| EMOTIC 静态图无时序 | Loss 1 无法用 | 只用 EMOTIC 跑 Loss 4 Head 2，不参与 Step 1 |

---

## 5. 训练 Schedule

### 5.1 Step 1 — VEATIC Self-supervised 预训练（5/27-5/30，4 天）

| 项 | 值 |
|----|-----|
| 数据 | VEATIC 全量约 52w 帧 |
| batch_size | 64（16 video × 4 frame group）|
| epoch 数 | 10 |
| 优化器 | AdamW (lr=3e-4, weight_decay=1e-4) |
| lr schedule | Cosine annealing 10 epoch warm 1 epoch |
| Loss | 1.0 * L_temporal + 0.3 * L_persona |
| backbone | trainable |
| readout | frozen (随机初始化保留，不训)  |
| 总时间预估 | ~17 min/epoch × 10 = ~3 hours/run × 多次调参（4 天预算）|
| ckpt 频率 | 每 epoch end + 每 1000 step（防训练中断）|

**期望产出**：`step1_backbone.pt`（仅 backbone + persona embedding）

### 5.2 Step 2 — EMOTIC Readout V-A 训练（5/31-6/2，3 天）

| 项 | 值 |
|----|-----|
| 数据 | EMOTIC 约 23K 标注图 |
| batch_size | 32 |
| epoch 数 | 30 |
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| lr schedule | Step decay：每 10 epoch ÷ 10 |
| Loss | 0.5 * L_readout（只 Head 2 V-A CCC + Head 4 conf）|
| backbone | **frozen**（assert 参数不变）|
| readout | trainable (Head 2 + Head 4) |
| 总时间预估 | ~5 min/epoch × 30 = ~2.5 hours/run |

**期望产出**：`step2_readout_va.pt`（merge backbone + readout Head 2/4）

### 5.3 Step 3 — mst Readout 8-state + intent 训练（6/3-6/6，4 天）

| 项 | 值 |
|----|-----|
| 数据 | mst ~30K 标注图 + LLM 离线打 intent_type label 子集 |
| batch_size | 32 |
| epoch 数 | 30 |
| Loss | 0.5 * L_readout（Head 1 8-state CE + Head 3 intent CE）|
| backbone | frozen |
| readout | trainable (Head 1 + Head 3) |

**期望产出**：`step3_readout_full.pt`（merge backbone + 全 readout）

### 5.4 Step 4 — 评估 + 调参 + 导出（6/7-6/10，4 天）

- 跑 §6 全部评估指标
- 必要时回到 Step 1/2/3 调超参
- 最终导出 `base_v3.pt`（≤ 5MB）+ ONNX/CoreML 兼容性测试

### 5.5 训练崩溃恢复（Studio nohup + supervisor）

```bash
# 启动命令模板
nohup python scripts/train_base_v3_step1.py \
  --resume-from latest \
  --ckpt-dir ~/lnmem-ckpts/base_v3_step1/ \
  --log /tmp/base_v3_step1.log \
  > /tmp/base_v3_step1_nohup.out 2>&1 &
```

- ckpt 每 1000 step 自动落盘（防 64GB unified memory OOM）
- supervisor 用 `pgrep -f train_base_v3` 守护 + auto-resume
- 每天 23:00 cron rsync ckpt 到 Air 备份（Studio 单点风险）

---

## 6. 评估指标（LNN-FIRST 守则）

### 6.1 主指标（必须看，禁止用 accuracy 替代）

#### M1: KG 召回质量（top）

**计算方式**：

1. 从 graphiti KG 取 100 个已标注 event（带 text query 和 ground-truth 关联 event ID）
2. 用 base v3 forward 出每个 event 的 64-dim hidden_repr
3. 用 query event 的 hidden 做 cosine 相似度检索 KG
4. 计算 Top-K hit rate（K=5, 10, 20）

**期望阈值**：
- Top-5 hit ≥ 40%
- Top-10 hit ≥ 60%
- 对比基线：base v1 用 V-A 2 维 直接当 query 的 Top-5 估 < 20%

**失败定义**：Top-5 < base v1 → 整个 v3 路径失败，回退

#### M2: 跨样本 hidden 聚类清晰度（top）

**计算方式**：

1. 对 mst 8-class 测试集 forward 出 hidden state
2. 计算 silhouette score（按 ground-truth label 分簇）
3. t-SNE 降到 2D 视觉检查

**期望阈值**：silhouette ≥ 0.25（base v2 应 < 0.05，因为 collapse）

**失败定义**：silhouette < 0.10 → 表征空间 collapse

#### M3: 时序一致性（top）

**计算方式**：

对 VEATIC val 集每 100 个 video 段：
- `dist_near = mean(||h_t - h_{t+30}||)` (1 秒内)
- `dist_far  = mean(||h_t - h_{t+900}||)` (30 秒外)
- ratio = dist_far / dist_near

**期望阈值**：ratio ≥ 1.5

**失败定义**：ratio < 1.1 → 时序信息没学到

### 6.2 附属指标（次要，仅用于 sanity check）

| 指标 | 计算 | 期望阈值 |
|------|------|---------|
| Head 1 8-state acc (mst val) | top-1 accuracy | ≥ 35%（base v2 是 48% 但是 collapsed，v3 真实学习预期低但不 collapse）|
| Head 2 V-A CCC (EMOTIC val) | Concordance Correlation Coefficient | V ≥ 0.25, A ≥ 0.15 |
| Head 3 intent acc | top-1 | ≥ 50% |
| Head 4 confidence calibration | ECE (Expected Calibration Error) | ≤ 0.15 |

**Why（附属指标不能当主指标）**：v2 已证明 supervised acc 0.48 但川哥真人 64/64 mismatch，acc 与真实可用性脱钩。v3 主指标必须看「表征质量」，acc 只防训崩。

### 6.3 LNN-FIRST 合规检查

每 epoch end 跑：

- [ ] `assert backbone trainable in Step 1, frozen in Step 2/3`
- [ ] `assert h_t std > 0.05` (不 collapse)
- [ ] `assert ratio(far/near) > 1.0` (时序信号存在)
- [ ] `assert persona embedding 多样性 > baseline` (Loss 3 生效)

---

## 7. Studio 训练环境

### 7.1 硬件 / 软件

| 项 | 配置 |
|----|------|
| 机器 | Mac Studio M2 Max 64GB unified memory，38-core GPU |
| 主框架 | PyTorch 2.1+（ncps CfC 官方实现是 PyTorch）|
| 备选 | MLX（实验性，CfC 端口可能不全，仅做对比）|
| Python | 3.9（与 Air poc-fresh venv 对齐）|
| 关键依赖 | torch, ncps>=1.0, pandas, numpy<1.24, opencv<4.10（py-feat 兼容）, scikit-learn (silhouette), umap-learn (t-SNE 备选) |
| 数据存储 | `/Users/fu-121/lnmem-data/` 9.8GB |
| ckpt 存储 | `~/lnmem-ckpts/base_v3_*/` |

### 7.2 训练守护

- **nohup + supervisor**：见 §5.5
- **ckpt resume**：torch.save 含 `model_state_dict + optimizer_state_dict + step + epoch + best_metric`
- **OOM 防御**：batch_size 自适应降级（OOM 时自动 ÷ 2 重试，最多 3 次）
- **进度可视化**：每 1000 step 写一行到 `~/lnmem-ckpts/base_v3_*/progress.jsonl`，包含 loss / lr / hidden std / 时序 ratio
- **每天 23:00 cron rsync** 到 Air `/Users/rayman.chen/lnmem-ckpts-backup/`

### 7.3 失败兜底

- Studio 死机 → Air 拉最近 ckpt 接着跑（但 Air 显存只够 batch=16，会慢 4×）
- VEATIC 数据损坏 → 单 video 跳过不停训
- ncps 库更新破坏 API → 锁版本 `ncps==1.0.7`，requirements.txt 强制

---

## 8. 实施任务拆解（Phase γ 任务清单）

| 任务 ID | 内容 | Owner | 依赖 | 估时 |
|---------|------|-------|------|------|
| γ-1 | 数据 pipeline spike：用 tiny VEATIC 子集（10 video, 1w 帧）跑通端到端 | cc | ADR-018 schema | 1 天 |
| γ-2 | 模型架构 PyTorch 实现（fusion + CfC backbone + 5 head + FiLM/τ-bias）| codex 5.4 high | γ-1 数据格式确定 | 1.5 天 |
| γ-3 | Loss 1 (Temporal Contrast) 实现 + tiny 训练（10 video，验证不 collapse）| cc + codex 对抗审 | γ-2 | 1 天 |
| γ-4 | Step 1 全量预训练（VEATIC 52w 帧）| cc 启动 + 守护 | γ-3 | 4 天 |
| γ-5 | Step 2 + Step 3 readout 弱监督训练 | codex 5.4 high 实施 | γ-4 step1 ckpt | 3 天 |
| γ-6 | 评估脚本（M1 KG 召回 / M2 silhouette / M3 时序 ratio） | cc | γ-4 step1 ckpt（M2/M3 可先跑）| 2 天 |
| γ-7 | ckpt 导出 + 序列化 + Air demo 接入 / daemon 切换 base v3 | cc | γ-4/γ-5 final ckpt + γ-6 评估通过 | 2 天 |

**总工时**：14.5 天，与 Phase γ 5/27-6/10 的 15 天预算贴合。

**并行可能**：γ-2 / γ-6（评估脚本）可与 γ-3 并行；γ-1 完成后 γ-2 立即启动；γ-6 不必等 γ-5 完成（M2/M3 用 step1 ckpt 已可跑）。

---

## 9. 风险登记

继承 ADR-016 §风险登记 5 项，追加实施细节风险：

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| 1（继承）| self-supervised 表征「看起来学到了」但下游 KG 召回 0 | 中 | M1 KG 召回硬指标 + 用 base v1 V-A 当 baseline 对比 |
| 2（继承）| 多模态融合数据不足 | 高 | Phase γ Loss 2 weight=0，留 M1+ 加 |
| 3（继承）| VEATIC 跨域到川哥真人 | 中 | Phase δ 微调 + AUCaliper |
| 4（继承）| 多 head readout 不稳定 | 中 | 每 head 独立 lr + 严格 frozen backbone 哨兵 |
| 5（继承）| cc/codex 又用 LLM 思路评 v3 | 高 | 评估脚本默认输出 M1/M2/M3，accuracy 仅放附属表 |
| 6（新增）| Loss 1 mode collapse（hidden 全收敛到一点）| 中 | 监控 `std(h) < 0.05` 自动停训 + 加 BarlowTwins 风格的去相关正则项备用 |
| 7（新增）| Persona Loss 3 学到 ID 而非 persona drift | 中 | Batch sampler 跨视频 + 同 batch 不同 persona × 不同时间 |
| 8（新增）| `requires_grad=False` 漏挂 backbone 被意外更新 | 中 | 训练循环开头/结尾各跑一次 `assert_backbone_frozen()` 哨兵 |
| 9（新增）| Studio 训练中断 ckpt 损坏 | 低 | 双 ckpt 轮替（last_a.pt / last_b.pt 交替写）+ rsync 到 Air |
| 10（新增）| ncps CfC 库 API 不稳定/更新破坏 | 中 | 锁版本 `ncps==1.0.7`，写自适配 wrapper 隔离 |
| 11（新增）| t-SNE/silhouette 结果不稳定（随机性高）| 低 | 评估时固定 seed + 跑 5 次取中位数 |
| 12（新增）| EMOTIC 标注质量（V-A 不均衡 + 多人 bbox 混淆）| 中 | Step 2 数据清洗：过滤 multi-person 帧 + 平衡 V-A 象限采样 |

---

## 10. 不可妥协 6 项检查（v0.3.3 §10.1）

| # | 不可妥协项 | base v3 落实状态 |
|---|-----------|-----------------|
| 1 | 万能语法 20 槽 | ✅ 作为 input 一部分（+ ADR-018 intent_type 5 槽扩展）|
| 2 | 多时间尺度 LNN 心智核心 | ✅ **强化**：4 层 CfC，每层 τ 跨度 0.1s ~ 600s，本 spec 直接落地此项 |
| 3 | CLS 双系统离线 consolidation | ⏳ 不在 Phase γ 范围（M2+）|
| 4 | per-user 个性化栈 | ✅ Persona embedding + FiLM + τ-bias 三件套 backbone 准备好（Phase δ 启动真人微调）|
| 5 | 居家+车载双场景统一架构 | ⏳ M2+ |
| 6 | 记忆 UI 可见可治理 | ⏳ 砖 3 已在做，与 base v3 解耦 |

**Why 检查这个**：CLAUDE.md §5 「红线（强制停下）」第 1 条：任何对 §10.1 六项的妥协建议必须停下问老板。本 spec 不妥协任何一项，#3/#5/#6 是「不在本阶段范围」不是「妥协」。

---

## 11. 验收门（Phase γ 完成定义）

Phase γ 视为完成，须同时满足：

- [ ] M1 KG 召回 Top-5 ≥ 40%
- [ ] M2 silhouette ≥ 0.25
- [ ] M3 时序 ratio ≥ 1.5
- [ ] ckpt ≤ 5MB
- [ ] 单帧推理延迟 ≤ 15ms (CPU)
- [ ] LNN-FIRST 合规检查 4 项 全 ✅
- [ ] codex 对抗审查 PR 通过
- [ ] 川哥签字 + ADR-019 收尾文档落盘

**未通过任一**：
- M1/M2/M3 任一不达标 → 不导出 `base_v3.pt`，回到 Step 1 调超参（不上线）
- LNN-FIRST 合规失败 → 立即停训 + 写 root cause ADR
- 推理延迟超标 → 模型压缩（参数量减半或量化 INT8）

---

## Summary

ADR-016 base v3 实施 spec v0.1 落盘。本 spec 覆盖 Phase γ（5/27-6/10）工程细化：4 层多时间尺度 CfC backbone（参数量 ~30K，ckpt ≤5MB） + 5 head readout（state/V-A/intent/conf/hidden_repr） + 4 个 loss（Temporal Contrast 主线 weight=1.0，Persona Contrast 0.3，Readout 0.5，Crossmodal 本阶段 weight=0）。3 步 schedule：VEATIC 自监督预训练（4 天）→ EMOTIC readout V-A（3 天）→ mst readout 8-state+intent（4 天）→ 评估调参 4 天。评估指标禁 accuracy，主线看 KG 召回 Top-5 ≥ 40%、hidden 聚类 silhouette ≥ 0.25、时序一致性 ratio ≥ 1.5。12 项风险登记 + 11 项验收门。等川哥过目 + codex 对抗审查。

---

**v0.1 起草：cc（Claude Opus 4.7）| 2026-05-21 | 等川哥拍板**
