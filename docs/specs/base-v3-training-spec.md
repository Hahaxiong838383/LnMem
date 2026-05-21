---
version: v0.2.4
audience: cc / codex / 工程师 / 川哥
purpose: ADR-016 base v3 multimodal self-supervised LNN 的工程实施细化 spec
status: 川哥钦定（2026-05-21 v0.2.4，吸收 codex 5.4 + gemini 3.5 双评审修订）
last_updated: 2026-05-21
changelog:
  - v0.1（2026-05-21 起草）: 初稿，Agent 后台写
  - v0.2（2026-05-21 修订）: 吸收双评审 → Loss 3 默认关 / 评估三联替 silhouette / 2 层先跑通 / collapse 四联监控 / 10-video preflight 门控
  - v0.2.4（2026-05-21 晚 修订）: 吸收 codex+gemini 双 REVISE → M1 Visual-Only 子指标 / M5 三层梯度 / Loss3 关闭态 N/A 标注 / ncps wrapper τ 真落地 γ-2 验证项 / Studio 14d 磁盘预算表
related:
  - ../decisions/016-base-v3-multimodal-self-supervised-lnn.md
  - ../decisions/013-base-v2-failure-and-finetune-plan.md
  - ../decisions/018-eventtensor-schema-v1_1-intent-type-multimodal-context.md
  - ../decisions/019-spec-v0_2-revision-from-academic-review.md
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
4. **Frozen-feature Linear Probe 表征质量**（v0.2.1 取代 silhouette 主指标）：M4 EMOTIC V-A CCC + mst Linear acc 优于 base v2；视频 ID Shortcut 反测 acc < 30%（M5）
5. ckpt **≤ 20MB**（v0.2.1 实测修正，v0.1 5MB 不实），单帧推理延迟 ≤ 15ms（Studio M2 Max CPU 路径，daemon 兼容）

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
    - （v0.2.4 加）γ-2 必做「τ-bias 真进 ODE」验证：若 ncps 高层 API 不暴露 τ 注入通道，则改为手写 CfCCell 单步隐式欧拉实现

[CfC Backbone] (ncps 库的 CfC，v0.2 改 2 层先跑通 / 4 层等 γ-2 实测后升)
  Layer 1: τ ∈ [0.1s, 10s]   — 快速表情 + 情绪窗口（v0.2 合并 v0.1 层 1+2）
  Layer 2: τ ∈ [10s, 600s]   — 短期 mood + 长时段落级（v0.2 合并 v0.1 层 3+4）
  # 4 层 hierarchical 实证 ckpt ≤20MB 且训练稳定后才升（γ-2 实测决定）

  ↓ 64-dim hidden state h_t (主输出)

[Readout Heads] (附属，弱监督，backbone 冻结时才训)
  Head 1 (state_8)  : Linear 64 → 8, softmax     — 8-state emotion
  Head 2 (va_cont)  : Linear 64 → 2, tanh        — V-A 连续回归
  Head 3 (intent)   : Linear 64 → 5, softmax     — intent type
  Head 4 (conf)     : Linear 64 → 1, sigmoid     — confidence
  Head 5 (hidden_repr): Identity 64 → 64         — 直出给 KG
```

### 2.2 各层细节

| 层 | 模块 | input dim | output dim | 参数量预估（v0.1 拍脑袋）| 实测（v0.2，codex 核验）|
|----|------|-----------|------------|--------------------------|------------------------|
| Input projection | nn.Linear + LayerNorm + GELU | 52 | 32（α 试跑）/64（β 全量） | 3.4K | 待 γ-1 spike 落实 |
| Persona embedding | nn.Embedding(num_personas=64, dim=16) | — | 16 | 1K | 1K（不依赖框架）|
| FiLM head | MLP 16 → 2×fusion_dim | 16 | 128 | 4K | 待实测 |
| **CfC stack（2 层）** | ncps.torch.CfC | 64 | 64 | ~10K | **≈ 100K** ⚠️（codex 实测单层 ~50K，spec v0.1 严重低估）|
| Head 1 state_8 | Linear | 64 | 8 | 0.5K | 0.5K |
| Head 2 va_cont | Linear | 64 | 2 | 0.13K | 0.13K |
| Head 3 intent | Linear | 64 | 5 | 0.33K | 0.33K |
| Head 4 conf | Linear | 64 | 1 | 0.07K | 0.07K |
| Head 5 hidden_repr | Identity | 64 | 64 | 0 | 0 |

**v0.2 实测修正**（codex 5.4 评审本地实证）：
- 单层 CfC(64,64) 实测 **57,736 参数**（不是 v0.1 拍脑袋的 5K）
- 2 层 stack 估 **~100K 参数**
- 4 层 stack 估 **~198K 参数**

**总参数量预估（v0.2）：≈ 110-120K**（2 层 CfC + 其他）。ckpt 序列化估 **≤ 20MB**（FP32 权重 + persona embedding 表 + 元信息）。

**ckpt 上限调整**：v0.1 拍脑袋 5MB → v0.2 改 **20MB**。如果 γ-2 实测 ckpt > 20MB → 用 INT8 量化或 LoRA 减参数。daemon 侧 ckpt 加载延迟在 20MB 内仍可接受（M2 Max NVMe < 200ms）。

### 2.3 Studio M2 Max 推理延迟预估

- CPU 路径（daemon 兼容）：单帧 forward 估 8-12ms（v0.2.1 改 **2 层 CfC** + 5 head readout，4 层 γ-2 实测后再决定升）
- MPS 路径（训练）：batch=64 单步 forward 估 25-40ms
- 训练吞吐目标：≥ 500 frames/sec（VEATIC 52w 帧约 17 min/epoch）

**Why（参数量压缩比 v2 小）**：base v2 用 9-state Russell margin 训练，readout 重，引入大量冗余。v3 backbone 主线学表征，readout 是附属轻量 head，单 head 0.07-0.5K 参数。

---

## 3. 4 个 Loss 函数

### 3.1 Loss 1 — Temporal Contrast（主目标，weight=1.0）

**目标**：同一视频内时间相近的 hidden state 互相接近，时间远离的远离。让 backbone 学到「时序连续性」这个第一性原理。

**形式**：InfoNCE（v0.2 加 VICReg 双路 spike 备选 · 见 §3.1.bis）

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
- Mode collapse：所有 hidden 都收敛到同一点 → 升级为**四联监控**（v0.2，见 §6.3）：std + 协方差秩 + per-dim variance + 输出能量
- 学到 ID 而非语义：t-SNE 按视频 ID 分簇而非按语义分簇 → 加 batch-level video shuffle + 混合多视频 batch
- **视频 ID Shortcut Learning（v0.2 加，Geirhos 2020 Nature MI 警告）**：模型可能匹配背景/光影/相机噪声而非用户连续表征 → 加**视频 ID 反测**（§6.1 **M5**）：把 video_id 当 label 跑 linear probe，acc > 50% → 学到 ID 不是表征 → 加强 augmentation 或换 SSL 目标

### 3.1.bis VICReg 备选路径（v0.2 加，gemini + codex 双评审建议）

**Why 加备选**：
- Gemini 评审（D1 学术 B 分）：InfoNCE 在连续动力学有「假阴性惩罚」→ 撕裂相空间轨迹
- Codex 评审（学术 6.5/10）：纯 InfoNCE 缺 anti-collapse 辅助，2024 SOTA 倾向 VICReg/V-JEPA

**形式**：VICReg (Bardes et al. ICLR 2022, arXiv:2105.04906)

```
L_vicreg = λ_v * Variance(h) + λ_i * Invariance(h, h') + λ_c * Covariance(h)
  Variance:    max(0, γ - std(h_i, dim=batch)) for each i  (γ=1)
  Invariance:  ||h - h'||² where h' = h at t+δ
  Covariance:  Σ_{i≠j} C_ij² / d                          (C=cov matrix)
  λ_v=25, λ_i=25, λ_c=1（VICReg 论文默认）
```

**spike 双路**（γ-1 必跑）：
- Path A: 纯 InfoNCE（v0.1 默认）
- Path B: 纯 VICReg
- Path C: InfoNCE + VICReg 联合（weight 0.5/0.5）

**γ-1 末 5/26 决策**：哪条 path 在 10-video preflight 上 std(h) 稳定 + linear probe acc 高 → 锁定 Path 上 Step 1 全量。

**Why 不直接全量切 VICReg**：v0.1 InfoNCE 范式已熟 + 工程实现少坑 + VICReg 在时序 LNN 上尚无大规模实证（gemini 引 V-JEPA 也未在 ncps CfC 上验证过）。先 spike 不押注。

### 3.2 Loss 2 — Cross-modal Contrast（M1+，weight=0 本阶段）

**目标**：同一时间窗的视觉/文本/音频 hidden 互相接近。

**本阶段不实施**：Phase γ 没有同步的多模态对齐数据（VEATIC 是纯视频 + ratings，没文本不没音频片段对齐）。Phase γ weight=0，留接口和注释占位。

### 3.3 Loss 3 — Persona Contrast（v0.2 默认关 weight=0，重命名 Observer-style Regularizer）

**🚨 v0.2 关键修订**（codex 5.4 评审打脸 + gemini D1 学术批评双重证据）：

**v0.1 错误**：把 VEATIC 的 `viewer_id` 当 user persona。但 **VEATIC viewer 是给视频打分的标注员（observer），不是被拍的人（subject）**——viewer 的"persona"只反映标注偏好（如打 V-A 偏高/偏低），不是 LnMem 想学的「主人多模态连续表征」。继续用 viewer 当 persona 会让 Loss 3 把"标注员偏好"硬塞进 hidden state，污染表征。

**学术补充**（gemini）：SupCon 假设离散类别，但 LnMem 的 persona 是**连续演化非平稳的**（用户情绪、认知疲劳是 transient state）。强行 SupCon 拉近会抹杀 LNN 捕获情绪波动的能力，违反 P-1。

**v0.2 决策**：
- **默认 weight=0**，本阶段不启用
- 重命名为 **Observer-style Regularizer**（如果未来要启用，明文限定到 VEATIC_Psych 0-82 标注员，作为正则项而不是 persona 学习信号）
- **真 persona 学习留 Phase δ**（6/11+ 川哥真人 daemon 60w 帧数据，那才是真的 per-user 信号）

**Phase δ 真 persona 路径**（产品定义 §10.1 #4 落地点）：
- 川哥真人 daemon 累积事件 → persona embedding online learning
- τ-bias 三件套 + FiLM 启动
- 不用对比学习，用 daemon 反馈直接监督（已在 daemon V0.3 Phase B 跑通）

**v0.1 SupCon 公式段保留（封存）**：作为参考，未来连续 persona 真有信号时可参考改造（如改用 SoftCLR / DynCL 等连续对比方法）。

**形式（保留，但 weight=0）**：SupCon (Khosla et al. 2020)

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
- 同 batch 内 persona 多样性约束：≥ 8 个不同 persona_idx（DataLoader 自定义 sampler 保证）**[v0.2.4 加：N/A when Loss3 weight=0]**

**启用条件**：v0.2 起**默认关 weight=0**（参见上方 v0.2 关键修订段）。Step 1 不启用，本段公式为封存参考。

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

### 3.5 三 loss 联合训练流程（v0.2 更新 weight）

```python
# Pseudo-code（v0.2 修订，Loss 3 默认关）
total_loss = (
  1.0 * L_temporal +          # 主目标（InfoNCE 或 VICReg，γ-1 决策）
  0.0 * L_crossmodal +        # M1+
  0.0 * L_persona_observer +  # v0.2 默认关（VEATIC viewer≠user persona，留 Phase δ）
  0.5 * L_readout             # Step 2/3 才启用 + backbone frozen
)
```

**v0.2 简化**：γ Step 1 只跑 Loss 1（temporal contrast 单一强 SSL），减少梯度干扰风险（gemini D1 警告：4 loss 联合 → ODE 求解器刚性爆炸）。Step 2/3 backbone 冻结后再加 Loss 4 readout。

**Why（loss weight 选这组）**：Loss 1 是主目标必须 dominant，Loss 3 防 backbone 忽略 persona 信息（v2 失败教训之一），Loss 4 只用 0.5 因为它走 detach 路径不能反传 backbone，但权重不能太低否则 readout 收敛慢。

---

## 4. 数据 Pipeline

### 4.1 三数据集状态与路径

| 数据 | 大小 | Studio 路径 | 当前预处理状态 | Phase γ 用途 |
|------|------|------------|--------------|-------------|
| VEATIC | 6 GB | `/Users/fu-121/lnmem-data/veatic/` | 部分帧已抽 AU parquet（ADR-007 Phase 1.2），剩余约 30% 未抽 | Step 1 主：自监督预训练（**v0.2.1 仅 Loss 1 Temporal Contrast**，Loss 3 默认关）|
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

**Step 1（v0.2.1 仅 Loss 1 Temporal Contrast，Loss 3 默认关）必须保证**：

1. ~~每个 batch 内 ≥ 8 个不同 persona_idx（Loss 3 多样性）~~ — **v0.2.1 Loss 3 关闭，本约束删除**
2. 每个 batch 内 **≥ 16 个不同 video_id**（Loss 1 跨视频 negative）
3. 同 video 内连续 60 帧序列做 anchor + positive window（Loss 1 时序）
4. **v0.2.1 加**：train/val 按 `video_id` 严格隔离（preflight §11.1 第 9 项防数据泄漏要求）

实施：自定义 `VeaticSeqSampler`，batch_size=64 时切成 16 video × 4 frames × 1 persona_group。

**Step 2/3**：标准 random batch（弱监督 readout 不需要时序结构）。

### 4.4 跨域风险与缓解

| 风险 | 现象 | 缓解 |
|------|------|------|
| 西方人 → 亚洲人 AU 分布偏移 | VEATIC/EMOTIC 西方人为主，川哥亚洲人 baseline 差异（py-feat 实测 AU01 偏高 0.85 / AU04 衰减）| **AUCaliper 校准层** 用 base v1 训练时方案（lnn-first.md §4.1）—— 推理时也用，Step 3 加入校准的合成 calibration 样本 |
| VEATIC viewer ratings 标注稀疏 | 不是每帧都有 V-A label | Loss 1 不需要 label（self-supervised），不影响。**v0.2.1 修订**：Loss 3 已默认关，viewer_id 不再当 persona 信号 |
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
| Loss | **1.0 * L_temporal**（v0.2.1 单一强 SSL，L_persona 默认关 weight=0）|
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
- 最终导出 `base_v3.pt`（**≤ 20MB**，v0.2.1 修正）+ ONNX/CoreML 兼容性测试

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

## 6. 评估指标（LNN-FIRST 守则 · v0.2 大幅重写）

**v0.2 关键修订**（吸收 codex + gemini 双评审 D2 评分 C+ 批评）：

| v0.1 主指标 | v0.2 修订 |
|------------|----------|
| ❌ Silhouette score 当主指标 | ⚠️ 移到 §6.2 附属 sanity（SSL 阶段 invalid，gemini 警告：高 silhouette 反而证明 collapse）|
| ❌ KG 召回 / 时序 ratio | ✅ 保留 + 加 **M4 Linear Probe 表征质量** + **M5 视频 ID 反测** |

### 6.1 主指标（v0.2 三联 · 必须看，禁止用 accuracy 替代）

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

**M1-V Visual-Only 压力测试（v0.2.4 加，防文本捷径）**：

- 评估时将 25d 文本特征（20 槽 + 5 intent）强制置零，仅保留 24d 视觉 AU + 2d 时间戳 + 6d 音频占位
- 子指标阈值：Top-5 hit ≥ 20%
- 失败定义：M1-V < 20% → 判定多模态融合无效，**整个 v3 路径失败回退**

#### M3: 时序一致性（top）

**计算方式**：

对 VEATIC val 集每 100 个 video 段：
- `dist_near = mean(||h_t - h_{t+30}||)` (1 秒内)
- `dist_far  = mean(||h_t - h_{t+900}||)` (30 秒外)
- ratio = dist_far / dist_near

**期望阈值**：ratio ≥ 1.5

**失败定义**：ratio < 1.1 → 时序信息没学到

**v0.2 加防护**：监控 `||h_t||` 是否常数（防模型输出 0 vector game ratio 指标 — gemini D2 警告）

#### M4: Frozen-Feature Linear Probe（v0.2 加，codex 强烈建议）

**计算方式**：

1. base v3 backbone 冻结，forward 出 64-dim hidden
2. 在 EMOTIC val 上训一个 Linear classifier（V-A 回归）和 mst val 上训 8-state 分类
3. 报告 **CCC（V-A）+ acc（8-state）**

**期望阈值**：
- EMOTIC V-A CCC: V ≥ 0.30 / A ≥ 0.20
- mst 8-state Linear Probe acc ≥ 40%

**Why 这个比 silhouette 靠谱**（codex 评审）：Linear probe 是 SSL 表征评估的金标准（He et al. MAE CVPR 2022 用的就是这个）。Silhouette 在连续 LNN 上反向激励 collapse；Linear probe 直接验证「表征是否对下游任务可分」。M4 是表征可分性主指标，**视频 ID Shortcut 反测见 M5**（下方）。

#### M5: 视频 ID Shortcut 反测（v0.2 加，gemini D2 警告）

**计算方式**：

1. 把 VEATIC 的 `video_id` 当 label 跑 Linear probe（不是 emotion）
2. 看 backbone hidden 能否高准确率分出 video_id

**期望阈值（v0.2.4 加：三层梯度统一）**：
- `green`: acc < 30%（pass）
- `yellow`: 30% ≤ acc ≤ 50%（强制做 1 次 augmentation 重跑或切换 SSL 目标后复测）
- `red`: acc > 50%（reject Phase γ）

**门控规则（v0.2.4 加）**：
- preflight 允许 `green/yellow`，但 `yellow` 不得直接放行到 final
- final 必须 `green`

### 6.2 附属指标（次要，仅用于 sanity check · v0.2 silhouette 降级到这）

| 指标 | 计算 | 期望阈值 | v0.2 备注 |
|------|------|---------|----------|
| **Silhouette score（mst val）** | sklearn.metrics.silhouette_score | ≥ 0.10 | ⚠️ 仅 sanity，**禁止当主指标**（gemini D2 警告：连续 LNN 上高分反而证明 collapse）|
| **t-SNE 视觉检查（mst val）** | sklearn TSNE + matplotlib | 8 类视觉分离 | 仅人眼定性 sanity |
| Head 1 8-state acc (mst val) | top-1 accuracy | ≥ 35%（base v2 是 48% 但是 collapsed，v3 真实学习预期低但不 collapse）| |
| Head 2 V-A CCC (EMOTIC val) | Concordance Correlation Coefficient | V ≥ 0.25, A ≥ 0.15 | |
| Head 3 intent acc | top-1 | ≥ 50% | 依赖 Phase β intent_type LLM 离线打标 |
| Head 4 confidence calibration | ECE (Expected Calibration Error) | ≤ 0.15 | |

**Why（附属指标不能当主指标）**：v2 已证明 supervised acc 0.48 但川哥真人 64/64 mismatch，acc 与真实可用性脱钩。v3 主指标必须看「表征质量」，acc 只防训崩。

### 6.3 LNN-FIRST 合规检查（v0.2 升级为四联 collapse 监控）

**每 epoch end + 每 200 step**（v0.2 增加频率，codex 强烈建议）：

四联 collapse 监控（取代 v0.1 的单 std）：

- [ ] **std**: `assert h_t.std() > 0.05`（不全 collapse 到一点）
- [ ] **协方差秩**: `assert torch.linalg.matrix_rank(cov(h_t), tol=1e-4) >= 32`（hidden state 至少有 32 个独立方向，64-dim 一半）
- [ ] **per-dim variance**: `assert (h_t.var(dim=0) > 0.01).sum() >= 40`（≥ 40 维有有效信号，防 60+ 维死掉）
- [ ] **输出能量**: `assert h_t.norm(dim=-1).mean() in [0.5, 50.0]`（防输出全 0 或爆炸）

其他守门检查（每 epoch）：

- [ ] `assert backbone trainable in Step 1, frozen in Step 2/3`
- [ ] `assert ratio(far/near) > 1.0` (时序信号存在)
- [ ] `assert video_id_probe_acc < 0.50` (无视频 ID shortcut)

**任一不过**：自动停训 + 报警 + 落盘当前 ckpt 给 cc 排查。

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

**14 天最坏磁盘容量预算（v0.2.4 加）**：

| 占用项 | 单次 | 14 天累 | 备注 |
|---|---|---|---|
| ckpt 每 1000 step 存 20MB | 20MB | ~30GB | 14d × Step1-4 各 100 ckpt |
| TensorBoard / wandb 日志 | — | ~5GB | 每天 0.4GB |
| 数据集（已落盘） | — | 10GB | 不变 |
| 临时 cache (pip / torch.hub) | — | ~10GB | 安装 + NCPS + Python wheel |
| 实验中间产物（embedding cache, 特征图） | — | ~20GB | 每轮调参累积 |
| **14d 累积下限** | — | **≥75GB** | — |
| **安全边际** | — | **+75GB** | 应急 + 双备份 |
| **γ-0 必清磁盘到 free ≥ 150GB** | — | — | — |

### 7.2 训练守护

- **nohup + supervisor**：见 §5.5
- **ckpt resume**：torch.save 含 `model_state_dict + optimizer_state_dict + step + epoch + best_metric`
- **OOM 防御**：batch_size 自适应降级（OOM 时自动 ÷ 2 重试，最多 3 次）
- **进度可视化**：每 1000 step 写一行到 `~/lnmem-ckpts/base_v3_*/progress.jsonl`，包含 loss / lr / hidden std / 时序 ratio
- **每天 23:00 cron rsync** 到 Air `/Users/rayman.chen/lnmem-ckpts-backup/`
- **磁盘哨兵（v0.2.4 加）**：训练脚本每 5 分钟检测可用空间，`free < 10GB` 自动暂停训练并报警

### 7.3 失败兜底

- Studio 死机 → Air 拉最近 ckpt 接着跑（但 Air 显存只够 batch=16，会慢 4×）
- VEATIC 数据损坏 → 单 video 跳过不停训
- ncps 库更新破坏 API → 锁版本 `ncps==1.0.1`（v0.2.4 实测纠正：PyPI 最高版本是 1.0.1 而非 1.0.7；available versions: 1.0.1, 1.0.0, 0.0.7..0.0.1）→ requirements.txt 强制

---

## 8. 实施任务拆解（Phase γ 任务清单）

| 任务 ID | 内容 | Owner | 依赖 | 估时 |
|---------|------|-------|------|------|
| γ-1 | 数据 pipeline spike：用 tiny VEATIC 子集（10 video, 1w 帧）跑通端到端 | cc | ADR-018 schema | 1 天 |
| γ-2 | 模型架构 PyTorch 实现（fusion + CfC backbone + 5 head + FiLM/τ-bias）+ **v0.2.4 加：ncps τ 注入通道拆解验证（支持则 wrapper；不支持则手写 CfCCell 单步隐式欧拉）** | codex 5.4 high | γ-1 数据格式确定 | 1.5 天 |
| γ-3 | Loss 1 (Temporal Contrast) 实现 + tiny 训练（10 video，验证不 collapse）| cc + codex 对抗审 | γ-2 | 1 天 |
| γ-4 | Step 1 全量预训练（VEATIC 52w 帧）| cc 启动 + 守护 | γ-3 | 4 天 |
| γ-5 | Step 2 + Step 3 readout 弱监督训练 | codex 5.4 high 实施 | γ-4 step1 ckpt | 3 天 |
| γ-6 | 评估脚本（**v0.2.1 主指标三联：M1 KG 召回 / M3 时序 ratio / M4 Linear Probe + M5 视频 ID Shortcut 反测**；silhouette 仅作附属 sanity）| cc | γ-4 step1 ckpt（M3/M5 可先跑）| 2 天 |
| γ-7 | ckpt 导出 + 序列化 + Air demo 接入 / daemon 切换 base v3 | cc | γ-4/γ-5 final ckpt + γ-6 评估通过 | 2 天 |

**总工时**：14.5 天，与 Phase γ 5/27-6/10 的 15 天预算贴合。

**并行可能**：γ-2 / γ-6（评估脚本）可与 γ-3 并行；γ-1 完成后 γ-2 立即启动；γ-6 不必等 γ-5 完成（M3/M5 用 step1 ckpt 已可跑）。

---

## 9. 风险登记

继承 ADR-016 §风险登记 5 项，追加实施细节风险：

| # | 风险 | 概率 | 缓解 |
|---|------|------|------|
| 1（继承）| self-supervised 表征「看起来学到了」但下游 KG 召回 0 | 中 | M1 KG 召回硬指标 + 用 base v1 V-A 当 baseline 对比 |
| 2（继承）| 多模态融合数据不足 | 高 | Phase γ Loss 2 weight=0，留 M1+ 加 |
| 3（继承）| VEATIC 跨域到川哥真人 | 中 | Phase δ 微调 + AUCaliper |
| 4（继承）| 多 head readout 不稳定 | 中 | 每 head 独立 lr + 严格 frozen backbone 哨兵 |
| 5（继承）| cc/codex 又用 LLM 思路评 v3 | 高 | 评估脚本默认输出 **M1/M3/M4/M5 主指标三联**（v0.2.1 同步），accuracy 仅放附属表 |
| 6（新增）| Loss 1 mode collapse（hidden 全收敛到一点）| 中 | 监控 `std(h) < 0.05` 自动停训 + 加 BarlowTwins 风格的去相关正则项备用 |
| 7（新增）| Persona Loss 3 学到 ID 而非 persona drift | 中 | Batch sampler 跨视频 + 同 batch 不同 persona × 不同时间 **[v0.2.4 加：N/A when Loss3 weight=0，启用时风险才生效]** |
| 8（新增）| `requires_grad=False` 漏挂 backbone 被意外更新 | 中 | 训练循环开头/结尾各跑一次 `assert_backbone_frozen()` 哨兵 |
| 9（新增）| Studio 训练中断 ckpt 损坏 | 低 | 双 ckpt 轮替（last_a.pt / last_b.pt 交替写）+ rsync 到 Air |
| 10（新增）| ncps CfC 库 API 不稳定/更新破坏 | 中 | 锁版本 `ncps==1.0.1`（v0.2.4 实测纠正自 1.0.7），写自适配 wrapper 隔离 |
| 11（新增）| t-SNE/silhouette 结果不稳定（随机性高）| 低 | 评估时固定 seed + 跑 5 次取中位数 |
| 12（新增）| EMOTIC 标注质量（V-A 不均衡 + 多人 bbox 混淆）| 中 | Step 2 数据清洗：过滤 multi-person 帧 + 平衡 V-A 象限采样 |
| 13（v0.2.4 加）| τ-bias 未进入 ODE 内核，仅在 ncps 外层后处理（假 LNN-FIRST） | 高 | γ-2 必做 ncps 源码通道验证；不支持外部 τ 注入则手写 CfCCell 单步隐式欧拉；preflight #12 做显著性检验 |

---

## 10. 不可妥协 6 项检查（v0.3.3 §10.1）

| # | 不可妥协项 | base v3 落实状态 |
|---|-----------|-----------------|
| 1 | 万能语法 20 槽 | ✅ 作为 input 一部分（+ ADR-018 intent_type 5 槽扩展）|
| 2 | 多时间尺度 LNN 心智核心 | ✅ **强化**：v0.2.1 **2 层 CfC spike**（τ 0.1s~10s + 10s~600s），4 层 hierarchical 等 γ-2 实测后再升 |
| 3 | CLS 双系统离线 consolidation | ⏳ 不在 Phase γ 范围（M2+）|
| 4 | per-user 个性化栈 | ✅ Persona embedding + FiLM + τ-bias 三件套 backbone 准备好（Phase δ 启动真人微调）|
| 5 | 居家+车载双场景统一架构 | ⏳ M2+ |
| 6 | 记忆 UI 可见可治理 | ⏳ 砖 3 已在做，与 base v3 解耦 |

**Why 检查这个**：CLAUDE.md §5 「红线（强制停下）」第 1 条：任何对 §10.1 六项的妥协建议必须停下问老板。本 spec 不妥协任何一项，#3/#5/#6 是「不在本阶段范围」不是「妥协」。

---

## 11. 验收门（v0.2 重构：先 preflight 门控，再 Phase γ DoD）

### 11.1 Phase γ 启动门控（v0.2 加 · 5/26 必通过才能 5/27 启动 Step 1）

**10-video preflight checklist · v0.2.4 升级 12 项**（含 τ-bias 真落地验证）：

**基础 8 项**（v0.2 原版）：
- [ ] AU 抽取吞吐 ≥ 500 frame/sec（Studio M2 Max CPU）
- [ ] Sampler 能稳定取到 ±60 帧正样本 + |Δt|>30s 负样本
- [ ] **VEATIC viewer_id 实际去重数（必须验证）**（若 <8 → Loss 3 永久关闭）
- [ ] **EMOTIC vs VEATIC V-A 标注口径**（Bland-Altman 一致性分析）
- [ ] ncps CfC 实际参数量 + τ-bias 多尺度注入 spike OK（v0.2.4 加：必须给出 τ 注入路径证据）
- [ ] InfoNCE vs VICReg 双路 spike 跑 1 epoch，**四联监控全过**（std + 秩 + per-dim var + 能量，不只 std）
- [ ] 视频 ID Shortcut 反测 (M5) 达到 `green` 或 `yellow`（`green`: <30%，`yellow`: 30%-50%，`red`: >50%）
- [ ] Studio ssh 通 + nohup + ckpt 落盘双轮替（防训练中途崩）

**v0.2.1 新加 3 项**（codex 第二轮验收 §11.1.C 强建议）：
- [ ] **数据切分防泄漏**：按 `video_id` 严格 train/val 隔离（防 M4/M5 结果虚高）
- [ ] **四联 collapse 入 preflight 门控**（不只是训练时跑，preflight 阶段就锁定阈值）
- [ ] **双 seed 可复现性**：preflight 跑 2 次（seed=0/42），核心指标（std/Linear probe acc）波动 < ±15%
- [ ] **v0.2.4 加 #12：τ-bias 真进 ODE 验证**：参数化 `τ_i` 前后输出差异显著性 `p < 0.05`，且证据链包含 forward 注入点

**任一不通过**：
- 不启动 5/27 全量 Step 1
- 推迟到 5/29 或重写有问题部分

### 11.2 Phase γ 完成 DoD（v0.2 主指标三联）

Phase γ 视为完成，须同时满足：

- [ ] **M1 KG 召回 Top-5 ≥ 40%**（vs base v1 V-A baseline）
- [ ] **M1-V Visual-Only Top-5 ≥ 20%**（v0.2.4 加：25d 文本特征置零后仍需达标）
- [ ] **M3 时序一致性 ratio ≥ 1.5**
- [ ] **M4 Frozen-feature Linear Probe**（与 §6.1 口径对齐）：EMOTIC V-A CCC V ≥ 0.30 / A ≥ 0.20，mst 8-state acc ≥ 40%
- [ ] **M5 视频 ID Shortcut 反测必须 `green`（acc < 30%）**（v0.2.4 加：`yellow` 仅限 preflight，final 不可通过）
- [ ] ckpt ≤ 20MB（v0.2 调整，v0.1 5MB 不实）
- [ ] 单帧推理延迟 ≤ 15ms (Studio M2 Max CPU)
- [ ] LNN-FIRST 合规四联 monitor 每 200 step 不报警
- [ ] codex 对抗审查 PR 通过 + gemini 学术评审通过
- [ ] 川哥签字 + ADR-019 决策记录已立 + ADR-020 收尾报告落盘

**未通过任一**：
- M1/M3/M4 任一不达标 → 不导出 `base_v3.pt`，回到 Step 1 调超参（不上线）
- M5 video_id acc 处于 `yellow`（30%-50%）→ 强制做 1 次 augmentation 重跑或切换 SSL 目标后复测
- M5 video_id acc 处于 `red`（>50%）→ 直接 reject Phase γ
- LNN-FIRST 合规失败 → 立即停训 + 写 root cause ADR
- 推理延迟超标 → 模型压缩（INT8 量化或 LoRA 减参数）

### 11.3 Phase δ 加保 — TTA Fail-safe（v0.2 加，gemini D4 强建议）

**Why 加**：VEATIC 西方人 → 川哥真人是 spec §9 风险 #3 高概率项。Phase δ 用 daemon 数据 fine-tune 是主路径，但 fine-tune 需要 60w+ 帧累积（达标前 demo 仍可能跨域 fail）。

**TTA 备份机制**（推理时熵最小化，Wang et al. ICLR 2021 Tent / arXiv:2006.10726）：

- 推理时仅微调 backbone 的 BatchNorm/LayerNorm 参数
- 最小化 `H(softmax(head_state_8(h_t)))` 让模型对自己输出更自信
- daemon 侧每 1000 帧跑一次 TTA，更新归一化参数（不动主权重）
- Phase δ ADR-021 落地详细方案

---

## Summary

**v0.2（2026-05-21 修订，吸收 codex 5.4 high + gemini 3.5 high 双评审 5 条核心建议）**

ADR-016 base v3 实施 spec。本 spec 覆盖 Phase γ（5/27-6/10）工程细化：**2 层 CfC backbone**（v0.2 改 4→2 层 spike，γ-2 实测后再决定升 4 层；参数量待 spike 实证，ckpt ≤20MB） + 5 head readout（state/V-A/intent/conf/hidden_repr） + **3 个 loss 实际生效**（Temporal Contrast 主线 weight=1.0 / Readout weight=0.5 / Persona Loss 3 默认关 weight=0，Crossmodal 0 留 M1+）。

**v0.2 5 核心修订**：
1. **Loss 3 默认关**：VEATIC viewer 是观察者不是用户 persona（codex 工程实证 + gemini 学术批评双重证据），真 persona 留 Phase δ 川哥真人 daemon
2. **主指标三联替 silhouette**：M1 KG 召回 + M3 时序 ratio + **M4 Frozen-feature Linear Probe + M5 视频 ID Shortcut 反测**（silhouette 降级附属，因 SSL 阶段高分反而证明 collapse）
3. **架构 4 层 → 2 层 spike**：v0.1 拍脑袋参数 30K 实测 198K（4 层）/100K（2 层），ckpt 上限调 5MB→20MB
4. **collapse 四联监控**：std + 协方差秩 + per-dim variance + 输出能量，每 200 step 检查
5. **加 InfoNCE vs VICReg 双路 γ-1 spike**：5/26 决策日二选一锁定主路径
6. **新增 10-video preflight 门控**：5/26 前必通过 8 项 checklist 才能 5/27 启动 Step 1
7. **Phase δ 加 TTA Tent**（fail-safe，跨域兜底）

3 步 schedule 不变：VEATIC self-supervised 预训练（4 天）→ EMOTIC readout V-A（3 天）→ mst readout 8-state+intent（4 天）→ 评估调参 4 天。GO 判定 **CONDITIONAL GO**，5/22-5/26 修订周完 → 5/27 启动 Step 1。

---

**v0.2.1 补丁（2026-05-21 20:30）**：codex 第二轮验收 REVISE 后口径对齐补丁
- 修 8 处一致性矛盾（ckpt 5MB→20MB / 4 层→2 层 / Loss 3 默认关 全文统一）
- §11.1 preflight 8 项 → 11 项（codex 补 3 条：防数据泄漏 / 四联入 preflight / 双 seed 可复现）
- §6.1 vs §11.2 M4 阈值口径对齐

**v0.2.4（2026-05-21 晚 修订）**：吸收 codex+gemini 双 REVISE
- **M1 增 M1-V Visual-Only**（25d 文本置零后 Top-5 ≥ 20%，否则整路径回退）
- **M5 统一 green/yellow/red 三层梯度**（preflight 可 green/yellow，final 必须 green）
- **Loss 3 关闭态收口**（persona_idx 约束与风险项加 `N/A when weight=0`）
- **γ-2 + preflight #12 增 τ-bias 真进 ODE 验证**（含显著性检验 `p < 0.05`）
- **§7.1 补 14 天磁盘预算**（γ-0 free ≥ 150GB + `<10GB` 自动暂停报警）
- **§9 #10 ncps 版本修正**：1.0.7 → 1.0.1（PyPI 实测最高版本，1.0.7 拍脑袋错）

**v0.2.4 后续触发条件（gemini 3.5 终审 APPROVE-with-conditions 留 + cc 实测补）**：
- 🔮 **5/26 决策日触发**：若 γ-3 spike 最终选 Path A 纯 InfoNCE → 必须加时序梯度截断（相邻帧 AU 变化率 `||x_t - x_{t+1}||₂ > μ+2σ` 时断 positive pair，防低通滤波抹杀微表情）
- 🔮 **6/10 Phase γ→δ 演进时**：M5 阈值从 30% 收紧到 ≤15%（写 ADR-020，作为长期泛化保障）
- ✅ **ncps 1.0.1 实测确认**（5/21 22:55，Air+Studio）：`__version__` 字段仍显 0.0.2 是上游 bug（实际参数量从 57736 → 49536 证 API 已升）；`CfCCell` 暴露 `time_a` + `time_b` Linear(64→128) 两个可微 τ 学习层 + `forward(timespans=...)` 外部时间步注入 → gemini C.1 担心的"完全黑盒"部分错误，γ-2 spike 有 3 条可行路径：(A) timespans 注入 per-persona 时间步 / (B) hook time_a.weight 注 persona-bias / (C) 手写 CfCCell 单步隐式欧拉

**v0.2 修订：cc 主线 + codex 5.4 + gemini 3.5 双评审 | 2026-05-21 | 等川哥钦定**
**v0.1 起草：Agent 后台 | 2026-05-21**
