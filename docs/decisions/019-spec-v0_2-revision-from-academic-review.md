---
adr-id: 019
title: base v3 spec v0.1→v0.2 修订（吸收 codex 5.4 + gemini 3.5 双评审 5 条核心建议）
status: accepted（2026-05-21 19:55，川哥宏观钦定 GO 方案 A+）
date: 2026-05-21
deciders: 川哥（产品守门员）+ cc（主线 LNN-FIRST 守门）+ codex-5.4-high（工程对抗审）+ gemini-3.5-flash-high（学术深审）
supersedes: docs/specs/base-v3-training-spec.md v0.1（同日起草）
related:
  - ../specs/base-v3-training-spec.md
  - ./016-base-v3-multimodal-self-supervised-lnn.md
  - ./013-base-v2-failure-and-finetune-plan.md
  - ./018-eventtensor-schema-v1_1-intent-type-multimodal-context.md
---

## 上下文

2026-05-21 18:30 Agent 后台写出 base v3 实施 spec v0.1（534 行，10 段齐备）。同日 19:30 派 codex 5.4 high + gemini 3.5 high 双评审（学术 + 工程对抗审查并行）。

**双评审独立得出 CONDITIONAL GO**，但 7 条具体建议中有 5 条**两路评审一致命中**，且包含 codex 本地代码实证打脸 spec v0.1 工程预估（参数量从 30K 实测到 198K，6.6x 偏差）。

川哥 19:55 钦定：「能符合这个项目最初设定的目标，达到我的基于 LNN 的多模态识别与预测，就可以 go」。

cc 主线宏观对账 4 维度（基于 LNN / 多模态 / 识别 / 预测），结论：**修订项是项目目标护卫者，不是绕路**，方案 A+ GO。

## 决策

### 1. CONDITIONAL GO Phase γ（5/27 启动门控 5/26）

| 阶段 | 日期 | 内容 |
|------|------|------|
| **修订周** | 5/22-5/26（5 天）| spec v0.2 落地 + γ-1/γ-2 spike + 10-video preflight |
| **启动门控** | 5/26 | preflight 8 项 checklist 全通过 → 5/27 启动 Step 1 |
| **Phase γ Step 1** | 5/27-5/30 | VEATIC 全量 self-supervised 预训练（Studio M2 Max）|
| **Phase γ Step 2-3** | 5/31-6/6 | EMOTIC + mst readout 弱监督 |
| **Phase γ Step 4** | 6/7-6/10 | 评估 + 调参 + 导出 |

### 2. spec v0.1 → v0.2 五大核心修订

#### 修订 #1：Loss 3 Persona Contrast 默认关（weight=0）

**v0.1**：用 VEATIC `viewer_id` 当 user persona，SupCon weight=0.3
**v0.2**：weight=0 默认关，重命名为 Observer-style Regularizer

**双评审证据**：
- **codex 5.4（工程实证）**：VEATIC viewer 是给视频打分的标注员（observer），不是被拍的人（subject）。viewer 的"persona"只反映标注偏好（V-A 打分倾向），不是 LnMem 想学的「主人多模态连续表征」。继续用 → Loss 3 把"标注员偏好"硬塞进 hidden state 污染表征。
- **gemini 3.5（学术批评）**：SupCon 假设离散类别，LnMem 的 persona 是**连续演化非平稳**的（情绪、认知疲劳是 transient state）。强行 SupCon 拉近会抹杀 LNN 捕获情绪波动能力，**违反 P-1 LNN-FIRST**。

**cc 主线决策**：
- 完全采纳「默认关」（不冒概念错误风险）
- 真 persona 学习路径：**Phase δ 川哥真人 daemon 60w 帧累积**（产品定义 §10.1 #4 真正落地点）
- v0.1 SupCon 公式段保留封存，未来若启用改造为 SoftCLR/DynCL 等连续对比方法

#### 修订 #2：评估指标三联替代 silhouette

**v0.1**：M1 KG 召回 + M2 silhouette + M3 时序 ratio
**v0.2**：M1 KG 召回 + M3 时序 ratio + **M4 Frozen-feature Linear Probe + M5 视频 ID Shortcut 反测**（silhouette 降级 §6.2 sanity 附属）

**双评审证据**：
- **gemini 3.5（学术批评 D2 评分 C+，4 维最低分）**：「Silhouette 假设各向同性超球聚类，LNN 学的是连续轨迹/极限环。高 silhouette 反而证明 backbone collapse」（Falcon et al. arXiv:2206.04674）
- **codex 5.4**：M2 silhouette 会把 supervised mst 域的结构误当 backbone 质量；应该用 frozen-feature linear probe（He et al. MAE CVPR 2022 SSL 评估金标准）

**额外 gemini 警告**：视频 ID Shortcut Learning（Geirhos 2020 Nature Machine Intelligence）—— InfoNCE 易让模型匹配背景/光影而非用户表征。

**cc 主线决策**：
- 完全采纳 Linear Probe 主指标化（M4）
- 完全采纳视频 ID 反测（M5）防 Shortcut
- silhouette 保留但降级（人眼定性 sanity，不当主指标）—— 守住 LNN-FIRST P-1 准则 V1/V3「评估必须看表征质量」

#### 修订 #3：架构 4 层 CfC → 2 层 spike（ckpt 5MB → 20MB）

**v0.1**：4 层 CfC（τ 0.1s~600s），参数量预估 ~30K，ckpt ≤5MB
**v0.2**：2 层 CfC（τ 0.1s~10s + 10s~600s），参数量待 spike 实证，ckpt ≤20MB

**codex 本地实证打脸（决定性证据）**：
- 当前单层 CfC(64,64) 实测 **57,736 参数**（v0.1 拍脑袋 5K）
- 4 层 stack 实测约 **198,144 参数**（v0.1 拍脑袋 30K，6.6x 偏差）
- 2 层 stack 估约 **100K 参数**

**gemini 工程警告（D3 评分 B+）**：600s 单级 BPTT 梯度爆炸风险，建议 Hierarchical CfC（低层高频高层下采样）。

**cc 主线决策**：
- 默认 2 层 spike（先跑通）
- 4 层等 γ-2 实测 ckpt ≤20MB 且训练稳定后才升
- ckpt 上限调 5MB→20MB（实证后）；若 γ-2 实测 ckpt > 20MB → INT8 量化或 LoRA 减参数
- 不偏离 daemon 加载延迟约束（M2 Max NVMe 20MB < 200ms 仍可接受）

#### 修订 #4：collapse 监控 std → 四联监控（每 200 step）

**v0.1**：`assert h_t std > 0.05`（单一阈值）
**v0.2**：四联监控 + 每 200 step 检查（v0.1 是每 epoch）：
- std: `h_t.std() > 0.05`（基础）
- 协方差秩: `matrix_rank(cov(h_t)) >= 32`（防低维 collapse）
- per-dim variance: `(h_t.var(dim=0) > 0.01).sum() >= 40`（防 60+ 维死掉）
- 输出能量: `h_t.norm(dim=-1).mean() ∈ [0.5, 50.0]`（防输出全 0 或爆炸）

**双评审证据**：
- **codex**：std<0.05 单一阈值不足，模型可以 maintain 单一方向的 std 但其他维度全死
- **gemini**：Loss 1 mode collapse 是连续 LNN 的高风险失败模式

**cc 主线决策**：完全采纳四联，频率从 epoch 升到 200 step（早期预警）

#### 修订 #5：加 InfoNCE vs VICReg γ-1 双路 spike

**v0.1**：纯 InfoNCE（Loss 1 weight=1.0）
**v0.2**：γ-1 双路 spike，5/26 决策日二选一锁定

**双评审证据**：
- **gemini D1（B 分）**：「InfoNCE 在连续动力学有假阴性惩罚，撕裂相空间轨迹」。建议 VICReg/V-JEPA（Bardes et al. ICLR 2022 + 2024）
- **codex（学术 6.5/10）**：纯 InfoNCE 缺 anti-collapse 辅助，2024 SOTA 倾向 VICReg + Barlow Twins

**cc 主线决策（CLAUDE.md §3.1 冲突诊断）**：
- **不预 lock-in**（既不押注 v0.1 InfoNCE，也不无脑切 VICReg）
- γ-1 跑双路 spike，看哪个 std(h) 稳定 + Linear Probe acc 高
- 5/26 决策日基于实测数据二选一锁定主路径
- 备选 InfoNCE + VICReg 联合（weight 0.5/0.5）

### 3. 启动门控 10-video preflight（v0.2 加，5/26 必通过）

8 项 checklist（spec §11.1）：
1. AU 抽取吞吐 ≥ 500 frame/sec
2. Sampler ±60 帧 + |Δt|>30s 稳定
3. **VEATIC viewer_id 实际去重数验证**（决定 Loss 3 启用与否）
4. **EMOTIC vs VEATIC V-A 标注口径** Bland-Altman 一致性
5. ncps CfC 实际参数量 + τ-bias 多尺度注入 spike
6. InfoNCE vs VICReg 双路 spike 1 epoch
7. 视频 ID Shortcut 反测 acc < 50%
8. Studio ssh + nohup + ckpt 双轮替

### 4. Phase δ 加 TTA Tent fail-safe（gemini D4 强建议）

跨域风险（VEATIC 西方人 → 川哥真人）是 spec §9 风险 #3 高概率项。Phase δ daemon 数据 fine-tune 是主路径，但累积 60w 帧需要时间。

**TTA Tent**（Wang et al. ICLR 2021 / arXiv:2006.10726）：
- 推理时仅微调 backbone BN/LN 参数
- 最小化输出熵 H(softmax(head_state_8(h_t)))
- daemon 侧每 1000 帧跑一次更新归一化参数（不动主权重）
- 详细方案留 ADR-021 Phase δ 启动前

## 影响范围

### spec 文件
- `docs/specs/base-v3-training-spec.md` v0.1 → v0.2
- 534 行 → 656 行（+122 行 / +22.8% 内容）
- 主要新增：§3.1.bis VICReg / §3.3 Loss 3 关闭说明 / §6.1 主指标三联 / §6.3 四联监控 / §11.1 preflight 门控 / §11.3 TTA

### 任务列表（cc 主线 + codex 5 天修订周）
- 5/22 周四：spec v0.2 commit + push（本 ADR-019 同步）
- 5/23 周五：γ-1 数据 pipeline + ncps 实测 spike（cc + codex）
- 5/24 周六：γ-2 模型架构 2 层 CfC 实测 + collapse 四联监控实现
- 5/25 周日：InfoNCE vs VICReg 双路 spike + 视频 ID Shortcut 反测 + Studio 远程验证
- 5/26 周一：10-video preflight 全跑通 + 8 项 checklist 川哥钦定

## 与项目目标对账（川哥钦定 GO 依据）

| 项目最高准则维度 | v0.2 服务度 |
|----------------|------------|
| **基于 LNN（不可妥协 §10.1 #2）** | ✅ **强化** — CfC backbone 主路径，4 loss 全围绕 hidden state |
| **多模态识别**（视觉 + 文本 + 时间）| ✅ fusion 52-dim 全保留（视觉 AU + 文本 20 槽 + intent_type + 时间戳）|
| **预测心理活动** | ✅ Loss 1 Temporal Contrast 主目标 = 学时间演化 = 预测能力的数学基础 |
| **per-user 个性化**（不可妥协 §10.1 #4）| ⏳ Phase δ 真正落地（v0.2 backbone + FiLM + τ-bias 已就位）|

**结论**：v0.2 修订项**全是项目目标护卫者，不是绕路**。

## 风险登记

| # | 风险 | 缓解 |
|---|------|------|
| 1 | 修订周 5 天不够，5/26 preflight 不过 | 推迟启动 5/29，total 时间仍 < ADR-016 21 天 buffer |
| 2 | VICReg γ-1 spike 比 InfoNCE 差 | 锁回 InfoNCE，v0.2 同样不损 |
| 3 | 2 层 CfC 表征能力不够 | γ-2 实测 ckpt < 20MB + 训稳后升 4 层（spec §2.2 留接口）|
| 4 | TTA Tent 在 LNN 上未实证 | Phase δ ADR-021 详细设计前先 spike，不达标改用 LoRA adapter |
| 5 | M4 Linear Probe 阈值（CCC ≥ 0.25 / acc ≥ 40%）拍脑袋 | γ-2 spike 末校准（基于 base v1 baseline 反推合理阈值）|

## 与现有 ADR 关系

| ADR | 关系 |
|-----|------|
| ADR-013 base v2 失败 | 教训承袭：supervised collapse 教训驱动 v3 自监督，v0.2 进一步守 silhouette/acc 不当主指标 |
| ADR-016 base v3 范式 | spec 本身是工程实施细化，v0.2 不改 ADR-016 范式骨架（仅细化和加防护）|
| ADR-018 EventTensor v1.1 | intent_type Phase β 已上线，v0.2 spec §3.5 readout Head 3 直接消费 |

## Summary

base v3 spec v0.1→v0.2 修订决策记录。吸收 codex 5.4 high + gemini 3.5 high 双评审 5 条核心建议：(1) Loss 3 默认关 / (2) 评估三联替 silhouette + 视频 ID Shortcut 反测 / (3) 架构 4 层→2 层 spike + ckpt 5MB→20MB / (4) collapse 监控 std→四联 200 step / (5) InfoNCE vs VICReg γ-1 双路 spike + 10-video preflight 门控。Phase δ 加 TTA Tent fail-safe。川哥宏观钦定 CONDITIONAL GO 方案 A+：5/22-5/26 修订周 → 5/27 启动 Phase γ Step 1。所有修订项守护川哥项目最高准则「基于 LNN 的多模态识别与预测」零妥协。

---

**模板版本：v0.1 · 2026-05-03**
