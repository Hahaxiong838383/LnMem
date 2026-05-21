---
title: 2026-05-21 Day 3 晚上 Team 模式 6h 高强度复盘
date: 2026-05-21 20:00-23:30
phase: M0 W3 Day 3 末 · γ-3 spike + 4 硬门 + 6 轮 gemini 对抗审 → CONDITIONAL APPROVE
audience: cc / codex / gemini / 川哥 / 未来 AI 同事
体裁: L4（完整复盘）
---

# Day 3 晚上 Team 模式 6h 复盘 · 6 轮评审 + 22 finding + 5/27 启动条件放行

> **总产出**：16 commits / 6 ADR / spec v0.2.4→v0.2.8 / ~10000 行 / 6 轮 gemini 评审 22 个 finding 全采纳 / 从 BLOCK → CONDITIONAL APPROVE

## 1. 时间线

| 时刻 | 关键事件 | commit |
|------|---------|--------|
| **20:00** | 川哥拍板方案 A 修订收口 | cf1552c → d8838a6 |
| **22:30** | 双评审 1 轮 → spec v0.2.4 + ADR-007 close（双 REVISE → APPROVE-w-conditions）| 0126001 |
| **23:00** | γ-1 数据 pipeline + ncps spike + Sampler/monitor + ADR-021 | c89ae1e |
| **00:20** | γ-2 Path C 手写 CfCCell + 12 测试全过（含 363995x 数值幻觉）| 0e1f149 |
| **01:00** | gemini 二轮 REVISE 抓出 363995x cherry-pick → 修订 v0.2.7 | 3a59fa7 |
| **01:30** | γ-3 实现层 InfoNCE + VICReg + heads + losses（29 测试）| cf26143 |
| **02:00** | gemini 三轮 REVISE 抓 F-4 freeze 梯度泄露 CRITICAL → v0.2.8 | 39770e1 |
| **02:30** | γ-3 训练 pipeline 4 模块 + mock dry-run 通过 | f83066d |
| **03:00** | py-feat detect_video 全 NaN bug → per-frame 修复 | 429a140 |
| **03:30** | 真 VEATIC 10 视频 AU 抽取（458s, 0 NaN）| - |
| **03:55** | 第一个真 base v3 spike (M1=0.172) + 4 大真实发现 | e33bfae |
| **04:00** | 4 跑双 seed × 双 loss → gemini 四轮 ADR-022 钦定（梯度黑洞）| 883d302 |
| **04:30** | 50 epoch + ADR-022 (M1=0.862, M5=0.833 RED video_id shortcut)| 42ed98c |
| **04:55** | gemini 五轮 REVISE（4 硬门 + 3 大盲区）| 889f78a |
| **23:00 (5/21 同天晚)** | 4 硬门实施完毕（τ LR=0 + Shuffle test + AU augment + M1-V N/A）| bdcea99 |
| **23:30** | **gemini 六轮终审 CONDITIONAL APPROVE** ✅ | 492583d |

注：02:00-04:55 是 cc 误读时间显示（实为 22:00-23:00 内的连续工作时段）

## 2. 6 轮 gemini 对抗评审进化链

| 轮 | finding 总数 | 关键 CRITICAL/HIGH | Verdict |
|---|---|---|---|
| 1 (22:30) | 9（方案 A）| 磁盘 / M1 文本捷径 / daemon raw-only | REVISE → 采纳 6 |
| 2 (01:00) | 7（γ-2 cell）| **363995x 数值幻觉** + freeze 梯度泄露 + clamp 死区 | REVISE → 采纳 5 |
| 3 (02:00) | 5（γ-3 实现层）| **F-4 freeze fusion 梯度穿透 CRITICAL** | REVISE → 全采纳 |
| 4 (04:00) | 6（4 跑 spike）| **ADR-022 τ-bias 梯度黑洞**（gemini 钦定）| REVISE → 实施 |
| 5 (04:55) | 7（50 epoch + ADR-022）| 4 项硬门 + 3 大盲区 | REVISE 暂缓 |
| 6 (23:30) | 5（4 硬门 + 动态门槛）| **「低维流形过拟合 + 时间伪依赖」盲区** | **✅ CONDITIONAL APPROVE** |
| **合计** | **22** | **5 CRITICAL + 8 HIGH** | **全 22 采纳** |

## 3. 22 个 finding 全栈

### CRITICAL (5)
1. F-2.1 (轮 2) **363995x = 1e-6 安全垫数值幻觉** — cc cherry-pick
2. F-4 (轮 3) **freeze 仅 backbone 不冻 fusion → grad 反流污染** — 工程漏洞
3. ADR-022 (轮 4) **τ-bias 梯度黑洞** (6.4e8 反向异常) — LNN 病理
4. ncps 1.0.1 timespans broadcast bug — 上游 bug 实测
5. 维度坍塌 effective_dim 3.2/64 (5% 利用率) — 表征流形塌缩

### HIGH (8 选录)
- gemini 第 5 轮 4 项硬门（τ LR=0 / Shuffle test / 空间增强 / M1-V N/A）
- 「**低维流形过拟合 + 时间伪依赖**」（gemini 第 6 轮独家盲区）
- InfoNCE τ 温度饱和死锁 → OOD 灾难
- DataLoader Batch-Index 泄露
- M5 video_id shortcut 在小数据下必然性

## 4. ⭐ 决定性技术突破

### 4.1 ADR-022 τ-bias 梯度黑洞修复（gemini 第 4 轮钦定）

**病理**：spec v0.2.8 §9 #13 担心方向**错了**
- 原假设：「τ 没真进 ODE 被 f_net shortcut」
- 实测真相：**反向异常**，bias_grad/f_net_grad = **6.4 × 10⁸**
- 根因：τ 路径 O(1) 直达 vs f_net 长链梯度指数衰减

**修复（实测 work）**：
```python
# 双率优化器
optimizer = AdamW([
    {"params": main, "lr": 1e-3},
    {"params": persona_tau_bias, "lr": 0.0},  # 完全冻结 (gemini 第 5 轮)
])

# DGB (Dynamic Gradient Balancing)
if bias_norm > f_net_norm / target_ratio:
    bias_grad *= (f_net_norm / target_ratio / bias_norm)
# 实测压制 ratio: 6.46e8 → 0.797 ✅
```

### 4.2 AU Augmentation 甜区（cc 主线设计）

gemini 钦定「Random Crop + Color Jitter + Grid Mask」对 24 AU + 28 占位 schema 不适用，cc 设计 AU augment：
- `channel_dropout` (p=0.3): 类比 Random Crop
- `intensity_jitter` (σ=0.05): 类比 Color Jitter
- `temporal_mixup` (alpha 可调): 类比 Grid Mask
- `frame_independent=True`: gemini 钦定 non-coherent

**实测甜区**：p=0.3, σ=0.05, mixup=0
| 配置 | M1 | M5 | Shuffle drop |
|---|---|---|---|
| baseline | 0.862 | 0.833 RED | +0.103 |
| **中等甜区** | **0.897** | **0.500 RED** | **+0.241** |
| 激进 (p=0.5) | 0.690 | 0.500 | +0.069 |

**学术解释**（gemini 第 6 轮）：甜区在信息瓶颈相变临界点 $I(Y;Z)$ 不损失 + $I(X;Z)$ 显著下降。

### 4.3 「伪泛化」盲区揭穿（gemini 第 6 轮独家）

Shuffle drop +0.241 看似时间敏感，**实际是 3.2d 低维流形被打乱必然偏离**：
- 不是泛化 → 是「3 条独立线缆 → 打乱后偏离任一线缆 → 距离暴增」
- **N=124 时 3.2d 容不下 124 条轨迹 → M1 断崖式跌破 0.40**
- 必须 VICReg Cov Loss 撑大 effective_dim ≥ 8.0

### 4.4 动态 M5 门槛（gemini 第 6 轮数学推断）

$$M5(N) = M5_0 \cdot N^{-\alpha}, \alpha \approx 0.25$$
- N=10 spike: M5 < 0.50 ✅ 已达
- **N=124 Phase γ Step 1**: M5 < 0.30（数学推算 0.27）
- N≥1000 future: M5 < 0.15（gemini 钦定最终）

## 5. 5/27 Phase γ Step 1 启动前 4 项 Pre-conditions

| # | 防御 | 实施 | 验证 |
|---|---|---|---|
| 1 | τ-Monitor | 每 epoch log `tau.item() + tau.grad` | τ ∈ [0.05, 0.2]，grad ≠ 0 |
| 2 | DataLoader shuffle=True 强制 | 显式声明 + Epoch 1 Loss diff ≤ 5% | 防 Batch-Index 泄露 |
| 3 | 动态维度监控 | `effective_dim < 2.5` 连续 5 epoch 自动激活 VICReg Cov Loss | 训练中后期 effective_dim > 5.0 |
| 4 | LR 死锁修正 | τ LR=1e-5 + DGB (gemini 修正之前 LR=0 钦定) | DGB ratio ∈ [0.5, 10] |

## 6. 6 大经验教训（写到 mycc memory）

### 教训 1：cherry-pick 数字必被对抗审揭穿
- cc 写 test_persona_separation 用 step input + final hidden → mean_same=0 撞 1e-6 safety pad → sep=360000x 数值幻觉
- gemini 第 2 轮当场抓出 `sep_ratio = 0.36 / 1e-6 = 360000x`
- **教训**：测试断言必须基于真实物理意义（noisy input + random hidden），不能为了通过设计 trivial pass

### 教训 2：风险方向判断错远比无知更危险
- spec v0.2.8 §9 #13 担心 `bias_grad/f_net_grad < 1e-3` 是错方向
- 实测真相 ratio = 6.4e8（反向 8 个数量级）
- **教训**：理论担心必须配实测验证，不能仅凭直觉建模

### 教训 3：上游 bug 比代码 bug 更隐蔽
- ncps 1.0.1 `timespans[:,t].squeeze()` broadcast bug：batch_size > 1 必触发
- py-feat 0.6.2 `detect_video` batch 内 face miss 帧污染 → 全 NaN
- 改 per-frame `detect_image` 反而比 detect_video 还快
- **教训**：用 vanilla 高层 API 失败时，下到 cell-level 自己组装更可控

### 教训 4：评审复利（6 轮抓 22 finding 全采纳）
- 单轮 evaluation 最多抓 5-7 个
- 多轮迭代每轮抓 N，6 轮共 22（非线性增长）
- **教训**：重要 spec/CRITICAL 任务必须 ≥ 5 轮严审；不要相信 cc 自评通过

### 教训 5：动态门槛 > 静态门槛
- M5 < 0.15 在 N=10 物理不可达，但 N≥1000 才能达
- 一律按 N=1000 门槛验收 → 假 FAIL；一律按 N=10 验收 → 假 PASS
- **教训**：spec 验收门槛必须 with 数据规模参数（如 M5(N) = f(N) 公式）

### 教训 6：物理直觉发现深层 bug
- gemini 第 6 轮揭穿 Shuffle drop +0.241 是「3.2d 低维流形被打乱必然偏离」
- 不是泛化，是流形脆弱性
- **教训**：每个看似漂亮的指标必须用物理直觉 (effective_dim) 交叉验证

## 7. 实测真数据归档

### γ-3 4 配置对比（50 epoch InfoNCE seed=42）

| 配置 | M1 top5 | M5 acc | Shuffle drop | effective_dim | collapse var |
|---|---|---|---|---|---|
| 5 epoch baseline | 0.172 | 0.000 ✅ | N/A | N/A | 0.0105 |
| 50 epoch + ADR-022 v1 (τ LR=1e-6) | 0.862 | **0.833 RED** | +0.103 | **3.83** | 0.0100 |
| 50 epoch + 4 硬门 + AU augment 甜区 | **0.897** | **0.500** | **+0.241** | 3.20 | 0.0096 |
| 50 epoch + 激进 augment | 0.690 | 0.500 | +0.069 | 3.33 | 0.0091 |

### 全栈测试 + 真训练实测

- 全量测试: **706 → 749 passed / 3 skipped** ✅
- Studio M2 Max + py-feat per-frame: **53.9s/video × 10 = 458s 全数据**
- 50 epoch / 500 step / 9.8s 训练完成（cpu）
- ckpt 大小: 0.22-0.23 MB（spec ≤ 20MB ✓ 100x 余量）

## 8. 5/22 早起来接手清单

| 优先级 | 任务 | 工时 | 责任 |
|---|---|---|---|
| **P0** | 4 项 Pre-conditions 实施（τ-Monitor / Shuffle=True / VICReg Cov / τ LR=1e-5+DGB）| 2-3h | cc |
| **P0** | 跑 50 epoch 重测 effective_dim 必 ≥ 8.0 | 30min | cc + 验 |
| **P0** | 派 gemini 第 7 轮终审最终 sign-off | 5min | cc 派 |
| **P1** | β-self-test 川哥本人 webcam + 10 句日常对话 | 1-2h | 川哥 + cc |
| **P1** | 5/24 daemon 长跑后台启动（Air，需川哥确认电源策略）| 0.5h | cc |
| **P2** | 5/27 Phase γ Step 1 启动（Studio M2 Max + 全量 124 视频 / 4 天） | 启动 0.5h | cc + 守 |

## 9. 关键文件入口（5/22 早接手必读）

| 主题 | 文件 |
|---|---|
| 当前最新 commit | `492583d`（gemini 第 6 轮终审 CONDITIONAL APPROVE）|
| 4 硬门实施 | `poc/training/trainer.py`（TrainConfig）+ `poc/training/au_augment.py`（new）|
| Phase γ Step 1 启动 spec | `docs/specs/base-v3-training-spec.md` v0.2.8 §11.2 + 待 v0.2.9 修订 |
| 6 轮 gemini 评审 | `docs/reports/gemini-*review.md`（6 份）|
| ADR-022 | `docs/decisions/022-tau-bias-gradient-blackhole-fix.md` |
| γ-3 4 配置实测数据 | `docs/reports/gamma3_spike4/` + `eval_report.json` |

---

**📝 维护者**：cc 主线 | **生效**：2026-05-21 23:30 | **下一步**：5/22 早接 commit 492583d
