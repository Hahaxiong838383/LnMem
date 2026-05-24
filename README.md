# LnMem · 以 LNN 为心脏的伴侣型 AI 记忆系统

> 一个能「陪你数年」的 AI 伙伴 · 居家+车载持续运行 · 用万能语法编码事件 · 用液态神经网络（LNN）作永远在线心智核心。
>
> **🔥 5/23 v0.1 战略转折**：LNN 角色重定义为 **online 实时意图预测**（不再是 AU 序列预测），见 [ADR-028](docs/decisions/028-lnn-role-redefine-online-intent.md) 川哥钦定签字。

---

## 🤖 给 AI 同事的 30 秒上手（必读）

| 想要了解 | 读这个 |
|---|---|
| 🔥 **v0.1 最新方向（最重要）** | [`docs/decisions/028-lnn-role-redefine-online-intent.md`](docs/decisions/028-lnn-role-redefine-online-intent.md) |
| 🔥 **今日复盘 + 踩坑大全** | [`docs/reports/2026-05-23-v0.1-direction-shift-complete.md`](docs/reports/2026-05-23-v0.1-direction-shift-complete.md) |
| **项目目标 + LNN-FIRST 原则** | [`docs/onboarding-ai.md`](docs/onboarding-ai.md)（30 秒）|
| **项目状态 + TODO** | [`ROADMAP.md`](ROADMAP.md)（5 分钟全局）|
| **技术全栈架构** | [`docs/architecture-overview.md`](docs/architecture-overview.md)（5 分钟懂栈）|
| **产品定义（投资人友好）** | [`docs/product-definition.md`](docs/product-definition.md) §0-§1 |
| **路线图 M0-M12** | [`roadmap/milestones.md`](roadmap/milestones.md) |
| **28 个决策记录 ADR** | [`docs/decisions/`](docs/decisions/) |
| **AI 协作纪律** | [`CLAUDE.md`](CLAUDE.md)（cc）/ [`AGENTS.md`](AGENTS.md)（codex）/ [`GEMINI.md`](GEMINI.md)（gemini）|

---

## 📍 当前阶段（2026-05-23 凌晨）

**🚀 v0.1 实时意图预测 · Sprint 1 完成 · Sprint 2 启动 · M0 6/10 收口在跑**

### 5/22-23 单晚 16 commits Team 模式连续 11 小时

| 类别 | 数字 |
|---|---|
| commits | **16+**（c96a12f → d80137d）|
| ADR | **4 新**（025/026/027/028）|
| 测试 | **113/113 全过零回归** |
| gemini 对抗审 | **7 轮 / 30+ finding 全采纳** |
| Team 模式 | cc 主线 + codex × 5 + gemini × 7 + 川哥钦定 3 次签字 |

### 5/23 v0.1 战略转折关键事实

1. **三方战略复盘**（cc + codex + gemini 独立思考）→ "LNN 应降级为子模块"
2. **川哥点睛纠偏**："14 天记忆界面 ChatGPT 也能做，我要前十几秒到几分钟意图预测"
3. **真相**：cc 5/22-23 全程在 spec 字面任务上验证 LNN，没追产品意图
4. **ADR-028**：LNN 角色从"AU rollout 心智核心"→"online intent estimator"（不 override LNN-FIRST P-1，只是 #2 内涵更新）

### v0.1 6/10 demo 目标

- LNN 30Hz 多模态 → 三分类意图（正/负/无反馈）
- AI 主动推断 → UI 弹出 → 川哥 [确认/纠正] → τ-bias 漂移
- "7 天准确率 35% → 65%+"

完整路径见 [ADR-028](docs/decisions/028-lnn-role-redefine-online-intent.md)。

---

## 📍 历史阶段（2026-05-21 23:30 - 2026-05-22）

---

## 📍 当前阶段（2026-05-21 23:30）

**M0 W3 Day 3 末 · γ-3 4 硬门通过 · gemini 6 轮评审 CONDITIONAL APPROVE · 5/27 Phase γ Step 1 ready**

### 单晚 6h Team 模式成果（commit `cf1552c` → `492583d`，16 commits）

| 类别 | 数字 |
|---|---|
| commits | **16** |
| ADR | **6 新增**（016/017/018/021/022 + spec 修订 v0.2.4→v0.2.8）|
| 代码 + 文档 | ~10000 行 |
| gemini 对抗审 | **6 轮** / 22 个 finding / **全采纳** |
| 全量测试 | 749 passed / 3 skipped ✅ |

### 6 轮 gemini 评审 verdict 进化

```
轮 1 REVISE → 轮 2 REVISE → 轮 3 REVISE → 轮 4 REVISE → 
轮 5 REVISE (4 硬门钦定) → 轮 6 CONDITIONAL APPROVE ✅
```

### γ-3 4 硬门状态

| # | 硬门 | 状态 | 实测值 |
|---|---|---|---|
| 1 | DGB ratio ∈ [0.1, 50] | ✅ | 0.797-0.836（pre-DGB 6.46e8）|
| 2 | Shuffle Test M1 drop > 0.05 | ✅ | **+0.241**（134% 提升）|
| 3 | AU augment M5 显著降 | ⚠️ | 0.833 → **0.500**（-40%，N=10 物理极限）|
| 4 | M1-V Spike 标 N/A | ✅ | 已标 |

### γ-3 实测数据（50 epoch InfoNCE seed=42，10 视频 / 494 帧）

| 配置 | M1 top5 | M5 acc | Shuffle drop | effective_dim |
|---|---|---|---|---|
| baseline (无 augment) | 0.862 | 0.833 RED | +0.103 | 3.83 |
| **+ ADR-022 + AU augment 甜区** ⭐ | **0.897** | **0.500** | **+0.241** | 3.20 |

---

## 🎯 它是什么 + 不是什么

### 是什么

**伴侣型 AI 的记忆系统**：跟 LLM Agent / 数字孪生 不同，LnMem 是**永远在线运行**的——

```
传统 LLM + RAG          LnMem
└─ 对话时检索过去   vs  └─ 永远在线连续时间状态持有当下
└─ 静态 embedding 库     └─ 动态 LNN 心智核心 (LTC/CfC ODE)
└─ 用户主动查询          └─ 系统主动写入 + 夜间巩固
```

### 不可妥协 6 项（项目 IP 红线）

| # | 不可妥协 | 当前状态 |
|---|---|---|
| 1 | 万能语法 20 槽全系统共享 schema | ✅ schema_v1.1 封版 |
| 2 | **多时间尺度 LNN 心智核心**（LTC/CfC 微分方程）| ⏳ Phase γ 关键 14 天 |
| 3 | CLS 双系统离线 consolidation（夜间巩固）| ✅ v0 已上线 |
| 4 | per-user τ-bias + FiLM + 槽偏好向量 | ⏳ Phase δ（6/11+）|
| 5 | 居家+车载双场景统一架构 | ⏳ M6+ |
| 6 | 记忆 UI 用户可改可删可治理 | ✅ v0.1 删除已上线 |

### P-1 最高准则：LNN-FIRST

> 凡涉及「**时间 / 演化 / 记忆 / 个性化**」，第一性原理必须是 **LTC/CfC 微分方程**。
> 用 LLM/RNN/规则替代 LTC 必须有 ADR + 川哥签字。

详细：[`docs/principles/lnn-first.md`](docs/principles/lnn-first.md)

---

## 🏗 技术全景（5 分钟懂栈）

```
┌─────────────────────────────────────────────────────────────────────┐
│                         User Reality                                │
│  对话 / 表情 / 行为 / 视觉信号 / 上下文 (居家/车载)                   │
└──────────────┬──────────────────┬──────────────────┬───────────────┘
               ▼                  ▼                  ▼
        ┌──────────┐       ┌──────────┐       ┌──────────┐
        │ Parser   │       │ Webcam   │       │ Audio    │
        │ 20 槽    │       │ py-feat  │       │ M1+ 接   │
        │ + intent │       │ 20 AU    │       │          │
        └────┬─────┘       └────┬─────┘       └────┬─────┘
             ▼                  ▼                  ▼
       ┌──────────────────────────────────────────────────┐
       │   EventTensor (52d schema_v1.1)                   │
       │   24 AU + 20 槽 + 2 time + 6 audio + intent_type │
       └─────────────┬────────────────────────────────────┘
                     ▼
        ┌─────────────────────────────────────┐
        │  Fusion Layer (52d → 64d) + GELU    │
        └─────────────────┬───────────────────┘
                          ▼
       ┌──────────────────────────────────────────┐
       │  LNN Backbone (MultiLayerCustomCfCCell)  │ ← Phase γ 训练目标
       │  2 layer × CfC + per-persona τ-bias      │
       │  ODE: τ·dh/dt = -h + f(x_t, h_prev)      │
       │  半隐式欧拉离散化 + DGB 防梯度黑洞       │
       └──────────────────────────────┬───────────┘
                                      ▼
              ┌──────────────────────────────────────┐
              │  hidden_state (64d, 实际 eff~3-8d)   │
              └───┬───────────┬───────────┬──────────┘
                  ▼           ▼           ▼
            ┌─────────┐ ┌──────────┐ ┌──────────┐
            │ 5 heads │ │ Graphiti │ │ daemon   │
            │ readout │ │ KG SQLite│ │ realtime │
            │弱监督   │ │bi-tempor│ │  + CLS   │
            └─────────┘ └──────────┘ └──────────┘
```

---

## 📊 路线图（M0 → M12）

| 阶段 | 时间 | 主题 | 状态 |
|------|------|------|------|
| **M0 W3** | 5/21-5/27 | Phase α 多模态 demo + Phase β schema 完工 + γ-3 spike 验收 | ✅ 完工 |
| **Phase γ Step 1** | **5/27-5/30** | **VEATIC 全量 124 视频 self-supervised 预训练** | **🚀 启动 ready** |
| Phase γ Step 2 | 5/30-6/2 | EMOTIC readout V-A 弱监督 | ⏳ |
| Phase γ Step 3 | 6/2-6/6 | mst readout 8-state + intent_type | ⏳ |
| Phase γ Step 4 | 6/6-6/10 | 评估 + 调参 + 导出 base_v3.pt | ⏳ |
| Phase δ | 6/11+ | 川哥真人 daemon 微调 + per-user 三件套启动 | ⏳ |
| M2 | 7 月+ | 多用户系统化 + 居家 v1 | ⏳ |
| M6 | Y2 | 车载场景 + 双场景统一架构 | ⏳ |

详细：[`ROADMAP.md`](ROADMAP.md)

---

## 🚀 5/27 Phase γ Step 1 启动 4 项 Pre-conditions（gemini 6 轮终审钦定）

| # | 防御 | 实施 | 验证 |
|---|---|---|---|
| 1 | **τ-Monitor** | 每 epoch log `tau.item() + tau.grad` | τ ∈ [0.05, 0.2]，grad ≠ 0 |
| 2 | **DataLoader shuffle=True 强制** | 显式声明 + Epoch 1 Loss diff ≤ 5% | 防 Batch-Index 泄露 |
| 3 | **动态维度监控** | `effective_dim < 2.5` 连续 5 epoch 自动激活 VICReg Cov Loss | 训练中后期 effective_dim > 5.0 |
| 4 | **LR 死锁修正** | τ LR=1e-5 + DGB | DGB ratio ∈ [0.5, 10] |

5/22 早接手即实施这 4 项。完整 list: [`docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md`](docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md) §5

---

## 🛠 代码主干

| 模块 | 路径 | 状态 |
|---|---|---|
| 万能语法 parser | [`universal_grammar_parser/`](universal_grammar_parser/) | ✅ schema v1.1 + intent_type 5 类 100% |
| Graphiti KG | [`graphiti_zh_adapter/`](graphiti_zh_adapter/) | ✅ SQLite bi-temporal |
| EventTensor schema | [`contracts/`](contracts/) | ✅ schema v1.1 封版 |
| **LNN Backbone (γ-2 Path C)** | [`poc/lnn/`](poc/lnn/) | ✅ 手写 CfCCell 半隐式欧拉 |
| **训练 pipeline** | [`poc/training/`](poc/training/) | ✅ trainer + dataset + losses + au_augment |
| Sampler + Monitor | [`poc/data/temporal_sampler.py`](poc/data/temporal_sampler.py) + [`poc/monitoring/collapse_monitor.py`](poc/monitoring/collapse_monitor.py) | ✅ |
| Streamlit demo | [`apps/demo_e2e/`](apps/demo_e2e/) | ✅ Phase α 多模态 + Phase β 6 维状态 JSON |
| daemon (live LNN) | [`poc/daemon/`](poc/daemon/) | ✅ raw 路径 ADR-005 |

---

## 🧪 启动 Demo（30 秒跑起来）

```bash
cd /Users/rayman.chen/mycc/2-Projects/LnMem-poc
export GEMINI_PROXY_TOKEN=<token>

# Phase α + β demo (川哥可真实体验)
PYTHONPATH=. .venv-poc-fresh/bin/streamlit run apps/demo_e2e/main.py
# → http://localhost:8501

# 跑测试
PYTHONPATH=. .venv-poc-fresh/bin/python -m pytest poc/ apps/ contracts/ \
  graphiti_zh_adapter/ universal_grammar_parser/ -q
# 期望: 749 passed / 3 skipped ✅
```

---

## 👥 团队配置

| 引擎 | 角色 | 状态 |
|------|------|------|
| **川哥** | 架构 lead / LNN-FIRST 守门员 / 最终决策 | 主战 |
| **cc** (Claude Sonnet 4.6) | 主线实现 / 文档 / 终验 / commit | 主战 |
| **codex** (gpt-5.4 high / 5.3-codex high) | 对抗审查 + 大模块实现 | 配额限制时备用 |
| **gemini** (3.5-flash high / 3-pro-preview) | 学术外脑 + 战略评审 | 6 轮对抗审 22 finding 全采纳 |

---

## 📚 关键文档索引

### AI 同事路径
1. [`docs/onboarding-ai.md`](docs/onboarding-ai.md) — 30 秒入门
2. [`docs/project-status.md`](docs/project-status.md) — 3 分钟全局
3. [`docs/architecture-overview.md`](docs/architecture-overview.md) — 5 分钟懂栈
4. [`docs/specs/base-v3-training-spec.md`](docs/specs/base-v3-training-spec.md) — Phase γ 详细 spec v0.2.8
5. [`docs/principles/lnn-first.md`](docs/principles/lnn-first.md) — P-1 LNN-FIRST 准则

### 复盘 + 决策记录
- [`docs/retrospectives/`](docs/retrospectives/) — 每日复盘（最新 [Day 3 6h Team 模式](docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md)）
- [`docs/decisions/`](docs/decisions/) — 22 个 ADR（最新 [ADR-022 τ-bias 梯度黑洞](docs/decisions/022-tau-bias-gradient-blackhole-fix.md)）
- [`docs/reports/`](docs/reports/) — gemini 6 轮评审报告 + γ-3 实测数据

### 工程纪律
- [`CLAUDE.md`](CLAUDE.md) — cc 工作纪律（三道门 + 1 LNN-FIRST + 6 不可妥协）
- [`AGENTS.md`](AGENTS.md) — codex 工作纪律
- [`GEMINI.md`](GEMINI.md) — gemini 工作纪律
- [`docs/STYLE.md`](docs/STYLE.md) — 文档写作规范（L1-L4 四级体裁）
- [`docs/development-harness.md`](docs/development-harness.md) — BRO-PDCA × 目标驱动验收

---

## 🏆 Day 3 晚上 Team 模式总结

### 决定性突破

1. **真训练 pipeline 完整跑通** · py-feat → 真 VEATIC → InfoNCE/VICReg → 5 readout heads → evaluation
2. **ADR-022 τ-bias 梯度黑洞**修复（双率优化器 + DGB 压制 6.4e8 → 0.797）
3. **22 个 finding 全采纳** · 单晚 6 轮 gemini 对抗审从 BLOCK → CONDITIONAL APPROVE
4. **5/27 Phase γ Step 1 启动 ready** · 4 项 Pre-conditions 清单已落 ADR-022 v2

### 6 大经验教训（mycc memory 已落盘）

| # | 教训 | 案例 |
|---|---|---|
| 1 | cherry-pick 数字必被对抗审揭穿 | 363995x sep 实际是 0.36/1e-6 数值幻觉 |
| 2 | 风险方向判断错远比无知更危险 | spec §9 #13 担心方向错（τ 过强不是过弱）|
| 3 | 上游 bug 比代码 bug 更隐蔽 | py-feat detect_video 全 NaN + ncps timespans broadcast |
| 4 | 评审复利 6 轮抓 22 finding 非线性 | 单轮 5-7，6 轮 22 |
| 5 | 动态门槛 > 静态门槛 | M5(N) = 0.50·(N/10)^-0.25 |
| 6 | 物理直觉发现深层 bug | Shuffle drop +0.241 是「3.2d 流形伪泛化」 |

---

**📝 主要维护者**：cc 主线 + 川哥 LNN-FIRST 守门 | **当前 commit**：`492583d` | **下一站**：5/27 Phase γ Step 1 启动 🚀
