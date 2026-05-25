# LnMem ROADMAP & TODO（GitHub 首页快速一览）

> **AI 同事进入项目第一站**：本文件是项目整体进度 + 后续优先级总览，3 分钟看完即可掌握项目当下状态。
>
> 详细决策记录见 [`docs/decisions/`](docs/decisions/)，工作纪律见 [`CLAUDE.md`](CLAUDE.md)。

---

## ⭐ 当前阶段：v0.1 Sprint 2 **已部署实战中**（5/25 09:30 真闭环跑通）

> **2026-05-23 ADR-028 战略转折**：LNN 从「24h AU 序列预测」改为「online 三分类意图估计 + 用户反馈闭环」。
> 复盘：[`docs/reports/2026-05-23-v0.1-direction-shift-complete.md`](docs/reports/2026-05-23-v0.1-direction-shift-complete.md)
> Sprint 2 spec：[`docs/decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md`](docs/decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md)
> **实战部署完整复盘（5/25 凌晨+早晨 7h）**：[`docs/reports/2026-05-25-sprint2-real-deployment-complete.md`](docs/reports/2026-05-25-sprint2-real-deployment-complete.md)

**真闭环实证**（5/25 08:44:03）：川哥点 [→正] → 1s 内 `~/.lnmem/data/intent_feedback/2026-05-25.jsonl` 落一条含 `hidden_state_window` 64-dim phase2 特征向量。Sprint 2 最小可用单位 ✅

**主分支**：`poc/lnn-validation`

**川哥 5/24 凌晨钦定三选 A**：
- Q1=A 数据流：Air daemon `.lnmem/data/*.jsonl` 每小时 rsync 到 Studio（排除 raw_video）
- Q2=A Studio 角色：纯训练（夜间 03:00 cron 训 intent classifier，Air 主动 pull ckpt）
- Q3=A Sprint 2 节奏：立即启动（不等数据，UI 和 classifier 同步上）

**Sprint 2 进度（5/25 09:30 实时更新）**：

| 任务 | 状态 | 产出 |
|------|------|------|
| S2-1 ADR-029 Sprint 2 spec | ✅ | `docs/decisions/029-*.md` |
| S2-2 codex 一轮：intent_head + lnn_inferencer + `/intent_feedback` | ✅ | commit `578baf3`，19 测试过 |
| S2-2b codex 二轮 retrofit：gemini H1/H2/H3/D4/U2 | ✅ | commit `2e260c6`，24 测试过 |
| S2-3 rsync 脚本 + sync_daemon_to_studio.sh | ✅ | Air→Studio 公网通 |
| S2-4 Studio 夜训入口 + Air pull ckpt | ✅ | `studio_nightly_train.sh` + `pull_ckpt_from_studio.sh` |
| S2-5 Streamlit UI v0 | ✅ | http://localhost:8520 跑中 |
| S2-6 gemini 8 finding 对抗审 | ✅ | CONDITIONAL GO + 10s hidden state 钦定 |
| S2-7 commit + push GitHub | ✅ | 8 个 commit 推到 `poc/lnn-validation` |
| S2-8 train_intent_classifier.py | ✅ | Dynamic Focal + Pos/Neg 10x oversample |
| S2-9 Studio venv 装依赖 | ✅ | uv + python 3.11 + torch 2.12 + ncps + sklearn |
| S2-10 Air 14 天 daemon 数据传 Studio | ✅ | 659MB jsonl + daemon_au_cache 19MB |
| S2-11 Studio dtaware 真训练 10 epoch | ✅ | loss 0.20→0.04，dt-aware actual_dt=0.994 |
| S2-12 拉 phase2 ckpt 回 Air + daemon 加载 | ✅ | `/health` ckpt_sha8=59cd1d9c 持续输出 |
| S2-13 notify_watcher.sh 系统弹窗 | ✅ | osascript 3 按钮 + 60s gave up |
| S2-14 第一条 intent_feedback 落盘 | ✅ | 5/25 08:44:03 含 hidden_state_window |
| S2-15 旧 ProactiveQueryEngine 静默化 | ✅ | enable_prompt=False 旁路 |
| S2-16 修 5 个 UX/Bug（详见复盘 §4-5） | ✅ | 8 个 commit 全 push |
| **S2-17 川哥反馈累计 50 条** | 🔄 | 当前 5 条，继续给反馈 |
| **S2-18 Studio 03:00 cron 训第一个 intent_head ckpt** | ⏳ | 等 ≥ 50 条触发 |
| S2-19 装 LaunchAgent 让 rsync/pull/夜训自动跑 | ⏳ | `scripts/install_launchd_air.sh` 待跑 |

**5/24 凌晨 Sprint 2 启动归档**：
- 战略锚：[ADR-028](docs/decisions/028-lnn-role-redefine-online-intent.md) + [ADR-029](docs/decisions/029-sprint2-intent-classifier-daf-studio-pipeline.md)
- 决策矩阵：cc + codex + gemini 三方并行 team 模式
- gemini 钦定建议：10s hidden state 窗口必须落盘（动力学 vs 统计学）

**项目哲学锚**：[ADR-009 拒绝 RAG](docs/decisions/009-no-rag-shortcut-stick-to-lnn-first.md) + [ADR-028 LNN 角色重定义](docs/decisions/028-lnn-role-redefine-online-intent.md)
**🏗 技术全景**：[`docs/architecture-overview.md`](docs/architecture-overview.md)
**🤖 AI 同事 30 秒上手**：[`docs/onboarding-ai.md`](docs/onboarding-ai.md)

---

## 历史归档：M0 W3 Day 4（**🟢 5/22 11 轮 Team 模式 + ADR-022/023/024 + v3.0 200ep 全门破阈值**）

> ⚠️ 这一段记录 5/22 凌晨的训练侧实证。**5/23 ADR-028 已把 LNN 重新定位**，
> 这里的 effective_dim / shuffle_ratio / vs_persistence 指标在 v0.1 已不是主验收维度。
> 保留作为「为什么转折」的因果证据。

**最近核心 commit（v0.1 转折前）**：`7d1fc5f` (ADR-024 AU 物理上限 · 5/22 晚) → `e38d28f` (R1+R3 LNN-FIRST 物理松绑 · 5/22 下) → `f3ee848` (gemini 11 stride=3 准静态死局 · 5/22 中) → `8e38bde` (ADR-023 范式转换 · 5/22 早)
**📊 11 轮完整复盘**：[`docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md`](docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md)
**📊 项目状态总览**：[`docs/project-status.md`](docs/project-status.md)
**📜 Day 3 6h Team 模式复盘**：[`docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md`](docs/retrospectives/2026-05-21-day3-evening-team-mode-retro.md)

## Day 3 晚上 Team 模式（5/21 20:00-23:30，6h）

**16 commits / 6 ADR / spec v0.2.4→v0.2.8 / ~10000 行 / gemini 6 轮 22 finding 全采纳**

### 6 轮 gemini 对抗评审进化

```
轮 1 REVISE → 轮 2 REVISE → 轮 3 REVISE → 轮 4 REVISE → 
轮 5 REVISE (4 硬门钦定) → 轮 6 CONDITIONAL APPROVE ✅
```

### γ-3 4 硬门状态

| # | 硬门 | 状态 |
|---|---|---|
| 1 | DGB ratio ∈ [0.1, 50] | ✅ 实测 0.797-0.836 |
| 2 | Shuffle Test M1 drop > 0.05 | ✅ **+0.241** 134% 提升 |
| 3 | AU augment M5 < 0.15 | ⚠️ N=10 物理极限 (0.833→0.500) |
| 4 | M1-V Spike 标 N/A | ✅ |

### γ-3 实测对比（50 epoch InfoNCE seed=42）

| 配置 | M1 top5 | M5 acc | Shuffle drop | effective_dim |
|---|---|---|---|---|
| baseline (无 augment) | 0.862 | 0.833 RED | +0.103 | 3.83 |
| **+ ADR-022 + AU augment 甜区** ⭐ | **0.897** | **0.500** | **+0.241** | 3.20 |

### ⭐ 决定性突破

- **ADR-022 τ-bias 梯度黑洞修复**（双率优化器 + DGB 压制 6.4e8 → 0.797）
- **AU augment 甜区 p=0.3**（信息瓶颈相变临界点，M5 -40% + Shuffle drop +134%）
- **gemini 第 6 轮揭穿「3.2d 低维流形伪泛化」**深度盲区

## 5/27 Phase γ Step 1 启动 5 项 Pre-conditions（gemini 第 7 轮终审 ✅ 全绿 · commit `96e47e2`）

| # | 防御 | 实施 | 5/22 实测验收 |
|---|---|---|---|
| 1 | **τ-Monitor + F-5 grad 时序** | epoch log τ.mean/std/range/grad_norm/mean_abs_delta；grad 在 backward 后 / step 前捕获 | ✅ τ=1.000 ∈ [0.1, 10]s (spec §86), grad_norm=4e-2, delta=1.8e-3 |
| 2 | **DataLoader shuffle=True + per-epoch reseed** | WindowDataset + `seed+epoch_idx` generator | ✅ epoch 0 vs 1 first-batch diff=27/32 (84% 变化) |
| 3 | **VICReg Std+Cov auto-trigger（软门控）** | `λ_cov=sigmoid(4·(2.5-d))` + std+cov 1:1 联合（防自我静音作弊） | ✅ λ=0.88 @ d=2.0, silenced loss > normal loss 4.3x |
| 4 | **ADR-022 v3 释放 tau_lr=1e-3** | DGB 单项控制（撤销 v2 双重锁死） | ✅ τ.mean_abs_delta=1.8e-3 物理位移 |
| 5 | **N=124 工程防御**（新增 F-4） | PC-5a memory cache + PC-5b truncated BPTT (max=32 步) | ✅ BPTT t≤7 input.grad 100%=0.0 数学证明 |

> 7 轮 gemini 评审 finding 总数：22 (Day 3) + 5 (Day 4) = **27 findings 全采纳**。详见 [`docs/reports/gemini-round7-final-signoff-2026-05-22.md`](docs/reports/gemini-round7-final-signoff-2026-05-22.md)。

## 5/22 早接手清单（已全完成）

| 优先级 | 任务 | 工时 | 完成 |
|---|---|---|---|
| ~~P0~~ | ~~4 项 Pre-conditions 实施~~ | 2-3h | ✅ |
| ~~P0~~ | ~~50 epoch 重测 effective_dim ≥ 8.0~~ | 30min | ⚠️ **发现 8.0 物理不可达 → ADR-024 改判** |
| ~~P0~~ | ~~gemini 第 7 轮终审 sign-off~~ | 5min | ✅（但 7 → 14 轮持续迭代）|

---

## 5/22 整日 11 轮 Team 模式（5/22 早 - 晚，**单日 13 commits / 3 ADR / 7 轮 gemini 对抗审**）

### 核心成就：**5/27 Phase γ Step 1 verdict 第 9 次更新 → 🟢 GREEN LIGHT 首次过线**

### v3.0 200ep 全 5 启动门破阈值（按 ADR-024 数据驱动相对阈值）

| 门 | 200ep 实测 | 阈值 | 状态 |
|---|---|---|---|
| τ.std > 0.5（多时间尺度真分化）| 0.577 | > 0.5 | ✅ |
| effective_dim / input_effective_dim | **0.97** (4.92/5.07) | ≥ 0.80 | ✅ **+21%** |
| Shuffle ratio（时序贡献）| **1.73** | ≥ 1.5 | ✅ **+15%** |
| vs_naive baseline | **0.632** | < 0.7 | ✅ |
| vs_persistence baseline | **0.50** | < 0.7 | ✅ |
| trivial_recon_check_pass | **True** | True | ✅ |

### 三大架构级 ADR 单日落地

- **ADR-022 v3** — τ-bias 释放 1e-3 + DGB 单项控制（撤回 v2 双重锁死）
- **ADR-023** — 范式转换 InfoNCE+L2 → Latent World Model Rollout（Dreamer 风格）
- **ADR-024** — AU 数据物理上限锁死（input effective_dim=5.07，相对阈值替代绝对 8.0）

### 5/23+ 接手清单（更新）

| 优先级 | 任务 | 工时 | 状态 |
|---|---|---|---|
| **P0** | gemini 第 14 轮 sign-off + ROADMAP/memory 沉淀 | 完成中 | ✅ |
| **P0** | **罕见 AU 重建监控加入 evaluate.py**（gemini 14 N=124 警告 → ADR-025） | 1h | ✅ 5/22 21:45 完成 |
| **P0** | ~~β-self-test 川哥本人 webcam + 10 句~~ → daemon 14 天 378k 帧已收割 | — | ✅ 5/22 22:15 替代完成 |
| **P0** | **TODO-1 dt-aware ODE 实施 + 200ep 验证**（ADR-027）| 2 人日 | ✅ 5/23 凌晨完成 |
| **P0** | **TODO-2 Phase 2 multi-fps curriculum**（spec V0.2 + gemini 5 finding 整改 ready）| 3 人日 | ⏳ **待川哥拍 launch** |
| **P0** | **τ clamp 加入防长训失控**（ADR-027 §5.6）| 0.5 人日 | ⏳ |
| **P1** | 5/24 daemon 长跑后台启动 | — | ✅ 已在跑 14 天 |
| **P2** | 5/27 **Phase γ Step 1 启动**（推迟，需 TODO-2 验收过 6 道门）| 4 天 | ⏳ 推迟 |

### 5/27 启动 7 道门 V3（ADR-024 + ADR-025/026 + ADR-027 dt-aware/cross-dist）

| # | 门 | 来源 | 当前 dt-aware 200ep 实测 |
|---|---|---|---|
| 1 | τ.std > 0.5 **且 τ.max < 100s（防爆炸 ADR-027）** | ADR-024 + 027 | ❌ τ.max=3.5e9（200ep 失控）|
| 2 | vs_persistence < 0.7 | ADR-024 | ✅ VEATIC 0.54 / ❌ daemon 17.65 |
| 3 | effective_dim / input ≥ 0.8 | ADR-024 | ⚠️ VEATIC 0.71 |
| 4 | Shuffle ratio ≥ 1.5 | ADR-024 | ⚠️ VEATIC 0.51 / ❌ daemon 0.057 |
| 5 | rare_au_verdict ∈ {green, yellow} | ADR-025/026 | n/a (daemon 整体崩盘) |
| 6 | **cross-distribution shuffle ratio ≥ 0.5（daemon 1Hz 必测）** | **ADR-027 新增** | ❌ 0.057 |
| 7 | **Variance-Aware shuffle 防伪**（std(pred) ≥ 0.1*std(target)） | **ADR-027 Phase 2** | 未实施 |

**Phase 1 (dt-aware) 实测**：τ 分布从 [0.991, 1.009] 解放到 [0.96, 34]（30ep）= LNN-FIRST §1.2 多时间尺度第一次实现 ✅；
但 200ep τ 失控爆炸到 3.5e9（死神经元）；daemon 跨分布仍崩。

---

## 🔥 5/23 凌晨方向重定义（川哥钦定 + ADR-028 签字）

**LnMem v0.1 = 实时意图预测助手**（不再 AU 序列预测）：
- LNN 实时观察 30Hz 多模态信号 → 预测三分类意图（正反馈/负反馈/无反馈）
- AI 主动推断 → UI 弹出 → 川哥 [确认/纠正] → 增量 supervised → per-user τ-bias 漂移
- "陪你越久越懂你" = 7 天后准确率 35% → 65%+

**5/22-23 经验**：4 个训练方案在 daemon shuffle ≈ 0 → 三方战略复盘 → 川哥点睛"我要的是意图预测不是 AU rollout"

### 5/24+ Sprint 路径（ADR-028 钦定）

| Sprint | 时间 | 主任务 | 状态 |
|--------|------|--------|------|
| **Sprint 0** | 5/24 一天 | ADR-028 + ROADMAP + 冻结旧训练 | ⏳ in progress |
| **Sprint 1** | 5/25-5/31 | 30Hz daemon + 实时 inference server + UI v0 观察期 | ⏳ |
| **Sprint 2** | 6/1-6/7 | 意图三分类头 + 主动弹窗 + 反馈闭环 | ⏳ |
| **Sprint 3** | 6/8-6/10 | M0 收口 demo + 24h 学习曲线 | ⏳ |

### 已冻结的 5/22-23 主线（基础设施沉淀，未来可能复用）

| 路径 | 状态 |
|------|------|
| ode_loss rollout 训练 / multi-fps curriculum / invariance loss | 🗑️ 冻结但代码留 |
| shuffle_ratio 主目标 | 🗑️ 降级为内部诊断 |
| **dt-aware ODE / τ 解放 / daemon adapter / per-user τ-bias 数学** | ✅ **100% 复用 v0.1** |
| ADR-025/026/027（rare AU / dt-aware / 跨分布）结论 | ✅ 保留 |

### 5/27 不启动 N=124 / 6/10 M0 收口验收

| # | 项 | 6/10 目标 |
|---|---|---|
| 1 | 万能语法 20 槽 | ⏳ M1（v0.1 简化）|
| **2** | **多时间尺度 LNN 心智核心** | ✅ **online intent estimator 真正落地** |
| 3 | CLS 离线 consolidation | ⏳ M1 |
| **4** | **per-user 个性化栈** | ✅ **第一次端到端 work** |
| 5 | 居家+车载双场景 | ⏳ M2 |
| **6** | **用户可见可治理界面** | ✅ **实时意图 UI + 反馈按钮** |

完整设计：[`docs/decisions/028-lnn-role-redefine-online-intent.md`](docs/decisions/028-lnn-role-redefine-online-intent.md)

### 11 轮 6 次 gemini 自我弹劾 + 5 大经验教训

详见 [`docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md`](docs/reports/2026-05-22-team-mode-11-rounds-retrospective.md)

## 5/22 修订周启动（方案 A · 已 99% 完成于 5/21 晚单晚 6h Team 模式）

> 5/21 22:30 川哥拍板方案 A。**目标对账**：每个任务都用「能否服务于基于 LNN 的多模态识别与预测」单一维度过滤。
>
> **5/21 23:30 实际进度**：原 5 天计划 5 天压缩到 6h 完成 80%+：γ-0a/γ-0b/γ-1/γ-2 全完工 + γ-3 实施层 + 4 硬门 + 6 轮 gemini 评审。剩 4 项 Pre-conditions 留 5/22 早 2-3h。
>
> **公网链路实测**：Mini=192.168.100.2 / Studio=192.168.100.3 雷电网桥实际通（cc 之前 IP 误判，川哥纠偏）。Studio ncps 1.0.1 + torch 2.8 MPS + 9.8GB 数据 + py-feat 真训练通过 ✅。

**5 天工作流**（详见 [`docs/project-status.md`](docs/project-status.md) §6）：

```
5/22 上 γ-0 Studio 前置就绪（70% 已预热）
5/22 下 β-self-test 川哥 10 句自体验
5/22 晚-5/23 γ-1 数据 pipeline + ncps spike
5/23-5/24 γ-2 2 层 CfC 实测 + collapse 四联监控
5/24 晚 daemon 长跑后台启动（→ 6/10 累积 17 天真人数据）
5/25 上 γ-3 InfoNCE vs VICReg（半天压缩）
5/25 下 v2 collapse reproducer 落盘
5/26 γ-preflight 10-video 11 项 checklist
5/27 🚀 Phase γ Step 1 启动
```

## 2026-05-21 Day 2 进展 🎯

> **L4 W+ 三砖 + Phase α 多模态 + Phase β 5 砖 + γ-spec v0.2.3 = 单日 15 commits / 4 ADR**

### 单日 15 commits / 4 ADR (016/017/018/019) / γ-spec codex 4 轮 APPROVE / ~5000 行
### 详细复盘：[docs/retrospectives/2026-05-21-day2-phase-alpha-retro.md](docs/retrospectives/2026-05-21-day2-phase-alpha-retro.md)

### 关键 commits（按时间倒序）

```
265c8f4 docs(γ-spec v0.2.3 final): codex 4 轮 APPROVE 通过
1c580db docs(γ-spec v0.2.2): codex 第三轮 REVISE 抓 3 条 gap 全修
99b6a5e docs(γ-spec v0.2.1): codex 第二轮 REVISE 抓 6 条 gap 全修
e5753c0 docs(γ-spec v0.2 + ADR-019): 吸收 codex 5.4 + gemini 3.5 双评审
04f7aae feat(β-3+β-5): demo 完整状态描述 + multimodal 持久化
fc69ccb fix(multimodal): py-feat 3 bug 修复 + V-A 投影 v3 + mock 强警告
27b9361 fix(demo): 修 3 个割裂感 bug + UI 全面友好化
f4a3340 feat(L4-3+phase-α): Streamlit demo 4-in-1 升级
e51eedc feat(β-1/2/4): EventTensor v1.1 + parser intent_type + graphiti SQLite
5efdd37 feat(L4-2): CLS v0 夜间整合最小可行
720732b docs(ADR-016): 撤回 ADR-015 + 立 base v3 范式
ed744d1 feat(L4-1): M0 端到端 demo CLI 跑通
df5002e docs(ADR-015): 视觉感知路径冻结（被 ADR-016 撤回）
795bef4 feat(ADR-013/014): base v2 失败分析 + v2.1 DFAT 决策检查点
```

### 7 commits / 3 ADR / ~1400 行 · 详细复盘：[docs/retrospectives/2026-05-21-day2-phase-alpha-retro.md](docs/retrospectives/2026-05-21-day2-phase-alpha-retro.md)

| Commit | 内容 |
|---|---|
| [`27b9361`](../../commit/27b9361) | **fix(demo)** 修 3 个割裂感 bug + UI 全面友好化 |
| [`f4a3340`](../../commit/f4a3340) | **feat(L4-3+phase-α)** Streamlit demo 4-in-1 升级 + ADR-017 |
| [`5efdd37`](../../commit/5efdd37) | **feat(L4-2)** CLS v0 夜间整合最小可行 |
| [`720732b`](../../commit/720732b) | **docs(ADR-016)** 撤回 ADR-015 + 立 base v3 多模态 self-supervised LNN |
| [`ed744d1`](../../commit/ed744d1) | **feat(L4-1)** M0 端到端 demo CLI 跑通 |
| [`df5002e`](../../commit/df5002e) | **docs(ADR-015)** 视觉感知路径阶段性冻结（被 016 撤回）|
| [`795bef4`](../../commit/795bef4) | **feat(ADR-013/014)** base v2 失败分析 + v2.1 DFAT 决策检查点 |

### Day 2 关键决策

1. **ADR-016 范式转换**：撤回 ADR-015 视觉冻结，立 base v3 multimodal self-supervised LNN（4 loss / 9.8GB 数据全留）
2. **L4 W+ 三砖完工**：M0 e2e demo / CLS v0 / 记忆 UI v0.1
3. **Phase α 落地**：webcam snapshot → V-A + 9-state + 20 AU 实时显示（ADR-016 input signal 预览）
4. **3 bug 修复**：LLM thinking 默认 / streamlit cwd 失锚 / streamlit 14+ deprecation
5. **UI 友好化范式**：HELP 字典 11 术语 + 4 步引导 + tooltip + 友好错误提示

### Day 2 实测指标（Phase α demo）

- LLM parse: **4.91s/句**（修复前 45s 超时，**9x 改进**）
- P0 槽位填充: **4/4（100%）**
- 全量测试: **30 passed / 1 skipped** ✅
- CLS 精华显示: ✅ 昨日 5 events → 3 highlights

### Day 2 Team 协作纪实

| 引擎 | 角色 | 状态 |
|------|------|------|
| **川哥** | 守门员 / 2 次关键纠偏（视觉范式 + UI 友好化） | 主战 |
| **cc 主线** | 实现 + 文档 + 终验 + commit | 主战 |
| **gemini-3.5-flash** | 5 维度评审 high reasoning（18.7KB 输出，3 处分歧记 ADR-017）| 评审 |
| **codex 5.3** | 后台写 multimodal | 14:00 quota 用尽，cc 接管 |

---

## 2026-05-20 Day 1 进展（Team 模式启动）

> 8 commits / 2710 行 / 11h 主战。详细复盘：[`docs/retrospectives/2026-05-20-day1-team-mode-retro.md`](docs/retrospectives/2026-05-20-day1-team-mode-retro.md)

主要产出：ADR-007 Phase 1.2-1.4（VEATIC quality probe + AU extraction + train_base_v2）+ ADR-012 V-A→state Russell + graphiti Phase A InMemory backend。

---

## 2026-05-16 单日重大进展 🚀

> **PoC 假设决定性验证日**：5/12 LNN 首次自然触发 + 5/16 揭穿 calibration polarization + LNN realtime 接入

### 6 commits / 7 ADRs / 522 tests / 5 轮 codex + 3 轮 gemini 协作

| Commit | 内容 | 影响 |
|---|---|---|
| [`4031ee0`](../../commit/4031ee0) | watchdog HTTP 假死修复 + SIGUSR1 + Phase B 复盘 | daemon 健康度从"假活"→真活 |
| [`168d7ec`](../../commit/168d7ec) | fit-cron + 增量水位线防漂移 | PoC 持续学习闭环真闭合 |
| [`9a5c9ad`](../../commit/9a5c9ad) | **ADR-005 dual signal_path** (raw_au.AU04) | 修 calibration polarization 22% 信息熵损失 |
| [`602506d`](../../commit/602506d) | dual replay 离线 A/B 对照 | 揭穿 calibrated 通道 99.6% 是假皱眉信号 |
| [`ec0e109`](../../commit/ec0e109) | 4 个 ADR (006/007/008/009) + 词表 + schema | 锁定规模化 5 阶段路线 |
| [`3c10834`](../../commit/3c10834) | **Phase 4 LNN realtime 接入 daemon** | margin/entropy 落盘 signals jsonl |

### 决定性 PoC 假设证据

1. **D-1 信号熵分析**：raw_au.AU04 熵 83% vs calibrated polarized 61%（损失 22%）
2. **Dual Replay**：calibrated 与 raw 两路 trigger 共识仅 0.4% → 99.6% 假皱眉
3. **Spike 暴露 base ckpt OOD**：LNN 在川哥真人脸上 state 准确率 0%（说 focused 时 LNN 预测 emotional_negative）
4. **Phase 4 验证 LNN realtime 工作**：daemon /health 显示 LNN forward margin=0.9994 但**虚假自信** — 完全证实 ADR-007 必要性

---

## 7 个 ADR 决策链（007-009 是今日新增的规模化路线）

| ADR | 主题 | 状态 |
|---|---|---|
| [ADR-001](docs/decisions/001-route-b-single-founder-ai-mode.md) | Route B 单创业者 AI 模式 | accepted |
| [ADR-002](docs/decisions/002-adversarial-review-m0-skeleton.md) | M0 骨架对抗审查 | accepted |
| [ADR-003](docs/decisions/003-calibration-violates-lnn-first-bypass-to-raw.md) | calibration 信息退化（口径修订） | superseded by 005 |
| [ADR-004](docs/decisions/004-ab-experiment-raw-vs-calibrated.md) | A/B 实验框架 | accepted |
| [ADR-005](docs/decisions/005-trigger-uses-raw-au04-signal-path-dual.md) | dual signal_path 最小改动方案 | accepted |
| [ADR-006](docs/decisions/006-threshold-auto-recalibration.md) | trigger 阈值自动重校准 | placeholder |
| **[ADR-007](docs/decisions/007-base-ckpt-realhuman-affectnet.md)** | **base ckpt 真人化（AffectNet V-A）** | **superseded by [ADR-016](docs/decisions/016-base-v3-multimodal-self-supervised-lnn.md)**（2026-05-21 双评审统一） |
| **[ADR-008](docs/decisions/008-user-lora-multi-user-schema.md)** | **User-LoRA + multi-user schema** | accepted（M3 启动前必做） |
| **[ADR-009](docs/decisions/009-no-rag-shortcut-stick-to-lnn-first.md)** | **拒绝 RAG 捷径，坚守 LNN-FIRST** | accepted（哲学级） |

---

## TODO（按 PoC 资源分配铁律排序）

### 🔥 立即（M0 W3 早期，本周）

- [ ] **TODO-1 Schmitt Trigger + EMA + dynamic quantile threshold**（codex F-04 + gemini #2）
  - margin/entropy 阈值不再拍脑袋，用 sweep 分位数 + 滞回防震荡
  - 工时：1-2h
  - 前置：scripts/sweep_margin_quantile.py 已就绪
- [ ] **TODO-2 dual replay v2 — 包含 LNN 预测对照**
  - 现有 dual replay 只对比 trigger 通道
  - 加 LNN forward 预测，对比"calibrated ckpt vs raw ckpt 在同 events 上预测差异"
  - 工时：1.5h
- [ ] **TODO-3 daemon stable 跑 7 天累积 raw 路径真实信号**
  - 不动代码，纯数据收集
  - 数据用于 ADR-007 base ckpt 真人化训练
  - 目标：≥1000 events with user feedback（当前 < 100）

### ⭐ 短期（M0 W3-W4）

> **2026-05-21 双评审统一**：原 TODO-4/5（ADR-007 AffectNet）已被 ADR-016 base v3 multimodal self-supervised LNN 取代。Phase γ (5/27-6/10) 直接走 VEATIC+EMOTIC+mst 自监督路径，不再走 AffectNet 重训路径。TODO-6 改 P2 等 Phase β 修改卡片 UI 一并做。

- ~~TODO-4 ADR-007 实施：AffectNet 数据准备~~ → **superseded by ADR-016 Phase γ**
- ~~TODO-5 ADR-007 实施：base ckpt v2 重训~~ → **superseded by ADR-016 base v3**
- [ ] **TODO-6 对话流自然询问替代 macOS 弹窗**（cognitive.md 合规，P2，与 Phase β 修改卡片 UI 一并做）

### 🌅 中期（M1-M2 / 1-3 个月）

- [ ] **TODO-7 ADR-008 实施：multi-user schema 工程化**
  - users.json 路由层
  - feedback / signals jsonl 加 user_id 字段（schema 已就绪，工程化路由层未做）
  - fit_orchestrator 多 persona batch 改造
  - 工时：1 周
- [ ] **TODO-8 ADR-008 实施：User-LoRA per persona**
  - OnlineLoRAAdapter 改造为多 user LoRA 路由
  - 验证 5-10 user 同时学习不互相污染
  - 工时：1 周
- [ ] **TODO-9 SIGUSR1 dump 不工作根因排查**（known issue）
  - faulthandler.register 在 daemon ctx 不写文件（OpenCV 抢占？）
  - 不阻塞 watchdog 重启核心，但调试时需要
  - 工时：1h 调研

### 🚀 长期（M3+ / 3 个月以上）

- [ ] **TODO-10 联邦学习（Flower 框架）**
  - 多用户 LoRA 加密上传聚合
  - gemini 警告 Apple sandbox 限制需先验证
- [ ] **TODO-11 边缘推理 + 云端集中学习**
  - 推理在用户端（隐私 + 低延迟）
  - 类似 Apple Private Federated Learning
- [ ] **TODO-12 居家 + 车载场景化分层 ckpt**
  - §10.1 第 5 项 "居家+车载双场景统一架构"

---

## 当前 daemon 实时状态（2026-05-16 20:00 CST）

```bash
$ curl localhost:9876/health | jq
{
  "status": "ok",
  "uptime_s": 110,
  "frames_processed": 110,
  "feedback": {
    "observed": 110,
    "triggered": 0,
    "queried": 0
  },
  "lnn": {                          # ← Phase 4 新增
    "enabled": true,
    "ckpt_sha8": "10cc41c3",        # 最新 raw 路径 ckpt
    "persona_idx": 30,              # vboss_self
    "signal_path": "raw",           # ADR-005 raw 通道
    "history_len": 60,              # warmup 完成
    "fail_count": 0                 # 无错
  }
}

$ tail -1 ~/.lnmem/data/signals_*.jsonl | jq '.lnn'
{
  "status": "ok",
  "predicted_state": "emotional_positive",  # ← LNN 实时预测（虚假自信，证实 ADR-007 必要）
  "top_prob": 0.9995,
  "margin": 0.9994,
  "entropy": 0.0066,
  "ckpt_sha8": "10cc41c3",
  "signal_path": "raw"
}
```

---

## 经验教训（一天 5 轮 codex + 3 轮 gemini 协作沉淀）

### Team 协同的真实价值

1. **codex 对抗审查救了 3 个 CRITICAL bug**：cc v0.1 LNN 接入方案直接落地会全链路崩溃（state_model.forward 签名漏传 persona_idx + fusion 输入 schema 错位 + LNMEM_SIGNAL_PATH 切换变量虚构）
2. **gemini 给出 outside-in 关键视角**：margin > entropy（8 分类下）+ AffectNet > FER2013 + RAG 捷径警告 + Flower 联邦推荐
3. **cc 撤回 3 次错误方案**：D-2 calibration table（违反 P-1） → v0.1 全链路重写（codex 否决）→ v0.2 raw 旁路 LoRA（过度工程）→ v0.3 最小改动（最终成功）

### 哲学锚定 > 工程便利

- **ADR-009 拒绝 RAG 捷径**：gemini 直接建议 "M3 前插 RAG/长上下文捷径"（Pi/Character.ai 都用），但 LnMem 差异化恰好是"不靠 RAG"。**坚守 P-1 LNN-FIRST 比 5-10x 工程难度更重要**。
- **lnn-first.md §4.1 明确视觉感知层不受 P-1 约束**：cc 把 calibration 定性为"违反 P-1"是论证不准确，codex F-03 揪出并修订为"AU04 算法信息退化"。**哲学条款使用要精确**。

### 数据驱动 > 拍脑袋

- **dual replay** 用 4 天 241K 帧数据证明 calibrated/raw 触发共识仅 0.4% — 比任何理论分析都有说服力
- **quantile sweep** 给出数据驱动阈值（thr=0.55 P91 → 每天 5-6 次触发），不再凭直觉
- **spike-then-implement** 把 5-7h 工程缩到 2h（spike 验证后才进实施）

### codex 多轮审查的复利

| 任务 | codex 一审 finding | 修后 codex 二审 finding | 修后 codex 三审 finding |
|---|---|---|---|
| daemon watchdog | 5 (1 CRITICAL) | 3 (2 MEDIUM) | - |
| fit cron | 5 (3 HIGH) | 3 (2 HIGH) | 3 (1 HIGH + 1 MED) |
| raw 路径设计 | 7 (3 CRITICAL) | 2 HIGH (F-08/F-09) | - |
| X 方案 v0.2 | 8 (1 CRITICAL + 5 HIGH) | - | - |

每轮 codex 都揪出实质问题，**默认采纳 + 修复 + 再审**是最稳模式。

---

## 给 AI 同事的快速 onboarding（3 步）

1. 读本文件了解当前阶段
2. 读 [`CLAUDE.md`](CLAUDE.md) 工作纪律（三道门 + P-1 LNN-FIRST）
3. 看最近 3 个 ADR：[007 base 真人化](docs/decisions/007-base-ckpt-realhuman-affectnet.md) / [008 multi-user](docs/decisions/008-user-lora-multi-user-schema.md) / [009 拒绝 RAG](docs/decisions/009-no-rag-shortcut-stick-to-lnn-first.md)

---

**最后更新**：2026-05-16 20:05 CST（cc 单日 6 commits + 7 ADRs + 522 tests 后落盘）
