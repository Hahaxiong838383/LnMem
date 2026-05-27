# ADR-038: 路径 D++ 现场可交互 demo 工程基建合订本

**状态**：accepted（v1.1 经 codex gpt-5.4 high + gemini-3-pro-preview 二轮对抗审）
**日期**：2026-05-27
**决策者**：川哥（5/27 钦定路径 D 三站方案 + 节点可推后 + team 模式开干） / cc（主案 + 修订执行） / codex（对抗审 + schema 设计） / gemini（深度审 + LNN-FIRST 违例检出）
**触发**：5/27 川哥定调 demo 必须现场可交互证明 LNN 差异化（非录屏），节点可推后；cc 起草路径 D 三站方案 → team 模式（codex gpt-5.4 high + gemini-3-pro-preview）评审挖出 8 处代码级 / 演示物理级硬伤 → cc 综合修正为路径 D++ v1.0 → 二轮对抗审挖出 LNN-FIRST 违例 + 数学非法 + 验证不可复核 → 修订为 v1.1（本文）
**关联**：ADR-033（LTC ODE 真预测）、ADR-036（A+B 自回归外推）、ADR-037（state 投影对齐）
**体裁**：L3 决策文档（依 [`docs/STYLE.md`](../STYLE.md) §L3）
**父仓全局规则参考**：`~/.claude/rules/no-guessing.md`、`~/.claude/rules/failure-escalation.md`、`~/.claude/rules/testing.md`（双门验证）——cc/codex/gemini 三引擎全局执行
**本仓工程纪律**：[`docs/development-harness.md`](../development-harness.md)（BRO-PDCA × 目标驱动验收）、[`CLAUDE.md`](../../CLAUDE.md) §3 三道门、§5 红线、§3.1 意见冲突诊断

---

## 背景

### 项目阶段对账

LnMem M0 W3 末，技术 85% / 产品 35% / 商业化 0%。原 M0 demo 6/3 节点不可达 demo 真正目的（向技术朋友证明 LNN 不可替代）。川哥 5/27 拍板四条：

1. 不录视频，要现场可交互
2. 必须展现 LNN 在系统里的差异化效果
3. 创新项目交付节点可推后
4. 围绕项目内核（LNN-FIRST P-1 准则）重新设计

cc 据此起草路径 D（三个对照实验站，让观众亲手验证 LNN ≠ ChatGPT Memory），川哥钦定：
- 顺序站 1 → 站 2 → 站 3
- 观众先内圈技术朋友（含老板）→ 稳定后投资人
- Team 模式调最高能力评审，cc 综合后再开干

### 一轮评审挖出的 5 类硬伤

#### A. 代码级硬伤（codex gpt-5.4 high 仓库证据）

| 代码位置 | 现状 | 对站 3 的影响 |
|---|---|---|
| [`daemon_main.py:610`](../../poc/src/realtime/daemon_main.py:610) | 实时推理路由是单个 `LNMEM_PERSONA_IDX` 环境变量 | 双人接力"B 接手立即识别"runtime 层 0% 实现 |
| [`fit_orchestrator.py:124`](../../poc/src/lnn/fit_orchestrator.py:124) + [`feedback_to_batch.py:241`](../../poc/src/lnn/feedback_to_batch.py:241) | 所有反馈被视为同一 persona | 多用户路由 = 0% |
| [`online_adapter.py:89`](../../poc/src/lnn/online_adapter.py:89) | 在线学习只更新 `tau_drift`，FiLM 不动 | 画像演化幅度小到现场看不见 |
| [`apps/demo_e2e/main.py:445`](../../apps/demo_e2e/main.py:445) | 现有删除是 KG `invalidate`，注释明写"数据未物理删除" | "删除 = 抑制 τ-bias"目前是口号不是能力 |
| [`online_contracts.py:108`](../../poc/src/lnn/online_contracts.py:108) | 在线学习触发门槛 10 条总反馈 + 10 条 mismatch | 5-10min demo 内根本触发不到 drift |
| [`inference_server.py:93`](../../poc/realtime/inference_server.py:93) | τ 信息只暴露 `short_tau_mean/long_tau_mean` 粗粒度 | 强行画 30s/5min/1h/1d/7d/∞ 整齐光谱 = 叙事先行 |

#### B. 演示物理级硬伤（gemini-3-pro-preview 架构审）

1. **大 τ 不动悖论**：7d/∞ 神经元在 10min demo 内正常不会动，画出来是直线
2. **1.5s 延迟 + state 锁死 emotional_negative** 在交互 demo 现场 = 公关灾难
3. **"上帝视角"反噬**：把 τ-bias 像心电图剖出来反而破坏"伴侣感"幻觉
4. **跨模态冲突未处理**：观众一边大笑一边愤怒说"我很生气"，融合层会精神分裂

#### C. 公平对照硬伤
- 站 1"RNN 单 τ 必崩"会被技术朋友抓住——GRU 不是显式连续时间但不等于只有一个时间尺度
- 站 2 把 AU 离散化喂 Mem0 = "你多模态转文本损失不是 Mem0 的错"

#### D. N=1 数据硬伤
- 当前只有川哥一个真人 daemon 数据，算法隐式过拟合；触发门槛 10 条 mismatch 演示窗口内达不到

#### E. 哲学一致性硬伤
- [`cognitive.md:11`](../capability-specs/cognitive.md:11) 明写"评估应不打扰，与多模态理解同构"
- README 当前主线仍重度依赖弹窗反馈/纠正

### 二轮评审挖出的 3 类违例（v1.0 → v1.1 必改）

**违例 1 · F5 高斯扰动违反 LNN-FIRST**（gemini + codex 共识）：v1.0 "看门狗注入隐状态高斯扰动破吸引子"在数学上是离散状态跳变，违反 [`lnn-first.md`](../principles/lnn-first.md) I1 规则"涉及时间的状态变量禁止用阶跃"。本仓 ADR-037 也已诚实把 state 锁死归因到投影对齐+intent 欠阻尼，不是用 heuristic 注入解决。

**违例 2 · F1 反向抑制数学非法**（gemini + codex 共识）：在线训练 loss 是 reward-weighted NLL + clipping + argmax 补 label + batch 顺序相关 ([`online_trainer.py:91-156`](../../poc/src/lnn/online_trainer.py:91))，简单"参数取负"不是合法逆运算。LTC 非线性系统的"遗忘"唯一物理合法路径是 **Surviving Feedback Replay**（从最近 clean ckpt 重放剔除该 event 的所有反馈）。

**违例 3 · 验证指标不可复核 + 与 cognitive.md 冲突**：v1.0 含"懂行人能看出"、"≥ 2/3 技术朋友主动说"、"70% 观众调研没感到延迟"——既是主观调研（与 cognitive.md "评估不打扰" 直接冲突），也无法外部复核。development-harness §AC 要求"可量化且可复核"。

---

## 决策（v1.1）

**采纳路径 D++ v1.1**：6/24 内圈 demo = 站 1-lite + 站 2-full + 站 3-roadmap（含真实可操作中间物），7 月 3 站全开版才面向投资人。

具体决策 9 条：

1. **站结构重排**：站 1 缩为 2-3min 证据前言；站 2 升为绝对主战场（60% 时长）；站 3 降为 roadmap 但**必须展示真实可操作中间物**（SQLite schema + 真实样例记录 + CLI 查询命令），不是 PPT 设计图
2. **公平性产品化为双 lane**：每站 UI 内置 `Strict Fair Lane`（同输入信息边界）+ `Full Product Lane`（明告"whole product 对比"）两按钮；每站 baseline 模型 / 参数量 / 输入范围 / 延迟预算直接写 UI
3. **必做 6 项工程基建前置**（见 §工程基建清单 v1.1）
4. **N≥3 数据底线**：先合成 3 段 persona（有 baseline 漂移 + 局部 burst + 转场平滑段 + 跨 AU 协方差结构，不是常值模板），再尽力补真人；6/24 不宣称 generalization 只说 feasibility
5. **τ 神经元先 audit 再画**：当前 short/long 粗粒度先做物理含义标注，每个神经元给出"为什么属于 X τ"的代码证据；站 1 现场展示的 τ 光谱**每个神经元都有 ADR 级出处**
6. **删除三阶段化**（替代原"反向抑制"）：
   - **Phase A**（6/24 ship）：lineage 可查（fit_run_id → event_uuid_list → ckpt_in/out_sha8）
   - **Phase B**（7 月 ship）：Surviving Feedback Replay 可重建
   - **Phase C**（不承诺日期）：反向抑制仅作 demo-safe 近似可视化，永不写回训练账本
7. **演示安全网拆两层**（替代 v1.0 F5）：
   - **Layer A · Render-layer smoothing**：前端 1s 平滑插值，UI 显式标注 `display-smoothed`；strict fair lane 同时显示 raw latency
   - **Layer B · Core-state intervention**：**默认关闭**；若开启则仅在 sandbox lane 用 ODE 合法方式（注入持续 Neutral 控制信号经 `dx/dt` 积分平滑引导出 negative lock），perturbed 状态**不写回训练账本 / 不进 RL 反馈窗口**
8. **时间曲率旋钮严格边界**：dt 倍速 1× / 10× / 100×（**取消 1000×**，避免打爆 KG 写入速率）；倍速模式下 RL 反馈窗口暂停采集（防 cognitive.md 反馈污染）；UI 显示真实时间 + 模拟时间双时间轴
9. **不在 demo 中宣称"立刻删掉某条记忆并完全逆转 τ-bias"**——只承诺 Phase A lineage 已闭环

---

## 工程基建清单 v1.1（6 项 P0 前置）

> **决策权矩阵**：每个 F 项的 schema/口径**最终决策权归属**已明确，其他角色提建议但不阻塞。冲突按 [`CLAUDE.md`](../../CLAUDE.md) §3.1 走 ADR。

### F1 · Attribution Ledger（lineage 优先，不做反向抑制）

| 维度 | 内容 |
|---|---|
| Owner | cc 主案 |
| Schema 决策权 | **codex**（gpt-5.4 high effort 提案，cc 实施） |
| 工时 | **5 天**（v1.0 估 3 天过乐观，codex 指出是全链条改造） |
| 数学合法路径 | 仅 Surviving Feedback Replay；反向抑制不进治理闭环 |

**Schema（3 层，codex 设计）**：

```sql
-- 真源：SQLite（事务 / 一致性 / 删除标记 / 增量查询）
CREATE TABLE fit_runs (
  fit_run_id TEXT PRIMARY KEY,
  ckpt_in_sha8 TEXT NOT NULL,
  ckpt_out_sha8 TEXT NOT NULL,
  fired_at_utc TEXT NOT NULL,
  persona_idx INTEGER NOT NULL,
  event_uuid_list_hash TEXT NOT NULL,
  batch_size INTEGER,
  config_snapshot_json TEXT,
  success_flag INTEGER NOT NULL
);
CREATE TABLE event_attribution (
  fit_run_id TEXT NOT NULL REFERENCES fit_runs(fit_run_id),
  event_uuid TEXT NOT NULL,
  event_order INTEGER NOT NULL,
  reward REAL,
  label_source TEXT,
  window_frame_ids TEXT,
  contribution_mode TEXT,
  PRIMARY KEY (fit_run_id, event_uuid)
);
CREATE TABLE delta_summary (
  fit_run_id TEXT PRIMARY KEY REFERENCES fit_runs(fit_run_id),
  delta_l2_total REAL,
  delta_l2_by_persona_json TEXT,
  topk_dims_json TEXT
);
```

**Parquet 角色**：仅做离线分析导出，异步 materialize，**不参与主路径 ack**（写失败只降级分析，不回滚训练）。

**Fail-safe 顺序**：先原子写 ckpt → 再事务写 SQLite ledger → 异步 parquet materialize。

**DoD（外部可验证）**：
- ✅ 任意 `ckpt_out_sha8` 可反查 `fit_run_id → event_uuid 列表 → ckpt_in_sha8`
- ✅ CLI 命令 `lnmem ledger lineage <ckpt_sha8>` 输出完整 lineage 链
- ✅ Phase A 测试：跑 3 个连续 fit_run，断电模拟，重启后 ledger 完整无孤儿记录
- ✅ Phase B 测试（不阻塞 6/24）：剔除某 event 从最近 clean ckpt replay，新 ckpt 与原 ckpt 在剔除该 event 影响后的预测分布 KL ≤ 0.1

### F2 · Matched Baseline 训练（公平性 6 口径 operationalize）

| 维度 | 内容 |
|---|---|
| Owner | codex high effort |
| 口径决策权 | **cc**（公平性是 demo 信用地基，cc 负责对外） |
| 工时 | **3 天**（v1.0 估 2 天过乐观，含调参轮次对齐） |

**6 个公平口径（必须全部满足，UI 显式展示）**：

| # | 口径 | 阈值 |
|---|---|---|
| 1 | 参数量误差 | ≤ 5% |
| 2 | 输入模态 | 完全同源（AU + head_pose + screen 或同一文本流） |
| 3 | 输入预处理 | 完全同源（同 normalization / 同 window） |
| 4 | 训练 budget | 固定 wall-clock 或固定 step（择一） |
| 5 | 调参轮次 | 一致（baseline 不能"只默认参数跑几轮"） |
| 6 | 推理延迟 | 同硬件同 batch size 实测 |

**Baseline 模型**：Phased LSTM（真 time-gate 输入 `dt`，不偷固定步长）+ Hier-GRU（明示层级时间下采样）。

**LNN-FIRST 兼容性声明**：RNN baseline 在此**仅作 comparator**，不替代主路径，不违反 [`lnn-first.md`](../principles/lnn-first.md) §3 Override 协议。

**对比展示**：站 1 同屏 3 张图 — `accuracy/correction rate vs time` + `latency vs model` + `参数量 vs improvement` + 一张"same-input strict lane"截图。

**DoD（外部可验证）**：
- ✅ 6 个口径在 [`docs/realtime/baseline-fairness.md`](../realtime/baseline-fairness.md)（待建）逐项列实测数值
- ✅ 3 个模型在 3 段合成 persona + N=1 真人数据上的 frozen vs adapted 曲线 commit 为静态 PNG + CSV 数据
- ✅ 复跑命令 `make baseline-bench` 可重现，CI 跑通

### F3 · 合成 Persona 数据流（双轨：合成 + Live Camera）

| 维度 | 内容 |
|---|---|
| Owner | cc 主案 |
| 数据合成策略决策权 | **cc**（参考 [`lnn-first.md`](../principles/lnn-first.md) §合成数据警告） |
| 工时 | **2 天**（v1.0 估 1 天过乐观，含 4 维约束实现） |

**合成数据 4 维约束（防特征饱和，gemini+codex 共识）**：
1. **Baseline 漂移**：5min 内 mean ± 0.1 缓慢游走
2. **局部 burst**：episode 片段（3-10s）AU activation 显著但非饱和
3. **状态切换平滑段**：burst 间 1-2s linear interpolation
4. **跨 AU 协方差结构**：参考 [`day14-progress`](../day14-progress.md) 真人协方差矩阵

**3 段 persona**：多动型（高频 baseline 漂移 + 多 burst） / 面瘫型（低 baseline 漂移 + 极少 burst） / 抑郁型（低 baseline + 长低活 episode）。

**双轨设计**（gemini 钦定，破"现场交互"叙事矛盾）：
- **Lane A · Reference Baseline**：明示标注 `Reference Synthetic Persona — In-Distribution Feasibility Check`
- **Lane B · Live Camera**：现场摄像头直连，明示 `Live Field Test`
- 观众可一键切换；F1 ledger 的 `persona_idx` 字段必须真路由（不能仍是 env 变量共享）

**DoD（外部可验证）**：
- ✅ 3 段 5min AU 时序的 inter-persona centroid distance ≥ 真人 day-split distance（用 N=1 真人 day-1 vs day-2 做基线）
- ✅ intra-persona variance ratio ≤ 1.5× 真人 day-split variance（合成不能比真人更平整）
- ✅ Live Camera lane 切换响应 ≤ 1s，feedback 链 persona_idx 字段实测不串

### F4 · 时间曲率旋钮（严格边界）

| 维度 | 内容 |
|---|---|
| Owner | cc 主案 |
| 工时 | **1 天** |

**边界**：dt 倍速 `1× / 10× / 100×`（**取消 1000×**，避免 KG 写入打爆）；倍速 ≥ 10× 自动暂停 RL 反馈窗口采集；UI 双时间轴显示。

**DoD（外部可验证）**：
- ✅ 100× 模式下 KG 写入速率峰值 ≤ 现有 1× 的 5×（避免 OOM）
- ✅ 100× 模式下 cognitive RL 反馈窗口（T+1s/T+3s/T+8s）显式标记 `paused`，[`memory_lnn`](../../poc/src/lnn/memory_lnn.py) 不接收倍速时段的 reward
- ✅ 大 τ（≥ 1h）神经元在 100× 跑 5min 后 hidden state 漂移可观测（≥ 真人 1h 漂移的 1/5）

### F5 · 演示安全网（拆两层）

| 维度 | 内容 |
|---|---|
| Owner | cc 主案 |
| Layer B 数学合法性决策权 | **gemini**（ODE 合法性审） + **cc 拒绝权**（违 LNN-FIRST 即否决） |
| 工时 | **2 天**（v1.0 估 1.5 天过乐观，含两层隔离实现） |

**Layer A · Render-layer smoothing**（核心机制）：
- 前端 1s 平滑插值（参考 [`lnn_inferencer.py:153`](../../poc/src/realtime/lnn_inferencer.py:153) 现有 30 帧过渡先例）
- UI 显式标注 `display-smoothed`
- strict fair lane 同步显示 raw latency
- **不改 hidden state，不污染训练账本**

**Layer B · Core-state intervention**（默认关闭 / sandbox lane 才开）：
- **替换 v1.0 高斯扰动**：用 ODE 合法方式 — 检测到 negative_lock ≥ 30s（不是 15s，给状态机更多自愈时间），向 LTC 输入注入持续 Neutral 基线控制信号 `u_neutral(t)`，通过 `dx/dt = -x/τ + f(input + u_neutral)` 平滑引导脱离吸引子
- 仅在 sandbox lane 启用（普通观众默认不见）
- **perturbed 状态不写回训练账本 / 不进 RL 反馈窗口**
- 配置 env var `LNMEM_NEGATIVE_LOCK_RECOVERY=off|sandbox|always`，默认 `off`

**DoD（外部可验证）**：
- ✅ Layer A：UI 抓包显示 `display-smoothed` 标签；ADR-037 风格 "intent EMA 引入反应延迟"诚实承认沿用
- ✅ Layer B：单测覆盖 negative_lock 30s 触发 → `u_neutral` 注入 → 5s 内 state 概率分布显著迁移（Δentropy ≥ 0.2）→ 训练账本无 perturbed 状态写入
- ✅ LNN-FIRST 兼容性：Layer B 实现走 `dx/dt` 积分路径，**单测断言无 hidden state 阶跃赋值**

### F6 · τ 神经元归类 Audit

| 维度 | 内容 |
|---|---|
| Owner | cc 主案 |
| 工时 | **1.5 天**（v1.0 估 1 天过乐观，含代码证据梳理） |

**任务**：当前仓库 [`inference_server.py:93`](../../poc/realtime/inference_server.py:93) 只暴露 `short_tau_mean/long_tau_mean`。本项需：
1. 从 [`lnn/model.py`](../../poc/src/lnn/model.py) grep 所有 τ 参数初始化 + clamp 范围
2. 每个神经元（或神经元组）给出物理时间含义的代码证据
3. 输出 [`docs/realtime/tau-spectrum-audit.md`](../realtime/tau-spectrum-audit.md)（待建）

**DoD（外部可验证）**：
- ✅ 站 1 现场展示的 τ 光谱**每个分组**都有 [`tau-spectrum-audit.md`](../realtime/tau-spectrum-audit.md) 行级出处
- ✅ 任何观众追问"为什么是 X τ 不是 Y τ"，cc 能在 30s 内打开文档定位代码行
- ✅ 不存在"叙事先行"的整齐刻度（30s / 5min / 1h / 1d / 7d / ∞）；按实际 τ 分布画

---

## 选项与对比（v1.1 不变）

| 选项 | 优点 | 缺点 | 推荐度 |
|------|------|------|--------|
| A. 路径 D 原版（3 站全开 6/24） | 一次到位 | 工程缺口 50%+，站 3 现场翻车概率 >70% | ❌ 否决 |
| B. 路径 D++（1 + 2 + 3-roadmap 6/24，7 月 3 站全开） | 工程契合度高、双版本切分对应"内圈→投资人"路径、站 3 留 roadmap 不丢分 | demo 体量收缩，要管理观众预期 | ⭐ 推荐 |
| C. 推迟到 7 月一次发 3 站全开 | "大而完美" | 内圈反馈晚一个月，错过 6/24 决策窗口；3 站全开仍要先验证 1+2 | ❌ 否决 |

## 为什么选 B

1. **代码级证据不允许 A**：codex grep 8 个文件证明站 3 当前是从 0 搭，不是补 20%
2. **双版本切分 = 把不同观众的不同要求工程化拆开**：内圈技术朋友看 1+2 全开已足够给出"思路对了"反馈；投资人才需要 3 站全闭环
3. **站 3 roadmap 用"真实可操作中间物"撑场**（codex 钦定）：SQLite schema + 真实样例记录 + CLI 查询，不是 PPT；技术朋友会因严谨给高分而非靠 invalidate 当删除鼓掌
4. **6/17 决策点是 fail-safe**：到点闭环 → 升 station 3 主菜；不闭环 → 降 roadmap 不翻车
5. **6 项基建有独立价值**：attribution ledger / matched baseline / 合成 persona / 时间曲率旋钮 / 演示安全网 / τ audit **都是后续 7 月版必备**，不是 6/24 一次性投入

## 后果

### 正面
- demo 信用地基稳：双 lane + 6 公平口径把"公平性一票否决"提前变成 cc 主动叙事
- 6/24 节点稳保：站 3 降级 + 中间物展示不让 demo 翻车
- 6 项基建沉淀为后续 7 月版本硬资产
- F1 ledger 是 LnMem 长期"可治理记忆系统"的关键基础设施
- F5 双层安全网 = 通用演示工程模式，未来其他 demo 复用
- LNN-FIRST 准则在 F5 修订后**强化**而非削弱（用 ODE 合法方式破死锁是范式胜利）

### 负面 / 风险
- demo 体量收缩，观众预期管理压力增大
- F1 5 天 + F2 3 天 + F3 2 天 + F4 1 天 + F5 2 天 + F6 1.5 天 = **14.5 天纯基建**（v1.0 估 9 天过乐观，v1.1 修正）
- 三站实现 + 调试 + 现场脚本 ≈ 另 14 天，**累计 4-5 周工程量**
- F1 ledger schema 设计如出错污染 online learning → 用 git tag `before-adr-038` 回滚

### 触发的下游决策
- 6/10 第一道门 → 站 2 单用户 frozen vs adapted A/B 是否可量化（ADR-038.1 记录）
- 6/17 第二道门 → 站 3 multi-persona 路由 + ledger Phase A 是否闭环（ADR-039）
- 7 月版本：N=3 真人招募 + 多模态融合冲突处理 + Phase B replay（ADR-040 / ADR-041）
- 公平对照双 lane → 落 [`docs/rules/demo-fairness-playbook.md`](../rules/demo-fairness-playbook.md)（待建，**不**写进 STYLE.md）

## 反对意见 / 不同看法（含二轮评审）

### 一轮 gemini 主张（被部分采纳）
"开发顺序应 2 > 3 > 1（Risk-First）；演示顺序保 1→2→3"

**cc 不同意理由**：codex 用代码证据证明站 3 工程缺口 100%。但 gemini 的"风险优先"逻辑被采纳到 §F1 优先做。

### 一轮 codex 主张（v1.1 部分采纳）
"必须 N=3 真人才能证明 generalization"

**cc 部分采纳**：6/24 用 3 合成 persona + N=1 真人撑 feasibility；7 月版补 N=3 真人。**不在 6/24 demo 中宣称已证明 generalization**。

### 二轮 gemini 必改点（v1.1 全采纳）
1. F5 高斯扰动违反 LNN-FIRST → ✅ 改为 ODE 合法的 Neutral 控制信号注入（Layer B）
2. F3 后台回放违背"现场交互" → ✅ 改双轨（Reference Baseline + Live Camera）
3. F1 反向抑制数学非法 → ✅ 改 Surviving Feedback Replay 三阶段

### 二轮 codex 必改点（v1.1 全采纳）
1. F5 拆两层（render smoothing vs core intervention）→ ✅ Layer A / Layer B 拆分 + 默认关闭
2. F1 三阶段（lineage / replay / approx-unlearn）→ ✅ Phase A/B/C 拆分
3. 验证指标可复核 + 治理引用清理 → ✅ 删主观调研项 / 父仓规则显式标注 / 去 STYLE.md 误挂

### 老板可能的反对（预测）
"站 3 是颠覆性最强的，降级会让整个 demo 失去灵魂"

**cc 应对（codex 钦定话术）**：站 3 如果按原计划硬上，会暴露 demo 最核心的信用问题 — 现有删除只是 `invalidate` 不是动力学回退（[`apps/demo_e2e/main.py:445-448`](../../apps/demo_e2e/main.py:445)）。**降级不是保护研发舒适度，是保护 demo 信用**。展示中间物（schema + 样例 + CLI）反而把"在路上 + 可验证"做成可信叙事。

## 验证 / 复盘

### 何时知道这个决策对不对

- **6/3 第零道门**：F1 schema 单测 + F4/F5/F6 原型 ready？是 → 进站 2 实现；否 → 整体推迟 1 周
- **6/10 第一道门**：站 2 单用户 frozen vs adapted A/B 是否有可量化曲线？是 → 进站 1 实现；否 → 整体延期 + 重新规划
- **6/17 第二道门**：站 3 multi-persona 路由 + F1 Phase A 是否闭环？是 → 站 3 升主菜；否 → 站 3 降 roadmap（决策 B 保命）
- **6/24 demo 后**：内圈技术朋友 ≥ 2/3 在 1 周内 follow-up 给出书面反馈（≥ 200 字，含具体技术点对照）？是 → 决策 B 成功；否 → 复盘 demo 设计错位

### 验证指标 v1.1（全部外部可复核）

| 指标 | 阈值 | 验证方式 | 来源 |
|---|---|---|---|
| F1 Phase A: ckpt lineage 可查 | 100% | `lnmem ledger lineage <sha8>` CLI 输出非空 | F1 |
| F1 Phase A: 断电恢复完整性 | 0 孤儿记录 | 模拟断电 3 次后 ledger consistency check | F1 |
| F2: 6 公平口径 | 全部满足 | `docs/realtime/baseline-fairness.md` 逐项实测数值 | F2 |
| F2: adapted vs frozen 改进 | correction rate gap ≥ 现有 daemon 24h 实测 baseline noise band 上限 | CI `make baseline-bench` 输出 | F2 |
| F3: persona separation | inter ≥ 真人 day-split distance；intra ≤ 1.5× 真人 day-split variance | `docs/realtime/persona-separation.md` 自动生成 | F3 |
| F3: Live Camera 切换 | 响应 ≤ 1s，persona_idx 不串 | E2E 单测 | F3 |
| F4: 100× 模式 KG 写入 | 峰值 ≤ 1× 的 5× | 监控日志 | F4 |
| F4: 大 τ 漂移可观测 | ≥ 真人 1h 漂移的 1/5 | 自动对比脚本 | F4 |
| F5 Layer A: UI smoothed 标注 | 100% 现场可见 | 截图 commit | F5 |
| F5 Layer B: ODE 合法性 | 单测断言无阶跃赋值 | pytest `test_layer_b_ode_legal.py` | F5 |
| F5 Layer B: 训练账本隔离 | 0 perturbed 状态写入 | F1 ledger query 验证 | F5 |
| F6: τ 神经元归类 | 100% 有 audit 文档行级出处 | `tau-spectrum-audit.md` review | F6 |

**主观调研项删除**（与 cognitive.md 冲突）：
- ~~"70% 观众调研没感到延迟"~~ → 改 Layer A 工程实现验证
- ~~"≥ 2/3 技术朋友主动说"~~ → 改 "1 周内书面 follow-up ≥ 200 字"（非现场弹窗调研）
- ~~"懂行人能看出真 ODE"~~ → 改 F5 LNN-FIRST 单测断言

### 如果错了，回退路径

- F1 ledger schema 出错污染 online learning → 回滚到 git tag `before-adr-038`（已打）
- F2 公平性被技术朋友质疑穿帮 → 当场打开 `baseline-fairness.md` 现场修正，记 ADR-038-postmortem
- 6/10 第一道门未过 → 整体推迟 1 周，重评估 F1/F2 阻塞链
- 6/24 demo 现场翻车 → 当场降为"路线讨论"，邀技术朋友共同设计 7 月版

---

## 时间表 v1.1

| 节点 | 里程碑 | 决策点 |
|---|---|---|
| **5/27** | ADR-038 v1.0 起草 + 二轮对抗审 + 修订为 v1.1 + commit + git tag `before-adr-038` | — |
| **5/28 - 6/1** | F1 attribution ledger（cc 实施 + codex 5.4 high 提案 schema） — 5 天 | — |
| **6/1 - 6/3** | F4 时间曲率（1 天） + F6 τ audit（1.5 天） + dashboard Help + README — 并行 | **6/3 第零道门**：F1 schema + F4/F6 原型 ready？ |
| **6/3 - 6/6** | F2 matched baseline 训练（codex high effort） — 3 天 | — |
| **6/6 - 6/8** | F3 合成 persona 数据流 + 双轨 — 2 天 | — |
| **6/8 - 6/10** | F5 演示安全网两层 — 2 天 | — |
| **6/10** | 6 项基建全 ready；启动站 2 主线 | **第一道门**：6 项基建 DoD 全过？站 2 frozen vs adapted A/B 准备好？ |
| **6/10 - 6/17** | 站 2-full + 站 1-lite 主线实现 + Mem0 strict fair lane 对照 | **6/17 第二道门**：站 3 multi-persona 路由 + ledger Phase A 是否闭环？ |
| **6/17 - 6/24** | 站 3 roadmap 中间物（schema + CLI + 只读 dashboard） + 3 站联调 + 现场脚本 + 内圈邀请 | — |
| **6/24** | **内圈 demo**（站 1-lite + 站 2-full + 站 3-roadmap 含中间物） | 收书面 follow-up |
| **6/25 - 6/30** | 复盘 + F1 Phase B replay 实施 + N=3 真人招募启动 | — |
| **7 月** | 3 站全开版（含 Phase B replay 真闭环） → 投资人 / 种子用户 ready | — |

---

## cc 的责任承诺 v1.1

cc 在本任务中除了写代码外，承担：

1. **代码质量把控**：每个 PR 必经 codex 对抗审 + cc 主线自检 + 川哥签字三道门（[`CLAUDE.md`](../../CLAUDE.md) §2.1）
2. **双门验证**（父仓 `~/.claude/rules/testing.md` 全局规则）：每次实现完成跑 Verify（目标达成）+ Guard（无回归）两门；本仓具体实现按 [`docs/development-harness.md`](../development-harness.md) §AC + DoD
3. **交付验收**：每个工程基建项按 v1.1 DoD 严格验收，附 CI / 截图 / 单测三类证据；做不到 DoD = 不进下一项
4. **轨迹捕获**：M0 demo 工程是普通档+任务，必须留轨迹 `trajectory_open → step → decision → friction → close`
5. **失败升级**（父仓 `~/.claude/rules/failure-escalation.md`）：连续 3 次失败启 REFINE 重新分析，5 次启 PIVOT 回退稳定点
6. **HOT 元认知监控**（父仓 `~/.claude/rules/no-guessing.md` 实体锚定）：对外操作前自报 `[自检]` 三维分数 + 关键实体锚

---

## v1.0 → v1.1 变更摘要

| 项 | v1.0 | v1.1 |
|---|---|---|
| F1 删除路径 | "反向抑制 OR replay" 二选一 | Phase A lineage / Phase B replay / Phase C unlearn-approx 三阶段 |
| F1 Schema | 模糊 `event_ids → tau_drift delta` | 3 表 SQL（fit_runs / event_attribution / delta_summary） |
| F1 工时 | 3 天 | **5 天** |
| F2 公平口径 | "等参数 / 等输入 / 等训练预算"3 条 | 6 条 operationalize（参数量 5% / 模态同源 / 预处理同源 / budget 固定 / 调参一致 / 推理延迟同硬件） |
| F2 工时 | 2 天 | **3 天** |
| F3 数据流 | "后台回放"单轨 | 双轨（Reference Baseline + Live Camera） |
| F3 合成约束 | "多动 / 面瘫 / 抑郁"3 段 | 4 维约束（baseline 漂移 + 局部 burst + 转场平滑 + 跨 AU 协方差） |
| F3 工时 | 1 天 | **2 天** |
| F4 时间曲率上限 | 1000× | **100×**（避 KG 打爆） |
| F4 RL 反馈窗口 | 未提 | ≥10× 自动暂停（防 cognitive.md 污染） |
| F5 看门狗机制 | 高斯扰动注入（违 LNN-FIRST） | Layer A render smoothing + Layer B ODE 合法 Neutral 注入（默认关闭） |
| F5 工时 | 1.5 天 | **2 天** |
| F6 工时 | 1 天 | **1.5 天** |
| 总工时 | 9 天 | **14.5 天**（修正过乐观） |
| 验证指标 | 含主观调研（70% / 2-3 朋友 / 懂行人） | 全部外部可复核（CLI / 单测 / 截图 / 自动脚本） |
| 治理引用 | `testing.md` / `failure-escalation.md` 等当成本仓文件 | 显式标注"父仓 `~/.claude/rules/` 全局规则" |
| 双 lane 落档 | STYLE.md | `docs/rules/demo-fairness-playbook.md`（待建） |
| 决策权矩阵 | "cc 主案 + 并行建议" | 每个 F 项明确 schema / 口径决策权归属 |

---

**模板版本：v0.1 · 2026-05-03**
**本 ADR 版本：v1.1 · 2026-05-27**
**预计本 ADR 将被 ADR-039（6/17 站 3 升降级决策） / ADR-040（多模态融合冲突） / ADR-041（N=3 真人招募）/ ADR-042（F1 Phase B replay 数学合法性）等下游 ADR 引用**
