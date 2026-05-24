# ADR-028: LNN 角色重定义 — 从 AU 序列预测 → online 三分类意图预测

> **决策者**：川哥钦定签字 + cc 主线整合 + codex/gemini 战略复盘三方共识
> **状态**：accepted（川哥 2026-05-23 凌晨签字）
> **关联**：ADR-022/023/024/025/026/027 全部前作 · LNN-FIRST P-1（内涵更新，不 override）· 不可妥协 #2 #4 #6（重新对齐）

---

## 1. 触发事件

5/22-5/23 凌晨 36 小时 11 轮 Team 模式连夜冲刺：
- 4 个训练方案（v3.0 200ep / dt-aware 30ep / dt-aware 200ep / multifps+invariance 50ep）
- 全部在川哥 14 天 daemon 真人数据上 daemon shuffle_ratio ≈ 0
- multi-fps + invariance 反而把 VEATIC shuffle 从 1.73 削到 0.22（8x 退化）
- v3.0 200ep 在 daemon 上 vs_persistence=2.27（比 trivial 还差 2.3x）

cc + codex (gpt-5.3-codex high) + gemini (3-flash-preview) 战略复盘三方独立得出同一结论：
**LNN 应从"AU 序列预测心智核心"重新定位**。

川哥点睛指出三方初步结论（"LNN 降级为子模块 + 14 天回忆界面"）的根本错误：
> "14 天回忆 + 纠错界面，传统记忆系统也能做。我真正要的是基于前十几秒到几分钟状态变化，预测我的意图，并通过反馈让预测越来越准。"

这条提示直接戳穿 cc/codex/gemini 三方盲区——**不是 LNN 不行，是 cc 5/22-23 全程在错任务上验证 LNN**。

## 2. 川哥三参数（产品需求钦定）

| # | 参数 | 钦定值 |
|---|------|--------|
| 1 | 意图类型 | **三分类**：正反馈意图 / 负反馈意图 / 无反馈意图 |
| 2 | 反馈机制 | **AI 主动推断 → UI 弹推断结果 → 川哥 [确认/纠正]**（不是 RL）|
| 3 | 算力约束 | **Air M5 性能上限可用**（不再 1Hz 受限）|

## 3. 决策

### 3.1 LNN 真实角色重新定义

**从**：
- AU 序列预测器（30fps → 30fps 重建）
- ode rollout loss 优化 shuffle_ratio
- 评测指标：shuffle / vs_persistence / effective_dim

**到**：
- **Online intent estimator**：实时多模态流（30Hz AU + head_pose + 屏幕活动）→ 实时三分类意图状态
- supervised classification loss + per-user τ-bias 漂移
- 评测指标：意图预测准确率 + 反馈后改善幅度 + 跨 session 状态连续性

### 3.2 LNN 4 个不可替代能力的 100% 命中

| 能力 | 5/22-23 旧任务里 | **v0.1 新任务里** |
|------|----------------|------------------|
| 连续时间演化（任意采样率） | 鸡肋（1Hz 数据物理不足）| ✅ **核心**：30Hz 实时流 + 稀疏反馈事件 |
| 多时间尺度共存 | 仅作 metric 装饰 | ✅ **核心**：短 τ 秒级感知 + 长 τ 小时级累积 |
| 物理可解释 τ | 仅训练 log | ✅ **核心**：UI 直接展示"专注度 τ 剩 12 分钟" |
| per-user τ-bias | spec 钦定但未真用 | ✅ **核心**：每次川哥纠正驱动漂移 |

### 3.3 不可妥协 #2 内涵更新（不 override LNN-FIRST P-1）

**旧定义**：「多时间尺度 LNN 心智核心」← 字面强调"LNN 用 ODE 微分方程"

**新定义**：「多时间尺度 LNN 心智核心：online intent estimation + per-user 漂移」← 强调 LNN **真正擅长**的事

LNN-FIRST P-1 全文保留不动，只是把 #2 的"心智核心"内涵从模糊钦定具体到"online intent estimator"。

## 4. 5/22-23 工作复用矩阵

| 5/22-23 资产 | v0.1 复用 | 说明 |
|--------------|---------|------|
| dt-aware ODE | ✅ 100% | 30Hz 实时 inference dt_target=0.033s 直接用 |
| τ 多时间尺度解放 [0.96, 34] | ✅ 100% | v0.1 核心能力前提 |
| daemon adapter + DaemonAUDataset | ✅ 80% | schema 扩展加 head_pose + 屏幕活动 |
| per-user τ-bias 数学（custom_cfc_cell）| ✅ 100% | confirm 反馈驱动漂移 |
| rare AU 监控（ADR-025/026/027）| 🟡 50% | M1 评估意图分类时仍可用 |
| multifps curriculum | 🗑️ 暂搁置 | v0.1 单一 30Hz |
| invariance loss | 🗑️ 暂搁置 | 无需跨 fps 对齐 |
| ode_loss rollout 训练入口 | 🗑️ 冻结 | 不再 launch |

**60% 直接复用 + 20% 部分复用 + 20% 搁置（基础设施沉淀，未来可能用）**

## 5. 4 个 Sprint 路径

### Sprint 0（5/24 一天）：方向钉死
- 本 ADR-028 落盘（cc 已写）
- ROADMAP 修订（冻结旧训练 + 重写 5/27 + 6/10）
- 冻结 `scripts/dtaware_multifps_train.py` 等过时入口

### Sprint 1（5/25-5/31 一周）：30Hz daemon + 实时 inference server + UI v0 观察期
- daemon 升级 1Hz → 30Hz（M5 实测 + py-feat 实时）
- 新增模态：head_pose（py-feat 已支持）+ 屏幕活动信号（active_app + key/mouse rate）
- 实时 inference server（WebSocket：daemon → server → UI）
- UI v0：Tauri 桌面 widget，仅显示 LNN state（观察期不主动 prompt）
- 川哥日常用 1 周，手动标 ~50-200 个 [正/负/无] sample

### Sprint 2（6/1-6/7 一周）：意图三分类头 + 主动弹窗 + 反馈闭环
- LNN hidden state → MLP 3 类分类头
- Supervised training（Sprint 1 数据）
- 主动 prompt 系统（置信度 > 0.85 触发 + 5min cooldown 防骚扰）
- 反馈接口：[确认] / [纠正为正] / [纠正为负] / [纠正为无]
- 每次纠正 → mini batch (size=8) τ-bias finetune

### Sprint 3（6/8-6/10 三天）：M0 收口 demo + 学习曲线
- 实时 demo：1 周使用对比 Day 1 vs Day 7 准确率（35% → 65%+）
- 24h 学习曲线可视化（每 epoch τ-bias 漂移）
- 一键导出"AI 对我的画像"
- M0 收口报告

## 6. 6/10 M0 不可妥协 6 项达成度

| # | 项 | 6/10 状态 |
|---|---|---|
| 1 | 万能语法 20 槽 schema | ⏳ M1 启动（v0.1 简化 schema）|
| **2** | **多时间尺度 LNN 心智核心**（新定义）| ✅ **真正落地** |
| 3 | CLS 离线 consolidation | ⏳ M1 |
| **4** | **per-user 个性化栈** | ✅ **第一次端到端 work** |
| 5 | 居家+车载双场景 | ⏳ M2（v0.1 仅 Air 桌面）|
| **6** | **用户可见可治理界面** | ✅ **实时意图 UI + 反馈按钮** |

**3 项核心达成（#2 #4 #6），3 项 M1+ 推进。**

## 7. 6/10 demo 场景（川哥能向人演示的）

**场景**：川哥写代码 25 分钟，开始揉眼睛 + 眉头微皱 + 头微低

**LnMem 行为**：
```
┌─────────────────────────────────┐
│  🤖 LnMem 实时状态              │
│  当前: 负反馈倾向 (置信度 0.72) │
│  τ 衰减: 专注记忆 还剩 3 分钟  │
│  💡 建议: 起身走 5 分钟？        │
│  [接受] [拒绝] [稍后]            │
└─────────────────────────────────┘
```

- 川哥 [拒绝] → LNN 学到"在写代码深度状态下拒绝中断"
- 第 2 次同情境，AI 不立刻 prompt，等到揉眼睛 + 喝水 2 次叠加才提示

**24h 学习曲线**：Day 1 准确率 35% → Day 7 65%+

## 8. 风险与对冲

| 风险 | 对冲 |
|------|-----|
| 30fps 实时 LNN + py-feat 双跑 M5 性能不够 | Sprint 1 benchmark，必要降 15fps |
| 意图三分类标注主观性（川哥自己也不一致） | Sprint 1 累积 50 样本后做一致性分析 |
| Cold start 数据不足训分类头 | Day 1-3 规则 baseline，Day 4+ supervised finetune |
| 弹窗骚扰川哥 | 置信度 0.85 起步 + 5min cooldown |
| 屏幕活动信号涉隐私 | 仅记 active_app 名 + activity rate（无内容）|

## 9. mycc 教训新增

- `feedback_lnn_intent_vs_rollout`：LNN 真正擅长的是 online state estimation + 任意采样率，不是 long rollout AU prediction。5/22-23 spec 字面 vs 产品意图脱节是根本误读
- `feedback_strategic_retro_three_engines`：cc + codex + gemini 三方独立战略复盘是项目方向级拐点的可靠机制（5/22-23 三方独立得出同一结论后川哥精准纠偏）
- `feedback_user_intent_anchoring`：cc 必须周期性追问用户真实意图，不能只跟着 spec 字面跑

## 10. LNN-FIRST P-1 关系澄清

**本 ADR 不 override LNN-FIRST P-1**。

LNN-FIRST P-1 原文：「凡涉及时间/演化/记忆/个性化必用 LTC/CfC ODE 微分方程」—— v0.1 完全符合：
- 时间感知：30Hz AU 流 → LNN ODE 实时演化 hidden state
- 演化：状态实时更新
- 记忆：LNN hidden + per-user τ-bias 即"AI 对你的记忆"
- 个性化：τ-bias 漂移

ADR-028 只是把 #2 "心智核心"的实施任务从"AU rollout"具体到"online intent"。LNN 仍是 LnMem 心智核心，不替换为 LLM/RNN。

---

## 11. gemini 5/23 凌晨对抗审 5 finding（全采纳，加入 Sprint 设计）

| Finding | 风险 | 整改 |
|---------|------|------|
| **F1 努力度-效价混淆** | HIGH | Sprint 2 UI 标注扩展"我正忙（别理我）" vs "我烦了（帮我）"二选项，仍映射回三分类输出 |
| **F2 M5 热降频 → ODE 步长崩** | MED-HIGH | Sprint 1 inference server **必须按 frame 真实 timestamp 算 dt**（dt-aware 基础设施已具备，强调实施细节）|
| **F3 LnMem 自指循环** | MED | Sprint 1 screen_activity 数据必须 **屏蔽 active_app=="LnMem" 时段**（标 `context_ignore`）|
| **F4 多数类陷阱**（90% 无反馈→坍塌）| HIGH | Sprint 2 分类头训练用 **cost-sensitive loss**（正/负类权重 ≥ 10x）+ 考虑对比自监督预训练 |
| **F5 观察者效应**（弹窗污染心流数据）| CRITICAL | Sprint 1 **仅观察期不弹**（cc UI v0 已设计）；Sprint 2 主动弹窗改为 **DAF 异步延迟反馈**（仅在状态切换时请求）|

**最关键修正（gemini 钦定）**：F3 LnMem 自指特征屏蔽 — 不做这个会让模型陷入"自嗨式预测"。

---

**📝 修订**：2026-05-23 凌晨 gemini round 7 对抗审 5 finding 整改

---

**📝 决策者**：川哥钦定签字 · cc 主线整合 · codex/gemini 三方战略复盘共识  
**📅 签发**：2026-05-23 凌晨  
**🔗 川哥原话**：「我签字，接受方案。你开启 team 模式并行执行。」
