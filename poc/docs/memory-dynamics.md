---
version: v1.0
audience: cc / codex / gemini / 工程师 / 老板
purpose: 5 阶段记忆系统的 LTC 动力学公式 — 把"记忆"建模为多时间尺度的连续衰减系统
status: P-1 LNN-FIRST 准则下的 P0.5 设计稿（2026-05-08 cc 起草，待 Day 7 实现）
parent: ../README.md
references:
  - ../../docs/principles/lnn-first.md（**P-1 最高准则**）
  - ../../docs/lnn-math.md（LTC + CfC 数学基础）
  - ../../docs/memory-architecture.md（记忆架构）
  - ./poc-spec.md §4（PoC 训练任务定义）
---

# 5 阶段记忆 LTC 动力学

> **核心命题**：5 阶段记忆 = 5 个不同时间尺度的 LTC 单元串联。
> 记忆衰减 = LTC 的 leak 项；记忆迁移 = 上一级 LTC 的输出作为下一级的输入；
> 老板"确认进终身"= 把第 5 级 LTC 的 leak 关掉（τ→∞）。
>
> 这是 LnMem 区别于 mem0/zep/letta 的本质所在——**记忆不是文本，是动力系统**。

---

## 0. 公式概览（一页纸）

```
m₁(t)  刚发生：    τ₁ ≈ 30s         dm₁/dt = -m₁/τ₁ + I(t)
m₂(t)  短期：      τ₂ ≈ 5min        dm₂/dt = -m₂/τ₂ + g₁₂·m₁
m₃(t)  工作：      τ₃ ≈ 1hour       dm₃/dt = -m₃/τ₃ + g₂₃·m₂
m₄(t)  长期：      τ₄ ≈ 24hour      dm₄/dt = -m₄/τ₄ + g₃₄·m₃
m₅(t)  终身：      τ₅ → ∞ (老板确认) dm₅/dt = -m₅/τ₅ + g₄₅·m₄

I(t)            外部输入（fusion.py 的 18 维信号 → embedding 后投到 m₁ 的状态空间）
g_{i,i+1}       重要性门 ∈ [0,1]，由 LLM 重要性裁判输出（详见 §3）
m_i             N 维状态向量（PoC 默认 N=64，与 LNN hidden_dim 对齐）
```

每一级 m_i 是一个 N=64 维 LTC 单元，状态向量在连续时间上演化，
不是离散文本块。LLM 不参与 m_i 的演化主逻辑，只贡献 g_{i,i+1} 这个标量门。

---

## 1. 物理直觉 — 为什么是 5 阶段，为什么是 LTC

### 1.1 5 阶段对应人类记忆心理学

| 阶段 | 时间尺度 | 对应认知科学概念 | 物理意义 |
|------|---------|-----------------|---------|
| m₁ 刚发生 | 30s | 感觉记忆（sensory memory）| 当前感官信息余晖 |
| m₂ 短期 | 5min | 短期记忆（short-term）| 工作中的临时缓存 |
| m₃ 工作 | 1hour | 工作记忆（working memory）| 当前任务的活跃上下文 |
| m₄ 长期 | 24hour | 长期记忆（long-term）| 待巩固的事件 |
| m₅ 终身 | ∞ | 自传记忆（autobiographical）| 已确认的核心记忆 |

人类记忆的"遗忘曲线"（Ebbinghaus 1885）本质是 **leak 项主导的指数衰减**——这正是 LTC 的标准形式。

### 1.2 为什么不能用 LLM 模拟 5 阶段

**LLM 模拟方案（mem0/zep 走的路）**：
- 用 LLM prompt "请把这条记忆从短期升级到长期"
- 用文本字符串保存记忆，向量库做相似度检索
- 每次"想起"都要 LLM 重读全文

**问题**：
1. 没有"时间常数"概念，遗忘是离散触发不是连续演化
2. 重要性判断不可学习（每次都重新 prompt，30 个画像 30 套规则）
3. 个性化只能靠 prompt 模板，无法做到 per-persona τ
4. **本质是文本游戏，不是动力系统**

### 1.3 LTC 方案的优势

| 维度 | LLM 方案 | LTC 方案 |
|------|---------|---------|
| 遗忘曲线 | 拍脑袋阈值 | 物理上正确（指数衰减） |
| 个性化 | prompt 模板，难调 | τ_i^persona 可学 |
| 重要性 | 每次 LLM 调用 | 一次推理出 g 即可 |
| 真人微调 | 重新 fine-tune LLM | 只更新 τ + persona_embedding |
| 可视化 | 文本 dump | 5 通道连续时序曲线 |
| 运行成本 | 每帧调 LLM | 微分方程一次前向 |

---

## 2. 完整公式系统

### 2.1 单级 LTC 单元

每级 m_i ∈ ℝ^N（N=64），演化方程：

```
dm_i/dt = (-m_i + W_i · u_i(t)) / τ_i^persona

其中：
  τ_i^persona ∈ ℝ+      时间常数（per-persona, per-stage 可学）
  u_i(t)                第 i 级的输入（i=1 时是 fusion 信号，i≥2 时是 g·m_{i-1}）
  W_i ∈ ℝ^{N×?}         投影矩阵（把输入维度对齐到 N）
```

欧拉离散化（dt=1s）：
```
m_i[t+dt] = m_i[t] + dt × (-m_i[t] + W_i · u_i[t]) / τ_i^persona
```

**注意**：这是简化形式。完整 LTC 还可以加 `f(m_i, u_i, t)` 非线性门控（Hasani 2020），
PoC 阶段先用简化版跑通，P1 再开非线性。

### 2.2 5 级串联 + 重要性门

```
m₁[t+dt] = m₁[t] + dt × (-m₁[t] + W₁·I[t]) / τ₁^persona
m₂[t+dt] = m₂[t] + dt × (-m₂[t] + g₁₂[t]·W₂·m₁[t]) / τ₂^persona
m₃[t+dt] = m₃[t] + dt × (-m₃[t] + g₂₃[t]·W₃·m₂[t]) / τ₃^persona
m₄[t+dt] = m₄[t] + dt × (-m₄[t] + g₃₄[t]·W₄·m₃[t]) / τ₄^persona
m₅[t+dt] = m₅[t] + dt × (-m₅[t] + g₄₅[t]·W₅·m₄[t]) / τ₅^persona
```

`g_{i,i+1}[t] ∈ [0,1]` 是重要性门——LLM 输出（详见 §3）。

### 2.3 m₅ 终身记忆的特殊处理

**老板确认机制**：
- 默认 τ₅ = 168hours（一周）——若一周没确认，慢慢遗忘
- 老板点击 UI"进终身记忆"→ 该条记忆的 τ₅ → ∞（leak=0，永不遗忘）
- 实现：每条记忆有 `confirmed: bool` 字段，confirmed=True 时 leak 设为 0

```
if not confirmed:
    m₅[t+dt] = m₅[t] + dt × (-m₅[t] + g₄₅[t]·W₅·m₄[t]) / τ₅
else:
    m₅[t+dt] = m₅[t] + dt × (g₄₅[t]·W₅·m₄[t]) / τ₅   # 注意：去掉 -m₅ 项
```

这正好对应"被老板确认 = 这条记忆免疫遗忘"的产品语义。

---

## 3. 重要性门 g（LLM 在 LnMem 的合法使用 #1）

**核心思想**：LLM 不参与 m_i 演化主逻辑，只输出"是否该往下渗透"的标量。

### 3.1 g 的定义

```
g_{i,i+1}[t] = LLM_judge(m_i[t], context, persona) ∈ [0, 1]
```

含义：
- g=0：m_i 完全不渗透到 m_{i+1}（这一帧的内容不重要）
- g=1：m_i 完全渗透（这是关键时刻）
- g=0.5：部分渗透（一般重要）

### 3.2 LLM judge 的 prompt 模板

```
你是 LnMem 的"记忆重要性裁判"。
当前阶段：m_{stage_name}（衰减常数 τ={tau_value}）
内容摘要：{m_i_summary}（由 m_i 状态向量经轻量 decoder 解码）
主人画像：{persona_summary}

请输出 0-1 之间的浮点数，回答：
"这条内容值得从 m_{i} 渗透到 m_{i+1}（{next_stage_name}）吗？"

判断准则：
- 与主人长期关心的事相关 → 高
- 偶然事件、噪声 → 低
- 与已有 m_{i+1} 内容冲突 → 中（让 LNN 自己学冲突解决）

只输出一个浮点数，不要解释。
```

### 3.3 调用频率

- 不是每帧调 LLM（太贵）
- 每个时间尺度 τ 调一次：
  - g₁₂ 每 30s 调一次（与 τ₁ 对齐）
  - g₂₃ 每 5min 调一次
  - g₃₄ 每 1hour 调一次
  - g₄₅ 每 24hour 调一次（一天结束时统一处理）
- 帧间默认 g 保持上次值（zero-order hold）

### 3.4 g 也是可学的（P1 计划）

PoC 阶段 g 由 LLM 生成；
Day 13 真人数据微调时，把 g 训练成 nn.Module（输入 m_i + context_embed → sigmoid 输出）；
LLM 只在新场景 / 不确定时降级使用。这又一次体现"LNN 学 + LLM 兜底"的设计模式。

---

## 4. 个性化 — τ_i^persona 的来源

### 4.1 PoC Day 7 初始化方式

LLM 一次性推导（与表情 profile 同思路）：
```json
{
  "persona_id": "vboss_001",
  "memory_tau": {
    "m1": 30,        // seconds
    "m2": 300,       // 5 min
    "m3": 3600,      // 1 hour
    "m4": 86400,     // 1 day
    "m5": 604800     // 1 week (默认值，confirmed 后 → inf)
  }
}
```

### 4.2 HEXACO 推导规则示例（待 gemini 调研补充）

- **高 C（尽责）**：τ₂、τ₃ 偏长（工作记忆持久），τ₄ 偏长（细节记得久）
- **高 O（开放）**：τ₁ 偏短（信息流转快），但 τ₅ 偏长（重要的不忘）
- **高 N（神经质）**：τ₃ 偏短（工作记忆易被情绪打断）
- **高 A（宜人）**：g_{i,i+1} 推导偏高（容易记住别人的事）

### 4.3 真人微调（Day 13 + P1）

老板自用累积真实数据后：
1. 观察老板真实"记忆持续时间"（提到某事多久后还能想起）
2. 通过梯度下降更新 τ_i^persona
3. 老板的"确认行为"作为 g 的监督信号

这就是 LNN 在线学习的入口。

---

## 5. 与 LNN model.py 的关系

### 5.1 当前 model.py（StateRecognitionLNN）

```
fusion 18 维 → CfC + τ-bias + FiLM → state logits (8 类)
```

这是"瞬时状态识别"，没有跨时间的记忆。

### 5.2 P0.5 之后的 memory_lnn.py

```
fusion 18 维 → 5 阶段 LTC 链 → 5 个 m_i 状态向量
                                      ↓
                              组合后送 state recognition
```

state 识别可以查询 m₁-m₃ 来增强（"5 分钟前你说过 X，所以现在你说 Y 应该还在 X 的上下文中"）。

### 5.3 共享 LNNTrainConfig

**P-1 LNN-FIRST 准则 I4**：memory_lnn.py 必须 import LNNTrainConfig，
保证 hidden_dim、persona_embed_dim、设备等参数与 lnn/model.py 完全对齐。

---

## 6. 实现路线图（P0.5）

| 任务 | Agent | 文件 | 时长 |
|------|-------|------|------|
| 写本文档（公式 + 直觉） | cc | 当前文件 | ✅ 已完成 |
| `memory_lnn.py` LTC 链实现 | codex | poc/src/memory/memory_lnn.py | 30 min |
| `importance_judge.py` LLM 重要性裁判 | codex | poc/src/memory/importance_judge.py | 20 min |
| `memory_visualize.py` 5 通道时序图 | codex | poc/src/memory/visualize.py | 15 min |
| 老板确认 UI（Streamlit）| codex | poc/ui/memory_confirm.py | 20 min |
| 端到端测试（synthetic）| cc | poc/tests/test_memory_lnn.py | 30 min |
| live_verify v4（含记忆通道）| cc | poc/scripts/live_verify_memory.py | 25 min |

**总计 ~140 min。**

---

## 7. 验收标准

### 7.1 单元测试
- [ ] `MemoryLNN.step(input, dt)` 返回 5 个 N 维向量
- [ ] τ_i = 30s 时，m₁ 在无输入后 60s 衰减到 ≈ e⁻² × 初值（指数衰减验证）
- [ ] confirmed=True 时 m₅ 不衰减（24 小时后误差 < 0.001）
- [ ] 不同 persona τ 不同 → 同输入 → 不同 m_i 轨迹

### 7.2 端到端
- [ ] 跑 1 小时 synthetic 数据，5 通道曲线肉眼可见 "短期波动 + 长期累积"
- [ ] 重要事件（state=emotional_positive 持续 5min）能渗透到 m₃，普通事件停在 m₁
- [ ] 老板点"确认进终身"→ m₅ 该维度不再衰减

### 7.3 LNN-FIRST 准则
- [ ] grep `if.*hour\|if.*time\|if.*age` 不命中（没有时间硬编码 if-else）
- [ ] grep `LLM.*memory\|prompt.*memory_stage` 仅在 importance_judge.py 命中
- [ ] PR 描述回答"如何强化 LNN 内核"

---

署名：cc（Claude Opus 4.7）| 2026-05-08 v1.0 起草
