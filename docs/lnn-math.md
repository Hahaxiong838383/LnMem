# LNN 数学方程在 LnMem 的运用

> 这份文档是给"想看懂 LNN 心脏在跳什么"的人写的。两个核心方程：**LTC**（液态时间常数）和 **CfC**（闭式连续时间）。前者是训练时的物理本质，后者是推理时的工程形态。
>
> 读完你能回答三个问题：① 这俩方程到底说了什么？② LnMem 里哪几层用了它？③ 跟 v0.3.x 个性化栈（τ 偏置 + FiLM + 槽位偏好向量）什么关系？

---

## 1. LTC 方程：液态时间常数（训练时本质）

### 1.1 方程

$$
\frac{dx(t)}{dt} = -\left(\frac{1}{\tau} + NN\big(x(t), I(t), \theta\big)\right) \odot x(t) + \ldots
$$

省略号通常是输入驱动项 $+ NN(x(t), I(t), \theta) \odot A$，其中 $A$ 是输入的目标态。完整 LTC 方程见 Hasani et al. (AAAI 2021) "Liquid Time-constant Networks"。

### 1.2 每个符号在说什么

| 符号 | 含义 | 在 LnMem 的对应物 |
|------|------|------------------|
| $x(t)$ | 隐状态向量（Layer 1/2 的"当前感受/工作记忆"） | L1 感知输出 / L2 工作记忆 hidden |
| $\tau$ | 基础时间常数（**可学习参数**，每个 unit 一个） | 全用户共享的"基线衰减速度" |
| $1/\tau$ | 基础衰减率 | "信息忘多快"的下限 |
| $NN(\cdot)$ | 神经网络输出（含输入 $I$ + 当前状态 $x$） | 内容感知的"额外衰减/加速"调制 |
| $I(t)$ | 外部输入（音频/视频/文本/ EventTensor） | 万能语法 20 槽事件张量 |
| $\theta$ | 神经网络参数 | M0-M3 训练目标 |
| $\odot$ | Hadamard 积（逐元素乘） | per-unit 独立衰减，不是矩阵作用 |

### 1.3 工程语义：为什么"液态"

普通 RNN：$x_{t+1} = f(W \cdot x_t + U \cdot I_t)$，**衰减率 = 常数**（`σ(W)` 的谱）。

LTC：**衰减率 $= 1/\tau + NN(x, I, \theta)$，本身依赖输入和状态**。意味着：

- 输入"惊讶"时（NN 输出大）→ 衰减加速 → 状态被快速覆盖（关注新事件）
- 输入"平稳"时（NN 输出小）→ 只剩 $1/\tau$ 衰减 → 状态保留得久（持续观察）

这就是 **"液态"**：参数像液体一样**随情境流动**，而不是固定的固体网络。

### 1.4 在 LnMem 的位置

```
[麦克风/摄像头/文本输入 I(t)]
          ↓
[Layer 1 感知 LTC cell]   ← τ 偏置（per-user）注入这里
          ↓ x(t) [B, units]
[boundary_gate 事件分割]
          ↓
[Layer 2 工作记忆 LTC cell]   ← τ 偏置 + FiLM 调制注入这里
          ↓
[EventTensor 组装 → KG 写入 / consolidation]
```

- **L1（秒级）**：每个传感器帧调用 LTC.forward，$x(t)$ 编码"当前 0.1-1 秒的感受"
- **L2（分钟级）**：上一段感受流 + 新事件 → LTC，$x(t)$ 编码"近 5-30 分钟主题/情绪"
- **超过分钟尺度**：不再用 LTC，转 KG（L4 自传记忆）+ CLS consolidation

### 1.5 实例 A — τ 偏置如何区分两个用户

场景：用户「晚睡型」和用户「早起型」说同一句话"**我饿了**"。

| 步骤 | 晚睡型用户 | 早起型用户 |
|------|----------|----------|
| 输入 $I(t)$ | embedding("我饿了") | 同左 |
| `τ_offset[user_id]` | +0.5（衰减更慢） | −0.5（衰减更快） |
| 有效 $\tau$ | $\tau_{\text{base}} + 0.5$ → 大 | $\tau_{\text{base}} - 0.5$ → 小 |
| 状态衰减项 $1/\tau$ | 小 → 状态保留久 | 大 → 状态被快速冲掉 |
| L2 输出语义 | "饿了，但应该是一直没吃，常态" | "饿了，立即响应找吃的" |
| Agent 后续动作 | 记下来等下次复盘 | 主动建议"现在去吃饭吧" |

**为什么 τ 偏置就是 v0.3 §10.1#4 的个性化栈核心**：它直接作用在 $1/\tau$ 这个 LTC 方程的衰减常数上，是 LNN 数学层面的**唯一干预点**——不是改输入（FiLM 干那个）、不是改输出（LoRA 干那个），而是改"时间感知"本身。

### 1.6 实例 B — NN 项如何让 LTC 内容感知

场景：会议中 LTC 接收一段话："**昨天小李提了个新方案，我觉得很有意思，我们下周讨论**"。

LTC 处理过程（伪步骤）：

1. **token 级嵌入** $I(t_1)$ = embedding("昨天") → $NN(x, I, \theta)$ 输出小（普通时间词）→ 衰减率约 $1/\tau$ → 状态平稳更新
2. $I(t_2)$ = embedding("小李") → 实体！→ $NN$ 输出大 → 衰减率高 → 旧的"昨天"信息被压缩，新的"小李"实体特征写入 hidden
3. $I(t_3)$ = embedding("新方案") → 关键名词 → $NN$ 输出大 → 写入 + 与"小李"绑定（hidden 内积升高）
4. $I(t_4..)$ = embedding("我们下周讨论") → 时间引用 → $NN$ 输出中 → 标记"事件未来跟进"

最终 $x(t_{\text{end}})$ 编码：[小李—新方案—下周讨论] 这个语义簇 + 强度。boundary_gate 检测到事件边界后，把 $x$ 解码成 EventTensor 的 P0 槽位（who_actor=小李 / what_action=提方案 / when_time=昨天 / 等）写入 L4 KG。

**关键洞察**：LTC 不需要 attention head 也能做"重要词更影响状态"——通过 $NN$ 学到的内容相关衰减率自然实现。

---

## 2. CfC 方程：闭式连续时间（推理时形态）

### 2.1 方程

$$
x(t) = \sigma\big(-f(x, I) \cdot t\big) \odot g(x, I) + \big[1 - \sigma\big(-f(x, I) \cdot t\big)\big] \odot h(x, I)
$$

来自 Hasani et al. (NeurIPS 2022) "Closed-form continuous-time neural networks"。

### 2.2 每个符号在说什么

| 符号 | 含义 | 工程对应 |
|------|------|---------|
| $x(t)$ | 隐状态（同 LTC） | L1/L2 hidden |
| $\sigma(\cdot)$ | Sigmoid 函数 | 门控开关，输出 $[0, 1]$ |
| $t$ | 时间步长（多久） | 上次 forward 到现在的间隔 |
| $f(x, I)$ | 等价时间常数（小神经网络） | 控制"门切换速度" |
| $g(x, I)$ | 过去贡献（小神经网络） | "如果完全保留旧状态会怎样" |
| $h(x, I)$ | 新输入贡献（小神经网络） | "如果完全采用新输入会怎样" |
| $\odot$ | Hadamard 积 | per-unit 独立门 |

### 2.3 工程语义：为什么"闭式"

LTC 是 ODE：
$$
x(t+\Delta t) = x(t) + \int_t^{t+\Delta t} \frac{dx}{ds} \, ds
$$
**必须用 ODE solver 数值积分**（Euler / RK4 / Dopri5）。每步要 4-7 次 NN 调用 + 自适应步长，端侧推理慢 / 不稳。

CfC 是闭式：
$$
x(t) = \text{显式表达式}\big(f, g, h, t\big)
$$
**一次前向 = 三次小 NN 调用 + 一次 sigmoid + 两次乘加**，不需要 solver，无积分误差，端侧延迟 ms 级稳定。

**门控混合的几何**：当 $f \cdot t$ 大 → $\sigma \to 0$ → $x \approx h$（完全采用新输入）；当 $f \cdot t$ 小 → $\sigma \to 1$ → $x \approx g$（保留过去）。$f$ 学到"该切换多快"，等价于"自适应 $\tau$"。

### 2.4 在 LnMem 的位置

```
训练阶段（M0-M3 Studio M2 Max）
  ↓ 用 LTC + ODE solver（ncps.torch.LTCCell）
  ↓ 训练参数 θ_LTC
[蒸馏 / 参数共享]
  ↓ 把 LTC 行为压到 CfC 三个小 NN（f, g, h）
  ↓ 训练参数 θ_CfC

推理阶段
  ├─ 服务器端（mycc / brain）→ 仍可用 LTC（精度最好）
  ├─ 端侧 Mac App / 手机 → CfC（毫秒级，不耗电）
  └─ 车载 Jetson Orin / Android Automotive → CfC（强制，无 ODE solver 可用）
```

**LnMem 路线图绑定**：
- **M1-M3**：lnn_core 同时包含 LTC 实现（训练）+ CfC 实现（推理 export）
- **M6 车载试点**：强制走 CfC（[`roadmap/milestones.md`](../roadmap/milestones.md) M6）
- **M10 跨设备同步**：CfC 是手机端唯一能跑的形态

### 2.5 实例 C — CfC 在车载场景

场景：开车时麦克风持续 10ms 一帧采样 → ASR 输出 → CfC L1 处理。

| 步骤 | 时刻 | 输入 $I$ | $f(x,I)\cdot t$ | $\sigma(-ft)$ | 含义 |
|------|------|---------|----------------|--------------|------|
| 1 | 09:00:00 | "你今天" | 0.2（弱信号） | ≈0.55 | 50/50 平衡过去和新输入 |
| 2 | 09:00:01 | "感觉怎么样" | 0.8（情绪问询） | ≈0.31 | 偏向新输入 h（注意力切换） |
| 3 | 09:00:02 | （静默） | 0.1（无信号） | ≈0.52 | 保持 1 状态，缓慢衰减 |
| 4 | 09:00:03 | "我有点累" | 1.5（强情绪信号） | ≈0.18 | 大幅切换到 h（用户情绪状态） |

**对比 LTC**：同样 4 步，LTC 要跑 4 次 ODE solver（每次 4-7 次 NN forward），延迟 30-100ms 不稳；CfC 只需要 4 次"3 NN + sigmoid + 2 乘加"，延迟稳定 < 10ms，Jetson Orin 上 50W 功耗下也能实时跑。

---

## 3. LTC vs CfC 对照表

| 维度 | LTC | CfC |
|------|-----|-----|
| 形式 | ODE | 闭式表达式 |
| 计算依赖 | ODE solver（Euler/RK4/Dopri5） | 三个小 NN + sigmoid |
| 单步延迟 | 不定（4-7 次 NN）/ 不稳 | 稳定（≈3 NN） |
| 训练 | ✅ 物理本质，梯度通过 adjoint | ⚠️ 需要从 LTC 蒸馏或联合训 |
| 端侧推理 | ❌ solver 太重 | ✅ 唯一可行 |
| 表达力 | 强（连续时间 + 状态依赖衰减） | 中（门控混合 + 学到的等价 τ） |
| LnMem 用法 | M0-M3 训练 + 服务器推理 | M6+ 端侧 + 车载强制 |

---

## 4. 与 v0.3.x 个性化栈（三件套）的关系

[`docs/memory-architecture.md`](memory-architecture.md) §7 定义的三件套：**τ 偏置 + FiLM + 槽位偏好向量**。每件作用在数学方程的不同位置：

### 4.1 τ 偏置 — 直接改 LTC 的 $1/\tau$ 项

LTC 方程从：
$$
\frac{dx}{dt} = -\left(\frac{1}{\tau} + NN(x, I, \theta)\right) \odot x + \ldots
$$
改为：
$$
\frac{dx}{dt} = -\left(\frac{1}{\tau + \tau_{\text{offset}}[u]} + NN(x, I, \theta)\right) \odot x + \ldots
$$

其中 $\tau_{\text{offset}}[u] \in \mathbb{R}^{\text{units}}$ 是**用户 $u$ 专属向量**（< 1KB / 用户）。

**为什么这是合规的、不破坏 ODE 性质**：$\tau_{\text{offset}}$ 是常数（一旦训完冻结），ODE 仍然 well-posed，存在性 + 唯一性都保留。区别于 v0.1 否决的 "State-offset Tuning"（直接对 $x$ 加偏置），那个会破坏 ODE 几何。

### 4.2 FiLM 调制 — 改 $NN$ 的输入

FiLM (Feature-wise Linear Modulation) 在 NN 内部某层加 per-user 的 affine 变换：
$$
\text{hidden} \leftarrow \gamma[u] \odot \text{hidden} + \beta[u]
$$

作用在 LTC 方程的 $NN(x, I, \theta)$ 这一项内部，不改 LTC 的整体形式。**~ 2KB / 用户**（两个 H 维向量）。

### 4.3 槽位偏好向量 — 改 $I(t)$

$I(t)$ 是 EventTensor → embedding，加一个 **20 维 per-user 偏置向量**（80 字节 / 用户）作为 $I$ 的额外输入维度，让 LTC 能感知"这个用户对哪些槽位特别敏感"。

### 4.4 三件套总开销

$\underbrace{1\text{KB}}_{\tau\text{ 偏置}} + \underbrace{2\text{KB}}_{\text{FiLM}} + \underbrace{0.08\text{KB}}_{\text{槽位偏好}} \approx 3\text{KB / 用户}$

CPU 推理可行，符合 [`docs/product-definition.md`](product-definition.md) §10.1#4 不可妥协约束。

---

## 5. 工程实现指南

### 5.1 M0 W1 — spike

[`spikes/ncps_tau_bias/`](../spikes/ncps_tau_bias/) 用 LTC（`ncps.torch.LTCCell`）三阶段验证 τ 偏置可行性：

- **Day 1 烟雾**：LTC + tau_offset 能 forward + backward
- **Day 2-3 稳定性**：1000 步训练发散率 < 10%
- **Day 4-5 表达力**：5 主人区分度 ≥ 80%

任一红 → 触发 D5 回退（FiLM 单独 / BitFit+LoRA(2)）。

### 5.2 M1 — `lnn_core/`

模块包含两套实现：

```python
class LtcCellTrain(nn.Module):
    """ODE-based LTC，用于训练"""
    def forward(self, x, I, dt):
        # 调用 ODE solver
        ...

class CfcCellInfer(nn.Module):
    """闭式 CfC，用于推理"""
    def forward(self, x, I, t):
        f = self.f_net(x, I)
        g = self.g_net(x, I)
        h = self.h_net(x, I)
        gate = torch.sigmoid(-f * t)
        return gate * g + (1 - gate) * h
```

训练完 LTC 后用蒸馏把行为压到 CfC（M3 之前完成）。

### 5.3 M6 车载 — 强制 CfC

`automotive_adapter/` 装载的 LFM2-VL 内核就是 CfC 架构（Liquid AI 官方实现）。我们的 `lnn_core` 提供同 schema 的 CfC 推理 API，L1+L2 在 Jetson Orin 上 < 10ms / 帧。

---

## 6. 参考文献

| 论文 | 核心贡献 |
|------|---------|
| Hasani et al. AAAI 2021 — *Liquid Time-constant Networks* | 引入 LTC，证明状态依赖时间常数的表达力优势 |
| Hasani et al. NeurIPS 2022 — *Closed-form Continuous-time Neural Networks* | 给出 CfC 闭式解，去掉 ODE solver |
| Lechner & Hasani 2020 — *Neural Circuit Policies* | 19 神经元 NCP 控制车辆，端到端可解释 |
| Liquid AI 2024 — *LFM2 Technical Report* (arxiv 2511.23404) | 端侧 CfC + 蒸馏，350M/1.2B 参数级别 |

---

## 7. FAQ

**Q：为什么不直接用 Mamba / RWKV / GRU？**
答：Mamba/RWKV 的衰减率虽然可学，但仍是**离散时间**——不能处理"上次输入到现在间隔多久"这个连续时间维度。GRU 的门控是离散的，没有"积分到 $t$"的物理意义。**LnMem 需要的是"昨晚 8 点提的事 × 今晨 9 点的状态"这种跨时间尺度的连续推理**，这是 LTC/CfC 唯一可行的数学框架。

**Q：CfC 比 LTC 表达力弱，会不会影响 MVP 质量？**
答：M6 车载是 CfC 唯一强制场景，那时已经经过 M0-M5 LTC 训练 + 蒸馏。Liquid AI 在 LFM2 上的实测：CfC 在大多数任务上和 LTC 差距 < 3%，但延迟降一个数量级。

**Q：τ 偏置真的是 v0.3.x 个性化栈的"核心 IP"吗？**
答：见 [`docs/memory-architecture.md`](memory-architecture.md) §7.5 的学术埋点——若 M3 末看到 τ 容量曲线对数收敛（小预算大收益），M6 启动论文准备《Per-user τ-Bias for LTC Networks》。这是 LNN 第一篇 per-user 个性化论文，没人发过。

---

**v0.3.1 · cc 主线编制 · 2026-05-03**
