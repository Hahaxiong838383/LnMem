# LnMem 项目第 10 轮评审报告：直面范式转轨与数学硬伤

作为经历前 9 轮迭代、见证了从“表征冷凝”误判到“L2-norm 球面禁锢”真因剖析的评审器，本轮我将剥离一切“行业范式”滤镜，不留情面地直面 ADR-023 及 Spec v1。

---

## 战略评审决策：BLOCK 🟥 (严厉驳回)

### 1. 为什么必须 BLOCK？
离开 InfoNCE + L2-norm 的“自监督球面牢笼”是 **100% 正确的战略方向**。L2-norm 强行将 LNN 的连续状态轨迹压缩到非欧几里得球面（$S^{d-1}$），彻底阉割了线性/连续动力学的多维展开，这是 `effective_dim` 锁死在 3.95 的本质原因。

然而，**cc 在 ADR-023 和 Spec v1 中给出的具体数学解法，犯了灾难性的数学幼稚病**。你设计了一个**恒等式死局**和一套**与 LNN 物理内核完全冲突的刚性约束**。如果直接运行这套代码，模型将瞬间滑入“无记忆的空间前馈自编码器”塌陷。

---

## 数学层评审 Findings（致命伤暴露）

### 🛑 Finding 1 (FATAL): $L_{\text{predict}}$ 的“恒等式死局”（梯度为零）
cc 在 Spec 2.1 中设计的 Teacher-Forcing Rollout 公式如下：
*   **Rollout 状态**：$h_{\text{pred}}[N+k] = \text{backbone\_step}(x[N+k], h_{\text{pred}}[N+k-1], p)$
*   **基准状态**：$h_{\text{seq}}[N+k]$ 是对整段 $x$ 跑一遍 backbone 得到的状态。

**致命漏洞**：由于 CfC（以及所有因果循环网络）在生成 $h_{\text{seq}}$ 时，其前向传播公式也是：
$$h_{\text{seq}}[N+k] = \text{backbone\_step}(x[N+k], h_{\text{seq}}[N+k-1], p)$$
且你初始化指定 $h_{\text{pred}}[N-1] \equiv h_{\text{seq}}[N-1]$。

通过简单的数学归纳法：
*   当 $k=0$ 时：$h_{\text{pred}}[N] = \text{backbone\_step}(x[N], h_{\text{seq}}[N-1]) \equiv h_{\text{seq}}[N]$
*   当 $k=1$ 时：$h_{\text{pred}}[N+1] = \text{backbone\_step}(x[N+1], h_{\text{pred}}[N]) \equiv h_{\text{seq}}[N+1]$

**结论**：在任何参数下，对于任意 $k \ge 0$，**$h_{\text{pred}}[N+k]$ 恒等于 $h_{\text{seq}}[N+k]$**！
你的 $L_{\text{predict}} \equiv 0$。这个损失函数从训练的第一步开始就是绝对的零，**完全没有梯度传递给网络**。这是一个纸面上的“伪多步预测”。

---

### 🛑 Finding 2 (CRITICAL): CfC 动力学内核与一阶 Euler 离散化的“生态生克”
cc 试图用 $L_{\text{ode}}$ 强行约束 Euler 一致性：
$$L_{\text{ode}} = \| h[t+1] - (h[t] + dt \cdot (f_{\text{net}}(x_t) + \tau) \odot h[t]) \|^2$$

**致命漏洞**：
CfC (Closed-form Continuous-time) 的核心数学贡献就是**拒绝显式 Euler 离散化**！CfC 的内部状态演化是通过指数衰减门控（$e^{-f(x)\Delta t}$）实现的解析解：
$$h(t) \approx \text{gated}(x) \cdot e^{-[\tau + f(x)]\Delta t} + \text{conv}(x) \odot (1 - e^{-[\tau + f(x)]\Delta t})$$
这是一种**非线性、非局部的半隐式流形**。
你强行塞入一个一阶显式欧拉差分约束（$L_{\text{ode}}$），相当于逼迫一个连续的“液体”去套“刚性晶格”。这会造成严重的**梯度相互撕裂**（Gradient Conflict），逼迫 CfC 退化成一个普通的残差网络（ResNet），直接废掉了 ADR-022 刚刚释放的 $\tau$-bias 连续可解释性。

---

### 🛑 Finding 3 (HIGH): “瞬时捷径”（Instantaneous Shortcut）与空间前馈塌陷
由于 $L_{\text{predict}} \equiv 0$ 且 $L_{\text{ode}}$ 与网络内核冲突，模型唯一的有效驱动力将只剩下 $L_{\text{recon}}$。
*   $h_t$ 是 64 维，输入 $x_t$ 是 52 维。
*   当 $w_{\text{recon}} = 1.0$ 时，模型发现的最优捷径是：**放弃所有时序记忆，将 $h_t$ 降级为 $x_t$ 的恒等空间投影（$h_t \approx W_{\text{enc}} x_t$）**。
*   这能让 $L_{\text{recon}}$ 瞬间趋近于 0。因为 $h$ 变成了无记忆的瞬时快照，**Shuffle Test 的预测 drop 必归零**，记忆机制彻底死亡。

---

### ⚠️ Finding 4 (MEDIUM): $L_{\text{infonce\_aux}}$ 的尺度失衡（Sphere Leakage）
你移除了全局的 L2-norm，但保留了 $L_{\text{infonce\_aux}}$ 的“内部 L2-norm”。
这会导致极度危险的数值不匹配：主干网络 $h\_seq$ 没有模长约束，可以任意膨胀（以减小 MSE 重构误差），而在计算对比损失时又被强行投影回单位球面。尺度不一致会导致辅助梯度的方向与主梯度方向完全失调，轻则产生无用振荡，重则导致数值溢出。

---

## 盲区挖掘与彻底的 LNN-FIRST 方案改造

不要再套用传统的“时序自监督预测”范式。**真正的 LNN-FIRST，应该把 LNN 训练成一个潜空间世界模型（Latent World Model）**。

### 替代方案：自回归世界模型预测（Latent World Model Rollout）
完全抛弃 $L_{\text{predict}}$ 和 $L_{\text{ode}}$，只保留一个**统一的、非平凡的物理自回归重构损失**：

```
Prefix 阶段 (t = 0 ... N-1):
  h[t] = backbone_step(x[t], h[t-1])  -- 用真实输入建立历史记忆

Rollout 阶段 (t = N ... N+K):
  x_pred[t] = Decoder(h_pred[t-1])   -- 仅用前一步的潜状态，解码预测当前帧输入
  h_pred[t] = backbone_step(x_pred[t], h_pred[t-1])  -- 将“自己预测的输入”喂回给 LNN

Loss 函数 (仅对 Rollout 阶段的重构误差进行惩罚):
  L_world_model = (1/K) * Sum_{t=N}^{N+K} MSE(x_pred[t], x_actual[t])
```

#### 为什么这个方案完美对齐 LNN-FIRST 且无懈可击？
1.  **非平凡性**：因为 Rollout 阶段网络**拿不到**真实输入 $x_{\text{actual}}[t]$，它必须依靠 $h$ 中保留的连续动力学“模拟”未来的视频走向。$L_{\text{world\_model}}$ 绝不可能为 0。
2.  **抗前馈塌陷**：如果 $h$ 试图偷懒做瞬时空间投影（$h_t \approx W x_t$），在没有输入反馈的 Rollout 阶段，它将立刻因为失去动力学积分而无法预测未来，从而导致 $L_{\text{world\_model}}$ 爆炸。
3.  **释放连续时间内核**：CfC 的解析连续流被完全释放，用于模拟时序轨线的漂移。

---

## 参数与权重推荐（基于改造方案）

如果采用上述 **Latent World Model Rollout** 方案：

| 损失组件 | 推荐权重 | 物理意义与置信度 |
|---|---|---|
| **$L_{\text{recon}}$ (Prefix 帧逐帧重构)** | **0.5** | 提供基础空间特征锚定（防止潜空间漂移）。置信度: 90% |
| **$L_{\text{world\_model}}$ (自回归多步预测重构)** | **1.0** | 核心目标：逼迫 LNN 学习视频的物理演化轨迹。置信度: 95% |
| **$L_{\text{infonce\_aux}}$** | **0.0** | **彻底弃用**。不需要对比学习，动力学模拟本身就是最强的自监督。置信度: 99% |

*   **Rollout 步数推荐**：$K$ 从 5 开始，稳定后逐步退火至 15。

---

## 5/27 Verdict 判定第 4 次更新

*   **当前判定：🟥 RED LIGHT**
*   **原因**：ADR-023 的 Spec v1 数学公式存在逻辑闭环死锁（$L_{\text{predict}} \equiv 0$），且 Euler 离散化对 CfC 有结构性破坏。
*   **放行条件**：
    1.  废弃当前 Spec v1 的 $L_{\text{predict}}$ 和 $L_{\text{ode}}$。
    2.  改用上面提出的 **Latent World Model Rollout (自回归世界模型预测)** 框架，重写 Spec v2。
    3.  在 tiny 50 epoch 达成：`effective_dim` $\ge 8.0$ + `Shuffle drop` $\ge 0.15$。

---

## 给 cc 的最终建议（187字）

```
cc，不要再把“业界常识”包装成数学公式。你的 Spec v1 犯了两个低级错误：第一，用 Teacher-Forcing 算 Hidden 的多步 MSE，在因果网络里这只是把同一条公式算了两遍，Loss 恒等于 0，这叫“自欺欺人”；第二，用显式欧拉差分去约束 CfC 的解析指数流，这叫“买椟还珠”。

别再玩对比学习和辅助约束了。立刻把 LNN 升级为潜空间世界模型（Latent World Model）：用前 15 帧建立记忆，后 15 帧切断真实输入，让 LNN 靠自己的动力学“闭眼”画出未来的图像。这才是 liquid 的终极姿态。
```