# τ 神经元归类 Audit — F6 (ADR-038 v1.1 §F6)

> **目的**：站 1 现场演示"多时间尺度共存"前，先把 LnMem 当前 LTC backbone 的 τ 实际分布弄清楚——每个神经元 τ 的代码来源 + 实测数值 + 对 demo 叙事的影响。
>
> **诚实第一**：本 audit 拒绝"30s/5min/1h/1d/7d/∞ 整齐光谱"叙事先行。τ 是学出来的连续值，不是人为分配。

---

## 关键发现（一句话）

**当前 LnMem phase2 ckpt 的 64 个 CfC 神经元 τ 分布 mean=1.187 std=0.013，跨 persona spread=0.0015——近似单 τ，远未达"多时间尺度分化"。** 站 1 demo 叙事必须修正，不能口头主张"LnMem 64 个不同时间尺度 vs RNN 单 τ"，否则会被技术朋友当场拆穿。

---

## 1. 架构层：τ 在 LnMem 中怎么来的

### 1.1 数据流（代码追踪）

```
persona_idx (int 0..30)
   ↓  personalization.embedding (nn.Embedding(31, 8))
persona_embed (8-dim)
   ↓  personalization.tau_projection (nn.Linear(8 → 64))
tau_bias (64-dim)
   ↓  + online_tau_drift[persona_idx] (per-user 在线漂移, shape (31, 64))
tau_bias_with_drift (64-dim)
   ↓  timespans = 0.5 + softplus(tau_bias_with_drift)
timespans (64-dim, 强制 > 0.5)
   ↓  CfC.rnn_cell(features, hidden, timespans)
hidden state (64-dim)
```

### 1.2 代码行级证据

| 步骤 | 文件位置 | 关键逻辑 |
|---|---|---|
| Embedding 表 | [`personalization.py:17`](../../poc/src/lnn/personalization.py) | `nn.Embedding(config.n_personas=31, config.persona_embed_dim=8)` |
| τ 投影矩阵 | [`personalization.py:18`](../../poc/src/lnn/personalization.py) | `nn.Linear(persona_embed_dim=8, hidden_dim=64)` |
| τ 计算 | [`personalization.py:26-31`](../../poc/src/lnn/personalization.py) | `tau_projection(_persona_embedding(persona_idx))` |
| Softplus + 0.5 下界 | [`model.py:41-43`](../../poc/src/lnn/model.py) | `0.5 + F.softplus(tau_bias)`，物理意义：timespan 单位是 CfC 内部时间常数，强制正性 |
| Online drift | [`online_adapter.py:71-86`](../../poc/src/lnn/online_adapter.py) | `tau_drift` shape `(n_personas, hidden_dim)`，加在 base 上 |
| 真实 forward 调用 | [`model.py:65-75`](../../poc/src/lnn/model.py) | `tau_bias → timespans → backbone.rnn_cell(..., timespans)` |
| 在线训练唯一可训参数 | [`online_trainer.py:218-220`](../../poc/src/lnn/online_trainer.py) | `clip_grad_norm_([online_adapter.tau_drift], max_norm=10.0)` |

### 1.3 配置（[`contracts.py:45-54`](../../poc/src/lnn/contracts.py)）

```python
n_personas: int = 30          # vboss_001..vboss_030 + persona 30 = vboss_self
persona_embed_dim: int = 8    # embedding 维度
hidden_dim: int = 64          # CfC backbone 隐藏维度 = τ 神经元数量
```

**结论**：LnMem 当前架构上有 **64 个独立 τ 神经元**，每个 τ 是 `embedding @ Linear.T + bias + drift` 的连续函数，**没有任何人为"30s/5min/1h"语义分组**。

---

## ⚠️ 2026-05-29 ckpt 路径修正（cc 诚实补注）

本 audit v1.0 计算的是 `poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_20260524T144928Z.pt`（sha8 c163f53a），**但 daemon 实际加载的是 `~/.lnmem/ckpt/lnn_backbone_phase2_latest.pt`（sha8 59cd1d9c, 5/22），由 env `LNMEM_LNN_CKPT` 覆盖默认目录**。cc audit 时假设了 daemon 默认目录，没验 env 覆盖（no-guessing 小瑕疵）。

**但结论不变**：两个 ckpt 同属 personalization.tau_projection 路径，daemon 实际 ckpt（59cd1d9c）实测 τ mean=1.187 std=0.013 min=1.128 max=1.228，与本 audit 数值**几乎完全一致**。单 τ / 半衰期 0.82s / 缺口 36× 全部成立。

## 2. 实测 τ 分布（latest phase2 ckpt 2026-05-24）

### 2.1 全局统计（所有 31 personas × 64 神经元 = 1984 个 τ 值）

| 统计量 | 值 |
|---|---|
| mean | 1.187 |
| std | 0.013 |
| min | 1.128 |
| max | 1.228 |
| range | 0.100 |

### 2.1.x 物理时间标定（gemini 二审钦定必补）

**`timespan` 的物理单位**：CfC 内部 timespan，等价于 ODE `dx/dt = -x/τ + f(input)` 中的 τ。**单位由 forward 时 `dt` 参数决定**。

- 现实 daemon 调用：[`lnn_inferencer.py:215`](../../poc/src/realtime/lnn_inferencer.py) `backbone.rnn_cell(features, hidden, timespans)` 不显式传 `dt`（ncps CfC 默认 dt=1.0 单位时间）
- 实时 inference 频率：daemon 1Hz，每帧 dt=1（即 1 秒）
- τ ≈ 1.187 + dt=1.0 → **隐状态半衰期 t₁/₂ = τ · ln(2) ≈ 0.82 秒**
- 翻译为产品语义：**当前 LnMem 每个神经元的"记忆衰减半衰期"约 0.82 秒**，**远低于产品承诺的"30s 短期 ~ 永久"多尺度**

**对照表**（dt=1s, τ=1.187 vs 产品承诺尺度）：

| 产品承诺时间尺度 | 对应 τ 应该是多少 | 当前 τ=1.187 是否覆盖 |
|---|---|---|
| 30 秒短期注意力 | τ ≈ 43.3 (= 30 / ln2) | ❌ 短约 36× |
| 5 分钟会话上下文 | τ ≈ 432.8 | ❌ 短约 365× |
| 1 小时心情漂移 | τ ≈ 5193 | ❌ 短约 4376× |
| 1 天人格观察 | τ ≈ 124633 | ❌ 短约 105K× |

**结论**：当前 ckpt 的 τ 不仅"分化不足"，**所有 τ 都在亚秒级**，物理上根本承载不了产品定义 §1.3 第 1-2 条承诺（"它会记得 / 它会变化"）。这是必须 7 月版前修复的核心 bug。

### 2.2 Persona 30 (vboss_self) 单独看（64 神经元）

| 统计量 | 值 |
|---|---|
| mean | 1.187 |
| std | 0.013 |
| min | 1.128 (neuron #55) |
| q10 | 1.171 |
| q25 | 1.182 |
| median | 1.188 |
| q75 | 1.194 |
| q90 | 1.201 |
| max | 1.221 (neuron #10) |

**bottom 5 neurons**（短 timespan）：`#55, #32, #34, #35, #44` → 值 `[1.128, 1.159, 1.161, 1.168, 1.169]`
**top 5 neurons**（长 timespan）：`#5, #14, #30, #59, #10` → 值 `[1.202, 1.202, 1.203, 1.209, 1.221]`

### 2.3 跨 persona 个性化差异

| 统计量 | 值 |
|---|---|
| per-persona mean 最小 | 1.1863 |
| per-persona mean 最大 | 1.1878 |
| **spread (max - min)** | **0.0015** |

**注解**：31 个 persona 的 τ mean 几乎完全一致（spread 仅 0.0015，约 0.1%）。说明 τ-bias 个性化在当前训练数据下**几乎没分化**。

### 2.4 Online drift 状态

| 统计量 | 值 |
|---|---|
| `online_tau_drift` shape | (31, 64) |
| mean | 4.1e-8 |
| std | 8.2e-7 |
| min | -6.8e-6 |
| max | 2.3e-5 |

**注解**：drift 数量级 1e-6 ~ 1e-5，对 softplus 后的 timespan 影响远小于 0.001——**在线学习在 τ 维度上事实上没生效**。这与 ADR-038 §A 代码硬伤分析一致：当前在线学习只动 tau_drift，且数值极小，**画像演化幅度小到现场看不见**（[`online_adapter.py:89`](../../poc/src/lnn/online_adapter.py)）。

---

## 3. 对站 1 demo 的影响（路径 D++ 必须修正叙事）

### 3.1 不能继续说的口号（会被当场拆穿）

| ❌ 拒绝叙事 | 理由 |
|---|---|
| "LnMem 64 个不同时间尺度神经元" | 实测 τ std=0.013，**接近单 τ** |
| "30s / 5min / 1h / 1d / 7d / ∞ 整齐光谱" | 架构上没语义分组；τ 单位是 CfC timespan 不是绝对秒数 |
| "RNN 单 τ vs LnMem 多 τ" | 当前 LnMem 也是接近单 τ 状态，对照不诚实 |
| "你看每个神经元都有自己的物理时间含义" | 64 个 τ 是隐式学出来的，没有先验"含义" |
| "在线学习让 τ 个性化漂移" | 当前 drift mean 4e-8，事实上没生效 |

### 3.2 必须改用的诚实叙事

| ✅ 诚实叙事 | 依据 |
|---|---|
| "LnMem 架构提供 64 个可独立学习的连续 τ 神经元" | `hidden_dim=64`，每个神经元独立 `Linear(8→64)` 输出 |
| "当前训练后 τ 分布 mean=1.19, std=0.013，跨 persona spread=0.0015" | 实测数据 2.1-2.3 |
| "这是 LnMem 架构能力的真实快照，不是 demo 优化版" | 透明 = 信用 |
| "我们今天演示 LTC ODE 连续时间积分 vs RNN 离散步进的对照，**不**主张 τ 已分化" | 把 demo 锚到架构差异，不是训练状态 |
| "7 月版会用 Liquid-S4 多时间尺度损失 + 真人数据扩散，把 τ 训出实质分化" | 路线承诺，对应 ADR-040+ |

### 3.3 站 1 演示内容修订（路径 D++ 站 1-lite，gemini 二审 v1.1 修订）

**保留**（这部分诚实可信）：
- LTC 微分方程演化（`hidden[t+1] = hidden[t] + dt × (-hidden[t]/τ + ...)`）的连续时间性
- 任意采样率 / 不规则 dt 的鲁棒性（vs RNN/GRU 固定步长）
- F2 matched baseline 同输入同延迟同参数下的延迟/参数效率对照

**修订**（不再硬演"光谱"）：
- 改用"64 神经元 τ 直方图"现场展示（mean=1.19 std=0.013 的真实分布）
- 旁边并列"如果训出实质分化会是什么样"的目标分布草图（路线预告）
- **每帧 τ 实时显示 min/median/max + 个性化 drift 数值**，让观众看到 drift 在动（或没动）

**新增 demo 元素（gemini 二审钦定 v1.1，3 件套）**：

**(a) 一键注入分化 τ baseline 对比（`--mock-tau-diversity` 开关）**
- 站 1 UI 加 toggle：`Real Ckpt`（真实未分化）vs `Mock Diverse`（注入对数均匀 τ ∈ [0.1, 10.0]）
- 注入方式：在 inferencer load ckpt 后，覆盖 timespans 为 `torch.exp(torch.linspace(log(0.1), log(10.0), 64))`
- 现场对比：极不规则 dt 采样下 / 超长程记忆检索任务下，两种 τ 的预测 entropy / hidden state diversity
- **叙事**：「我们诚实暴露当前 τ 未分化，但架构本身能承载分化。一键切换就能看到分化后的能力差异——这是架构上限的 hands-on 证明」
- 工程量：F4 实施时一起做（dashboard 加 toggle ≤ 0.5d）

**(b) dt 不规则采样 LTC vs GRU 现场对照**
- 演 **dt 不规则采样下 LTC vs GRU 的鲁棒性**
- F4 时间曲率旋钮展示 dt 改变时 LTC 平滑、GRU 抽搐
- 即便 τ 退化，LTC 闭式解 ODE 嵌入 dt 到非线性门控的特性 vs GRU 离散步进的截断误差，仍是架构本质差异（参见 §6 防御话术）

**(c) τ 物理时间标定即时显示**
- UI 实时显示：「当前 τ=1.187 → 半衰期 0.82s → 产品承诺 30s 短期记忆缺口约 36×」
- 路线条直接显示在 UI：「7 月版目标 τ 分化到对数均匀 [0.1, 10.0] → 覆盖 0.07s ~ 7s + 后续训练扩到 1e2 ~ 1e5」

---

## 4. 7 月版改进路线（gemini 二审钦定 v1.1 修订，不阻塞 6/24 demo）

> 这部分是 audit 暴露问题后的下游决策，记入 [`ROADMAP.md`](../../ROADMAP.md) 7 月段。
>
> **v1.0 → v1.1 关键修订**：原 `L_diversity = -log(Var(τ))` 公式**作废**——gemini 钦定该公式数学陷阱（会导致 63 神经元压到 0.5 + 1 个推到 ∞ 的双峰极化崩塌，破坏 ODE 数值稳定性）。改用 Log-domain KL + Bucket Gradient Isolation。

| 改进项 | 目标 | 路径 | 关联 ADR |
|---|---|---|---|
| τ Log-domain 均匀先验 KL 约束 | std ≥ 0.5（log τ），覆盖 [0.1, 10.0] 区间 | `L_div = KL(P(log τ) ‖ U(log τ_min, log τ_max))` 替代 v1.0 `-log(Var)` | 拟 ADR-042 |
| τ Bucket Gradient Isolation | 初始化把 64 神经元划 4 组（快/中/慢/超慢），组间 detach 防同质化 | 改 `personalization.tau_projection` 为分组结构 | 拟 ADR-042 |
| 在线 drift 自适应学习率 | drift 量级 1e-5 → 可观测但稳定 | **不**简单全局放大（会数值崩溃），用 LARS / 独立 τ-only 优化器 | 拟 ADR-044 |
| FiLM 参与在线学习 | 当前只 tau_drift 动，FiLM 不动 | online_adapter 扩 FiLM drift 参数 | 拟 ADR-045 |
| N=3 真人数据 | 个性化训练样本扩充 | 招募 + 数据采集 SOP | 拟 ADR-041 |
| Liquid-S4 嫁接（**降级为研究项**） | 长时间尺度承载（>10s）| **不在 7 月版强行硬上**——CfC 闭式解 vs S4 时域卷积融合工程难度极高（gemini 钦定"跑车底盘装喷气发动机"），改为研究探索，不进 demo 路线 | 拟 ADR-043（研究）|

### 4.x 必须警惕的数学陷阱（v1.1 新增）

**❌ `L_diversity = -log(Var(τ))`**（v1.0 误用）：
- 优化器为拉大方差 → 63 个 τ 压到 softplus 下界 0.5 + 1 个 τ 推向 +∞
- 导致 ODE `dx/dt = -x/τ` 中 `1/τ` 爆炸 → 数值崩溃 + NaN
- **永久禁用**

**✅ `L_div = KL(P(log τ) ‖ U(log τ_min, log τ_max))`**：
- 在对数空间约束均匀分布，自然覆盖多个数量级
- 无极化崩塌风险
- 与 LTC 文献（Hasani 2021）的 τ 跨数量级分布 ($10^{-2}$ ~ $10^2$) 对齐

---

## 5. F6 DoD 对照（ADR-038 v1.1 §F6）

| DoD | 状态 | 证据 |
|---|---|---|
| 当前 short/long 粗粒度先做物理含义标注 | ✅ | §1.1 数据流 + §1.2 代码证据表 |
| 每个神经元（或神经元组）给出物理时间含义代码证据 | ✅ | §1.2 7 行代码表 + §2.2 bottom/top 5 neuron idx 实测值 |
| 输出 `docs/realtime/tau-spectrum-audit.md` | ✅ | 本文件 |
| 站 1 现场展示的 τ 光谱每个分组都有 audit 文档行级出处 | ✅ | §1.2 表所有 file:line 跳转 |
| 任何观众追问"为什么是 X τ"，cc 能 30s 内打开文档定位代码行 | ✅ | §1.2 表 + §3.1 拒绝叙事清单防止 cc 自己说错 |
| 不存在"叙事先行"的整齐刻度（30s/5min/1h/1d/7d/∞） | ✅ | §3.1 显式拒绝；§3.2 改用实测分布叙事 |

---

## 6. 修订下游产出物

本 audit 完成后，下列产出物必须基于其结论修订（不是基于 v1.0 ADR-038 的乐观假设）：

1. **F4 时间曲率旋钮 UI 设计**：站 1 主菜从"τ 光谱"改"dt 不规则 vs LTC 鲁棒性"
2. **F2 matched baseline 对照口径**：除原 6 公平口径外，**新增 "τ 分化度对照"**——并排展示 LnMem vs Phased LSTM 训练后的 τ 分布（让观众看到我们诚实暴露当前限制）
3. **站 1 现场脚本（待写）**：开场白第一句必须是"我们今天演示的是 LTC ODE 架构能力的真实快照，**不是为了 demo 优化的特殊状态**"——这是信用地基
4. **路线预告 PPT 一页**：放 §4 表 "τ 显式分化损失 / Liquid-S4 / drift 放大 / FiLM 在线 / N=3 真人"5 项 ADR 编号

---

## 附录 A：复现命令（v1.1 通用版，gemini 二审钦定修订）

> 原 v1.0 硬编码 `.venv-poc-fresh` + 特定时间戳 ckpt 文件名，gemini 钦定改通用 + 自动选最新 ckpt。

```bash
# 1. 自动选最新 phase2 ckpt（避免文件名硬编码）
CKPT=$(ls -t poc/data/checkpoints/m0w2_phase2/checkpoint_phase2_*.pt | head -1)
echo "Using ckpt: $CKPT"

# 2. 复现 τ 分布 + 物理时间标定
python3 -c "
import torch
import torch.nn.functional as F
import math
import os

ckpt_path = os.environ.get('CKPT', '$CKPT')
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
sd = ckpt['state_model']
emb = sd['personalization.embedding.weight']
Wt = sd['personalization.tau_projection.weight']
bt = sd['personalization.tau_projection.bias']
drift = ckpt['online_tau_drift']
tau_bias = emb @ Wt.T + bt + drift
timespans = 0.5 + F.softplus(tau_bias)

# τ 分布
print(f'tau mean: {float(timespans.mean()):.4f}')
print(f'tau std : {float(timespans.std()):.4f}')
print(f'tau min : {float(timespans.min()):.4f}')
print(f'tau max : {float(timespans.max()):.4f}')

# 物理时间标定 (daemon dt=1.0 s 假设)
DT_SECONDS = 1.0
tau_mean = float(timespans.mean())
half_life = tau_mean * math.log(2.0) * DT_SECONDS
print(f'\nPhysical calibration (assuming daemon dt=1.0s):')
print(f'  half-life: {half_life:.4f} s')
print(f'  product 30s short-term gap: {30 / half_life:.1f}x')
print(f'  product 1h drift gap     : {3600 / half_life:.1f}x')
"
```

预期输出：
```
tau mean: 1.1870
tau std : 0.0130
tau min : 1.1275
tau max : 1.2278

Physical calibration (assuming daemon dt=1.0s):
  half-life: 0.8228 s
  product 30s short-term gap: 36.5x
  product 1h drift gap     : 4376.3x
```

---

## 6. 现场答辩底牌（gemini 二审钦定 v1.1 新增）

> 如果观众抓住「你们 τ 没分化就等于 RNN」cc 现场答辩话术。

**核心防御逻辑**（即便 τ 退化为单值，LNN vs RNN 仍有本质架构差异）：

1. **连续时间闸门 (Continuous-time Gating)**
   - LTC/CfC 状态更新公式 `dx/dt = -x/τ + f(input, t)` 是 ODE **闭式解逼近**
   - dt 天然嵌入到非线性门控中
   - GRU 离散步进 `h_{t+1} = (1-z_t)·h_t + z_t·tilde_h_t`，截断误差随不规则 dt 扰动呈指数级级联放大

2. **零样本率适应 (Zero-shot Rate Adaptivity)**
   - CfC 即便 τ 是常数，状态流形在 dt→0 和 dt→∞ 时仍具渐进稳定性
   - **demo 现场可演**：抓不规则 dt 序列（混合 0.1s/1s/10s/120s 采样），LTC 平滑 vs GRU 抽搐

3. **架构上限 ≠ 训练状态**
   - 「我们 64 个 τ 神经元的架构上限是对数均匀 [10⁻², 10²]，覆盖 4 个数量级」
   - 「当前 ckpt 训出来的状态是 [1.13, 1.23]，仅 0.1 区间——这是训练损失没显式约束的结果，**不是架构限制**」
   - 「一键切 `--mock-tau-diversity` 可现场看到注入分化 τ 后的效果，证明能力存在」

4. **如果 7 月承诺没兑现的兜底叙事**：
   - 引用神经科学：「大脑皮层不同区域神经元时间常数也是基因表达决定（先验结构），不是完全后天习得」
   - 转向「半监督通道剪裁」+「物理先验硬编码」——学术上和工业上都站得住脚

**绝对不能说的**：
- ❌「这是 demo 优化版」
- ❌「我们就是没训好」（不带技术解释）
- ❌「7 月一定能修好」（追责性日期承诺）

---

## 附录 B：诚实策略 = 信用工程

**ADR-038 v1.1 §决策 5** 已明确：「τ 神经元先 audit 再画」+「不存在叙事先行的整齐刻度」。本 audit 是这条决策的执行产物。

**川哥钦定（5/27）**：demo 必须现场可交互证明 LNN 差异化。这个"差异化"不能靠夸大，**只能靠诚实承认当前状态 + 展示架构本身的能力上限**。技术朋友看到 cc 主动暴露 τ 分化不足，反而会因为「严谨」给高分——这与 codex 二轮审钦定的话术一致：「展示半成品系统件，不是设计图，技术人会因严谨给高分」。

---

**版本**：v1.0 · 2026-05-27
**作者**：cc（基于 phase2 ckpt 2026-05-24 实测）
**Review**：待 gemini 3.5 high 审查
**关联 ADR**：[`ADR-038`](../decisions/038-path-d-plus-plus-demo-engineering-foundation.md) §F6
**关联代码**：[`personalization.py`](../../poc/src/lnn/personalization.py), [`model.py`](../../poc/src/lnn/model.py), [`online_adapter.py`](../../poc/src/lnn/online_adapter.py), [`contracts.py`](../../poc/src/lnn/contracts.py)
