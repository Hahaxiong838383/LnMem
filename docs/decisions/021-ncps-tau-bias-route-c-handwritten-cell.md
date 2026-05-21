---
adr: ADR-021
title: ncps τ-bias 三路径 spike 结论 · γ-2 启动 Path C 手写 CfCCell
status: accepted（2026-05-21 23:40 cc 实测，川哥钦定 team 模式继续）
date: 2026-05-21
deciders: cc 主线 + γ-1.2 实测 + gemini C.1 担心
related:
  - 016-base-v3-multimodal-self-supervised-lnn.md
  - 019-spec-v0_2-revision-from-academic-review.md
  - ../specs/base-v3-training-spec.md §2.1 / §8 γ-2 / §9 #13 / §11.1 #12
parent: ../STYLE.md
体裁: L2（决策记录）
---

# ADR-021 · ncps τ-bias 三路径 spike 实测结论

## 1. Context

spec v0.2.4 §2.1 定义：`τ-bias: τ_i = τ_base * exp(persona_bias_i)`，作为 per-user 个性化 LNN 心智核心（不可妥协 §10.1 #4）的工程实现。

gemini 双评审 (5/21 22:30) 留 C.1 CRITICAL 担心：「ncps 1.0.1 高层 API 不暴露 τ 外部注入接口，包装一层 FiLM/MLP 而不调制内部 ODE 即假 LNN-FIRST」。

cc 主线 γ-1.2 spike (5/21 23:30 跑 `poc/spike/tau_bias_routes.py`) 实测三条路径。

## 2. 三路径实测结果

### Path A · forward(timespans=) 外部时间步

**结果：ABANDONED**

**根因（ncps 1.0.1 上游 bug）**：
```python
# ncps/torch/cfc.py:173
ts = 1.0 if timespans is None else timespans[:, t].squeeze()  # shape (batch,)

# ncps/torch/cfc_cell.py:162
t_interp = self.sigmoid(t_a * ts + t_b)  # t_a shape (batch, hidden)
# (batch,) * (batch, hidden) PyTorch 右对齐 broadcast 失败 → RuntimeError
```

`squeeze()` 把 dim=1 砍掉，导致 ts 不能 broadcast 到 (batch, hidden)。任何 `batch_size > 1` 都会触发。Path A 在 ncps 1.0.1 实测彻底不可用（除非 batch=1，或我们提 upstream PR 修 squeeze）。

### Path B · hook CfCCell.time_a.weight + persona-conditioned bias

**结果：WEAK**

**实测数据**（4 personas × 4 samples/persona = 16 batch，dt=1.0 default）：
- hidden_diff vs baseline mean = 2.04（有显著扰动）
- paired t-test p = 1.5e-3 < 0.05 ✓（hidden 被扰动是真的）
- **persona separation ratio = 1.003x**（不同 persona 间距 vs 同 persona 内 ≈ 1）

**根因**：
- monkey-patch time_a.forward 加 persona bias → bias 加到 t_interp 输入 → sigmoid 压缩到 (0,1)
- ff1/ff2 weighted combination 之后，**hidden 整体被扰动了但 persona 之间没有分化**
- 等价于「persona bias 在 ODE 中被稀释，不构成真 τ 调制」

**LNN-FIRST 评估**：触发 spec §9 #13 风险 — τ 没真进 ODE 内核，是假 LNN-FIRST。

### Path C · 手写 CfCCell 单步隐式欧拉求解器

**结果：必须启动（γ-2）**

**实施路径**：
```python
# 伪代码（spec §2.1 ODE 公式）
class CustomCfCCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_personas):
        self.tau_base = nn.Parameter(torch.ones(hidden_dim))
        self.persona_bias = nn.Embedding(num_personas, hidden_dim)
        self.f_net = MLP(input_dim + hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, persona_idx, dt=1.0):
        # τ_i = τ_base * exp(persona_bias_i)  per-neuron
        tau = self.tau_base * torch.exp(self.persona_bias(persona_idx))
        # 单步隐式欧拉 ODE: τ * dh/dt = -h + f(x, h)
        f = self.f_net(torch.cat([x_t, h_prev], dim=-1))
        h_new = (tau * h_prev + dt * f) / (tau + dt)  # 隐式欧拉解析解
        return h_new
```

**工时**：+1 天（spec §8 γ-2 1.5d → 2.5d）

**验收**：preflight #12 显著性检验 + sep > 1.5（比 Path B 严格）

## 3. Decision

**γ-2 启动 Path C 手写 CfCCell**，放弃 Path A/B。

理由：
1. **真 LNN-FIRST**：τ 直接进 ODE 解析解，不是外层包装
2. **per-neuron + per-persona 双维度调制**：spec §2.1 公式严格落地
3. **数学透明**：单步隐式欧拉是经典 ODE 求解器，行为可预测
4. **工时可控**：+1 天，Phase γ 整体仍在 5/27-6/10 窗口内（Step 4 调参可压缩）

## 4. Consequences

### 正面
- 不可妥协 §10.1 #4 per-user τ-bias 三件套真落地
- 摆脱 ncps 1.0.1 上游 bug 阻塞
- γ-2 实施透明度大幅提升（手写 cell 易 debug + 加 hook）

### 负面
- Phase γ 整体工时 +1 天（5/27→5/28 推迟启动 or Step 4 压 4d→3d）
- ncps 库依赖降级为「参考实现」而非 production backbone
- 需要重新跑参数量预算（手写 cell 参数量预估 50-70K vs ncps 49K，相近）

### 中性
- ncps 1.0.1 仍保留作 sanity check baseline（γ-2 同时跑两版本，对比 hidden 演化合理性）
- 未来若 ncps 上游修 squeeze bug 或暴露 cell hook 接口，可考虑 fallback

## 5. Trigger 条件

**5/22 γ-2 启动前**触发条件已就绪：
- ✅ Path A/B WEAK 证据落盘 `poc/spike/tau_bias_routes.py` + `poc/spike/tau_bias_routes_result.json`
- ✅ Path C 实施路径定义清晰（本 ADR §2 伪代码）
- ✅ Phase γ 工时 buffer 充足（Step 4 调参可压 4d→3d）

## 6. 关联记录

- spec v0.2.4 §9 #13 风险（τ 没真进 ODE）→ 本 ADR 实测确认 + 化解
- gemini C.1 担心（vanilla ncps 是黑盒）→ 部分否定（time_a/timespans 暴露）+ 部分确认（hook 不够）
- ncps 上游 PR 机会：fix `timespans[:, t].squeeze()` broadcast bug（社区贡献候选）

## 7. 实测修订（gemini 第二轮对抗审 REVISE 后，2026-05-22 01:00）

gemini 第二轮抓出 5 个修订项，cc 全部采纳 + 1 项 Reject。

### 7.1 数学定义诚实化（gemini F-1.1 HIGH）

**原版**：声称「单步隐式欧拉解析解」
**实际**：右端 `f(x_t, h_prev)` 作用于 `h_prev` 而非 `h_new` → **半隐式欧拉（Semi-Implicit Euler）**
**修正**：docstring + ADR + spec 全部改「半隐式欧拉离散化」（学术诚实，避免论文审稿人毁灭性打击）

### 7.2 τ clamp → scaled tanh（gemini F-1.2 MEDIUM）

`torch.clamp(bias, -3, 3)` 在边界处导数=0 → **梯度死区**
改用 `3 * tanh(bias/3)` 软有界，全域可微，物理范围等价。

### 7.3 363995x 数值幻觉化解（gemini F-2.1 CRITICAL）

**原版测试设计致命缺陷**：
- 输入完全确定性（前 0 后 1 step）
- 同 persona_idx → 同 persona_tau_bias → **同 persona 内 hidden 完全相同**
- mean_same 真实值 = 0 → 撞 `1e-6` 安全垫 → sep = 0.36 / 1e-6 = **360000x 数值幻觉**

**修正**：每样本独立 input noise（std=0.3）+ random init hidden → 真实 mean_same > 0

**真实 init 阶段 sep 实测**：
- 1 层 sep = **1.109x**（同 persona within=0.70, cross=0.78）
- 2 层 sep = **1.248x**

→ 之前 363995x **完全是测试设计 cherry-pick**，不是 cell 的真实物理性能。

### 7.4 var 阈值不放水（gemini F-3.1 HIGH）

**原版**：var > 1e-4（从 spec 0.01 放宽 100 倍）→ 工程妥协
**修正**：
- 测试阈值定 var > 5e-3
- `CustomCfCCellConfig` 调 `tau_init: 1.0 → 0.5` + `f_net_init_gain: 2.5`（增大 f_net 输出 gain）
- 实测：1 层 var = 0.049 / 2 层 var = 0.0072，自然达标不放水

### 7.5 f_net shortcut learning 量化证实（gemini F-4.1 HIGH）

**新测试** `test_gradient_flow_to_tau_bias` 实测：
```
bias_grad / f_net_grad ratio = 0.002
  (persona_tau_bias.grad.sum.abs = 25.33)
  (f_net.parameters.grad.sum.abs = 12343.27)
```

**结论**：未做任何缓解时 f_net 27K 强力 MLP **完全 shortcut** 4K 的 persona_tau_bias，grad 比例 1:500。

**γ-3 训练必须缓解（spec v0.2.7 加）**：
1. **f_net 加 L2 正则**：`weight_decay > 1e-3`
2. **persona_tau_bias 独立大学习率**：`lr_scaling = 5-10x` 主干 LR
3. **训练初期 freeze f_net 1k step**：强迫先学 τ-bias
4. **每 1k step 监控** `bias_grad/f_net_grad ratio`：< 1e-3 触发 alarm

### 7.6 长序列稳定性证实（gemini F-6.1 MEDIUM）

新测试 `test_long_sequence_stability` seq=1000 + 极端 τ：
- final |h|_max = 0.42（bounded 无爆炸）
- 无 NaN/Inf

半隐式欧拉公式 `h_new = (τ * h_prev + dt * f) / (τ + dt)` 的天然 bound：|h_new| ≤ max(|h_prev|, |f|)，长序列不会指数爆炸。

### 7.7 Reject: Loss 3 永久关「学术降级」（gemini F-5.1 HIGH）

**gemini 担心**：Loss 3 关 = 失去「多用户个性化对齐」核心学术卖点。提议合成 per-viewer 数据 + 邮件 VEATIC 作者。

**cc 主线 Reject**：
- VEATIC viewer = **annotator 标注者**，**不是 subject persona 被记录用户**（codex 4 轮严审 + 5/21 22:30 评审已确认）
- 合成 per-viewer 数据 = 凭空造假，**违反 LNN-FIRST P-1**
- Loss 3 真路径 = **Phase δ 川哥真人 daemon 数据**（不是 VEATIC）
- 当前 Phase γ 走自监督表征学习是正确方向，不退化为多 viewer 情感预测器

### 7.8 测试名重命名（认知诚实）

- `test_persona_separation` → `test_persona_init_sep_nonzero`（明确 init 阶段不验 > 1.5）
- 2 层同名重命名

spec §11.1 #12 `sep > 1.5` 验收**留 γ-3 训练 1 epoch 后做**（含 L2 reg + LR scaling）。

---

**📝 第二轮修订**：cc 主线 + gemini 3.5 high 对抗审 | 2026-05-22 01:00 | 14 测试全过

## 8. 第三轮 gemini 评审 γ-3 实现层修订（2026-05-22 02:00）

gemini 第三轮（评 γ-3 InfoNCE/VICReg/Fusion/heads）抓 5 个 finding，cc 全部采纳。

### 8.1 F-1 InfoNCE 可学温度坍塌风险 (HIGH)

**风险**：`learnable_sigma=True` 时 τ 可被压缩到 1e-4 截断值，logits 人为放大，CE 损失「作弊式」降低 → 梯度爆炸 / 表征退化。
**修正**：`InfoNCEConfig.learnable_sigma` 默认改 `False`，5/22 首跑必须 τ=0.1 固定。spike 双 seed 通过后再考虑开启。

### 8.2 F-2 VICReg N≤1 除零隐患 (HIGH)

**风险**：协方差分母 `N-1`，DataLoader 最后一个不完整 batch N=1 时整轮训练 NaN 崩溃。
**修正**：
- `if N <= 1: return zero loss + warning`
- 分母 `max(N-1, 1)`
- `N < 8` 时打 warning（Bardes 论文建议 batch ≥ 64）

### 8.3 F-3 test_infonce_perfect_positive 伪装 (MEDIUM)

**问题**：原版 `n = torch.randn() * 0.01` 试图制造「弱负样本」，但 `F.normalize(n)` 第一步就抹除了 0.01 缩放。
**修正**：构造**真正与 anchor 正交方向**的 negatives（`a_perp = n - (n·â)â`），normalize 后仍正交。这才是真「难度可控弱负样本」。

### 8.4 F-4 freeze 梯度泄露 (**CRITICAL**)

**漏洞**：仅 `freeze_module(backbone)` 不阻止梯度通过 unfrozen FusionLayer 反流。Readout Heads 弱监督信号会更新 fusion projection 矩阵，间接污染表征 → 违反 spec §3.4 Loss 4 弱监督铁律。
**修正**：新增 `freeze_representation_pipeline(fusion, backbone)` 同时 freeze。
**验证测试** `test_freeze_representation_pipeline_critical`：模拟 Loss 4 backward → fusion grads 全部 None ✓（无梯度穿透）。

### 8.5 F-5 时序截断未闭环 (HIGH)

**问题**：`detect_semantic_breakpoints` 只是 helper，没接入 `InfoNCELoss.forward`。gemini 终审条件 1 形同虚设。
**修正**：`InfoNCELoss.forward` 加 `breakpoint_mask: Optional[Tensor]` 参数，mask 为 True 的样本跳过 CE 计算（per-sample loss × keep mask / n_keep）。

### 8.6 关键测试新增

- `test_infonce_breakpoint_mask_closure`: 验证 mask=True 时 loss 跳过
- `test_infonce_default_no_learnable_sigma`: 验证默认温度 fixed
- `test_vicreg_n1_no_divide_by_zero`: 验证 N=1 边界 graceful
- `test_freeze_representation_pipeline_critical`: 验证 fusion grad 无穿透

**结果**：33 测试全过 ✓，spec v0.2.7 → v0.2.8.

---

**📝 第三轮修订**：cc 主线 + gemini 3.5 high γ-3 评审 | 2026-05-22 02:00 | 33 测试全过

---

**📝 维护者**：cc 主线 | **生效**：2026-05-21 23:40 | **下一步**：γ-2 启动 Path C 实现
