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

---

**📝 维护者**：cc 主线 | **生效**：2026-05-21 23:40 | **下一步**：γ-2 启动 Path C 实现
