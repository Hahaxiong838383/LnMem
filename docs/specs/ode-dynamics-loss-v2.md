# ODE 动力学 Loss · 实施 Spec **v2** Latent World Model（cc + gemini 第 10 轮收敛 2026-05-22）

> ## v1 → v2 整改原因
> gemini 第 10 轮抓出 v1 两个**致命数学错误**：
> 1. **L_predict ≡ 0 恒等死局**（teacher-forcing input + 共享 backbone 算 hidden MSE 数学归纳证明永远 0）
> 2. **L_ode Euler 自洽与 CfC 闭式解析解冲突**（CfC 用 `e^{-(τ+f)·dt}` 指数门控半隐式，强加 Euler 等于退回 ResNet）
>
> v2 完全采纳 gemini 提出的 **Latent World Model Rollout** 范式（Dreamer 风格）。

关联 ADR-023 范式级转换。本 spec 是 codex 实施的契约 + cc 整合测试的输入。

---

## 1. 核心范式：Latent World Model Rollout

### 1.1 训练逻辑

```
给一段时序 x[0..T-1]（T=30），切分 prefix + rollout：
  prefix_len N = 15  (前 50%)
  rollout_len K = 15 (后 50%)

【Phase A · Prefix 建立记忆】（拿真实 x 输入）
  for t in [0, N-1]:
    h[t] = backbone_step(x[t], h[t-1], persona_idx)

【Phase B · Rollout 闭眼预测】（不给真实 x，自回归）
  for t in [N, N+K-1]:
    x_pred[t] = decoder(h_pred[t-1])              # 解码上一步 h 预测当前 x
    h_pred[t] = backbone_step(x_pred[t], h_pred[t-1], persona_idx)  # 自回归喂回
  
  其中 h_pred[N-1] = h[N-1]（prefix 末态）

【Loss】
  L_world_model = (1/K) · Σ_{t=N..N+K-1} MSE(x_pred[t], x[t])  # 仅 rollout 阶段 input 预测误差
  L_recon_prefix = (1/N) · Σ_{t=0..N-1} MSE(decoder(h[t]), x[t])  # prefix 重构锚定
  
  L_total = w_world * L_world_model + w_recon_prefix * L_recon_prefix
```

**关键**：rollout 阶段 LNN **拿不到真实 x**，必须靠 h 中保留的连续动力学**模拟**未来。这从数学上排除了恒等死局，也排除了 L_recon 瞬时塌陷（rollout 没真实输入就崩）。

### 1.2 为什么这个范式无懈可击（gemini 第 10 轮论据）

| 路径 | 后果 |
|---|---|
| **L_predict v1 恒等 0** | ❌ 完全无梯度 |
| **L_recon 瞬时塌陷 h≈Wx** | 在 rollout 阶段 LNN 没真实 x，h≈Wx 会立刻让 x_pred 完全错，L_world_model 爆炸 |
| **CfC Euler 强约束** | ❌ 跟闭式解析解冲突 |
| **L_world_model（v2）** | ✅ 直接驱动 LNN 学连续动力学（必须能 forward 预测） |

## 2. 接口（Protocol）

### 2.1 OdeDynamicsLoss v2（`poc/lnn/ode_dynamics_loss.py`）

```python
from __future__ import annotations
import dataclasses
import torch
import torch.nn as nn

@dataclasses.dataclass(frozen=True)
class OdeDynamicsLossConfig:
    """v2 钦定（gemini 第 10 轮收敛）."""
    w_world_model: float = 1.0      # Rollout 主目标
    w_recon_prefix: float = 0.5     # Prefix 重构锚定（gemini 钦定 0.5 比 v1 1.0 弱）
    prefix_ratio: float = 0.5       # prefix 占 50%
    rollout_horizon: int = 15       # K=15 步 (gemini 钦定可退火 5→15)
    detach_prefix_grad: bool = False  # 是否在 prefix-rollout 边界 stop grad
    # 旧字段废弃（不再用 L_predict / L_ode / L_infonce_aux）


class Decoder(nn.Module):
    """64-d hidden → 52-d input space.
    
    gemini 第 10 轮警告：decoder 容量过大 → trivial 重构。
    限制：单层 Linear（无 hidden layer，强制 latent 必须保留信息）。
    """

    def __init__(self, hidden_dim: int = 64, input_dim: int = 52,
                 num_layers: int = 1) -> None:
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(hidden_dim, input_dim)
        else:
            # 1-hidden MLP (gemini 警告慎用)
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (..., hidden_dim) → x: (..., input_dim)"""
        return self.net(h)


class OdeDynamicsLoss(nn.Module):
    """v2 Latent World Model Rollout."""

    def __init__(self, cfg: OdeDynamicsLossConfig | None = None,
                 decoder: Decoder | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OdeDynamicsLossConfig()
        self.decoder = decoder or Decoder()

    def forward(
        self,
        x: torch.Tensor,             # (B, T, input_dim)
        backbone_step_fn,            # callable: (x_t, h_prev_list, persona_idx) → h_new_list
        persona_idx: torch.Tensor,   # (B,)
        init_hidden_fn,              # callable: (B, device) → h_prev_list (per layer initial)
    ) -> dict[str, torch.Tensor]:
        """
        Returns dict:
          - loss: total scalar
          - L_world_model: rollout MSE
          - L_recon_prefix: prefix decode MSE
          - prefix_len, rollout_len
          - x_pred_rollout: (B, K, input_dim) for diagnostics
          - h_seq_prefix: (B, N, hidden_dim) for evaluator (effective_dim 评估)
        """
        ...
```

### 2.2 trainer.py 整合

```python
# TinyTrainer.__init__
self.decoder = Decoder(hidden_dim=cfg.hidden_dim, input_dim=52).to(self.device)
self.ode_loss = OdeDynamicsLoss(OdeDynamicsLossConfig(), decoder=self.decoder).to(self.device)
# AdamW 加 decoder 参数到主干 LR 组（1e-3）

# train_one_epoch step loop
for step_idx, sampled_windows in enumerate(loader):
    x = torch.stack([w["event_tensor"] for w in sampled_windows]).to(self.device)  # (B, T, 52)
    persona_idx = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

    out = self.ode_loss(
        x=x,
        backbone_step_fn=lambda x_t, h_list, p_idx: self.backbone_step(x_t, h_list, p_idx),
        persona_idx=persona_idx,
        init_hidden_fn=lambda B, dev: [cell.init_hidden(B, dev) for cell in self.backbone.cells],
    )
    loss = out["loss"]
    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    self._capture_tau_grad(tau_grad_buffer)
    # DGB / step ...
```

### 2.3 MultiLayerCustomCfC 暴露 step 接口

```python
class MultiLayerCustomCfC:
    def step(self, x_t, h_prev_list, persona_idx, dt=1.0):
        """单步 forward（用于 ode_loss rollout）.
        
        Args:
            x_t: (B, input_dim)
            h_prev_list: list of (B, hidden_dim) for each layer
            persona_idx: (B,)
        Returns:
            h_new_list: list of (B, hidden_dim)
        """
        h_new_list = []
        for layer_idx, cell in enumerate(self.cells):
            input_for_layer = x_t if layer_idx == 0 else h_new_list[layer_idx - 1]
            h_new = cell(
                x_t=input_for_layer,
                h_prev=h_prev_list[layer_idx],
                persona_idx=persona_idx,
                dt=dt,
            )
            h_new_list.append(h_new)
        return h_new_list
```

注意：trainer 的 encode pipeline 现在是 fusion → backbone。step 接口需要把 fusion 也调一次：

```python
def backbone_step(self, x_t, h_prev_list, persona_idx, dt=1.0):
    """trainer 内部辅助：fusion + backbone single step."""
    x_t_fused = self.fusion(x_t.unsqueeze(1)).squeeze(1)  # (B, 52) → (B, 64)
    return self.backbone.step(x_t_fused, h_prev_list, persona_idx, dt)
```

## 3. 数学严格定义

### 3.1 完整训练 step（伪代码）

```
Input:
  x ∈ ℝ^{B×T×52}, T=30
  persona_idx ∈ ℤ^B
  prefix_len N = 15, rollout_len K = 15

Step 0: 初始化
  h_states = [cell.init_hidden(B) for cell in cells]  # list[(B, 64)]

Step 1: Prefix 阶段（拿真实 x）
  h_seq_prefix = []
  x_recon_prefix = []
  for t in [0, N-1]:
    h_states = backbone_step(x[:, t, :], h_states, persona_idx)
    h_seq_prefix.append(h_states[-1])  # 最后层 hidden
    x_recon_prefix.append(decoder(h_states[-1]))
  
  L_recon_prefix = mean over (B, N) of ||x_recon_prefix[t] - x[t]||²

Step 2: Rollout 阶段（闭眼自回归）
  h_pred_states = h_states  # 继续 prefix 的末态
  x_pred_rollout = []
  for t in [N, N+K-1]:
    x_pred_t = decoder(h_pred_states[-1])  # 用上一步末态预测当前 input
    x_pred_rollout.append(x_pred_t)
    h_pred_states = backbone_step(x_pred_t, h_pred_states, persona_idx)
  
  L_world_model = mean over (B, K) of ||x_pred_rollout[t] - x[t]||²

Step 3: 总 loss
  L_total = w_world_model * L_world_model + w_recon_prefix * L_recon_prefix
```

### 3.2 梯度路径分析

| Loss | 反传路径 |
|---|---|
| L_recon_prefix | decoder.weight + h_seq_prefix → backbone.cells → fusion |
| L_world_model | decoder.weight (rollout 多次)+ h_pred_states → backbone.cells（多次展开）→ prefix h_states → backbone.cells（再次）→ fusion |

**注意**：L_world_model 梯度沿 rollout K 步展开 + prefix N 步展开，总 BPTT 深度 = N+K = 30。**PC-5b truncated BPTT 在这里要小心**：可能需要在 prefix 阶段也截断。

### 3.3 vs v1 数学正确性对照

| v1 设计 | v2 设计 |
|---|---|
| h_pred[N+k] vs h_seq[N+k] MSE（恒等）| x_pred[N+k] vs x[N+k] MSE（必非零） |
| 用真实 x[N+k] teacher-force | 用 decoder(h_pred[N+k-1]) 自回归 |
| L_predict 数学 ≡ 0 | L_world_model 数学 > 0 且强约束动力学 |
| L_ode Euler 跟 CfC 解冲突 | L_ode 完全删除 |
| L_infonce_aux 保留 | L_infonce_aux 完全删除（gemini 钦定 99% 置信度） |

## 4. 验收单测

`poc/lnn/tests/test_ode_dynamics_loss.py`：

1. **smoke**：随机输入 forward 不崩
2. **L_world_model 非零**：随机初始化下 L_world_model > 0.01（**v1 bug 的反例验证**）
3. **L_world_model 梯度可达**：backward 后 backbone.cells[0].xh_weight.grad 非空
4. **Decoder 梯度可达**：backward 后 decoder.net.weight.grad 非空
5. **Rollout 自回归正确性**：x_pred[N+k] 应严格依赖 h_pred[N+k-1]（不依赖任何 x[N+k]）
6. **Shuffle Test**：x 帧打乱后 L_world_model 必显著抬高（>= +50%）

`poc/training/tests/test_ode_loss_integration.py`：

1. **trainer dry-run 1 epoch**：跑通 + L_world_model 在降
2. **effective_dim 测量**：1 epoch 后 effective_dim 不退化（≥ baseline）
3. **prefix vs rollout 切分**：N=15, K=15, T=30 边界正确

## 5. 评估改动 `poc/training/evaluate.py`

- **删除 retrieval-based M1/M5/M1V**（不再需要 contrastive 相似度）
- **新 metric**：
  - `L_world_model_train` / `L_world_model_unseen` → M5 改用 unseen-train gap
  - `L_world_model_shuffled` → shuffle test 改用 rollout drop
  - effective_dim 保留（用 prefix h_seq SVD）

## 6. codex 实施任务清单（v2）

1. 新建 `poc/lnn/ode_dynamics_loss.py`（v2 Latent World Model）
2. 改 `poc/lnn/multilayer_cfc.py`：暴露 `step()` 接口
3. 改 `poc/training/trainer.py`：
   - `TrainConfig` 新增 `use_ode_loss: bool = True` + `ode_loss_cfg` 字段
   - 加 `backbone_step()` 辅助方法（fusion + backbone.step 拼起来）
   - `train_one_epoch` 切换主损失，**完全删除** _build_contrastive_pairs 调用
   - 旧 InfoNCE 分支保留作 feature flag (`use_ode_loss=False`)
4. 改 `poc/training/evaluate.py`：加 v2 metrics
5. 单测 6+3 个
6. 跑全量 pytest 全绿
7. 跑 1 epoch dry-run 验 L_world_model 在降 + L_world_model > 0（**v1 bug 反例验证**）

## 7. 不准踩

- **不可妥协 §10.1 6 项必须保留**
- **ADR-022 v3 τ-bias 释放 + DGB**：不动
- **不要删旧 InfoNCE/VICReg 路径**（feature flag 保留）
- **不要 commit / push / ssh Studio**
- **rollout 阶段 backbone_step 必须用 x_pred[N+k]，不能 fallback 用 x[N+k]**（避免回到 v1 bug）
- **L_world_model 单测必须验 >0** 防止再走 v1 死局

## 8. 5/27 启动判定（gemini 第 10 轮第 4 次更新）

- 当前 🟥 RED LIGHT
- 放行条件（tiny 50 epoch）：
  1. `effective_dim` ≥ 8.0
  2. `L_world_model_shuffled / L_world_model_train` ≥ 1.5（shuffle drop ≥ 0.15 等价）
  3. `L_world_model_train` final 应 ∈ [0.05, 0.3]（不爆炸不塌陷）

---

**📝 维护者**：cc + gemini 第 10 轮收敛 | **状态**：v2（待 codex 实施 + cc 整合）
