# ODE 动力学 Loss · 实施 Spec v1（cc 主写 2026-05-22）

> 关联 ADR-023 范式级转换。本 spec 是 codex 实施的契约 + gemini 评审的输入。

---

## 1. 接口（Protocol 先行）

### 1.1 OdeDynamicsLoss 类（`poc/lnn/ode_dynamics_loss.py`）

```python
from __future__ import annotations
import dataclasses
import torch
import torch.nn as nn

@dataclasses.dataclass(frozen=True)
class OdeDynamicsLossConfig:
    """ADR-023 钦定权重默认值（codex/gemini 第 10 轮可调）."""
    w_predict: float = 1.0      # 多步 rollout 主目标
    w_recon: float = 1.0        # 重构自编码强信息约束
    w_ode: float = 0.1          # ODE 一致性轻量自洽
    w_infonce_aux: float = 0.1  # 时序对齐辅助
    predict_horizon: int = 5    # 预测多少步（codex 第 10 轮可推荐扩到 15）
    prefix_ratio: float = 0.5   # 前 50% 帧用作 prefix


class Decoder(nn.Module):
    """64-d hidden → 52-d input space (ADR-023 §4.3)."""

    def __init__(self, hidden_dim: int = 64, input_dim: int = 52,
                 num_layers: int = 1) -> None:
        super().__init__()
        if num_layers == 1:
            self.net = nn.Linear(hidden_dim, input_dim)
        else:
            self.net = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, input_dim),
            )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (B, T, hidden_dim) → x_recon: (B, T, input_dim)"""
        return self.net(h)


class OdeDynamicsLoss(nn.Module):
    """ADR-023 主损失：L_predict + L_recon + L_ode + L_infonce_aux."""

    def __init__(self, cfg: OdeDynamicsLossConfig | None = None,
                 decoder: Decoder | None = None) -> None:
        super().__init__()
        self.cfg = cfg or OdeDynamicsLossConfig()
        self.decoder = decoder or Decoder()

    def forward(
        self,
        x: torch.Tensor,             # (B, T, input_dim) 原始输入
        h_seq: torch.Tensor,         # (B, T, hidden_dim) backbone 输出
        backbone_step_fn,            # 函数：(x_t, h_prev, persona_idx) → h_next
        persona_idx: torch.Tensor,   # (B,)
    ) -> dict[str, torch.Tensor]:
        """Returns dict: {loss, L_predict, L_recon, L_ode, ...}."""
        ...
```

### 1.2 trainer.py 整合点

```python
# trainer.py::TinyTrainer.__init__
self.decoder = Decoder(hidden_dim=cfg.hidden_dim, input_dim=52).to(self.device)
self.ode_loss = OdeDynamicsLoss(OdeDynamicsLossConfig(), decoder=self.decoder).to(self.device)
# optimizer 加 decoder 参数（用主干 LR）
# - 不需要 contrastive pairs，直接整段 (B, T, 52)
# - 删除 _build_contrastive_pairs 调用（保留代码但 train_one_epoch 不用了）
```

```python
# trainer.py::train_one_epoch step loop（替换原 InfoNCE 主损失）
for step_idx, sampled_windows in enumerate(loader):
    x = torch.stack([w["event_tensor"] for w in sampled_windows]).to(self.device)  # (B, T, 52)
    persona_idx = torch.zeros(x.shape[0], dtype=torch.long, device=self.device)

    h_seq = self.encode(x, persona_idx)  # (B, T, 64) pre-normalize

    out = self.ode_loss(
        x=x,
        h_seq=h_seq,
        backbone_step_fn=lambda x_t, h_prev, p_idx: self.backbone_step(x_t, h_prev, p_idx),
        persona_idx=persona_idx,
    )
    loss = out["loss"]
    # ... backward / DGB / step
```

## 2. 数学具体公式

### 2.1 L_predict（多步 rollout）

```
给 prefix h[0..N-1] + x[0..N-1]，已经从 backbone 得到。
从 t=N-1 开始 rollout K 步：
  h_pred[N+k] = backbone_step(x[N+k], h_pred[N+k-1], persona_idx)  for k in [0, K-1]
  (注意：rollout 时 x 是 ground truth 输入，但 h 是预测的)

L_predict = (1/K) · Σ_{k=0..K-1} MSE(h_pred[N+k], h_seq[N+k])
```

其中 h_seq 是「真实」hidden（直接 encode 整段得到的）。

**注意**：rollout 用 ground truth 的 x[N+k]（teacher-forcing input），但 h 是迭代的。这是标准 next-state prediction 范式。

### 2.2 L_recon（重构自编码）

```
L_recon = (1/(B·T)) · Σ_{b,t} MSE(decoder(h_seq[b,t]), x[b,t])
```

整段时序逐帧重构。

### 2.3 L_ode（Euler 自洽）

```
对每个时间步 t < T-1:
  predicted_dh = backbone_step(x[t], h_seq[t], persona_idx) - h_seq[t]
  expected_dh = h_seq[t+1] - h_seq[t]
  L_ode_t = MSE(predicted_dh, expected_dh)

L_ode = (1/(T-1)) · Σ_t L_ode_t
```

注：这里其实跟 L_predict 在 K=1 时数学等价（一步预测），但保留作为「显式 ODE 自洽」诊断信号。**codex 第 10 轮决定是否合并到 L_predict**。

### 2.4 L_infonce_aux（保留时序对齐，weight=0.1）

复用现有 InfoNCELoss，但权重 0.1。同视频不同窗口仍构建 anchor/positive，但**不再是主损失**。

## 3. 数据流（trainer 简化）

```
Before (InfoNCE):
  windows pool → DataLoader → batch (B*4 windows)
              → _build_contrastive_pairs (anchor/pos/neg 三元组)
              → encode 3 次（h_a, h_p, h_n_flat）
              → InfoNCELoss(h_a, h_p, h_n)
              → + VICReg trigger aux

After (ODE-loss):
  windows pool → DataLoader → batch (B windows, 直接整段 (B, T, 52))
              → encode 1 次 (B, T, 64)
              → OdeDynamicsLoss(x, h_seq, backbone_step_fn, persona_idx)
              → out["loss"]
```

**简化点**：
- batch 从 `batch_size * 4` 改为 `batch_size`（不需要多抽 contrastive）
- encode 从 3 次（a/p/n_flat）变 1 次
- 步内不再调 _build_contrastive_pairs 和 InfoNCELoss
- VICReg trigger 完全 bypass（保留代码，TrainConfig 关闭）

## 4. 验收单测（cc 写测试 + codex 实施）

`poc/lnn/tests/test_ode_dynamics_loss.py`：

1. **smoke test**：随机输入 OdeDynamicsLoss forward 不崩
2. **L_recon 数学正确性**：当 decoder = identity（hidden_dim = input_dim）且 h = x 时，L_recon = 0
3. **L_predict 数学正确性**：当 backbone_step 完美预测时，L_predict = 0
4. **梯度可达 decoder**：backward 后 decoder.weight.grad 非空
5. **梯度可达 backbone**：backward 后 backbone.cells[0].xh_weight.grad 非空
6. **L_total 量级合理**：在随机初始化下，L_total ∈ [1.0, 20.0]（防止 explode/vanish）

`poc/training/tests/test_ode_loss_integration.py`：

1. **trainer dry-run 1 epoch**：跑通不崩 + loss 在降
2. **effective_dim 测量**：经过 1 epoch 训练后 effective_dim ≥ baseline（不退化）
3. **L_recon 训练曲线**：50 step 内 L_recon 必降 ≥ 30%

## 5. 与现有代码的兼容性

| 现有 | 处理 |
|---|---|
| `poc/lnn/losses.py::InfoNCELoss` | 保留（L_infonce_aux 用）|
| `poc/lnn/losses.py::VICRegLoss` | 保留但不调用 |
| `poc/training/vicreg_trigger.py` | 保留代码，cfg.enabled=False |
| `poc/training/au_augment.py` | 保留代码，cfg.au_augment_enabled=False |
| `poc/training/trainer.py::_build_contrastive_pairs` | 保留代码不用 |
| `poc/training/dataset.py::VEATICDataset` | 不变 |
| `poc/training/evaluate.py` | 改：M1/M5 评估仍跑（用 L_recon-based metric 替换 retrieval）|

## 6. 评估指标改动

### 6.1 effective_dim 评估保留
仍用 SVD 谱衰减 metric（codex 第 9 轮指出对幅值不变，所以重构 loss 撑起来的是结构而非幅值）。

### 6.2 Shuffle Test 改用 L_predict-based
```
shuffled_x = shuffle_frames(x)
shuffled_h_seq = encode(shuffled_x, persona_idx)
L_predict_shuffled = ode_loss(shuffled_x, shuffled_h_seq, ...)
shuffle_drop = (L_predict_shuffled - L_predict) / L_predict  # 必 > 0.20（时序打乱后预测必崩）
```

### 6.3 M5 改 Recon-based
```
对 train video 和 unseen video 分别跑 L_recon
M5_recon = L_recon(unseen) - L_recon(train)  # 必 > 0.05（unseen 重构应更差）
```

## 7. codex 实施任务清单

1. 新建 `poc/lnn/ode_dynamics_loss.py`（Decoder + OdeDynamicsLossConfig + OdeDynamicsLoss）
2. 修改 `poc/training/trainer.py`：
   - `TrainConfig` 新增 `use_ode_loss: bool = True`、`ode_loss_cfg` 嵌入字段
   - `__init__` 实例化 decoder + ode_loss
   - `train_one_epoch` 切换主损失分支
   - 保留旧分支（feature flag），方便 A/B 对照
3. 修改 `poc/training/evaluate.py`：加 L_recon / L_predict / shuffle_drop 新 metrics
4. 写测试 6+3 个（章节 4）
5. 必跑 `pytest poc/ -q` 全绿
6. 跑 1 epoch dry-run 验 loss 在降

## 8. 不准踩（codex / gemini 都遵守）

- **不可妥协 §10.1 6 项**：必须全部保留
- **ADR-022 v3 τ-bias 释放**：DGB + tau_lr=1e-3 不动
- **ADR-016 base v3 网络结构**：CfC 内核不动，只加 Decoder
- **不要删旧代码**：所有 InfoNCE/VICReg 路径保留作为 feature flag 备选
- **不要 commit**：cc 主线 commit
- **不要 ssh Studio**：cc 主线推送
- **不要改 §10.1 / §13.1**：这是封版红线

## 9. 时间预算

- cc 写 ADR + spec：✅ done
- codex 实施 + 单测：~30 分钟
- gemini 评审 spec：~10 分钟（并行）
- 反向对抗：~10 分钟
- cc 整合 + dry-run：~15 分钟
- push + Studio 跑：~20 分钟
- **总计 ~1.5 小时**

---

**📝 维护者**：cc 主写 + codex 第 9 轮收敛 + gemini 第 9 轮收敛 | **状态**：v1（待 codex 实施 + gemini 评审）
