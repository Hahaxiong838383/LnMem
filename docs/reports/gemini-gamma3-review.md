<!-- gemini-3.5-flash high · γ-3 review · 55.4s -->

## γ-3 实现层评审报告

本轮评审针对 Phase γ-3 实现层代码（`losses.py`, `heads.py` 及其测试文件）进行深度审计。虽然代码结构清晰、注释详尽，但在**数学严谨性、测试真实性、架构防污染（Freeze 铁律）以及 5/22 实操可行性**上仍存在关键漏洞。

---

### 一、 核心审计意见

#### 1. InfoNCE 数学与温度坍塌风险 (SimCLR 偏差)
*   **数学定义偏差**：代码声称遵循 SimCLR (Chen et al. 2020)，但实现上是 **AMDIM/CPC 风格的显式负样本对比**（输入 `negatives` 维度为 `[B, N, D]`）。SimCLR 的核心在于**同 Batch 内多路对称对比**（无需显式负样本池）。当前设计高度依赖外部 Sampler 构造负样本，增加了内存开销与管道复杂度。
*   **可学习温度 $\tau$ 的坍塌风险**：代码引入了 `learnable_sigma`（通过 `softplus` 约束）。在无熵正则化或表征范数强约束下，网络极易通过**将 $\tau$ 缩至极小值（如接近 `1e-4` 截断值）**来人为放大 Logits，从而在不改变表征流的前提下“作弊式”降低 Cross Entropy 损失。这会导致梯度爆炸或表征退化。
*   **建议**：5/22 首跑必须默认关闭 `learnable_sigma`，使用固定的 $\tau = 0.1$。

#### 2. VICReg 25/25/1 超参及除零隐患
*   **超参对齐**：超参（`sim=25.0`, `std=25.0`, `cov=1.0`）严格对齐了 Bardes ICLR 2022 论文，数学公式还原度高。
*   **小 Batch 崩溃与除零漏洞**：
    *   代码中协方差分母为 `N - 1`（`cov_a = (z_a_c.T @ z_a_c) / (N - 1)`）。若 DataLoader 最后一个不完整 Batch 的大小 $N = 1$，将直接触发 **Division by Zero**，导致整轮训练 `NaN` 崩溃。
    *   VICReg 极度依赖大 Batch（建议 $\ge 64$）来估计稳定的标准差和协方差。若在 tiny VEATIC 上以小 Batch（如 $N < 8$）运行，协方差矩阵将严重秩亏，导致表征扭曲。
*   **建议**：分母改为 `max(N - 1, 1)`，并在初始化时增加 `batch_size` 安全断言。

#### 3. 测试用例中的“安慰剂”与边界漏洞 (Cherry-pick 审查)
*   **曝光 `test_infonce_perfect_positive` 中的数学伪装**：
    ```python
    n = torch.randn(batch, 16, d) * 0.01  # 乘以 0.01 试图制造“弱负样本”
    ```
    这是一个典型的**安慰剂代码**。因为在 `forward` 中，第一步就是 `n = F.normalize(negatives, dim=-1)`。L2 归一化会直接抹去 `0.01` 的缩放因子！该测试实际上是在与**标准单位圆上的随机向量**做对比，而非其声称的“极弱负样本”。这属于测试设计上的不严谨。
*   **`detect_semantic_breakpoints` 的 `NaN` 崩溃边界**：
    若输入序列长度 `seq_len <= 2`，则 `diffs` 的时间维度长度 $\le 1$。此时 `diffs.std(dim=1)` 在 PyTorch 中会返回 `NaN`。这将导致整个突变检测机制失效（`diffs > NaN` 恒为 `False`）。

#### 4. 5 Heads Freeze 铁律的“物理泄露” (Critical)
*   **致命梯度回流**：`heads.py` 提供了 `freeze_module(cell)` 接口，并在测试中验证了 `CustomCfCCell` 的参数 `requires_grad = False`。**但是，这无法阻止梯度污染表征！**
*   **数学原理**：在 PyTorch 中，即使 Backbone 参数被冻结，如果其前驱模块 `FusionLayer` 的参数仍为 `requires_grad = True`，当 Readout Heads 的 Loss 传回时，Autograd 仍会**穿透**冻结的 Backbone，计算并更新 `FusionLayer` 的投影矩阵！
*   **后果**：Readout Heads 的弱监督信号（Loss 4）通过更新 `FusionLayer` 间接污染了输入表征，彻底违反了“Backbone 必须 Freeze”的铁律。
*   **修复方案**：必须将 `FusionLayer` 与 Backbone 作为一个整体（Encoder）进行统一冻结。

#### 5. 5/22 跑真训练前缺失的接口
*   **时序截断未闭环**：`detect_semantic_breakpoints` 仅作为一个孤立函数存在，**完全没有接入 `InfoNCELoss`**。在 `losses.py` 中，`InfoNCELoss.forward` 依然只接受扁平化的 `anchor` 和 `positive`，根本无法感知时序突变点并断开 Positive Pair。
*   **多任务损失聚合器缺失**：缺乏一个能够处理**缺失标签（Masked Loss）**的统一 Loss Wrapper。VEATIC 和 EMOTIC 标签不全，直接计算 5 heads 联合损失会导致未标注任务的梯度污染。

---

### 二、 改进代码 (Refactored)

#### 1. 修复 `losses.py` 中的除零、时序截断闭环与温度保护

```python
# losses.py (局部重构)
class InfoNCELoss(nn.Module):
    # ... 保留原有初始化 ...
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
        mask: torch.Tensor | None = None,  # 新增：时序突变掩码 [B]
    ) -> torch.Tensor:
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negatives, dim=-1)

        tau = self.temperature
        pos_logit = (a * p).sum(dim=-1) / tau
        neg_logits = torch.einsum("bd,bnd->bn", a, n) / tau

        logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        if mask is not None:
            # mask 为 True 表示是语义突变点，应断开（忽略该样本的损失）
            loss = F.cross_entropy(logits, target, reduction='none')
            loss = loss * (~mask).float()
            return loss.mean()
        
        return F.cross_entropy(logits, target)

class VICRegLoss(nn.Module):
    # ... 保留原有初始化 ...
    def forward(self, z_a: torch.Tensor, z_p: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.cfg
        N, D = z_a.shape
        # 防御 1: 样本数不足无法计算协方差
        if N <= 1:
            return {
                "loss": torch.tensor(0.0, device=z_a.device, requires_grad=True),
                "sim_loss": torch.tensor(0.0, device=z_a.device),
                "std_loss": torch.tensor(0.0, device=z_a.device),
                "cov_loss": torch.tensor(0.0, device=z_a.device),
                "mean_std_a": torch.tensor(0.0, device=z_a.device),
                "mean_std_p": torch.tensor(0.0, device=z_a.device),
            }

        sim_loss = F.mse_loss(z_a, z_p)

        std_a = torch.sqrt(z_a.var(dim=0) + cfg.std_eps)
        std_p = torch.sqrt(z_p.var(dim=0) + cfg.std_eps)
        std_loss = torch.mean(F.relu(cfg.std_target - std_a)) + \
                   torch.mean(F.relu(cfg.std_target - std_p))

        z_a_c = z_a - z_a.mean(dim=0, keepdim=True)
        z_p_c = z_p - z_p.mean(dim=0, keepdim=True)
        
        # 防御 2: 修正分母，防止 N=1 时除零
        denom = max(N - 1, 1)
        cov_a = (z_a_c.T @ z_a_c) / denom
        cov_p = (z_p_c.T @ z_p_c) / denom
        
        cov_loss = (
            (cov_a - torch.diag(torch.diag(cov_a))).pow(2).sum() / D +
            (cov_p - torch.diag(torch.diag(cov_p))).pow(2).sum() / D
        )

        total = (cfg.sim_weight * sim_loss
                 + cfg.std_weight * std_loss
                 + cfg.cov_weight * cov_loss)

        return {
            "loss": total,
            "sim_loss": sim_loss.detach(),
            "std_loss": std_loss.detach(),
            "cov_loss": cov_loss.detach(),
            "mean_std_a": std_a.mean().detach(),
            "mean_std_p": std_p.mean().detach(),
        }
```

#### 2. 修复 `heads.py` 中的梯度泄露漏洞

```python
# heads.py (局部重构)
def freeze_representation_pipeline(fusion: nn.Module, backbone: nn.Module) -> None:
    """严格冻结表征生成流水线（Fusion + Backbone），防止 Loss 4 弱监督梯度反流污染"""
    for m in [fusion, backbone]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

def unfreeze_representation_pipeline(fusion: nn.Module, backbone: nn.Module) -> None:
    """联合微调阶段统一解冻"""
    for m in [fusion, backbone]:
        for p in m.parameters():
            p.requires_grad = True
        m.train()
```

---

## Summary
1. **InfoNCE 偏离 SimCLR**：采用显式负样本设计而非 Batch 内互对比，且可学习温度 $\tau$ 存在无约束坍塌风险。
2. **VICReg 存在除零隐患**：未防御 $N \le 1$ 的边界 Batch，在小 Batch 训练时协方差估计极易失真。
3. **测试用例存在“伪装”**：`test_infonce_perfect_positive` 中对负样本乘以 `0.01` 的操作在 L2 归一化后完全失效，属于无效测试。
4. **梯度防污染泄露**：仅冻结 Backbone 无法阻止梯度通过 Autograd 传导并更新未冻结的 `FusionLayer`。
5. **5/22 训练脱节**：时序突变检测函数未与 InfoNCE 损失函数闭环，缺乏多任务 Masked Loss 聚合器。

## Verdict
**REVISE**（必须按照上述改进方案重构，阻断梯度泄露并完成时序截断闭环后方可执行 5/22 真实训练）