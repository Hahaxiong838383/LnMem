"""Phase γ 自监督表征损失实现（spec §3.1 + §3.1.bis）

两条路径并存，γ-3 (5/25) 用 tiny VEATIC 1 epoch 双路 spike 后选主路径：
- Path A (InfoNCE Temporal Contrast)：spec §3.1 默认路径
- Path B (VICReg Variance-Invariance-Covariance)：spec §3.1.bis 备选（防 InfoNCE 低通滤波）

**学术诚实记号（gemini F-1.1）**：本模块所有数学公式遵循论文原版，避免简写
- InfoNCE: Chen et al. SimCLR (ICML 2020) + Bachman et al. AMDIM
- VICReg: Bardes et al. (ICLR 2022) https://arxiv.org/abs/2105.04906

**已知风险（gemini F-4.1，γ-3 训练时缓解）**：
- InfoNCE 在长 pos_window (±60 frames @ 30fps = ±2s) 内会强行拉近所有相邻帧 hidden
  → 微表情瞬态被低通滤波抹杀（gemini B.1 CRITICAL 担心）
- 缓解：γ-3 spike 时跑 VICReg 备选 + 5/26 决策日如果选纯 InfoNCE 必加「时序梯度截断」
  （||x_t - x_{t+δ}|| > μ+2σ 时断开 positive pair）— gemini 终审条件 1
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Path A: InfoNCE Temporal Contrast
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class InfoNCEConfig:
    """InfoNCE 温度 + 配对策略."""
    temperature: float = 0.1     # SimCLR 经典 τ=0.1
    # soft assignment σ 可学（spec LNN-FIRST 一性原理 + feedback_lnn_first_universal_principle）
    learnable_sigma: bool = True


class InfoNCELoss(nn.Module):
    """时序对比损失（Temporal Contrast，spec §3.1 Loss 1）

    数学：
        L = -log( exp(sim(z_a, z_p) / τ) / Σ_n exp(sim(z_a, z_n) / τ) )
        sim(u, v) = u·v / (||u|| ||v||)   余弦相似度

    输入约定（与 poc/data/temporal_sampler.TemporalContrastSampler 对接）：
        - anchor: (batch, hidden_dim)
        - positive: (batch, hidden_dim)  同视频 ±60 帧
        - negatives: (batch, num_negs, hidden_dim)  跨视频 或 同视频 >30s

    soft assignment σ（spec 一性原理）：
        如果 learnable_sigma=True，τ 改为 softplus(raw_sigma)，让训练自适应温度
    """

    def __init__(self, cfg: InfoNCEConfig | None = None):
        super().__init__()
        self.cfg = cfg or InfoNCEConfig()
        if self.cfg.learnable_sigma:
            # 初始化 raw_sigma 使 softplus(raw_sigma) ≈ cfg.temperature
            import math
            init = math.log(math.expm1(self.cfg.temperature))
            self.raw_sigma = nn.Parameter(torch.tensor(init, dtype=torch.float32))
        else:
            self.register_buffer("_fixed_tau", torch.tensor(self.cfg.temperature))

    @property
    def temperature(self) -> torch.Tensor:
        if self.cfg.learnable_sigma:
            return F.softplus(self.raw_sigma).clamp(min=1e-4)
        return self._fixed_tau

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        """Args:
            anchor: (batch, hidden_dim)
            positive: (batch, hidden_dim)
            negatives: (batch, num_negs, hidden_dim)

        Returns:
            scalar loss（batch 平均）
        """
        # L2 normalize 拿余弦相似度（防 magnitude shortcut，VICReg 论文也指出）
        a = F.normalize(anchor, dim=-1)
        p = F.normalize(positive, dim=-1)
        n = F.normalize(negatives, dim=-1)

        tau = self.temperature
        # 正样本 logit: (batch,)
        pos_logit = (a * p).sum(dim=-1) / tau
        # 负样本 logits: (batch, num_negs)
        neg_logits = torch.einsum("bd,bnd->bn", a, n) / tau

        # 拼起来 cross entropy
        # logits shape (batch, 1 + num_negs)，target 全 0（pos 在第一列）
        logits = torch.cat([pos_logit.unsqueeze(1), neg_logits], dim=1)
        target = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        return F.cross_entropy(logits, target)


# -----------------------------------------------------------------------------
# Path B: VICReg Variance-Invariance-Covariance
# -----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class VICRegConfig:
    """spec §3.1.bis + Bardes 论文超参（gemini F-A.1 担心超参敏感）."""
    sim_weight: float = 25.0      # invariance (anchor ≈ positive)
    std_weight: float = 25.0      # variance (per-dim std ≥ 1)
    cov_weight: float = 1.0       # covariance (off-diagonal ≈ 0)
    std_eps: float = 1e-4         # std 计算稳定项
    std_target: float = 1.0       # 目标 std


class VICRegLoss(nn.Module):
    """VICReg 三联损失（备选 Path B，spec §3.1.bis）

    数学（Bardes et al. ICLR 2022）：
        L = λ_v · invariance + λ_i · variance + λ_c · covariance

        invariance = MSE(z_a, z_p)
        variance = Σ_d max(0, 1 - std(z[:, d]))     防表征坍缩
        covariance = (1/d) · ||off_diag(Cov(Z))||²_F   去相关性

    优势 vs InfoNCE：
        - 不需要 negative samples（省 batch 内存）
        - variance 项天然防 collapse
        - 缺点：3 超参敏感（gemini F-A.1 担心），γ-3 spike 必跑双 seed

    输入：
        - z_a: (batch, hidden_dim)  anchor representations
        - z_p: (batch, hidden_dim)  positive representations (paired)
    """

    def __init__(self, cfg: VICRegConfig | None = None):
        super().__init__()
        self.cfg = cfg or VICRegConfig()

    def forward(self, z_a: torch.Tensor, z_p: torch.Tensor) -> dict[str, torch.Tensor]:
        """Returns: {loss, sim, std_a, std_p, cov_a, cov_p}（dict for monitoring）"""
        cfg = self.cfg
        N, D = z_a.shape

        # 1. Invariance: anchor ≈ positive 的 MSE
        sim_loss = F.mse_loss(z_a, z_p)

        # 2. Variance: 每个维度的 std 应 ≥ std_target
        std_a = torch.sqrt(z_a.var(dim=0) + cfg.std_eps)
        std_p = torch.sqrt(z_p.var(dim=0) + cfg.std_eps)
        std_loss = torch.mean(F.relu(cfg.std_target - std_a)) + \
                   torch.mean(F.relu(cfg.std_target - std_p))

        # 3. Covariance: 去 batch mean 后 off-diagonal F-norm
        z_a_c = z_a - z_a.mean(dim=0, keepdim=True)
        z_p_c = z_p - z_p.mean(dim=0, keepdim=True)
        cov_a = (z_a_c.T @ z_a_c) / (N - 1)
        cov_p = (z_p_c.T @ z_p_c) / (N - 1)
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


# -----------------------------------------------------------------------------
# Helper: 时序梯度截断（gemini 终审条件 1，γ-3 决策日若选 Path A 必加）
# -----------------------------------------------------------------------------
def detect_semantic_breakpoints(
    x_seq: torch.Tensor,
    sigma_threshold: float = 2.0,
) -> torch.Tensor:
    """spec v0.2.4 后续触发：纯 InfoNCE 必须配的「时序梯度截断」.

    计算相邻帧 AU 变化率 ||x_t - x_{t+1}||₂，超过 μ + sigma_threshold * σ 视为
    语义突变点（微表情/意图转折），返回布尔 mask 让 InfoNCE 跳过该 positive pair.

    Args:
        x_seq: (batch, seq_len, input_dim)
        sigma_threshold: 阈值倍数（默认 2σ）

    Returns:
        breakpoints: (batch, seq_len-1) bool，True 表示该位置是突变点（应断开 positive pair）
    """
    # 相邻帧距离
    diffs = (x_seq[:, 1:, :] - x_seq[:, :-1, :]).norm(dim=-1)  # (batch, seq_len-1)
    # 每个 batch 单独的均值/方差（避免跨 batch 污染）
    mean = diffs.mean(dim=1, keepdim=True)
    std = diffs.std(dim=1, keepdim=True) + 1e-6
    return diffs > (mean + sigma_threshold * std)
