"""测试 Phase γ self-supervised losses · spec §3.1 + §3.1.bis"""
import pytest
import torch

from poc.lnn.losses import (
    InfoNCELoss, InfoNCEConfig,
    VICRegLoss, VICRegConfig,
    detect_semantic_breakpoints,
)


# -----------------------------------------------------------------------------
# InfoNCE Tests
# -----------------------------------------------------------------------------
def test_infonce_forward_shape():
    """InfoNCE forward 维度正确"""
    loss = InfoNCELoss()
    batch, d, num_negs = 4, 64, 16
    a = torch.randn(batch, d)
    p = torch.randn(batch, d)
    n = torch.randn(batch, num_negs, d)
    l = loss(a, p, n)
    assert l.dim() == 0, f"loss should be scalar, got {l.shape}"
    assert l.item() > 0, "InfoNCE loss > 0"


def test_infonce_perfect_positive():
    """anchor==positive 时 loss 应远低于 random"""
    loss = InfoNCELoss(InfoNCEConfig(temperature=0.1, learnable_sigma=False))
    torch.manual_seed(42)
    batch, d = 4, 64
    a = torch.randn(batch, d)
    # 完美正样本：positive = anchor
    p_perfect = a.clone()
    # 完美 negatives：所有 negatives 与 anchor 几乎正交
    n = torch.randn(batch, 16, d) * 0.01

    loss_perfect = loss(a, p_perfect, n).item()

    # 对照：positive 是 random
    p_rand = torch.randn(batch, d)
    loss_rand = loss(a, p_rand, n).item()
    print(f"\n  loss_perfect={loss_perfect:.4f}, loss_rand={loss_rand:.4f}")
    assert loss_perfect < loss_rand, \
        f"完美 pos 应 loss 更低, perfect={loss_perfect:.4f} rand={loss_rand:.4f}"


def test_infonce_learnable_sigma_grad():
    """learnable_sigma=True 时 raw_sigma 必有 grad"""
    loss = InfoNCELoss(InfoNCEConfig(learnable_sigma=True))
    a = torch.randn(4, 64, requires_grad=True)
    p = torch.randn(4, 64, requires_grad=True)
    n = torch.randn(4, 8, 64, requires_grad=True)
    l = loss(a, p, n)
    l.backward()
    assert loss.raw_sigma.grad is not None
    assert loss.raw_sigma.grad.abs().item() > 0, "raw_sigma grad 为 0"


# -----------------------------------------------------------------------------
# VICReg Tests
# -----------------------------------------------------------------------------
def test_vicreg_forward_shape():
    """VICReg returns dict with 'loss' scalar"""
    loss = VICRegLoss()
    z_a = torch.randn(8, 64)
    z_p = torch.randn(8, 64)
    out = loss(z_a, z_p)
    assert "loss" in out
    assert out["loss"].dim() == 0
    assert "sim_loss" in out and "std_loss" in out and "cov_loss" in out


def test_vicreg_collapse_penalty():
    """如果所有 z 都相同（collapse）→ std_loss 应大 → total loss 大"""
    loss = VICRegLoss()
    # Collapse: 所有 sample 完全一样
    z_collapse = torch.ones(8, 64) * 0.5
    out_collapse = loss(z_collapse, z_collapse)

    # 正常: random
    torch.manual_seed(42)
    z_normal = torch.randn(8, 64)
    out_normal = loss(z_normal, z_normal)  # 完美 invariance 但 random，variance OK

    print(f"\n  collapse loss={out_collapse['loss'].item():.4f}, "
          f"normal loss={out_normal['loss'].item():.4f}")
    print(f"  std_loss: collapse={out_collapse['std_loss'].item():.4f}, "
          f"normal={out_normal['std_loss'].item():.4f}")
    assert out_collapse["std_loss"].item() > out_normal["std_loss"].item(), \
        "collapse 状态 std_loss 应更大"
    assert out_collapse["loss"].item() > out_normal["loss"].item(), \
        "collapse 状态 total loss 应更大"


def test_vicreg_invariance_works():
    """anchor == positive 时 sim_loss 应为 0"""
    loss = VICRegLoss()
    torch.manual_seed(7)
    z = torch.randn(16, 64)
    out = loss(z, z)
    assert out["sim_loss"].item() < 1e-6, \
        f"a==p 时 sim_loss 应 0, got {out['sim_loss'].item()}"


# -----------------------------------------------------------------------------
# Semantic Breakpoint Detection (gemini 终审条件 1)
# -----------------------------------------------------------------------------
def test_detect_semantic_breakpoints_smooth():
    """平滑输入应无突变点"""
    # 完全光滑的输入（线性渐变）
    x = torch.linspace(0, 1, 30).unsqueeze(0).unsqueeze(-1).expand(2, 30, 52)
    bp = detect_semantic_breakpoints(x, sigma_threshold=2.0)
    print(f"\n  smooth input breakpoints: {bp.sum().item()}/{bp.numel()}")
    assert bp.sum().item() == 0, f"smooth input should have 0 breakpoints"


def test_detect_semantic_breakpoints_step():
    """带 step 的输入应检测到突变点"""
    torch.manual_seed(42)
    # 大部分时间光滑 + 一个跳变
    x = torch.randn(1, 30, 52) * 0.05  # 极小噪声
    x[0, 15, :] += 10.0  # t=15 处剧烈跳变
    bp = detect_semantic_breakpoints(x, sigma_threshold=2.0)
    # 应该至少检测到 t=14→15 或 t=15→16 是突变
    print(f"\n  step input breakpoints: {bp.sum().item()}/{bp.numel()}")
    assert bp.sum().item() >= 1, "step 处应至少 1 个突变点"
