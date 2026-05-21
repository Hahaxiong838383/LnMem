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
    """anchor==positive 时 loss 应远低于 random（gemini F-3 修正：不依赖 negative magnitude）

    gemini F-3：原版 `n = torch.randn() * 0.01` 试图制造「弱负样本」，但 InfoNCE forward
    第一步就 F.normalize(n)，0.01 缩放被抹除，测试退化为「随机单位圆向量」。
    修正：构造**与 anchor 正交方向**的 negatives（在单位球面上远离 anchor），才是真「弱负样本」。
    """
    loss = InfoNCELoss(InfoNCEConfig(temperature=0.1, learnable_sigma=False))
    torch.manual_seed(42)
    batch, d = 4, 64
    a = torch.randn(batch, d)
    a_norm = a / a.norm(dim=-1, keepdim=True)
    # 完美 pos: 同 anchor
    p_perfect = a.clone()
    # 真「弱负样本」: 与 anchor 正交方向（构造 a_perp ⊥ a）
    n_raw = torch.randn(batch, 16, d)
    proj = (n_raw * a_norm.unsqueeze(1)).sum(dim=-1, keepdim=True) * a_norm.unsqueeze(1)
    n_orth = n_raw - proj  # 与 a 正交分量，normalize 后仍正交

    loss_perfect = loss(a, p_perfect, n_orth).item()

    # 对照：随机正样本（与 anchor 无关）
    p_rand = torch.randn(batch, d)
    loss_rand = loss(a, p_rand, n_orth).item()
    print(f"\n  loss_perfect={loss_perfect:.4f}, loss_rand={loss_rand:.4f}")
    assert loss_perfect < loss_rand, \
        f"完美 pos 应 loss 更低, perfect={loss_perfect:.4f} rand={loss_rand:.4f}"
    # 完美 pos 与正交 neg：loss 应接近 0
    assert loss_perfect < 0.01, f"完美 pos + 正交 neg, loss 应近 0，得 {loss_perfect:.4f}"


def test_infonce_breakpoint_mask_closure():
    """gemini F-5 闭环测试：突变 mask 真的让 InfoNCELoss 跳过样本"""
    loss = InfoNCELoss(InfoNCEConfig(learnable_sigma=False))
    batch, d = 8, 64
    torch.manual_seed(42)
    a = torch.randn(batch, d)
    p = torch.randn(batch, d)  # 随机 → 每样本 loss 都不小
    n = torch.randn(batch, 16, d)

    # 全量 loss
    full_loss = loss(a, p, n, breakpoint_mask=None).item()

    # 前 4 个样本 mask 掉
    mask = torch.tensor([True, True, True, True, False, False, False, False])
    masked_loss = loss(a, p, n, breakpoint_mask=mask).item()

    print(f"\n  full={full_loss:.4f}, masked(half)={masked_loss:.4f}")
    # mask 后样本数减半，loss 仅算后 4 个样本，可能大于或小于 full 取决于具体样本
    # 关键：mask all True 时 loss 应为 0
    mask_all = torch.ones(batch, dtype=torch.bool)
    loss_all_masked = loss(a, p, n, breakpoint_mask=mask_all).item()
    print(f"  all-masked={loss_all_masked:.4f}")
    assert loss_all_masked < 0.01, f"全 mask 时 loss 应为 0, 得 {loss_all_masked}"


def test_infonce_learnable_sigma_grad():
    """learnable_sigma=True 显式开启时 raw_sigma 必有 grad（默认已关 by gemini F-1）"""
    loss = InfoNCELoss(InfoNCEConfig(learnable_sigma=True))
    a = torch.randn(4, 64, requires_grad=True)
    p = torch.randn(4, 64, requires_grad=True)
    n = torch.randn(4, 8, 64, requires_grad=True)
    l = loss(a, p, n)
    l.backward()
    assert loss.raw_sigma.grad is not None
    assert loss.raw_sigma.grad.abs().item() > 0, "raw_sigma grad 为 0"


def test_infonce_default_no_learnable_sigma():
    """gemini F-1 修正：默认 learnable_sigma=False 防温度坍塌"""
    loss = InfoNCELoss()  # default config
    assert not loss.cfg.learnable_sigma, "默认应关闭可学温度"
    assert not hasattr(loss, "raw_sigma"), "默认不应注册 raw_sigma 参数"


def test_vicreg_n1_no_divide_by_zero():
    """gemini F-2: N=1 不应崩溃"""
    loss = VICRegLoss()
    z = torch.randn(1, 64)
    out = loss(z, z)  # N=1 边界
    assert "warning" in out or out["loss"].item() == 0.0, "N=1 应被 graceful 处理"


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
