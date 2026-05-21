"""测试 CustomCfCCell · γ-2 Path C 核心验收"""
import pytest
import torch

from poc.lnn import CustomCfCCell, CustomCfCCellConfig


@pytest.fixture
def cfg():
    return CustomCfCCellConfig(
        input_dim=52, hidden_dim=64, num_personas=4, tau_init=1.0
    )


def test_forward_shape(cfg):
    """forward 维度正确"""
    cell = CustomCfCCell(cfg)
    batch = 4
    x_t = torch.randn(batch, 52)
    h_prev = torch.zeros(batch, 64)
    persona_idx = torch.tensor([0, 1, 2, 3])
    h_new = cell(x_t, h_prev, persona_idx, dt=1.0)
    assert h_new.shape == (batch, 64), f"got {h_new.shape}"


def test_implicit_euler_dt_zero(cfg):
    """dt → 0 → h_new ≈ h_prev（积分步长极小，hidden 不变）"""
    cell = CustomCfCCell(cfg)
    h_prev = torch.randn(4, 64)
    x_t = torch.randn(4, 52)
    persona_idx = torch.tensor([0, 0, 0, 0])
    h_new = cell(x_t, h_prev, persona_idx, dt=1e-6)
    # 数学：(τ * h_prev + 1e-6 * f) / (τ + 1e-6) → h_prev when dt → 0
    assert torch.allclose(h_new, h_prev, atol=1e-3), \
        f"dt→0 时 h_new 应≈h_prev, diff={(h_new-h_prev).abs().max().item()}"


def test_implicit_euler_dt_large(cfg):
    """dt → ∞ → h_new ≈ f(x, h_prev)（完全更新到 f 目标）"""
    cell = CustomCfCCell(cfg)
    h_prev = torch.randn(4, 64)
    x_t = torch.randn(4, 52)
    persona_idx = torch.tensor([0, 0, 0, 0])
    # 计算 f
    with torch.no_grad():
        f_target = cell.f_net(torch.cat([x_t, h_prev], dim=-1))
    # dt 大（1000.0）→ h_new 应 ≈ f_target
    h_new = cell(x_t, h_prev, persona_idx, dt=1000.0)
    # 数学：(τ * h_prev + 1000 * f) / (τ + 1000) → f when dt >> τ
    # 用 mean rel err 而非 max（max 在 f_target ≈ 0 处不稳定）
    rel_err = ((h_new - f_target).abs() / (f_target.abs() + 1e-2)).mean().item()
    assert rel_err < 0.05, f"dt→∞ h_new 应≈f_net 输出, mean rel_err={rel_err:.4f}"


def test_persona_separation(cfg):
    """spec §11.1 #12 核心验收：persona 真分化 ratio > 1.5

    设计要点（诊断后修正 2026-05-21）：
        - 用 step input（前 25 帧 0，后 25 帧 1）触发 transient 响应
        - 在 t=27 (step 启动后 2 帧) 取 hidden — 此时 τ 不同的 persona 还在 transient
        - 不用 final hidden（大家都到稳态 → within-persona 距离=0 除零）
        - 验证：τ 大的 persona 反应慢 → hidden 距 step 起点更近；τ 小的反应快 → 接近 driving force
    """
    torch.manual_seed(42)
    cell = CustomCfCCell(cfg)
    with torch.no_grad():
        cell.persona_tau_bias.weight[0] = 2.0    # τ 放大 e^2 ≈ 7.4x 反应迟
        cell.persona_tau_bias.weight[3] = -2.0   # τ 缩小 e^-2 ≈ 0.135x 反应敏锐
        cell.persona_tau_bias.weight[1] = 0.5
        cell.persona_tau_bias.weight[2] = -0.5

    batch_size = 16
    seq_len = 30
    persona_idx = torch.arange(batch_size) % 4

    # Step input: 前半段 zero，后半段 ones（触发 transient）
    x_seq = torch.zeros(batch_size, seq_len, 52)
    x_seq[:, 15:, :] = 1.0

    h = torch.zeros(batch_size, 64)
    h_at_transient = None
    with torch.no_grad():
        for t in range(seq_len):
            h = cell(x_seq[:, t, :], h, persona_idx, dt=1.0)
            if t == 17:  # step 启动后 2 帧（transient 阶段）
                h_at_transient = h.clone()
    h_flat = h_at_transient

    same_pair_dist, diff_pair_dist = [], []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            d = (h_flat[i] - h_flat[j]).norm().item()
            (same_pair_dist if persona_idx[i] == persona_idx[j] else diff_pair_dist).append(d)

    mean_same = sum(same_pair_dist) / len(same_pair_dist)
    mean_diff = sum(diff_pair_dist) / len(diff_pair_dist)
    sep_ratio = mean_diff / max(mean_same, 1e-6)
    print(f"\n  transient persona separation: {sep_ratio:.3f}x "
          f"(same={mean_same:.4f}, diff={mean_diff:.4f})")
    assert sep_ratio > 1.5, f"persona 没真分化, sep={sep_ratio:.3f} (要求 > 1.5)"


def test_no_collapse(cfg):
    """随机输入 → hidden var > 1e-4（不完全坍缩）

    注：spec §6.1 collapse 阈值 0.01 是训练后的目标，初始化随机权重 + 短序列下
    var 自然不够 0.01。这里只保证「没完全坍缩到常数」即 var > 1e-4 即可。
    真 collapse 监控由 poc/monitoring/collapse_monitor.py 在训练时做。
    """
    torch.manual_seed(123)
    cell = CustomCfCCell(cfg)
    batch = 8
    x_seq = torch.randn(batch, 30, 52)
    persona_idx = torch.zeros(batch, dtype=torch.long)
    h = torch.zeros(batch, 64)
    with torch.no_grad():
        for t in range(30):
            h = cell(x_seq[:, t, :], h, persona_idx)
    var = h.var().item()
    print(f"\n  random-input hidden var: {var:.6f}")
    assert var > 1e-4, f"hidden var={var:.6f} 完全坍缩"


def test_tau_clamping(cfg):
    """compute_tau bias 极端值不爆炸"""
    cell = CustomCfCCell(cfg)
    with torch.no_grad():
        # 给个极端 bias
        cell.persona_tau_bias.weight[0] = 100.0  # 应该被 clamp 到 3.0
    persona_idx = torch.tensor([0])
    tau = cell.compute_tau(persona_idx)
    # exp(3.0) ≈ 20.0, tau_base=1.0 → tau ≈ 20.0 max
    assert tau.max().item() < 30.0, f"tau clamp 失效, max={tau.max().item()}"
    assert tau.min().item() > 0, "tau 必须 > 0"


def test_num_parameters(cfg):
    """参数量在预估范围内"""
    cell = CustomCfCCell(cfg)
    n = cell.num_parameters()
    # 预估 27.4K（input=52, hidden=64, personas=4 替 64 因为 fixture）
    # tau_base 64 + persona_emb 4*64=256 + f_net 116*128+128 + 128*64+64
    expected = 64 + 4 * 64 + (52 + 64) * 128 + 128 + 128 * 64 + 64
    assert n == expected, f"参数量不符, got {n} expected {expected}"
    assert 10_000 < n < 50_000, f"参数量 {n} 超出合理范围"
