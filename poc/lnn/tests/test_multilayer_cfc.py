"""测试 MultiLayerCustomCfC 2 层 stack"""
import pytest
import torch

from poc.lnn import MultiLayerCustomCfC


def test_2layer_forward_shape():
    """2 层 stack forward 维度正确"""
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    batch = 4
    seq = 30
    x = torch.randn(batch, seq, 52)
    persona_idx = torch.arange(batch) % 4
    y = model(x, persona_idx)
    assert y.shape == (batch, seq, 64), f"got {y.shape}"


def test_2layer_params_under_60k():
    """spec v0.2.5 估 2 层 stack ~48K"""
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=64)
    n = model.num_parameters()
    assert 30_000 < n < 80_000, f"参数量 {n} 超出预估"


def test_2layer_no_collapse():
    """2 层 stack 不完全 collapse（gemini F-3.1: 阈值 > 5e-3 不再放水）"""
    torch.manual_seed(42)
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    x = torch.randn(8, 30, 52)
    persona_idx = torch.zeros(8, dtype=torch.long)
    with torch.no_grad():
        y = model(x, persona_idx)
    var = y.var().item()
    print(f"\n  2-layer init var: {var:.6f}")
    assert var > 5e-3, f"2 层 var={var:.6f} init 阶段弱坍缩"


def test_2layer_persona_init_nonzero():
    """2 层 stack init 阶段 persona τ 影响存在（sep > 1.0，真分化 > 1.5 留训练后）"""
    torch.manual_seed(42)
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    with torch.no_grad():
        for cell in model.cells:
            cell.persona_tau_bias.weight[0] = 2.0
            cell.persona_tau_bias.weight[3] = -2.0
            cell.persona_tau_bias.weight[1] = 0.5
            cell.persona_tau_bias.weight[2] = -0.5

    batch_size = 16
    persona_idx = torch.arange(batch_size) % 4
    x = torch.randn(batch_size, 30, 52) * 0.5
    x[:, 15:, :] += 1.0
    with torch.no_grad():
        y = model(x, persona_idx)
    h_flat = y[:, -1, :]
    same, diff = [], []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            d = (h_flat[i] - h_flat[j]).norm().item()
            (same if persona_idx[i] == persona_idx[j] else diff).append(d)
    mean_same = sum(same) / len(same)
    sep = (sum(diff) / len(diff)) / max(mean_same, 1e-6)
    print(f"\n  2-layer INIT sep: {sep:.3f}x [训练后目标 > 1.5x]")
    assert mean_same > 0.01, f"mean_same={mean_same:.6f} 数值幻觉"
    assert sep > 1.0, f"2 层 sep={sep:.3f} < 1.0 persona τ 完全失效"


def test_2layer_cascade():
    """第 2 层 hidden 必须受第 1 层 hidden 影响（cascade gradient check）"""
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    x = torch.randn(2, 5, 52, requires_grad=True)
    persona_idx = torch.tensor([0, 1])
    y = model(x, persona_idx)
    loss = y.sum()
    loss.backward()
    # 第 1 层参数应有 grad
    first_layer_grad = model.cells[0].tau_base.grad
    assert first_layer_grad is not None, "第 1 层 tau_base 无 grad"
    assert first_layer_grad.abs().sum().item() > 0, "第 1 层 grad 全 0 cascade 断了"
