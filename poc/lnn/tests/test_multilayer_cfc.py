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
    """2 层 stack 不完全 collapse（init 阶段 var > 1e-4）"""
    torch.manual_seed(42)
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    x = torch.randn(8, 30, 52)
    persona_idx = torch.zeros(8, dtype=torch.long)
    with torch.no_grad():
        y = model(x, persona_idx)
    var = y.var().item()
    print(f"\n  2-layer init var: {var:.6f}")
    assert var > 1e-4, f"2 层 var={var:.6f} 完全坍缩"


def test_2layer_persona_separation():
    """2 层 stack 也必须 persona separation > 1.5（spec §11.1 #12）

    用 step input 触发 transient（同 1 层测试设计）
    """
    torch.manual_seed(42)
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    with torch.no_grad():
        for cell in model.cells:
            cell.persona_tau_bias.weight[0] = 2.0
            cell.persona_tau_bias.weight[3] = -2.0
            cell.persona_tau_bias.weight[1] = 0.5
            cell.persona_tau_bias.weight[2] = -0.5

    batch_size = 16
    seq_len = 30
    persona_idx = torch.arange(batch_size) % 4
    # Step input 触发 transient
    x = torch.zeros(batch_size, seq_len, 52)
    x[:, 15:, :] = 1.0
    with torch.no_grad():
        y = model(x, persona_idx)
    # 取 step 后 2-3 帧（transient 阶段，t=17）
    h_flat = y[:, 17, :]

    same, diff = [], []
    for i in range(batch_size):
        for j in range(i + 1, batch_size):
            d = (h_flat[i] - h_flat[j]).norm().item()
            (same if persona_idx[i] == persona_idx[j] else diff).append(d)
    sep = (sum(diff) / len(diff)) / max(sum(same) / len(same), 1e-6)
    print(f"\n  2-layer transient persona sep: {sep:.3f}x")
    assert sep > 1.5, f"2 层 stack persona sep={sep:.3f} < 1.5"


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
