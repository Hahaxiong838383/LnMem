"""PC-5b: truncated BPTT graph-cut tests."""
from __future__ import annotations

import torch

from poc.lnn import MultiLayerCustomCfC


def test_truncated_bptt_cuts_history_grad_flow() -> None:
    """With truncation, late-step loss should not backprop to early timesteps."""
    torch.manual_seed(42)
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=32, num_layers=2, num_personas=4)
    persona_idx = torch.zeros(1, dtype=torch.long)

    x_full = torch.randn(1, 10, 52, requires_grad=True)
    y_full = model(x_full, persona_idx, bptt_truncate=None)
    loss_full = y_full[:, -1, :].sum()
    grad_full = torch.autograd.grad(loss_full, x_full)[0]
    assert grad_full[:, :9, :].abs().sum().item() > 0.0

    x_trunc = torch.randn(1, 10, 52, requires_grad=True)
    y_trunc = model(x_trunc, persona_idx, bptt_truncate=4)
    loss_trunc = y_trunc[:, -1, :].sum()
    grad_trunc = torch.autograd.grad(loss_trunc, x_trunc)[0]
    # truncate=4 => cut points after t=3,7; last-step loss should not reach t<=7
    assert grad_trunc[:, :8, :].abs().sum().item() == 0.0
    assert grad_trunc[:, 8:, :].abs().sum().item() > 0.0


def test_truncated_bptt_keeps_forward_shape() -> None:
    """Truncation should not change output tensor shape contract."""
    model = MultiLayerCustomCfC(input_dim=52, hidden_dim=64, num_layers=2, num_personas=4)
    x = torch.randn(3, 12, 52)
    persona_idx = torch.zeros(3, dtype=torch.long)
    y = model(x, persona_idx, bptt_truncate=4)
    assert y.shape == (3, 12, 64)
