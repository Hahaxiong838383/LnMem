"""MultiLayerCustomCfC · 2 层 stack（spec §2.2 v0.2 改 2 层先跑通）

每 timestep：h1_t = cell1(x_t, h1_{t-1}, persona) → h2_t = cell2(h1_t, h2_{t-1}, persona)

参数预估（2 层 input=52→64→64）：
    Layer 1 (CustomCfCCell 52→64): ~27.4K
    Layer 2 (CustomCfCCell 64→64): ~21.0K
    Total: ~48.4K（与 ncps 2 层 stack 97K 相比 -50%，因为 f_net 单层 vs ncps 多层 backbone）
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .custom_cfc_cell import CustomCfCCell, CustomCfCCellConfig


class MultiLayerCustomCfC(nn.Module):
    """N-layer CustomCfCCell stack（默认 2 层）.

    spec §2.2 注：N=2 是 spec v0.2 改的「先跑通」配置；γ-2 实测后若 N=2 不够才升 N=4。

    Forward API:
        y = model(x, persona_idx, dt=1.0)
        # x: (batch, seq_len, input_dim)
        # persona_idx: (batch,)
        # returns: (batch, seq_len, hidden_dim) 最后一层每帧 hidden
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        num_personas: int = 64,
        tau_init: float = 1.0,
    ):
        super().__init__()
        assert num_layers >= 1, "num_layers must be >= 1"
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

        cells = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            cells.append(CustomCfCCell(CustomCfCCellConfig(
                input_dim=in_dim,
                hidden_dim=hidden_dim,
                num_personas=num_personas,
                tau_init=tau_init,
            )))
        self.cells = nn.ModuleList(cells)

    def forward(
        self,
        x: torch.Tensor,
        persona_idx: torch.Tensor,
        dt: float = 1.0,
        return_all_layers: bool = False,
    ) -> torch.Tensor:
        """Sequence forward.

        Args:
            x: (batch, seq_len, input_dim)
            persona_idx: (batch,)
            dt: scalar 时间步
            return_all_layers: True 返回所有层最后帧 hidden，False 只返回最后层全 seq

        Returns:
            if return_all_layers=False: (batch, seq_len, hidden_dim)
            else: list of (batch, hidden_dim) per layer
        """
        batch, seq_len, _ = x.shape

        # 各层初始 hidden
        h_states = [cell.init_hidden(batch, device=x.device) for cell in self.cells]

        # 收集最后一层每帧 hidden
        last_layer_seq = []

        for t in range(seq_len):
            x_t = x[:, t, :]   # (batch, input_dim)
            for layer_idx, cell in enumerate(self.cells):
                input_for_layer = x_t if layer_idx == 0 else h_states[layer_idx - 1]
                h_states[layer_idx] = cell(
                    x_t=input_for_layer,
                    h_prev=h_states[layer_idx],
                    persona_idx=persona_idx,
                    dt=dt,
                )
            last_layer_seq.append(h_states[-1])

        out = torch.stack(last_layer_seq, dim=1)  # (batch, seq_len, hidden_dim)
        if return_all_layers:
            return out, h_states
        return out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
