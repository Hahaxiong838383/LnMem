"""Cross-FPS hidden-state invariance loss."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def invariance_loss(
    h_high_fps: torch.Tensor,
    h_low_fps: torch.Tensor,
    fps_ratio: int,
) -> torch.Tensor:
    """Align stride-subsampled high-fps hidden states with low-fps hidden states."""
    if fps_ratio < 1:
        raise ValueError(f"fps_ratio must be >= 1, got {fps_ratio}")
    if h_high_fps.dim() != 3 or h_low_fps.dim() != 3:
        raise ValueError("h_high_fps and h_low_fps must be (B, T, H)")
    h_high_aligned = h_high_fps[:, ::fps_ratio, :]
    common_steps = min(h_high_aligned.shape[1], h_low_fps.shape[1])
    if common_steps < 1:
        raise ValueError("aligned sequence has zero overlap")
    return F.mse_loss(
        h_high_aligned[:, :common_steps, :],
        h_low_fps[:, :common_steps, :],
    )
