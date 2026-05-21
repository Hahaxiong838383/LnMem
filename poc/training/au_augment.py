"""硬门 #3 替代：AU augmentation（替代 gemini 图像 Spatial Aug）.

LnMem schema 是 24 AU + 28 占位（非图像），所以 gemini 钦定的
"Random Crop + Color Jitter + Grid Mask" 不能直接用。

对应的 AU augmentation 设计（破坏 video_id shortcut）:
1. **per-frame channel dropout** (p=0.3): 类比 Random Crop，随机置 0 一部分 AU 通道
2. **AU intensity jitter**: 类比 Color Jitter，每帧 AU 值加小 noise (σ=0.05)
3. **temporal mixup**: 类比 Grid Mask，两个视频帧概率 α 线性混合
4. **frame-level non-coherent**: 每帧独立增强 (vs 全 sequence 同一变换)
   这是 gemini 钦定的「破坏空间一致性」核心

预期效果（gemini §B.3 推测）:
- 模型无法依赖「视频 0 总有强 AU06」这种 video-static 特征
- 被迫学习跨帧时序结构 → M5 acc < 0.15

实测：5/22 23:30 实施后实测 M5 从 0.833 → 0.??（待跑）
"""
from __future__ import annotations

import dataclasses
from typing import Optional

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class AUAugmentConfig:
    """AU augmentation 超参（spike 阶段保守）"""
    channel_dropout_p: float = 0.3        # 随机置 0 AU 通道概率
    intensity_jitter_std: float = 0.05    # AU 值高斯噪声 std
    temporal_mixup_alpha: float = 0.0     # 0 表示不开启 mixup（保守）
    frame_independent: bool = True        # 每帧独立增强（gemini 钦定 non-coherent）
    enabled: bool = True


def au_channel_dropout(
    et: torch.Tensor,
    p: float = 0.3,
    frame_independent: bool = True,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """随机置 0 一部分 AU 通道（类比 Random Crop）.

    Args:
        et: (B, T, 52) 或 (T, 52)
        p: dropout 概率
        frame_independent: True=每帧独立掩码 / False=整 sequence 同掩码（一致 crop）

    Returns:
        same shape, AU 部分 [0:24] 按概率置 0
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None

    out = et.clone()
    if et.dim() == 2:
        T = et.size(0)
        au_dim = 24  # spec §2.1 前 24d 是 AU + head pose
        if frame_independent:
            mask = torch.rand(T, au_dim, generator=gen) > p
        else:
            mask = (torch.rand(au_dim, generator=gen) > p).unsqueeze(0).expand(T, -1)
        out[:, :au_dim] = out[:, :au_dim] * mask.float()
    else:
        B, T, _ = et.shape
        au_dim = 24
        if frame_independent:
            mask = torch.rand(B, T, au_dim, generator=gen) > p
        else:
            mask = (torch.rand(B, au_dim, generator=gen) > p).unsqueeze(1).expand(B, T, -1)
        out[..., :au_dim] = out[..., :au_dim] * mask.float()
    return out


def au_intensity_jitter(
    et: torch.Tensor,
    std: float = 0.05,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """对 AU 强度加高斯噪声（类比 Color Jitter）.

    Returns: same shape, AU 部分 [0:24] 加 noise，clip 到 [0, 1]
    """
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    out = et.clone()
    if et.dim() == 2:
        T, _ = et.shape
        noise = torch.randn(T, 24, generator=gen) * std
        out[:, :24] = (out[:, :24] + noise).clamp(0, 1)
    else:
        B, T, _ = et.shape
        noise = torch.randn(B, T, 24, generator=gen) * std
        out[..., :24] = (out[..., :24] + noise).clamp(0, 1)
    return out


def au_temporal_mixup(et: torch.Tensor, alpha: float = 0.3, seed: Optional[int] = None) -> torch.Tensor:
    """Temporal mixup: 把序列里相邻帧线性混合（破坏 sharp 时序边界）.

    Args:
        et: (B, T, D) or (T, D)
        alpha: mixing 强度，alpha=0.3 → 当前帧 70% + 下一帧 30%
    """
    if alpha <= 0:
        return et
    if seed is not None:
        gen = torch.Generator().manual_seed(seed)
    else:
        gen = None
    out = et.clone()
    if et.dim() == 2:
        T = et.size(0)
        # 把每帧与随机另一帧混合
        perm = torch.randperm(T, generator=gen)
        lam = torch.rand(T, 1, generator=gen) * alpha
        out = (1 - lam) * et + lam * et[perm]
    else:
        B, T, D = et.shape
        perm = torch.randperm(T, generator=gen)
        lam = torch.rand(B, T, 1, generator=gen) * alpha
        out = (1 - lam) * et + lam * et[:, perm, :]
    return out


def au_augment(
    et: torch.Tensor,
    cfg: Optional[AUAugmentConfig] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    """组合 augmentation"""
    cfg = cfg or AUAugmentConfig()
    if not cfg.enabled:
        return et
    out = et
    if cfg.channel_dropout_p > 0:
        out = au_channel_dropout(out, cfg.channel_dropout_p,
                                  cfg.frame_independent, seed=seed)
    if cfg.intensity_jitter_std > 0:
        out = au_intensity_jitter(out, cfg.intensity_jitter_std,
                                   seed=(seed + 1) if seed else None)
    if cfg.temporal_mixup_alpha > 0:
        out = au_temporal_mixup(out, cfg.temporal_mixup_alpha,
                                seed=(seed + 2) if seed else None)
    return out
