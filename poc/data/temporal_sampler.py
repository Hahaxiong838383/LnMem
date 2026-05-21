"""Temporal contrast batch sampler for VEATIC pretraining.

Implements spec §4.3 with anti-shortcut multi-video batching.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.nn import functional as F


class TemporalSamplerProtocol(Protocol):
    """Temporal contrast sampler protocol; 中文：统一采样接口契约。"""

    def sample_batch(self, batch_size: int = 64) -> dict[str, Tensor]:
        """Return one contrastive batch tensors."""


@dataclass(frozen=True)
class SamplerConfig:
    """Config for VEATIC temporal sampler; 中文：采样窗口与反 shortcut 参数。"""

    pos_window: int = 60
    neg_min_gap: int = 900
    videos_per_batch: int = 4


class TemporalContrastSampler:
    """Temporal contrast sampler with learnable soft assignment sigma."""

    def __init__(
        self,
        video_index: dict[str, list[int]],
        pos_window: int = 60,
        neg_min_gap: int = 900,
        videos_per_batch: int = 4,
    ) -> None:
        if not video_index:
            raise ValueError("video_index must not be empty")
        if pos_window <= 0 or neg_min_gap <= 0 or videos_per_batch <= 0:
            raise ValueError("pos_window/neg_min_gap/videos_per_batch must be positive")

        self.config = SamplerConfig(
            pos_window=pos_window,
            neg_min_gap=neg_min_gap,
            videos_per_batch=videos_per_batch,
        )

        self._video_ids = sorted(video_index.keys())
        self.video_to_idx = {vid: i for i, vid in enumerate(self._video_ids)}

        frames: dict[int, Tensor] = {}
        counts: list[float] = []
        for vid in self._video_ids:
            uniq_sorted = sorted(set(int(x) for x in video_index[vid]))
            if not uniq_sorted:
                raise ValueError(f"video {vid} has no frames")
            idx = self.video_to_idx[vid]
            frames[idx] = torch.tensor(uniq_sorted, dtype=torch.long)
            counts.append(float(len(uniq_sorted)))

        self._frames_by_idx = frames
        self._video_probs = self._build_video_probs(counts)

        # spec §4.3: soft assignment sigma 可学，保持正值以避免数值不稳定。
        self.raw_sigma = Parameter(torch.tensor(0.0, dtype=torch.float32))

    def sample_batch(self, batch_size: int = 64) -> dict[str, Tensor]:
        """Sample one batch for temporal contrast loss (spec §3.1 + §4.3)."""
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        selected = self._select_videos(batch_size)
        counts = self._allocate_counts(selected, batch_size)

        anchor_frames: list[Tensor] = []
        positive_frames: list[Tensor] = []
        negative_frames: list[Tensor] = []
        anchor_video_ids: list[Tensor] = []
        negative_video_ids: list[Tensor] = []

        for vid_idx, count in counts.items():
            for _ in range(count):
                anchor = self._sample_anchor(vid_idx)
                positive = self._sample_positive(vid_idx, anchor)
                neg_frame, neg_vid_idx = self._sample_negative(vid_idx, anchor)

                anchor_frames.append(anchor)
                positive_frames.append(positive)
                negative_frames.append(neg_frame)
                anchor_video_ids.append(torch.tensor(vid_idx, dtype=torch.long))
                negative_video_ids.append(torch.tensor(neg_vid_idx, dtype=torch.long))

        return {
            "anchor_frames": torch.stack(anchor_frames),
            "positive_frames": torch.stack(positive_frames),
            "negative_frames": torch.stack(negative_frames),
            "anchor_video_ids": torch.stack(anchor_video_ids),
            "negative_video_ids": torch.stack(negative_video_ids),
        }

    @property
    def sigma(self) -> Tensor:
        """Positive sigma used by soft assignment."""
        return F.softplus(self.raw_sigma) + 1e-6

    def _build_video_probs(self, counts: list[float]) -> Tensor:
        inv_sqrt = torch.tensor([1.0 / (c**0.5) for c in counts], dtype=torch.float32)
        return inv_sqrt / inv_sqrt.sum()

    def _select_videos(self, batch_size: int) -> Tensor:
        n_videos = len(self._video_ids)
        need = min(self.config.videos_per_batch, n_videos, batch_size)
        if need == n_videos:
            return torch.arange(n_videos, dtype=torch.long)
        return torch.multinomial(self._video_probs, num_samples=need, replacement=False)

    def _allocate_counts(self, selected: Tensor, batch_size: int) -> dict[int, int]:
        counts = {int(v.item()): 1 for v in selected}
        remain = batch_size - len(counts)
        if remain <= 0:
            return counts

        probs = self._video_probs[selected]
        probs = probs / probs.sum()
        extra = torch.multinomial(probs, num_samples=remain, replacement=True)
        for i in extra.tolist():
            vid_idx = int(selected[i].item())
            counts[vid_idx] += 1
        return counts

    def _sample_anchor(self, vid_idx: int) -> Tensor:
        frames = self._frames_by_idx[vid_idx]
        idx = torch.randint(0, frames.numel(), (1,), dtype=torch.long)
        return frames[idx].squeeze(0)

    def _sample_positive(self, vid_idx: int, anchor_frame: Tensor) -> Tensor:
        frames = self._frames_by_idx[vid_idx]
        diffs = (frames - anchor_frame).abs()

        # spec §4.3：同视频 ±pos_window 内作为正样本。
        mask = (diffs <= self.config.pos_window) & (diffs > 0)
        candidates = frames[mask]
        cand_diffs = diffs[mask].float()

        if candidates.numel() == 0:
            return anchor_frame.clone()

        # spec §4.3 + LNN-FIRST：用连续可导 soft assignment，避免硬阈值阶跃选择。
        sigma = self.sigma
        weights = torch.sigmoid((self.config.pos_window - cand_diffs) / sigma)
        weights = weights / (weights.sum() + 1e-8)
        pick = torch.multinomial(weights, num_samples=1, replacement=True)
        return candidates[pick].squeeze(0)

    def _sample_negative(self, anchor_vid_idx: int, anchor_frame: Tensor) -> tuple[Tensor, int]:
        if len(self._video_ids) > 1:
            cross_video = self._sample_cross_video_negative(anchor_vid_idx)
            if cross_video is not None:
                return cross_video

        frames = self._frames_by_idx[anchor_vid_idx]
        diffs = (frames - anchor_frame).abs()
        far_mask = diffs > self.config.neg_min_gap
        far_candidates = frames[far_mask]
        if far_candidates.numel() > 0:
            pick = torch.randint(0, far_candidates.numel(), (1,), dtype=torch.long)
            return far_candidates[pick].squeeze(0), anchor_vid_idx

        # 极端兜底：无跨视频且无 far negative，回退同视频最远帧。
        farthest = torch.argmax(diffs)
        return frames[farthest], anchor_vid_idx

    def _sample_cross_video_negative(self, anchor_vid_idx: int) -> tuple[Tensor, int] | None:
        other_ids = [i for i in range(len(self._video_ids)) if i != anchor_vid_idx]
        if not other_ids:
            return None
        other = torch.tensor(other_ids, dtype=torch.long)
        probs = self._video_probs[other]
        probs = probs / probs.sum()
        picked = other[torch.multinomial(probs, num_samples=1, replacement=True)].item()
        frames = self._frames_by_idx[int(picked)]
        fidx = torch.randint(0, frames.numel(), (1,), dtype=torch.long)
        return frames[fidx].squeeze(0), int(picked)
