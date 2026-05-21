"""Tests for VEATIC temporal contrast sampler behavior."""
from __future__ import annotations

import math

import torch

from poc.data.temporal_sampler import TemporalContrastSampler


def _build_index() -> dict[str, list[int]]:
    """Build deterministic frame index for three short videos and one long video."""
    return {
        "v_short_a": list(range(0, 1200, 10)),
        "v_short_b": list(range(0, 1500, 10)),
        "v_short_c": list(range(0, 1800, 10)),
        "v_long": list(range(0, 10000, 10)),
    }


def test_positive_within_window() -> None:
    """positive pair frame gap must stay within pos_window (spec §4.3)."""
    torch.manual_seed(7)
    sampler = TemporalContrastSampler(_build_index(), pos_window=60, neg_min_gap=900, videos_per_batch=4)
    batch = sampler.sample_batch(batch_size=64)

    gaps = (batch["anchor_frames"] - batch["positive_frames"]).abs()
    assert bool((gaps <= 60).all())


def test_negative_rule_respected() -> None:
    """negative must be cross-video or far-in-time in same video (spec §4.3)."""
    torch.manual_seed(13)
    sampler = TemporalContrastSampler(_build_index(), pos_window=60, neg_min_gap=900, videos_per_batch=4)
    batch = sampler.sample_batch(batch_size=64)

    anchors = batch["anchor_frames"]
    negatives = batch["negative_frames"]
    anchor_videos = batch["anchor_video_ids"]
    negative_videos = batch["negative_video_ids"]

    same_video = anchor_videos == negative_videos
    far_enough = (anchors - negatives).abs() > 900
    valid = (~same_video) | far_enough
    assert bool(valid.all())


def test_batch_mixes_minimum_video_count() -> None:
    """batch should contain at least videos_per_batch unique videos (anti-shortcut)."""
    torch.manual_seed(23)
    sampler = TemporalContrastSampler(_build_index(), pos_window=60, neg_min_gap=900, videos_per_batch=4)
    batch = sampler.sample_batch(batch_size=64)

    unique_videos = torch.unique(batch["anchor_video_ids"])
    assert unique_videos.numel() >= 4


def test_video_frequency_is_inverse_sqrt_weighted() -> None:
    """long video should be suppressed by 1/sqrt(frame_count) weighting."""
    torch.manual_seed(42)
    sampler = TemporalContrastSampler(_build_index(), pos_window=60, neg_min_gap=900, videos_per_batch=4)

    inv_map = {v: k for k, v in sampler.video_to_idx.items()}
    counts = {k: 0 for k in _build_index()}
    rounds = 400
    for _ in range(rounds):
        out = sampler.sample_batch(batch_size=64)
        for vid_idx in out["anchor_video_ids"].tolist():
            counts[inv_map[int(vid_idx)]] += 1

    # 理论比值：p(long)/p(short_a) = (1/sqrt(1000)) / (1/sqrt(120)) = sqrt(120/1000)
    expected_ratio = math.sqrt(120 / 1000)
    observed_ratio = counts["v_long"] / max(1, counts["v_short_a"])
    assert observed_ratio < 1.0
    assert abs(observed_ratio - expected_ratio) < 0.10
