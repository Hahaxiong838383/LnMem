"""ADR-024 单测：measure_input_effective_dim 数学正确性 + 数据上限锁死."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from poc.training.dataset import VEATICDataset
from poc.training.evaluate import measure_input_effective_dim


def _create_mock_veatic(tmp_dir: Path, n_videos: int = 3, n_frames: int = 50,
                         au_rank: int = 5):
    rating_dir = tmp_dir / "ratings"
    rating_dir.mkdir(parents=True, exist_ok=True)
    au_cache_dict = {}
    rng = np.random.RandomState(42)
    for i in range(n_videos):
        U = rng.randn(n_frames, au_rank).astype(np.float32)
        V = rng.randn(au_rank, 24).astype(np.float32)
        au = U @ V + rng.randn(n_frames, 24).astype(np.float32) * 0.05
        au_cache_dict[f"au_{i}"] = au
        np.savetxt(rating_dir / f"{i}_valence.csv", rng.randn(n_frames) * 0.5)
        np.savetxt(rating_dir / f"{i}_arousal.csv", rng.randn(n_frames) * 0.5)
    npz_path = tmp_dir / "au_cache.npz"
    np.savez(npz_path, **au_cache_dict)
    return npz_path, rating_dir


def test_measure_input_effective_dim_returns_positive() -> None:
    """measure_input_effective_dim 返回合理正值."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp_path, n_videos=3, n_frames=80, au_rank=5)
        dataset = VEATICDataset(npz, ratings, seq_len=30)
        eff_dim = measure_input_effective_dim(dataset)
        assert eff_dim > 0, f"effective_dim 必须 > 0, got {eff_dim}"
        assert eff_dim < 24, f"effective_dim < AU 维度 24 = 数据 rank 上限, got {eff_dim}"


def test_low_rank_strictly_less_than_high_rank() -> None:
    """ADR-024 核心论据：相对关系 — low-rank AU < high-rank AU 的 effective_dim.

    注：mock data 是 i.i.d. random，跟真 AU 高度相关分布不同，所以测相对，不测绝对值。
    真实 AU (FACS) 实测 5.07（10 视频），mock random 可能高很多但相对关系成立。
    """
    with tempfile.TemporaryDirectory() as tmp_low, tempfile.TemporaryDirectory() as tmp_high:
        npz_low, ratings_low = _create_mock_veatic(Path(tmp_low), n_videos=3, n_frames=80, au_rank=3)
        npz_high, ratings_high = _create_mock_veatic(Path(tmp_high), n_videos=3, n_frames=80, au_rank=24)
        dataset_low = VEATICDataset(npz_low, ratings_low, seq_len=30)
        dataset_high = VEATICDataset(npz_high, ratings_high, seq_len=30)
        eff_low = measure_input_effective_dim(dataset_low)
        eff_high = measure_input_effective_dim(dataset_high)
        # 相对关系：rank=3 必须 < rank=24（mock 噪声也无法翻转）
        assert eff_low < eff_high, (
            f"low-rank effective_dim {eff_low} 应 < high-rank {eff_high}"
        )
