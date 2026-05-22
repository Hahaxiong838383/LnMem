"""单测：罕见 AU 重建监控（ROADMAP P0-2 + gemini round 14 N=124 长尾警告）.

三道门 D2：先写测试定义验收标准，TDD 红→绿.

测试覆盖：
- compute_au_activation_frequency 数学正确性 + AU 段约束
- partition_au_by_frequency 分桶不重叠 + 频率单调 + 边界报错
- compute_per_bucket_mse 数值正确（手算对照）
- evaluate_rare_au_drop green / yellow / red 三档触发
- 端到端：mock dataset → freq → bucket → mse → verdict 跑得通
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from poc.training.daemon_dataset import DaemonAUDataset
from poc.training.dataset import VEATICDataset
from poc.training.rare_au_monitor import (
    AU_ACTIVATION_THRESHOLD,
    AU_DIM_END,
    AU_DIM_START,
    DEFAULT_FIXED_BOTTOM_INDICES,
    RATIO_DENOM_SMOOTHING,
    SPREAD_GATE_RATIO_BOTTOM,
    AuBuckets,
    compute_au_activation_frequency,
    compute_per_bucket_mse,
    evaluate_rare_au_drop,
    partition_au_by_frequency,
    verdict_to_dict,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------
def _create_mock_veatic_with_skewed_au(
    tmp_dir: Path,
    n_videos: int = 3,
    n_frames: int = 80,
    high_freq_cols: tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6),
    low_freq_cols: tuple[int, ...] = (13, 14, 15, 16, 17, 18, 19),
    seed: int = 42,
) -> tuple[Path, Path]:
    """构造 mock VEATIC：指定列高频激活，指定列罕见激活.

    高频列：值 ∈ [0.7, 1.0]（激活率 = 1.0）
    罕见列：99% frame 接近 0，1% frame ∈ [0.7, 1.0]
    其他列：值 ∈ [0.3, 0.6]（中等激活率）
    """
    rating_dir = tmp_dir / "ratings"
    rating_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed)
    au_cache = {}
    for vi in range(n_videos):
        au = rng.uniform(0.3, 0.6, size=(n_frames, 20)).astype(np.float32)
        for c in high_freq_cols:
            au[:, c] = rng.uniform(0.7, 1.0, size=n_frames).astype(np.float32)
        for c in low_freq_cols:
            mask = rng.uniform(0, 1, size=n_frames) > 0.99
            au[:, c] = 0.0
            au[mask, c] = rng.uniform(0.7, 1.0, size=mask.sum()).astype(np.float32)
        au_cache[f"au_{vi}"] = au
        np.savetxt(rating_dir / f"{vi}_valence.csv", rng.randn(n_frames) * 0.3)
        np.savetxt(rating_dir / f"{vi}_arousal.csv", rng.randn(n_frames) * 0.3)
    npz = tmp_dir / "au.npz"
    np.savez(npz, **au_cache)
    return npz, rating_dir


# -----------------------------------------------------------------------------
# compute_au_activation_frequency
# -----------------------------------------------------------------------------
def test_activation_frequency_shape_and_range():
    """频率向量形状 = AU 段长度，值 ∈ [0, 1]."""
    with tempfile.TemporaryDirectory() as tmp:
        npz, ratings = _create_mock_veatic_with_skewed_au(Path(tmp))
        ds = VEATICDataset(npz, ratings, seq_len=30)
        freq = compute_au_activation_frequency(ds)
    assert freq.shape == (AU_DIM_END - AU_DIM_START,)
    assert (freq >= 0).all() and (freq <= 1).all()


def test_activation_frequency_high_vs_low_columns():
    """高频列频率 ≈ 1，罕见列频率 ≈ 0（容忍 mock 噪声）."""
    high_cols = (0, 1, 2)
    low_cols = (17, 18, 19)
    with tempfile.TemporaryDirectory() as tmp:
        npz, ratings = _create_mock_veatic_with_skewed_au(
            Path(tmp), high_freq_cols=high_cols, low_freq_cols=low_cols
        )
        ds = VEATICDataset(npz, ratings, seq_len=30)
        freq = compute_au_activation_frequency(ds, threshold=0.5)
    for c in high_cols:
        assert freq[c] > 0.9, f"高频列 {c} 激活率 {freq[c]} 应 > 0.9"
    for c in low_cols:
        assert freq[c] < 0.1, f"罕见列 {c} 激活率 {freq[c]} 应 < 0.1"


# -----------------------------------------------------------------------------
# partition_au_by_frequency
# -----------------------------------------------------------------------------
def test_partition_buckets_disjoint_and_cover():
    """三桶不重叠 + 总长 = 期望（纯频率分桶，关闭 fixed_bottom）."""
    freq = np.linspace(0.05, 1.0, 20)  # 单调递增
    buckets = partition_au_by_frequency(freq, top_k=7, bottom_k=7, fixed_bottom_indices=())
    s_top, s_mid, s_bot = set(buckets.top), set(buckets.median), set(buckets.bottom)
    assert s_top.isdisjoint(s_mid)
    assert s_top.isdisjoint(s_bot)
    assert s_mid.isdisjoint(s_bot)
    assert len(s_top) + len(s_mid) + len(s_bot) == 20
    assert len(s_top) == 7
    assert len(s_mid) == 6
    assert len(s_bot) == 7


def test_partition_top_freq_greater_than_bottom():
    """top bucket 平均频率 > bottom bucket（纯频率分桶语义）."""
    rng = np.random.RandomState(123)
    freq = rng.uniform(0, 1, size=20)
    buckets = partition_au_by_frequency(freq, fixed_bottom_indices=())
    top_mean = np.mean([freq[i] for i in buckets.top])
    bot_mean = np.mean([freq[i] for i in buckets.bottom])
    assert top_mean > bot_mean, f"top 均频 {top_mean} 应 > bottom 均频 {bot_mean}"


def test_partition_raises_when_k_exceeds_dim():
    """top_k + bottom_k > au_dim 必须报错."""
    freq = np.linspace(0, 1, 10)
    with pytest.raises(ValueError, match="au_dim"):
        partition_au_by_frequency(freq, top_k=6, bottom_k=6, fixed_bottom_indices=())


def test_partition_fixed_bottom_enforced_even_when_high_freq():
    """V0.3 D'：若 fixed 已在 top-K，应被 evict 并由低频 AU 动态补位."""
    # 让索引 18, 19 频率人为最高，并显式作为 fixed 输入
    freq = np.linspace(0.05, 1.0, 20)  # 18, 19 是频率最高的两个
    buckets = partition_au_by_frequency(freq, top_k=7, bottom_k=7, fixed_bottom_indices=(18, 19))
    assert buckets.evicted_fixed == (18, 19)
    assert 18 not in buckets.bottom
    assert 19 not in buckets.bottom
    assert 18 in buckets.top
    assert 19 in buckets.top
    assert buckets.bottom == tuple(range(0, 7))
    # 三桶仍然不重叠
    assert set(buckets.top).isdisjoint(buckets.bottom)
    assert set(buckets.median).isdisjoint(buckets.bottom)
    assert len(set(buckets.top) | set(buckets.median) | set(buckets.bottom)) == 20


def test_partition_evicts_fixed_when_in_top_k():
    """当 fixed 进入 top-K 时必须 evict，bottom 按纯频率低位补齐."""
    freq = np.array([0.9, 0.85, 0.6, 0.5, 0.2, 0.1], dtype=np.float64)
    buckets = partition_au_by_frequency(freq, top_k=2, bottom_k=3, fixed_bottom_indices=(0, 1))
    assert buckets.evicted_fixed == (0, 1)
    assert buckets.top == (0, 1)
    assert buckets.bottom == (3, 4, 5)


def test_partition_keeps_fixed_when_truly_in_bottom():
    """当 fixed 本身是低频时保留在 bottom，evicted_fixed 为空."""
    freq = np.linspace(1.0, 0.2, 20)
    freq[18] = 0.01
    freq[19] = 0.0
    buckets = partition_au_by_frequency(freq, top_k=7, bottom_k=7, fixed_bottom_indices=(18, 19))
    assert buckets.evicted_fixed == ()
    assert 18 in buckets.bottom
    assert 19 in buckets.bottom


def test_partition_raises_when_fixed_bottom_exceeds_bottom_k():
    """fixed_bottom_indices 数量超过 bottom_k 必须报错."""
    freq = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="fixed_bottom_indices"):
        partition_au_by_frequency(freq, top_k=7, bottom_k=2, fixed_bottom_indices=(0, 1, 2, 3))


def test_partition_raises_when_fixed_bottom_out_of_range():
    """fixed_bottom_indices 越界必须报错."""
    freq = np.linspace(0, 1, 20)
    with pytest.raises(ValueError, match="越界"):
        partition_au_by_frequency(freq, fixed_bottom_indices=(99,))


# -----------------------------------------------------------------------------
# compute_per_bucket_mse
# -----------------------------------------------------------------------------
def test_per_bucket_mse_isolates_columns():
    """构造 pred/target，让 top bucket 完美预测，bottom bucket 全错；验证 bottom MSE > top MSE."""
    # 一个简化场景：D=52 维全 0 target，pred 在 top 桶完美 0，在 bottom 桶 1.0
    B, T, D = 1, 5, 52
    target = torch.zeros(B, T, D)
    pred = torch.zeros(B, T, D)
    buckets = AuBuckets(
        top=(0, 1, 2),
        median=(8, 9, 10),
        bottom=(17, 18, 19),
        activation_freq=tuple(np.linspace(0, 1, 20).tolist()),
    )
    # 让 bottom 列预测错（pred=1, target=0 → MSE=1）
    for c in buckets.bottom:
        pred[:, :, c] = 1.0
    out = compute_per_bucket_mse(pred, target, buckets, au_start=AU_DIM_START)
    assert out["top"] == pytest.approx(0.0)
    assert out["median"] == pytest.approx(0.0)
    assert out["bottom"] == pytest.approx(1.0)


def test_per_bucket_mse_shape_mismatch_raises():
    pred = torch.zeros(1, 3, 52)
    target = torch.zeros(1, 4, 52)
    buckets = AuBuckets(top=(0,), median=(5,), bottom=(10,), activation_freq=(0.0,) * 20)
    with pytest.raises(ValueError, match="x_pred"):
        compute_per_bucket_mse(pred, target, buckets)


# -----------------------------------------------------------------------------
# evaluate_rare_au_drop
# -----------------------------------------------------------------------------
def _mk_buckets() -> AuBuckets:
    return AuBuckets(
        top=(0, 1, 2, 3, 4, 5, 6),
        median=(7, 8, 9, 10, 11, 12),
        bottom=(13, 14, 15, 16, 17, 18),
        activation_freq=tuple(np.linspace(0, 1, 20).tolist()),
        evicted_fixed=(),
    )


def test_verdict_green_when_spread_small_and_gate_active():
    """ratio_bottom 在 gate 之上 + spread 小 → green."""
    # ratio_bottom=0.6 触发 gate (>=0.5)，spread=0.1 < 0.3 → green
    world = {"top": 0.5, "median": 0.55, "bottom": 0.6}
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "green", v.reason
    # smoothing 影响：ratio = 0.5 / (1.0 + 1e-3) ≈ 0.4995
    assert v.ratio_top == pytest.approx(0.5, rel=2e-3)
    assert v.ratio_bottom == pytest.approx(0.6, rel=2e-3)


def test_verdict_green_when_bottom_already_strong_skips_spread_check():
    """gemini D3 Finding 2：ratio_bottom < gate (0.5) 时跳过 spread 检查 → green."""
    # 即使 spread 巨大，只要 ratio_bottom 已经强（< 0.5）也不警告
    world = {"top": 0.05, "median": 0.2, "bottom": 0.4}  # spread = 0.35 > yellow
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "green", v.reason
    assert "gate" in v.reason or "强于" in v.reason


def test_verdict_yellow_when_spread_moderate_and_gate_active():
    """ratio_bottom >= gate + spread 落在 [yellow, red) → yellow."""
    # ratio_top=0.55, ratio_bottom=0.9, spread=0.35 → yellow
    world = {"top": 0.55, "median": 0.7, "bottom": 0.9}
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "yellow", v.reason
    assert 0.3 <= v.ratio_spread < 0.5


def test_verdict_red_when_bottom_ratio_above_1():
    """bottom ratio ≥ 1 → red (罕见 AU 比 trivial 还差)."""
    world = {"top": 0.2, "median": 0.4, "bottom": 1.05}
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "red", v.reason
    assert "bottom ratio" in v.reason


def test_verdict_red_when_spread_huge_and_gate_active():
    """ratio_bottom >= gate + spread ≥ spread_red → red."""
    # ratio_top=0.35, ratio_bottom=0.95, spread=0.6 → red
    world = {"top": 0.35, "median": 0.7, "bottom": 0.95}
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "red", v.reason
    assert "spread" in v.reason


def test_smoothed_ratio_no_explosion_when_pers_zero():
    """gemini D3 Finding 1：恒 0 罕见 AU 列分母塌不再触发数值爆表."""
    # pers_bottom = 0 (罕见 AU 整段恒 0)，world_bottom 也极小但非 0
    world = {"top": 0.2, "median": 0.3, "bottom": 1e-5}
    pers = {"top": 0.5, "median": 0.5, "bottom": 0.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    # 旧版会算成 1e-5 / 1e-9 = 1e4 触发 red，新版用 +1e-3 平滑 ≈ 0.01
    assert v.ratio_bottom < 0.1, f"分母平滑后 ratio_bottom 应 < 0.1, got {v.ratio_bottom}"
    assert v.verdict == "green", v.reason


def test_verdict_to_dict_keys():
    """verdict_to_dict 输出 key 完整（merge 进 evaluate 时不漏字段）."""
    world = {"top": 0.5, "median": 0.55, "bottom": 0.6}
    pers = {"top": 1.0, "median": 1.0, "bottom": 1.0}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    d = verdict_to_dict(v)
    expected_keys = {
        "L_world_top", "L_world_median", "L_world_bottom",
        "ratio_top", "ratio_median", "ratio_bottom",
        "ratio_spread", "rare_au_verdict", "rare_au_reason",
        "rare_au_buckets", "rare_au_activation_freq", "rare_au_evicted_fixed", "rare_au_n_samples",
    }
    assert expected_keys.issubset(d.keys())
    assert d["rare_au_verdict"] == "green"


def test_constants_match_v03d_v04_decisions():
    """V0.3 D' + V0.4 关键常量值（round 2 改良 fixed AU07/AU10 + cc Y2 整体崩盘门）."""
    from poc.training.rare_au_monitor import TOP_COLLAPSE_RATIO
    assert AU_ACTIVATION_THRESHOLD == 0.1
    assert RATIO_DENOM_SMOOTHING == 1e-3
    assert SPREAD_GATE_RATIO_BOTTOM == 0.5
    assert DEFAULT_FIXED_BOTTOM_INDICES == (5, 7)
    assert TOP_COLLAPSE_RATIO == 1.0


def test_verdict_na_when_top_ratio_collapses():
    """V0.4 cc Y2 finding：top ratio > 1 → n/a（整体崩盘，长尾监控不适用）.

    实测场景：v3.0 200ep 在川哥 daemon 1Hz 数据上跑 inference，因训练分布(VEATIC 30fps)
    与评估分布严重偏移，ratio_top=2.17 (model > trivial)，spread=-1.44 (top 比 bottom 烂)，
    旧逻辑 verdict=green 误判长尾未被牺牲。
    """
    # 实测数据简化：world_top=0.110 / pers_top=0.050 → ratio≈2.17（崩盘）
    world = {"top": 0.110, "median": 0.106, "bottom": 0.033}
    pers = {"top": 0.050, "median": 0.100, "bottom": 0.046}
    v = evaluate_rare_au_drop(world, pers, n_samples=10, buckets=_mk_buckets())
    assert v.verdict == "n/a", v.reason
    assert "top ratio" in v.reason
    assert "整体崩盘" in v.reason


# -----------------------------------------------------------------------------
# 端到端：mock dataset → freq → bucket → mse → verdict
# -----------------------------------------------------------------------------
def test_end_to_end_pipeline_runs():
    """完整 pipeline 跑得通且 verdict 是合法值."""
    with tempfile.TemporaryDirectory() as tmp:
        npz, ratings = _create_mock_veatic_with_skewed_au(Path(tmp))
        ds = VEATICDataset(npz, ratings, seq_len=30)
        freq = compute_au_activation_frequency(ds)
        buckets = partition_au_by_frequency(freq)

        # 假装跑了一轮 rollout，模型在 top 桶接近 target，bottom 桶接近 0（牺牲长尾）
        B, T, D = 4, 6, 52
        target = torch.randn(B, T, D) * 0.1
        for c in buckets.top:
            target[:, :, c] = torch.rand(B, T) * 0.5 + 0.5  # 高频
        # pred: 模型刷了 top 但 bottom 全 0
        pred = target.clone()
        for c in buckets.bottom:
            pred[:, :, c] = 0.0
        # persistence: 用 t=0 帧反复
        pers_pred = target[:, :1, :].expand_as(target)

        world = compute_per_bucket_mse(pred, target, buckets)
        pers = compute_per_bucket_mse(pers_pred, target, buckets)
        v = evaluate_rare_au_drop(world, pers, n_samples=B * T, buckets=buckets)
        assert v.verdict in ("green", "yellow", "red")


def test_daemon_cache_end_to_end_default_fixed_not_evicted():
    """真实 daemon cache dry-run：默认 fixed(AU07/AU10) 在川哥数据应不被 evict."""
    cache_path = Path("/tmp/daemon_au_cache.npz")
    if not cache_path.exists():
        pytest.skip("missing /tmp/daemon_au_cache.npz")
    ds = DaemonAUDataset(cache_path)
    freq = compute_au_activation_frequency(ds)
    buckets = partition_au_by_frequency(freq)
    assert buckets.evicted_fixed == ()
    assert 5 in buckets.bottom
    assert 7 in buckets.bottom
