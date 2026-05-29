"""ADR-040 Phase1: get_long_context 多尺度长程上下文采集单测

验证 inferencer.get_long_context 的多尺度下采样 + 边界 + ts_ns 保留。
不依赖真 ckpt (直接构造 inferencer 的 _recent_temporal 缓冲)。
"""
from __future__ import annotations

import pytest

from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer


def _make_inferencer_with_buffer(frames: list[tuple[int, list[float], list[float]]]):
    """构造 disabled inferencer (ckpt=None) 但手动填 _recent_temporal。"""
    inf = RealtimeLNNInferencer(ckpt_path=None)
    inf._recent_temporal.clear()
    for f in frames:
        inf._recent_temporal.append(f)
    return inf


def _frames_1hz(n: int, start_ns: int = 0):
    """生成 n 帧 1Hz 数据 (间隔 1e9 ns), au=[i]*18, h=[i]*4。"""
    return [
        (start_ns + i * 1_000_000_000, [float(i)] * 4, [float(i)] * 18)
        for i in range(n)
    ]


def test_empty_buffer_returns_empty():
    inf = _make_inferencer_with_buffer([])
    assert inf.get_long_context() == []


def test_long_s_le_recent_s_returns_empty():
    inf = _make_inferencer_with_buffer(_frames_1hz(130))
    assert inf.get_long_context(recent_s=120, long_s=10) == []


def test_downsamples_to_one_per_bucket():
    """130 帧 1Hz (130s), recent_s=10 long_s=120 long_hz=0.1 (10s/桶).

    区间 [latest-120s, latest-10s) = 110s 窗口, 每 10s 一桶 → ~11 帧.
    """
    inf = _make_inferencer_with_buffer(_frames_1hz(130))
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    # 110s 区间 / 10s 桶 ≈ 11 桶
    assert 10 <= len(long) <= 12, f"期望 ~11 帧, 实际 {len(long)}"


def test_excludes_recent_window():
    """长程上下文不应含最近 recent_s 内的帧 (那是 get_recent_window 的活)。"""
    frames = _frames_1hz(130)
    latest_ts = frames[-1][0]
    inf = _make_inferencer_with_buffer(frames)
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    recent_cut = latest_ts - 10 * 1_000_000_000
    for item in long:
        assert item["ts_ns"] < recent_cut, "长程帧不应落入 recent 10s 窗口"


def test_excludes_beyond_long_s():
    """超过 long_s 的老帧应被丢弃。"""
    frames = _frames_1hz(200)  # 200s 数据
    latest_ts = frames[-1][0]
    inf = _make_inferencer_with_buffer(frames)
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    long_cut = latest_ts - 120 * 1_000_000_000
    for item in long:
        assert item["ts_ns"] >= long_cut, "不应含超过 long_s 的老帧"


def test_ts_ns_ascending_and_preserved():
    """返回帧 ts_ns 升序 + 保留真实 ts (不等间距, 供下游 Δt)。"""
    inf = _make_inferencer_with_buffer(_frames_1hz(130))
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    ts_list = [item["ts_ns"] for item in long]
    assert ts_list == sorted(ts_list), "ts_ns 必须升序"
    assert len(set(ts_list)) == len(ts_list), "ts_ns 不应重复"
    # 真实间隔 ~10s (下采样后), 验证保留了真实 ts 而非重新编号
    if len(ts_list) >= 2:
        gap = ts_list[1] - ts_list[0]
        assert gap >= 9_000_000_000, f"下采样后帧间隔应 ~10s, 实际 {gap/1e9}s"


def test_each_frame_has_h_and_au():
    inf = _make_inferencer_with_buffer(_frames_1hz(130))
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    for item in long:
        assert "ts_ns" in item and "h" in item and "au" in item
        assert len(item["au"]) == 18
        assert len(item["h"]) == 4


def test_backward_compat_get_recent_window_unchanged():
    """ADR-040 不可妥协: 不碰 get_recent_window 行为。"""
    inf = _make_inferencer_with_buffer(_frames_1hz(130))
    recent = inf.get_recent_window(seconds=10)
    # recent window 仍是 1Hz 全帧, 最近 ~10 帧
    assert 9 <= len(recent) <= 11
    # 且和 long_context 区间不重叠
    long = inf.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
    recent_min_ts = min(r["ts_ns"] for r in recent)
    long_max_ts = max(l["ts_ns"] for l in long) if long else 0
    assert long_max_ts < recent_min_ts, "long 与 recent 窗口不应重叠"
