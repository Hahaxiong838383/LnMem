"""Tests for ADR-041 anchor_ts_ns slicing methods on RealtimeLNNInferencer.

修复 intent_feedback 标注-窗口 100% 时间错位(median 5min): 标签 ts_ns 可能比
POST 时刻早数秒到数分钟, 此前 get_recent_window 锚定 deque[-1].ts(当下)
必然错位。新方法 get_window_around_ts / get_long_context_around_ts 按 anchor
切窗, 锚定到判断时刻; 越界 fail-soft 返回 [] 让上层降级。
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer  # noqa: E402

NS = 1_000_000_000


def _make_inferencer() -> RealtimeLNNInferencer:
    """构造空骨架, 手动塞 _recent_temporal 用于切窗测试(不需 ckpt/backbone)。"""
    inf = RealtimeLNNInferencer(ckpt_path=None)
    inf._recent_temporal = deque()
    return inf


def _push(inf: RealtimeLNNInferencer, ts_s: float, marker: int) -> None:
    """oldest→newest 顺序追加; marker 嵌入 au[0] 用于回读校验帧身份。"""
    ts_ns = int(ts_s * NS)
    hidden = [0.0] * 64
    au = [float(marker)] + [0.0] * 17
    inf._recent_temporal.append((ts_ns, hidden, au))


# ---------- get_window_around_ts ----------


def test_get_window_around_ts_anchor_within_buffer_returns_recent_s_frames() -> None:
    """anchor 落在 deque 中段, 切窗 [anchor-10s, anchor+2s], 仅返回该区间帧。"""
    inf = _make_inferencer()
    for i in range(30):  # 帧在 ts=100..129s
        _push(inf, 100.0 + i, marker=i)
    win = inf.get_window_around_ts(anchor_ts_ns=int(120 * NS), recent_s=10)
    # 期望: ts ∈ [110, 122] (anchor=120, recent_s=10, tolerance=2)
    assert win, "anchor 在缓冲范围内应有非空切窗"
    markers = [int(f["au"][0]) for f in win]
    # 110..122 对应 marker 10..22 (i=10..22), 共 13 帧
    assert markers == list(range(10, 23)), f"切窗 markers 不对: {markers}"
    # 排序: oldest→newest
    ts_list = [f["ts_ns"] for f in win]
    assert ts_list == sorted(ts_list)


def test_get_window_around_ts_anchor_in_past_only_returns_overlap() -> None:
    """anchor 比 deque 最旧帧还早一点点, 仍有部分交集时应返回交集内的帧。"""
    inf = _make_inferencer()
    for i in range(10):  # ts=100..109s
        _push(inf, 100.0 + i, marker=i)
    # anchor=105s → 切窗 [95, 107]; 与 [100, 109] 交集 = [100, 107]
    win = inf.get_window_around_ts(anchor_ts_ns=int(105 * NS), recent_s=10)
    markers = [int(f["au"][0]) for f in win]
    assert markers == list(range(0, 8)), f"交集帧不对: {markers}"


def test_get_window_around_ts_anchor_too_early_returns_empty() -> None:
    """anchor 远早于 deque(差距 > recent_s + tolerance), 完全无交集 → []。"""
    inf = _make_inferencer()
    for i in range(10):
        _push(inf, 1000.0 + i, marker=i)  # ts=1000..1009s
    # anchor=500s → 切窗 [490, 502], 与 [1000, 1009] 无交集
    assert inf.get_window_around_ts(anchor_ts_ns=int(500 * NS), recent_s=10) == []


def test_get_window_around_ts_anchor_too_late_returns_empty() -> None:
    """anchor 远晚于 deque(> tolerance_s) → []。"""
    inf = _make_inferencer()
    for i in range(10):
        _push(inf, 100.0 + i, marker=i)  # ts=100..109s
    # anchor=200s → 切窗 [190, 202], deque[-1].ts=109 < 190
    assert inf.get_window_around_ts(anchor_ts_ns=int(200 * NS), recent_s=10) == []


def test_get_window_around_ts_invalid_anchor_returns_empty() -> None:
    """非 int / 负数 / 0 anchor → [], 不抛 (fail-soft)。"""
    inf = _make_inferencer()
    _push(inf, 100.0, marker=0)
    assert inf.get_window_around_ts(anchor_ts_ns=0) == []
    assert inf.get_window_around_ts(anchor_ts_ns=-1) == []
    assert inf.get_window_around_ts(anchor_ts_ns="bad") == []  # type: ignore[arg-type]


def test_get_window_around_ts_empty_deque_returns_empty() -> None:
    inf = _make_inferencer()
    assert inf.get_window_around_ts(anchor_ts_ns=int(100 * NS)) == []


def test_get_window_around_ts_tolerance_boundary() -> None:
    """anchor 比最新帧晚 exactly tolerance_s, 应包含最新帧(边界包含)。"""
    inf = _make_inferencer()
    _push(inf, 100.0, marker=0)
    _push(inf, 101.0, marker=1)
    # anchor = 103s, recent_s=10, tolerance=2 → 切窗 [93, 105], 包含 100/101
    win = inf.get_window_around_ts(anchor_ts_ns=int(103 * NS), recent_s=10, tolerance_s=2.0)
    assert [int(f["au"][0]) for f in win] == [0, 1]
    # anchor = 103.001s, tolerance=2 → 边界 [-,103.001+2=105.001], 仍包含 101
    win2 = inf.get_window_around_ts(
        anchor_ts_ns=int(103.001 * NS), recent_s=10, tolerance_s=2.0
    )
    assert len(win2) == 2


def test_get_window_around_ts_returns_immutable_copies() -> None:
    """返回的 h/au 列表是 copy, 修改不应反向污染 deque。"""
    inf = _make_inferencer()
    _push(inf, 100.0, marker=7)
    win = inf.get_window_around_ts(anchor_ts_ns=int(100 * NS), recent_s=10)
    win[0]["au"][0] = 999.0
    win2 = inf.get_window_around_ts(anchor_ts_ns=int(100 * NS), recent_s=10)
    assert win2[0]["au"][0] == 7.0, "返回值变动污染了底层 deque"


# ---------- get_long_context_around_ts ----------


def test_get_long_context_around_ts_anchor_buckets_correctly() -> None:
    """anchor 切窗 [anchor-120s, anchor-10s], 每 10s 桶取最近一帧。"""
    inf = _make_inferencer()
    # 100..1200s 每秒一帧, marker=秒数(便于断言)
    for i in range(1101):
        _push(inf, 100.0 + i, marker=int(100 + i))
    # anchor=1200s → 长程区间 [1080, 1190] (s), 11 个 10s 桶, 每桶最后一帧
    long_win = inf.get_long_context_around_ts(
        anchor_ts_ns=int(1200 * NS), recent_s=10, long_s=120, long_hz=0.1
    )
    assert long_win, "anchor 在缓冲深处长程应有内容"
    markers = [int(f["au"][0]) for f in long_win]
    # 每 10s 桶最新一帧: ts=1089, 1099, 1109, ..., 1189 (差 10), 共 11 个
    assert markers == list(range(1089, 1190, 10)), f"桶下采样错: {markers}"


def test_get_long_context_around_ts_invalid_anchor_returns_empty() -> None:
    inf = _make_inferencer()
    for i in range(200):
        _push(inf, 100.0 + i, marker=i)
    assert inf.get_long_context_around_ts(anchor_ts_ns=0) == []
    assert inf.get_long_context_around_ts(anchor_ts_ns=-1) == []


def test_get_long_context_around_ts_uses_anchor_not_deque_newest() -> None:
    """关键: latest_ts 必须用 anchor 而非 deque[-1].ts (否则切窗仍锚到当下)。

    若 bug: 用 deque[-1]=ts=300s 当 latest, 长程区间 [180,290], 包含整段 deque 旧帧;
    正确: 用 anchor=200s, 长程区间 [80, 190], 只覆盖 100..189(deque 内)。
    """
    inf = _make_inferencer()
    for i in range(201):
        _push(inf, 100.0 + i, marker=int(100 + i))  # ts=100..300s
    long_win = inf.get_long_context_around_ts(
        anchor_ts_ns=int(200 * NS), recent_s=10, long_s=120, long_hz=0.1
    )
    markers = [int(f["au"][0]) for f in long_win]
    # 正确切窗 ts ∈ [80, 190); deque 实际从 100 开始, 故 [100, 190)
    # 桶: 每 10s 一个 (从 long_cut=80 起), 桶内最后一帧
    # bucket (ts - 80) // 10: ts=100→b=2 marker=100..109 取 109
    #                          ts=110→b=3 取 119, ..., ts=180→b=10 取 189
    # 期望 markers = [109, 119, 129, ..., 189]
    expected = list(range(109, 190, 10))
    assert markers == expected, f"anchor 长程切窗不对(可能没用 anchor 做 latest_ts): got {markers}, want {expected}"
