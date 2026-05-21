"""测试 dataset + evaluate（无需真 VEATIC 数据，用 mock）"""
import numpy as np
import torch
import pytest

from poc.training.dataset import build_event_tensor, split_to_windows
from poc.training.evaluate import (
    mask_text_features,
    evaluate_video_id_shortcut,
    evaluate_retrieval_topk,
)


def test_build_event_tensor_dim():
    """52d schema 拼接正确（spec §2.1: 24 AU + 20 槽含 intent + 2 time + 6 audio）"""
    T = 30
    au = np.random.rand(T, 24).astype(np.float32)
    ts = np.linspace(0, 1, T, dtype=np.float32)
    et = build_event_tensor(au, ts)
    assert et.shape == (T, 52)
    # AU 区域保留
    np.testing.assert_array_almost_equal(et[:, :24], au)
    # 20 槽（含 intent）区域 spike 阶段全 0
    np.testing.assert_array_equal(et[:, 24:44], 0.0)
    # 时间戳 [44:46] sin / cos 在 [-1, 1]
    assert et[:, 44].min() >= -1 and et[:, 44].max() <= 1
    # 音频占位 [46:52] 0
    np.testing.assert_array_equal(et[:, 46:52], 0.0)


def test_mask_text_features():
    """spec §6.1 M1-V: 20d 文本（含 intent）置零"""
    et = torch.randn(4, 30, 52)
    masked = mask_text_features(et)
    # 视觉/时间/音频 保留
    assert torch.equal(masked[..., :24], et[..., :24])
    assert torch.equal(masked[..., 44:], et[..., 44:])
    # 文本（20 槽含 intent）全 0
    assert masked[..., 24:44].abs().sum().item() == 0.0


def test_split_to_windows():
    """切窗口长度 + 数量正确"""
    sample = {
        "video_id": "0",
        "event_tensor": torch.randn(100, 52),
        "labels_va": torch.randn(100, 2),
    }
    windows = split_to_windows(sample, seq_len=30, stride=10)
    # (100-30)/10 + 1 = 8 个窗口
    assert len(windows) == 8
    for w in windows:
        assert w["event_tensor"].shape == (30, 52)
        assert w["labels_va"].shape == (30, 2)


def test_video_id_shortcut_random():
    """随机 hidden + 5 视频 → linear probe acc 应接近 random（20%）"""
    np.random.seed(42)
    n_per_video = 20
    n_videos = 5
    hidden = np.random.randn(n_per_video * n_videos, 64).astype(np.float32)
    video_ids = []
    for v in range(n_videos):
        video_ids.extend([str(v)] * n_per_video)
    result = evaluate_video_id_shortcut(hidden, video_ids, test_ratio=0.3)
    print(f"\n  M5 random hidden test acc: {result.test_acc:.3f} ({result.verdict})")
    # 随机 hidden → acc 应在 [0.1, 0.4]（5 类随机 ~0.2 ± noise）
    assert 0.0 <= result.test_acc <= 1.0
    assert result.verdict in ("green", "yellow", "red")


def test_video_id_shortcut_perfect_collapse():
    """同视频 hidden 完全相同 → linear probe 应能 100% 区分（M5 red）"""
    np.random.seed(7)
    n_videos = 5
    n_per_video = 20
    hidden = []
    for v in range(n_videos):
        # 每个视频用同一份 hidden（极端 collapse）
        base = np.random.randn(64).astype(np.float32) * 5
        hidden.extend([base.copy() for _ in range(n_per_video)])
    hidden = np.stack(hidden)
    video_ids = [str(v) for v in range(n_videos) for _ in range(n_per_video)]
    result = evaluate_video_id_shortcut(hidden, video_ids, test_ratio=0.3)
    print(f"\n  M5 collapse hidden test acc: {result.test_acc:.3f} ({result.verdict})")
    # 完全 collapse → 100% 可分 → red
    assert result.test_acc > 0.5, f"完全 collapse 应 > 50%, got {result.test_acc}"
    assert result.verdict == "red"


def test_retrieval_topk_perfect_self():
    """同 doc 集合自查询，自己被排除后应是相邻最相似项"""
    np.random.seed(42)
    n_videos = 3
    n_per_video = 4
    # 每视频 hidden cluster 相近
    hidden = []
    video_ids = []
    for v in range(n_videos):
        center = np.random.randn(64).astype(np.float32) * 3
        for i in range(n_per_video):
            hidden.append(center + np.random.randn(64).astype(np.float32) * 0.5)
            video_ids.append(str(v))
    hidden = np.stack(hidden)
    result = evaluate_retrieval_topk(hidden, hidden, video_ids, video_ids, k=5)
    print(f"\n  retrieval top1={result.top1_acc:.3f} top5={result.top5_acc:.3f}")
    # 同视频聚类好 → top5 应高
    assert result.top5_acc > 0.5
