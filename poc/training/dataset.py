"""γ-3 Step 2: VEATIC Dataset 包装 AU + V-A 标签（spec §4.1）

输入: AU cache .npz + VEATIC rating_averaged/{id}_valence|arousal.csv
输出: 52d EventTensor 序列 (T, 52) + V-A 标签 (T, 2)

EventTensor 52d schema（spec §2.1 实际维度计算）:
    [0:24]    24 AU from py-feat
    [24:44]   20 万能语法槽（**含 intent_type 编码**，spec §2.1 "intent_type 加 5 dim" 意指融入 20 槽 not 独立）
              spike 阶段: all 0
    [44:46]   2 时间戳 [sin(2πt/T), cos(2πt/T)]
    [46:52]   6 音频占位（spike: all 0，M1+ 接）

总: 24 + 20 + 2 + 6 = 52 ✓

注：ADR-018 EventTensor v1.1 把 intent_type 加为独立字段是 schema 层面，
模型 input 编码时 intent 信息 embed 进 20 槽（slot[15..19] 等位置），不另起 5 维.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_rating_averaged(rating_dir: str | Path, video_id: str) -> dict[str, np.ndarray]:
    """读取 VEATIC rating_averaged/{video_id}_valence.csv + arousal.csv.

    csv schema: frame_id, value 两列，frame_id 从 0 开始连续
    Returns: {"valence": (n_frames,), "arousal": (n_frames,)}
    """
    rating_dir = Path(rating_dir)
    out = {}
    for axis in ("valence", "arousal"):
        path = rating_dir / f"{video_id}_{axis}.csv"
        df = pd.read_csv(path, header=None, names=["frame_id", "value"])
        out[axis] = df["value"].values.astype(np.float32)
    return out


def build_event_tensor(
    au: np.ndarray,
    timestamps_norm: np.ndarray,
    slot_dim: int = 20,
    audio_dim: int = 6,
) -> np.ndarray:
    """从 AU + 时间戳 拼 52d EventTensor（spec §2.1）.

    Args:
        au: (T, 20) AU 强度（py-feat 实际输出 20 AU；spec §2.1 24d 包含 4 head pose
            待 M1+ 集成，本阶段 zero-pad 占位）
        timestamps_norm: (T,) 归一化时间戳 [0, 1]

    Returns:
        (T, 52) EventTensor: 20 AU + 4 head pose(占位) + 20 槽(含 intent) + 2 time + 6 audio
    """
    T = au.shape[0]
    # py-feat 实际 20 AU → zero-pad 到 24 维（包含未来 4 head pose 维度占位）
    if au.shape[1] == 20:
        au = np.concatenate([au.astype(np.float32),
                             np.zeros((T, 4), dtype=np.float32)], axis=1)
    assert au.shape[1] == 24, f"au shape {au.shape} expected (T, 24) after padding"

    t_sin = np.sin(2 * np.pi * timestamps_norm).reshape(-1, 1)
    t_cos = np.cos(2 * np.pi * timestamps_norm).reshape(-1, 1)

    slot_zero = np.zeros((T, slot_dim), dtype=np.float32)
    audio_zero = np.zeros((T, audio_dim), dtype=np.float32)

    event_tensor = np.concatenate([
        au.astype(np.float32),       # [0:24]
        slot_zero,                    # [24:44] 20 槽（含 intent 编码 in slots[15:19]）
        t_sin.astype(np.float32),    # [44]
        t_cos.astype(np.float32),    # [45]
        audio_zero,                   # [46:52]
    ], axis=1)
    assert event_tensor.shape[1] == 52, f"got {event_tensor.shape[1]} expected 52"
    return event_tensor


class VEATICDataset(Dataset):
    """每 sample 是一个视频的 (T, 52) EventTensor + (T, 2) V-A 标签.

    Args:
        au_cache_path: .npz 缓存路径
        rating_dir: VEATIC rating_averaged/ 目录
        seq_len: 训练时切片长度（默认 30，约 1s @ 30fps）
        normalize_au: 是否对 AU 做 z-score 归一化
    """

    def __init__(
        self,
        au_cache_path: str | Path,
        rating_dir: str | Path,
        seq_len: int = 30,
        normalize_au: bool = False,
    ):
        from .extract_au import load_au_cache
        self.au_cache = load_au_cache(au_cache_path)
        self.rating_dir = Path(rating_dir)
        self.seq_len = seq_len
        self.normalize_au = normalize_au
        self.video_ids = sorted(self.au_cache.keys(), key=lambda x: int(x))
        self._validate()

    def _validate(self):
        for vid in self.video_ids:
            au = self.au_cache[vid]
            # 接受 20 AU (py-feat 实际输出) 或 24 (含 head pose)
            assert au.ndim == 2 and au.shape[1] in (20, 24), \
                f"video {vid} AU shape {au.shape} 异常（期望 (T, 20) 或 (T, 24)）"

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        vid = self.video_ids[idx]
        au = self.au_cache[vid].copy()  # (n_frames, 24)
        T = au.shape[0]

        # 加载 V-A 标签（rating 帧数可能与 AU 抽样帧数不一致，需对齐）
        rating = load_rating_averaged(self.rating_dir, vid)
        # AU 抽样间隔 N → 取 rating 的 frame[0::N]
        # spike 阶段简化：用 AU 帧数对齐 rating（取 rating 前 T 帧）
        valence = rating["valence"][:T]
        arousal = rating["arousal"][:T]
        if len(valence) < T:
            # rating 不够长 → 末尾 pad 0
            pad_n = T - len(valence)
            valence = np.concatenate([valence, np.zeros(pad_n)])
            arousal = np.concatenate([arousal, np.zeros(pad_n)])

        # 时间戳归一化 [0, 1]
        timestamps = np.linspace(0, 1, T, dtype=np.float32)

        # AU 归一化（可选）
        if self.normalize_au:
            au = (au - au.mean(axis=0, keepdims=True)) / (au.std(axis=0, keepdims=True) + 1e-6)

        # 构造 52d EventTensor
        event_tensor = build_event_tensor(au, timestamps)
        labels_va = np.stack([valence, arousal], axis=1).astype(np.float32)

        return {
            "video_id": vid,
            "event_tensor": torch.from_numpy(event_tensor),  # (T, 52)
            "labels_va": torch.from_numpy(labels_va),         # (T, 2)
            "n_frames": T,
        }


def split_to_windows(
    sample: dict,
    seq_len: int = 30,
    stride: int = 10,
) -> list[dict]:
    """把一个视频切成多个 seq_len 长度的训练窗口（spec §4.3 Sampler 备选）.

    Args:
        sample: VEATICDataset.__getitem__ 输出
        seq_len: 窗口长度（默认 30 帧）
        stride: 窗口步长（默认 10 → 重叠 20 帧增加样本）

    Returns:
        list of {video_id, event_tensor (seq_len, 52), labels_va (seq_len, 2), start_frame}
    """
    et = sample["event_tensor"]
    labels = sample["labels_va"]
    T = et.shape[0]
    windows = []
    for start in range(0, T - seq_len + 1, stride):
        windows.append({
            "video_id": sample["video_id"],
            "event_tensor": et[start:start + seq_len],
            "labels_va": labels[start:start + seq_len],
            "start_frame": start,
        })
    return windows
