"""Daemon JSONL to VEATIC-compatible AU dataset adapter.

将 LnMem daemon signals_*.jsonl 转为 VEATICDataset 可消费的 AU cache 与 Dataset。
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import build_event_tensor
from .extract_au import PYFEAT_AU_COLUMNS

def _iter_signal_paths(signal_files: Iterable[str | Path]) -> list[Path]:
    """Resolve sorted JSONL paths. 英文一句话 + 中文说明：标准化输入文件列表。"""
    return sorted(Path(p).expanduser() for p in signal_files)

def _session_short_id(session_id: str, used: set[str], prefix_len: int = 12) -> str:
    """Build stable short id. 英文一句话 + 中文说明：优先前缀，冲突时附加哈希后缀。"""
    base = session_id[:prefix_len] if session_id else "unknown"
    key = base
    if key in used:
        digest = hashlib.sha1(session_id.encode("utf-8")).hexdigest()[:6]
        key = f"{base}_{digest}"
    used.add(key)
    return key


def _safe_float(value: object, default: float = 0.0) -> float:
    """Convert to float safely. 英文一句话 + 中文说明：解析失败时返回默认值。"""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if np.isfinite(out) else default


def build_daemon_au_cache(
    signal_files: Iterable[str | Path],
    out_path: str | Path,
    use_calibrated: bool = True,
    include_head_pose: bool = True,
    include_screen: bool = True,
) -> dict[str, np.ndarray]:
    """Convert daemon JSONL files into VEATIC-style AU cache.

    返回 `au_<session_short>` -> (T, 20) float32，列序严格按 `PYFEAT_AU_COLUMNS`。
    """
    source_key = "calibrated_au" if use_calibrated else "raw_au"
    by_session: dict[str, list[tuple[int, np.ndarray, np.ndarray, np.ndarray]]] = defaultdict(list)
    paths = _iter_signal_paths(signal_files)

    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if not row.get("face_detected", False):
                    continue
                au_dict = row.get(source_key) or {}
                if len(au_dict) < len(PYFEAT_AU_COLUMNS):
                    continue
                try:
                    au_vec = np.array([float(au_dict[col]) for col in PYFEAT_AU_COLUMNS], dtype=np.float32)
                except (KeyError, TypeError, ValueError):
                    continue
                # cc 整改：NaN/inf 过滤（codex 原版漏，导致 SVD 不收敛）
                if not np.isfinite(au_vec).all():
                    continue
                pose = row.get("head_pose") if include_head_pose else None
                if isinstance(pose, dict):
                    head_pose = np.array(
                        [
                            _safe_float(pose.get("yaw_deg")),
                            _safe_float(pose.get("pitch_deg")),
                            _safe_float(pose.get("roll_deg")),
                        ],
                        dtype=np.float32,
                    )
                else:
                    head_pose = np.zeros((3,), dtype=np.float32)
                activity = row.get("screen_activity") if include_screen else None
                if isinstance(activity, dict):
                    app = str(activity.get("active_app") or "").strip()
                    screen_vec = np.array(
                        [
                            1.0 if app else 0.0,
                            _safe_float(activity.get("key_rate_per_s")),
                            _safe_float(activity.get("mouse_rate_per_s")),
                        ],
                        dtype=np.float32,
                    )
                else:
                    screen_vec = np.zeros((3,), dtype=np.float32)
                session_id = str(row.get("session_id") or "")
                if not session_id:
                    continue
                frame_id = int(row.get("frame_id", 0))
                by_session[session_id].append((frame_id, au_vec, head_pose, screen_vec))

    cache: dict[str, np.ndarray] = {}
    frame_cache: dict[str, np.ndarray] = {}
    headpose_cache: dict[str, np.ndarray] = {}
    screen_cache: dict[str, np.ndarray] = {}
    used_short: set[str] = set()
    for session_id, frames in sorted(by_session.items()):
        frames.sort(key=lambda x: x[0])
        frame_ids = np.array([it[0] for it in frames], dtype=np.int64)
        au_arr = np.stack([it[1] for it in frames], axis=0).astype(np.float32)
        head_arr = np.stack([it[2] for it in frames], axis=0).astype(np.float32)
        screen_arr = np.stack([it[3] for it in frames], axis=0).astype(np.float32)
        short = _session_short_id(session_id, used_short)
        cache[f"au_{short}"] = au_arr
        frame_cache[f"frame_{short}"] = frame_ids
        headpose_cache[f"headpose_{short}"] = head_arr
        screen_cache[f"screen_{short}"] = screen_arr

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache, **frame_cache, **headpose_cache, **screen_cache)
    return cache

class DaemonAUDataset(Dataset):
    """VEATIC-compatible dataset over daemon AU cache.

    注意 daemon 常见采样率约 1Hz，本类不做重采样，交由上层策略处理。
    """

    def __init__(self, au_cache_path: str | Path):
        data = np.load(au_cache_path)
        self.au_cache = {k[3:]: data[k] for k in data.files if k.startswith("au_")}
        self.frame_cache = {k[6:]: data[k] for k in data.files if k.startswith("frame_")}
        self.headpose_cache = {k[9:]: data[k] for k in data.files if k.startswith("headpose_")}
        self.screen_cache = {k[7:]: data[k] for k in data.files if k.startswith("screen_")}
        self.video_ids = sorted(self.au_cache.keys())
        self._validate()

    def _validate(self) -> None:
        for vid, au in self.au_cache.items():
            if au.ndim != 2 or au.shape[1] != len(PYFEAT_AU_COLUMNS):
                raise ValueError(f"video {vid} AU shape {au.shape} expected (T, 20)")
            if vid in self.frame_cache:
                if self.frame_cache[vid].ndim != 1 or self.frame_cache[vid].shape[0] != au.shape[0]:
                    raise ValueError(f"video {vid} frame ids not aligned with AU rows")
            if vid in self.headpose_cache and (
                self.headpose_cache[vid].shape != (au.shape[0], 3) or self.headpose_cache[vid].dtype != np.float32
            ):
                raise ValueError(f"video {vid} head pose shape invalid: {self.headpose_cache[vid].shape}")
            if vid in self.screen_cache and (
                self.screen_cache[vid].shape != (au.shape[0], 3) or self.screen_cache[vid].dtype != np.float32
            ):
                raise ValueError(f"video {vid} screen shape invalid: {self.screen_cache[vid].shape}")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        vid = self.video_ids[idx]
        au = self.au_cache[vid].copy().astype(np.float32)
        T = au.shape[0]
        frame_ids = self.frame_cache.get(vid)
        head_pose = self.headpose_cache.get(vid)
        if head_pose is None:
            head_pose = np.zeros((T, 3), dtype=np.float32)
        else:
            head_pose = head_pose.copy().astype(np.float32)
        screen = self.screen_cache.get(vid)
        if screen is None:
            screen = np.zeros((T, 3), dtype=np.float32)
        else:
            screen = screen.copy().astype(np.float32)
        if frame_ids is None:
            frame_ids = np.arange(T, dtype=np.float32)
        else:
            frame_ids = frame_ids.astype(np.float32)
        max_frame = float(np.max(frame_ids)) if T > 0 else 0.0
        if T <= 1 or max_frame <= 0.0:
            timestamps = np.zeros((T,), dtype=np.float32)
        else:
            timestamps = frame_ids / max_frame

        event_tensor = build_event_tensor(au, timestamps)
        labels_va = np.zeros((T, 2), dtype=np.float32)
        return {
            "video_id": vid,
            "event_tensor": torch.from_numpy(event_tensor),
            "labels_va": torch.from_numpy(labels_va),
            "head_pose": torch.from_numpy(head_pose),
            "screen": torch.from_numpy(screen),
            "start_frame": 0,
            "fps": 1.0,
        }
