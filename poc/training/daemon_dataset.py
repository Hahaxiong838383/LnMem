"""Daemon JSONL to VEATIC-compatible AU dataset adapter.

将 LnMem daemon signals_*.jsonl 转为 VEATICDataset 可消费的 AU cache 与 Dataset。
"""
from __future__ import annotations

import argparse
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


def build_daemon_au_cache(
    signal_files: Iterable[str | Path],
    out_path: str | Path,
    use_calibrated: bool = True,
) -> dict[str, np.ndarray]:
    """Convert daemon JSONL files into VEATIC-style AU cache.

    返回 `au_<session_short>` -> (T, 20) float32，列序严格按 `PYFEAT_AU_COLUMNS`。
    """
    source_key = "calibrated_au" if use_calibrated else "raw_au"
    by_session: dict[str, list[tuple[int, np.ndarray]]] = defaultdict(list)
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
                session_id = str(row.get("session_id") or "")
                if not session_id:
                    continue
                frame_id = int(row.get("frame_id", 0))
                by_session[session_id].append((frame_id, au_vec))

    cache: dict[str, np.ndarray] = {}
    frame_cache: dict[str, np.ndarray] = {}
    used_short: set[str] = set()
    for session_id, frames in sorted(by_session.items()):
        frames.sort(key=lambda x: x[0])
        frame_ids = np.array([it[0] for it in frames], dtype=np.int64)
        au_arr = np.stack([it[1] for it in frames], axis=0).astype(np.float32)
        short = _session_short_id(session_id, used_short)
        cache[f"au_{short}"] = au_arr
        frame_cache[f"frame_{short}"] = frame_ids

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cache, **frame_cache)
    return cache


class DaemonAUDataset(Dataset):
    """VEATIC-compatible dataset over daemon AU cache.

    注意 daemon 常见采样率约 1Hz，本类不做重采样，交由上层策略处理。
    """

    def __init__(self, au_cache_path: str | Path):
        data = np.load(au_cache_path)
        self.au_cache = {k[3:]: data[k] for k in data.files if k.startswith("au_")}
        self.frame_cache = {k[6:]: data[k] for k in data.files if k.startswith("frame_")}
        self.video_ids = sorted(self.au_cache.keys())
        self._validate()

    def _validate(self) -> None:
        for vid, au in self.au_cache.items():
            if au.ndim != 2 or au.shape[1] != len(PYFEAT_AU_COLUMNS):
                raise ValueError(f"video {vid} AU shape {au.shape} expected (T, 20)")
            if vid in self.frame_cache:
                if self.frame_cache[vid].ndim != 1 or self.frame_cache[vid].shape[0] != au.shape[0]:
                    raise ValueError(f"video {vid} frame ids not aligned with AU rows")

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str | int]:
        vid = self.video_ids[idx]
        au = self.au_cache[vid].copy().astype(np.float32)
        T = au.shape[0]
        frame_ids = self.frame_cache.get(vid)
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
            "start_frame": 0,
            "fps": 1.0,
        }


def _collect_signal_files(signal_dir: str | Path) -> list[Path]:
    """Collect daemon files. 英文一句话 + 中文说明：匹配 signals_*.jsonl 并排序。"""
    return sorted(Path(signal_dir).expanduser().glob("signals_*.jsonl"))


def _summarize_cache(cache: dict[str, np.ndarray]) -> tuple[int, int, dict[str, int]]:
    """Build cache stats. 英文一句话 + 中文说明：生成会话数/总帧数/逐会话帧数。"""
    per_session = {k[3:]: int(v.shape[0]) for k, v in sorted(cache.items())}
    total_frames = int(sum(per_session.values()))
    return len(per_session), total_frames, per_session


def main() -> None:
    """CLI entrypoint. 英文一句话 + 中文说明：构建 daemon AU cache 并打印统计。"""
    p = argparse.ArgumentParser()
    p.add_argument("--signal-dir", default="~/.lnmem/data")
    p.add_argument("--out", required=True)
    p.add_argument("--use-raw", action="store_true", help="Use raw_au instead of calibrated_au")
    args = p.parse_args()

    signal_files = _collect_signal_files(args.signal_dir)
    if not signal_files:
        raise FileNotFoundError(f"No signals_*.jsonl found under {args.signal_dir}")

    use_calibrated = not args.use_raw
    cache = build_daemon_au_cache(signal_files, args.out, use_calibrated=use_calibrated)
    n_sessions, total_frames, per_session = _summarize_cache(cache)
    print(f"saved: {args.out}")
    print(f"n_sessions={n_sessions}")
    print(f"total_frames={total_frames}")
    print("per_session_frames=")
    for sid, cnt in per_session.items():
        print(f"  {sid}: {cnt}")


if __name__ == "__main__":
    main()
