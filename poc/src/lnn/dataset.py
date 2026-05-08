"""Day 5 dataset: lazy JSONL loader with sliding window for LNN training.

Implements ``SignalsLoader`` Protocol from ``contracts.py``.

设计决策
--------
1. **预处理到 numpy**：build_loaders 时一次性解析 JSONL 为 compact numpy arrays
   （vectors float32 + states int64 + boundaries float32 = ~2GB for 900 files），
   避免训练时反复 JSON 解析。原始 JSON 不驻留内存（900 × 12MB = 10.8GB）。
2. **文件级 shuffle**：``_FileGroupedSampler`` 文件间 shuffle + 文件内顺序处理，
   训练随机性充分且 numpy 索引 O(1)。
3. **State → idx 映射**：硬编码 8 类列表与 ``event_schema.py`` ``State`` enum 顺序对齐。
4. **Boundary 判定**：``frame_idx == 0`` 或 ``source_event_id`` 与前帧不同 → 1，其余 → 0。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from poc.src.lnn.contracts import LNNDataError, LNNTrainConfig

# State enum ordering — matches poc/src/data/event_schema.py State class
_STATE_LABELS: tuple[str, ...] = (
    "low_energy",
    "focused",
    "distracted",
    "fatigued",
    "emotional_positive",
    "emotional_negative",
    "neutral",
    "transitioning",
)
_STATE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(_STATE_LABELS)}

# Filename pattern: signals_<persona>_day<N>.jsonl
_FILENAME_RE = re.compile(r"^signals_(.+)_day(\d+)\.jsonl$")


@dataclass(frozen=True)
class _FileInfo:
    """Metadata for one JSONL signals file (no data loaded)."""

    path: Path
    persona_id: str
    day_index: int
    n_frames: int


@dataclass(frozen=True)
class _FileArrays:
    """Pre-processed file data as compact numpy arrays."""

    vectors: np.ndarray    # (N, 12) float32
    states: np.ndarray     # (N,) int64
    boundaries: np.ndarray  # (N,) float32
    persona_idx: int       # scalar


def _parse_file_to_arrays(
    fi: _FileInfo,
    persona_to_idx: dict[str, int],
) -> _FileArrays:
    """Parse one JSONL file into compact numpy arrays. Raises LNNDataError on bad JSON."""
    vectors_list: list[list[float]] = []
    states_list: list[int] = []
    event_ids: list[str] = []
    frame_idxs: list[int] = []

    with fi.path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, 1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise LNNDataError(
                    f"JSON parse error in {fi.path}:{line_no}: {exc}",
                ) from exc
            vectors_list.append(row["vector"])
            states_list.append(_STATE_TO_IDX[row["ground_truth_state"]])
            event_ids.append(row.get("source_event_id", ""))
            frame_idxs.append(row.get("frame_idx", -1))

    n = len(vectors_list)
    vectors = np.array(vectors_list, dtype=np.float32)  # (N, 12)
    states = np.array(states_list, dtype=np.int64)       # (N,)

    # Pre-compute boundaries
    boundaries = np.zeros(n, dtype=np.float32)
    for t in range(n):
        if frame_idxs[t] == 0:
            boundaries[t] = 1.0
        elif t == 0:
            boundaries[t] = 1.0  # First frame of file
        elif event_ids[t] != event_ids[t - 1]:
            boundaries[t] = 1.0

    return _FileArrays(
        vectors=vectors,
        states=states,
        boundaries=boundaries,
        persona_idx=persona_to_idx[fi.persona_id],
    )


class _FileGroupedSampler(Sampler[int]):
    """Shuffle at file level, sequential within each file's windows.

    Gives full inter-file randomness while keeping numpy array access
    cache-friendly (consecutive windows index the same pre-loaded array).
    """

    def __init__(self, file_window_groups: list[list[int]]) -> None:
        self._groups = file_window_groups
        self._total = sum(len(g) for g in file_window_groups)

    def __iter__(self):  # type: ignore[override]
        order = torch.randperm(len(self._groups)).tolist()
        for g_idx in order:
            yield from self._groups[g_idx]

    def __len__(self) -> int:
        return self._total


class _SignalsDataset(Dataset):  # type: ignore[type-arg]
    """Pre-processed sliding-window dataset with numpy-backed O(1) indexing."""

    def __init__(
        self,
        file_arrays: tuple[_FileArrays, ...],
        seq_length: int,
        seq_stride: int,
    ) -> None:
        self._arrays = file_arrays
        self._seq_length = seq_length

        # Build window index: (array_idx, start_frame)
        windows: list[tuple[int, int]] = []
        # Also build per-file window groups for grouped sampler
        self._file_groups: list[list[int]] = []
        for arr_idx, arr in enumerate(file_arrays):
            n = arr.vectors.shape[0]
            group: list[int] = []
            start = 0
            while start + seq_length <= n:
                group.append(len(windows))
                windows.append((arr_idx, start))
                start += seq_stride
            if group:
                self._file_groups.append(group)
        self._windows = tuple(windows)

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(
        self,
        idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        arr_idx, start = self._windows[idx]
        arr = self._arrays[arr_idx]
        end = start + self._seq_length

        return (
            torch.from_numpy(arr.vectors[start:end].copy()),   # (T, 12)
            torch.from_numpy(arr.states[start:end].copy()),    # (T,)
            torch.tensor(arr.persona_idx, dtype=torch.long),   # scalar
            torch.from_numpy(arr.boundaries[start:end].copy()),  # (T,)
        )


def _count_lines(path: Path) -> int:
    """Count non-empty lines in a file efficiently (no JSON parsing)."""
    count = 0
    with path.open("rb") as f:
        for raw in f:
            if raw.strip():
                count += 1
    return count


def _scan_files(config: LNNTrainConfig) -> tuple[_FileInfo, ...]:
    """Scan signals_dir for valid JSONL files, count lines, skip short ones."""
    signals_dir = config.signals_dir
    if not signals_dir.exists():
        raise LNNDataError(f"Signals directory not found: {signals_dir}")

    infos: list[_FileInfo] = []
    for path in sorted(signals_dir.glob("signals_*_day*.jsonl")):
        match = _FILENAME_RE.match(path.name)
        if not match:
            continue
        persona_id = match.group(1)
        day_index = int(match.group(2))
        n_frames = _count_lines(path)

        if n_frames < config.sequence_min_frames:
            continue  # Skip short files

        infos.append(
            _FileInfo(
                path=path,
                persona_id=persona_id,
                day_index=day_index,
                n_frames=n_frames,
            ),
        )
    return tuple(infos)


class JsonlSignalsLoader:
    """Build train/val DataLoaders from signals JSONL files.

    Satisfies ``SignalsLoader`` Protocol in ``contracts.py``.
    """

    def build_loaders(
        self,
        config: LNNTrainConfig,
    ) -> tuple[DataLoader, DataLoader, dict[str, int]]:
        """Scan signals_dir, split by (persona, day), return loaders + persona_to_idx."""
        file_infos = _scan_files(config)
        if not file_infos:
            raise LNNDataError(
                f"No valid JSONL files found in {config.signals_dir}",
            )

        # Deterministic persona_to_idx: sorted by persona_id
        all_personas = sorted({fi.persona_id for fi in file_infos})
        persona_to_idx: dict[str, int] = {p: i for i, p in enumerate(all_personas)}

        # Split by (persona_id, day_index) unique key
        keys = sorted({(fi.persona_id, fi.day_index) for fi in file_infos})
        torch.manual_seed(config.seed)
        perm = torch.randperm(len(keys)).tolist()
        split_point = int(len(keys) * config.train_val_split)

        train_keys: set[tuple[str, int]] = set()
        val_keys: set[tuple[str, int]] = set()
        for i, p_idx in enumerate(perm):
            target = train_keys if i < split_point else val_keys
            target.add(keys[p_idx])

        train_infos = tuple(
            fi for fi in file_infos if (fi.persona_id, fi.day_index) in train_keys
        )
        val_infos = tuple(
            fi for fi in file_infos if (fi.persona_id, fi.day_index) in val_keys
        )

        # Pre-process all files to numpy arrays (one-time cost, ~2 min for 900 files)
        print(f"[dataset] pre-processing {len(train_infos)} train + "
              f"{len(val_infos)} val files...", flush=True)
        train_arrays = tuple(
            _parse_file_to_arrays(fi, persona_to_idx) for fi in train_infos
        )
        val_arrays = tuple(
            _parse_file_to_arrays(fi, persona_to_idx) for fi in val_infos
        )
        print("[dataset] pre-processing done", flush=True)

        train_ds = _SignalsDataset(
            train_arrays, config.sequence_length, config.sequence_stride,
        )
        val_ds = _SignalsDataset(
            val_arrays, config.sequence_length, config.sequence_stride,
        )

        # Use file-grouped sampler for train (inter-file shuffle + cache-friendly)
        train_sampler: Sampler | None = None
        if len(train_ds) > 0 and train_ds._file_groups:
            train_sampler = _FileGroupedSampler(train_ds._file_groups)

        train_loader: DataLoader = DataLoader(
            train_ds,
            batch_size=config.batch_size,
            sampler=train_sampler,
            num_workers=0,
            drop_last=len(train_ds) >= config.batch_size,
        )
        val_loader: DataLoader = DataLoader(
            val_ds,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
        return train_loader, val_loader, persona_to_idx


__all__ = ["JsonlSignalsLoader"]
