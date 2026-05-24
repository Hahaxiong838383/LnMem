"""Multi-FPS dataset composition for curriculum training."""
from __future__ import annotations

from dataclasses import dataclass
from torch.utils.data import Dataset


@dataclass(frozen=True)
class _SampleRef:
    """Global sample pointer. / 全局索引到子数据集样本。"""

    dataset_id: int
    sample_idx: int
    fps: float


class MultiFpsDataset(Dataset):
    """Combine heterogeneous datasets and expose global indices with fps tags."""

    def __init__(self, datasets: list[tuple[Dataset, float]]):
        if not datasets:
            raise ValueError("datasets must not be empty")
        self._datasets = datasets
        self._refs: list[_SampleRef] = []
        for ds_id, (dataset, fps) in enumerate(datasets):
            fps_value = float(fps)
            if fps_value <= 0:
                raise ValueError(f"fps must be > 0, got {fps_value}")
            for sample_idx in range(len(dataset)):
                self._refs.append(_SampleRef(ds_id, sample_idx, fps_value))

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, idx: int) -> dict:
        ref = self._refs[idx]
        dataset = self._datasets[ref.dataset_id][0]
        sample = dataset[ref.sample_idx]
        if not isinstance(sample, dict):
            raise TypeError(f"dataset sample must be dict, got {type(sample)!r}")
        out = dict(sample)
        out["fps"] = float(ref.fps)
        out["dataset_id"] = int(ref.dataset_id)
        out["dataset_sample_idx"] = int(ref.sample_idx)
        return out

    def get_fps_to_indices(self) -> dict[float, list[int]]:
        """Return global indices grouped by fps for bucket samplers."""
        fps_to_indices: dict[float, list[int]] = {}
        for global_idx, ref in enumerate(self._refs):
            fps_to_indices.setdefault(float(ref.fps), []).append(global_idx)
        return fps_to_indices


class VEATICDownsampled(Dataset):
    """VEATIC 30fps downsampled views with stride-N windows and L>=min_seq_len."""

    def __init__(self, base: Dataset, target_fps: float, min_seq_len: int = 8):
        self.base = base
        self.target_fps = float(target_fps)
        self.min_seq_len = int(min_seq_len)
        if self.target_fps <= 0.0:
            raise ValueError(f"target_fps must be > 0, got {self.target_fps}")
        if self.min_seq_len < 2:
            raise ValueError(f"min_seq_len must be >= 2, got {self.min_seq_len}")
        self._index: list[tuple[int, int, int]] = []
        self._build_index()
        if not self._index:
            raise ValueError("no downsampled windows produced; check min_seq_len/data length")

    def _build_index(self) -> None:
        # VEATIC baseline is 30fps in current pipeline.
        base_fps = 30.0
        stride = max(1, int(round(base_fps / self.target_fps)))
        raw_window = stride * self.min_seq_len
        for sample_idx in range(len(self.base)):
            sample = self.base[sample_idx]
            event_tensor = sample["event_tensor"]
            total_steps = int(event_tensor.shape[0])
            if total_steps < raw_window:
                continue
            for start in range(0, total_steps - raw_window + 1, stride):
                self._index.append((sample_idx, start, stride))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        sample_idx, start, stride = self._index[idx]
        sample = self.base[sample_idx]
        event = sample["event_tensor"]
        labels = sample["labels_va"]
        end = start + stride * self.min_seq_len
        event_down = event[start:end:stride]
        labels_down = labels[start:end:stride]
        if event_down.shape[0] < self.min_seq_len:
            raise RuntimeError("downsampled sequence shorter than min_seq_len")
        return {
            "video_id": str(sample["video_id"]),
            "event_tensor": event_down.clone().contiguous(),
            "labels_va": labels_down.clone().contiguous(),
            "start_frame": int(start),
            "fps": float(self.target_fps),
        }
