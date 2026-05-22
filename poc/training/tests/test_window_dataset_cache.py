"""PC-5a: WindowDataset in-memory cache tests."""
from __future__ import annotations

import torch

from poc.training.window_dataset import WindowDataset


def test_window_dataset_cache_avoids_second_disk_read() -> None:
    """Lazy loader should be called only once during dataset init cache build."""
    calls = {"count": 0}

    def loader() -> torch.Tensor:
        calls["count"] += 1
        return torch.ones(30, 52)

    windows = [{"video_id": "v0", "event_tensor_loader": loader, "persona_idx": 0}]
    ds = WindowDataset(windows)
    assert calls["count"] == 1

    _ = ds[0]
    _ = ds[0]
    assert calls["count"] == 1, "2nd access should hit memory cache, not disk loader"


def test_window_dataset_returns_cpu_tensor() -> None:
    """Cached tensors stay on CPU to avoid accidental GPU memory pressure."""
    windows = [{"video_id": "v1", "event_tensor": torch.randn(30, 52)}]
    ds = WindowDataset(windows)
    item = ds[0]
    assert item["event_tensor"].device.type == "cpu"
