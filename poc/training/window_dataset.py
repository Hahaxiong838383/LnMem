"""WindowDataset for shuffled training windows with in-memory cache."""
from __future__ import annotations

from typing import Any, Callable

import torch
from torch.utils.data import Dataset


class WindowDataset(Dataset):
    """Cache split windows in CPU memory to avoid repeated disk I/O."""

    def __init__(self, all_windows: list[dict[str, Any]]) -> None:
        self._cache: list[dict[str, Any]] = []
        for w in all_windows:
            event_tensor = self._resolve_tensor(w)
            event_tensor_cpu = event_tensor.detach().cpu().contiguous()
            if torch.cuda.is_available():
                event_tensor_cpu = event_tensor_cpu.pin_memory()

            item: dict[str, Any] = {
                "event_tensor": event_tensor_cpu,
                "video_id": w["video_id"],
                "persona_idx": int(w.get("persona_idx", 0)),
            }
            if "labels_va" in w and w["labels_va"] is not None:
                item["labels_va"] = torch.as_tensor(w["labels_va"]).detach().cpu().contiguous()
            if "start_frame" in w:
                item["start_frame"] = int(w["start_frame"])
            self._cache.append(item)

    @staticmethod
    def _resolve_tensor(window: dict[str, Any]) -> torch.Tensor:
        """Resolve eager tensor or lazy loader into a torch tensor."""
        if "event_tensor_loader" in window and callable(window["event_tensor_loader"]):
            raw = window["event_tensor_loader"]()
        else:
            raw = window["event_tensor"]
            if callable(raw):
                raw = raw()
        return torch.as_tensor(raw)

    def __len__(self) -> int:
        return len(self._cache)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self._cache[idx]
