"""Bucket sampler for multi-FPS curriculum exposure."""
from __future__ import annotations

import random
from typing import Callable

from torch.utils.data import Sampler


def linear_probability_ramp(epoch: int, start_epoch: int, end_epoch: int, target_prob: float) -> float:
    """Linearly ramp probability from 0 to target_prob in [start_epoch, end_epoch]."""
    if target_prob < 0.0:
        raise ValueError("target_prob must be >= 0")
    if epoch < start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return float(target_prob)
    span = max(1, end_epoch - start_epoch)
    return float(target_prob) * ((epoch - start_epoch) / span)


class EquallyWeightedFpsBucketSampler(Sampler[list[int]]):
    """Sample homogeneous-fps batches and balance exposure by fps buckets."""

    def __init__(
        self,
        fps_to_indices: dict[float, list[int]],
        batch_size: int,
        curriculum_ramp: Callable[[int], dict[float, float]],
        seed: int,
    ):
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")
        if not fps_to_indices:
            raise ValueError("fps_to_indices must not be empty")
        self.fps_to_indices = {
            float(fps): list(indices)
            for fps, indices in fps_to_indices.items()
            if indices
        }
        if not self.fps_to_indices:
            raise ValueError("fps_to_indices has no non-empty buckets")
        self.batch_size = int(batch_size)
        self.curriculum_ramp = curriculum_ramp
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, ep: int) -> None:
        self.epoch = int(ep)

    def _weights_for_epoch(self) -> dict[float, float]:
        weights = self.curriculum_ramp(self.epoch)
        out: dict[float, float] = {}
        for fps in self.fps_to_indices:
            w = float(weights.get(fps, 1.0))
            out[fps] = max(0.0, w)
        if all(v <= 0.0 for v in out.values()):
            for fps in out:
                out[fps] = 1.0
        return out

    def _batches_per_epoch(self, weights: dict[float, float]) -> int:
        # Equal-exposure baseline by category (not by dataset size), then scaled by weights.
        base_batches = max(1, min(len(v) // self.batch_size for v in self.fps_to_indices.values()))
        active = sum(1 for v in weights.values() if v > 0)
        return max(active, base_batches * active)

    def __len__(self) -> int:
        weights = self._weights_for_epoch()
        return self._batches_per_epoch(weights)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        weights = self._weights_for_epoch()
        fps_values = sorted(self.fps_to_indices.keys())

        # Pre-shuffle bucket pools; wrap-around sampling allows long epochs.
        pools: dict[float, list[int]] = {}
        cursors: dict[float, int] = {}
        for fps, indices in self.fps_to_indices.items():
            items = list(indices)
            rng.shuffle(items)
            pools[fps] = items
            cursors[fps] = 0

        total_batches = self._batches_per_epoch(weights)
        active_fps = [fps for fps in fps_values if weights[fps] > 0]
        if not active_fps:
            active_fps = fps_values

        probs = [weights[fps] for fps in active_fps]
        prob_sum = sum(probs)
        probs = [p / prob_sum for p in probs]

        for _ in range(total_batches):
            fps = rng.choices(active_fps, weights=probs, k=1)[0]
            batch: list[int] = []
            while len(batch) < self.batch_size:
                cursor = cursors[fps]
                pool = pools[fps]
                if cursor >= len(pool):
                    rng.shuffle(pool)
                    cursor = 0
                batch.append(pool[cursor])
                cursor += 1
                cursors[fps] = cursor
            yield batch
