"""PC-2: DataLoader shuffle/reseed tests."""
from __future__ import annotations

import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from poc.training.dataset import VEATICDataset
from poc.training.tests.test_trainer_dry_run import _create_mock_veatic
from poc.training.trainer import TinyTrainer, TrainConfig
from poc.training.window_dataset import WindowDataset


def _first_batch_vids(windows: list[dict], seed: int, batch_size: int = 12) -> list[str]:
    ds = WindowDataset(windows)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda x: x,
    )
    batch = next(iter(loader))
    return [str(x["video_id"]) for x in batch]


def test_first_batch_sequence_differs_across_epochs() -> None:
    """Consecutive epoch seeds should not produce identical first-batch order."""
    windows = [
        {"video_id": f"v{i:03d}", "event_tensor": torch.randn(30, 52)}
        for i in range(120)
    ]
    seq0 = _first_batch_vids(windows, seed=42)
    seq1 = _first_batch_vids(windows, seed=43)
    seq2 = _first_batch_vids(windows, seed=44)
    assert seq0 != seq1
    assert seq1 != seq2
    assert seq0 != seq2


def test_trainer_records_epoch_first_batch_vids() -> None:
    """Trainer should expose epoch-level first batch vid sequence for diagnostics."""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=120)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            num_negs=4,
            hidden_dim=32,
            num_layers=2,
            num_personas=4,
            freeze_f_net_steps=0,
            log_every_n=2,
            seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        ep0 = trainer.train_one_epoch(epoch_idx=0)["epoch_log"]["first_batch_vids"]
        ep1 = trainer.train_one_epoch(epoch_idx=1)["epoch_log"]["first_batch_vids"]

    assert ep0 is not None and ep1 is not None
    assert ep0 != ep1
