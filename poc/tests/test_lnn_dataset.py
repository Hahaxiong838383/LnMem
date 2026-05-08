"""Day 5 LNN dataset tests — JsonlSignalsLoader shape/split/boundary/error handling."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from poc.src.lnn.contracts import LNNDataError, LNNTrainConfig
from poc.src.lnn.dataset import JsonlSignalsLoader, _STATE_TO_IDX


def _make_jsonl_file(
    tmp_dir: Path,
    persona: str,
    day: int,
    n_frames: int,
    *,
    event_len: int = 30,
) -> Path:
    """Create a minimal but valid signals JSONL file."""
    path = tmp_dir / f"signals_{persona}_day{day}.jsonl"
    with path.open("w") as f:
        states = list(_STATE_TO_IDX.keys())
        event_idx = 0
        for i in range(n_frames):
            frame_in_event = i % event_len
            if frame_in_event == 0 and i > 0:
                event_idx += 1
            row = {
                "ts": f"2026-01-01T09:00:{i:02d}",
                "frame_idx": frame_in_event,
                "vector": [0.1 * (j + 1) + 0.001 * i for j in range(12)],
                "missing": [],
                "ground_truth_state": states[i % len(states)],
                "activity": "commute",
                "persona_id": persona,
                "source_event_id": f"{persona}_d{day}_e{event_idx:03d}",
                "day_index": day,
            }
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> Path:
    """Create a small dataset with 3 personas × 2 days."""
    for p in ["vboss_001", "vboss_002", "vboss_003"]:
        for d in range(2):
            _make_jsonl_file(tmp_path, p, d, n_frames=120)
    return tmp_path


def test_persona_to_idx_deterministic(tiny_dataset: Path) -> None:
    """persona_to_idx must be sorted by persona_id and deterministic."""
    cfg = LNNTrainConfig(
        signals_dir=tiny_dataset,
        sequence_length=30,
        sequence_stride=15,
        batch_size=2,
    )
    loader = JsonlSignalsLoader()
    _, _, p2i_a = loader.build_loaders(cfg)
    _, _, p2i_b = loader.build_loaders(cfg)

    assert p2i_a == p2i_b, "persona_to_idx not deterministic"
    keys = list(p2i_a.keys())
    assert keys == sorted(keys), "persona_to_idx not sorted"
    assert p2i_a == {"vboss_001": 0, "vboss_002": 1, "vboss_003": 2}


def test_batch_shapes(tiny_dataset: Path) -> None:
    """Verify (x, state, persona_idx, boundary) tensor shapes."""
    seq_len = 30
    cfg = LNNTrainConfig(
        signals_dir=tiny_dataset,
        sequence_length=seq_len,
        sequence_stride=15,
        batch_size=2,
    )
    loader = JsonlSignalsLoader()
    train_dl, _, p2i = loader.build_loaders(cfg)

    batch = next(iter(train_dl))
    x, state, pidx, boundary = batch

    assert x.shape == (2, seq_len, 12), f"x shape: {x.shape}"
    assert x.dtype == torch.float32
    assert state.shape == (2, seq_len), f"state shape: {state.shape}"
    assert state.dtype == torch.long
    assert pidx.shape == (2,), f"pidx shape: {pidx.shape}"
    assert pidx.dtype == torch.long
    assert boundary.shape == (2, seq_len), f"boundary shape: {boundary.shape}"
    assert boundary.dtype == torch.float32


def test_state_range(tiny_dataset: Path) -> None:
    """State indices must be in [0, 8)."""
    cfg = LNNTrainConfig(
        signals_dir=tiny_dataset,
        sequence_length=30,
        sequence_stride=15,
        batch_size=4,
    )
    loader = JsonlSignalsLoader()
    train_dl, val_dl, _ = loader.build_loaders(cfg)

    for dl in (train_dl, val_dl):
        for batch in dl:
            _, state, _, _ = batch
            assert state.min() >= 0, f"state min: {state.min()}"
            assert state.max() < 8, f"state max: {state.max()}"


def test_boundary_correctness(tmp_path: Path) -> None:
    """Boundary marks should be 1.0 at frame_idx==0 frames."""
    # Create enough files for split to work (need >= 2 keys for 80/20 split)
    for d in range(3):
        _make_jsonl_file(tmp_path, "vboss_001", d, n_frames=60, event_len=30)

    cfg = LNNTrainConfig(
        signals_dir=tmp_path,
        sequence_length=60,
        sequence_stride=30,
        batch_size=1,
        train_val_split=0.99,  # All to train
    )
    loader = JsonlSignalsLoader()
    train_dl, _, _ = loader.build_loaders(cfg)

    batch = next(iter(train_dl))
    _, _, _, boundary = batch
    boundary_flat = boundary.squeeze(0)  # (60,)

    # frame 0 = boundary (start of event 0)
    assert boundary_flat[0].item() == 1.0
    # frame 30 = boundary (start of event 1, frame_idx resets to 0)
    assert boundary_flat[30].item() == 1.0
    # frame 1 = not boundary
    assert boundary_flat[1].item() == 0.0
    # frame 29 = not boundary
    assert boundary_flat[29].item() == 0.0


def test_short_file_skipped(tmp_path: Path) -> None:
    """Files with fewer frames than sequence_min_frames should be skipped."""
    # Create 3 files for vboss_001 to ensure split works
    for d in range(3):
        _make_jsonl_file(tmp_path, "vboss_001", d, n_frames=120)
    _make_jsonl_file(tmp_path, "vboss_002", 0, n_frames=10)  # Too short

    cfg = LNNTrainConfig(
        signals_dir=tmp_path,
        sequence_length=30,
        sequence_stride=15,
        batch_size=1,
        sequence_min_frames=30,
    )
    loader = JsonlSignalsLoader()
    _, _, p2i = loader.build_loaders(cfg)

    # vboss_002 should be skipped (10 < 30)
    assert "vboss_002" not in p2i
    assert "vboss_001" in p2i


def test_json_parse_error(tmp_path: Path) -> None:
    """Invalid JSON should raise LNNDataError when loading."""
    # Create a bad file with invalid JSON on line 1
    path = tmp_path / "signals_vboss_001_day0.jsonl"
    with path.open("w") as f:
        f.write('{"vector": [0.1], "this is not valid\n')
        for i in range(60):
            row = {
                "ts": f"2026-01-01T09:00:{i:02d}",
                "frame_idx": i,
                "vector": [0.1] * 12,
                "missing": [],
                "ground_truth_state": "neutral",
                "activity": "commute",
                "persona_id": "vboss_001",
                "source_event_id": "vboss_001_d0_e000",
                "day_index": 0,
            }
            f.write(json.dumps(row) + "\n")

    # Also create valid files for split to work
    for d in range(1, 4):
        _make_jsonl_file(tmp_path, "vboss_001", d, n_frames=120)

    cfg = LNNTrainConfig(
        signals_dir=tmp_path,
        sequence_length=30,
        sequence_stride=15,
        batch_size=1,
        train_val_split=0.99,  # Force all to train so bad file gets loaded
    )
    loader = JsonlSignalsLoader()

    with pytest.raises(LNNDataError, match="JSON parse error"):
        loader.build_loaders(cfg)


def test_empty_dir_raises(tmp_path: Path) -> None:
    """Empty signals directory should raise LNNDataError."""
    cfg = LNNTrainConfig(
        signals_dir=tmp_path,
        sequence_length=30,
        sequence_stride=15,
        batch_size=1,
    )
    loader = JsonlSignalsLoader()
    with pytest.raises(LNNDataError, match="No valid JSONL"):
        loader.build_loaders(cfg)


def test_train_val_split_ratio(tiny_dataset: Path) -> None:
    """80/20 split: 6 files → ~5 train + ~1 val (by persona×day key)."""
    cfg = LNNTrainConfig(
        signals_dir=tiny_dataset,
        sequence_length=30,
        sequence_stride=15,
        batch_size=1,
        train_val_split=0.8,
    )
    loader = JsonlSignalsLoader()
    train_dl, val_dl, _ = loader.build_loaders(cfg)

    # Both should have data
    assert len(train_dl) > 0
    assert len(val_dl) > 0
    # Train should have more windows than val
    assert len(train_dl) > len(val_dl)
