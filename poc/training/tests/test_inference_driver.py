"""Tests for v3.0 daemon-cache inference driver."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from poc.training.inference_driver import load_v30_checkpoint, run_inference
from poc.training.trainer import TinyTrainer, TrainConfig


def _make_daemon_cache(path: Path, n_sessions: int = 3, n_frames: int = 90) -> Path:
    rng = np.random.RandomState(7)
    cache: dict[str, np.ndarray] = {}
    for i in range(n_sessions):
        base = rng.uniform(0.05, 0.25, size=(n_frames, 20)).astype(np.float32)
        if i == 0:
            base[:, 0] += 0.5
            base[:, 4] += 0.3
        elif i == 1:
            base[:, 2] += 0.4
        else:
            base[:, 10] += 0.35
        cache[f"au_s{i:02d}"] = np.clip(base, 0.0, 1.0).astype(np.float32)
    np.savez_compressed(path, **cache)
    return path


def _train_and_save_ckpt(cache_path: Path, ckpt_path: Path) -> None:
    from poc.training.daemon_dataset import DaemonAUDataset

    ds = DaemonAUDataset(cache_path)
    cfg = TrainConfig(
        seq_len=30,
        batch_size=2,
        hidden_dim=16,
        num_layers=1,
        num_personas=4,
        freeze_f_net_steps=0,
        log_every_n=2,
        seed=11,
        use_ode_loss=True,
        use_vicreg=False,
    )
    trainer = TinyTrainer(cfg=cfg, dataset=ds, device="cpu")
    train_out = trainer.train_one_epoch(epoch_idx=0)
    assert train_out["total_steps"] > 0
    trainer.save_checkpoint(ckpt_path)


def test_run_inference_smoke_end_to_end():
    """Train 1 epoch -> save ckpt -> load -> evaluate daemon cache -> write JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cache_path = _make_daemon_cache(tmpdir / "daemon_au_cache.npz", n_sessions=3, n_frames=90)
        ckpt_path = tmpdir / "v30_mock_ckpt.pt"
        out_path = tmpdir / "eval.json"

        _train_and_save_ckpt(cache_path, ckpt_path)
        result = run_inference(
            ckpt_path=ckpt_path,
            cache_path=cache_path,
            out_path=out_path,
            sample_per_video=2,
            seq_len=30,
        )

        assert out_path.exists()
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["ckpt_path"] == str(ckpt_path)
        assert payload["cache_path"] == str(cache_path)
        assert payload["n_sessions"] == 3
        assert payload["total_frames"] == 270
        assert isinstance(payload["elapsed_s"], float)

        gates = payload["gates_summary"]
        expected_keys = {
            "gate_1_tau_std",
            "gate_2_vs_persistence",
            "gate_3_effective_dim_ratio",
            "gate_4_shuffle_ratio",
            "gate_5_trivial_recon_check",
            "gate_6_rare_au_verdict",
        }
        assert set(gates.keys()) == expected_keys
        assert gates["gate_1_tau_std"].startswith("N/A")

        assert "eval_result" in payload
        assert "rare_au_verdict" in payload["eval_result"]
        assert result["n_sessions"] == 3


def test_load_v30_checkpoint_uses_default_cfg_when_missing():
    """Loader should recover when checkpoint cfg is absent and enforce ODE mode."""
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        cache_path = _make_daemon_cache(tmpdir / "cache.npz", n_sessions=1, n_frames=30)
        from poc.training.daemon_dataset import DaemonAUDataset

        ds = DaemonAUDataset(cache_path)
        src_cfg = TrainConfig(seq_len=30, hidden_dim=8, num_layers=1, use_ode_loss=True)
        src = TinyTrainer(cfg=src_cfg, dataset=ds, device="cpu")

        ckpt = tmpdir / "missing_cfg.pt"
        torch.save(
            {
                "fusion_state": src.fusion.state_dict(),
                "backbone_state": src.backbone.state_dict(),
            },
            ckpt,
        )

        restored = load_v30_checkpoint(ckpt, device="cpu")
        assert restored.cfg.use_ode_loss is True
        assert restored.cfg.seq_len == 30
        assert next(restored.fusion.parameters()).device.type == "cpu"
