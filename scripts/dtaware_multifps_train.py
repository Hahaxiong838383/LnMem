"""Phase-2 multi-FPS curriculum dry-run entrypoint."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.training.daemon_dataset import DaemonAUDataset
from poc.training.dataset import VEATICDataset, split_to_windows
from poc.training.evaluate import run_evaluation
from poc.training.multifps_dataset import MultiFpsDataset, VEATICDownsampled
from poc.training.multifps_sampler import EquallyWeightedFpsBucketSampler
from poc.training.trainer import TinyTrainer, TrainConfig


class WindowedDataset(Dataset):
    """Fixed-length windows for curriculum batches."""

    def __init__(self, base: Dataset, seq_len: int) -> None:
        self.items: list[dict] = []
        for i in range(len(base)):
            sample = base[i]
            windows = split_to_windows(sample, seq_len=seq_len, stride=seq_len)
            for w in windows:
                self.items.append(
                    {
                        "video_id": str(w["video_id"]),
                        "event_tensor": w["event_tensor"].clone().contiguous(),
                        "labels_va": w["labels_va"].clone().contiguous(),
                        "start_frame": int(w["start_frame"]),
                        "fps": float(w.get("fps", sample.get("fps", 30.0))),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]


def _write_mock_veatic(root: Path, n_videos: int = 4, n_frames: int = 360) -> tuple[Path, Path]:
    rating_dir = root / "rating_averaged"
    rating_dir.mkdir(parents=True, exist_ok=True)
    cache: dict[str, np.ndarray] = {}
    for vid in range(n_videos):
        rng = np.random.default_rng(1000 + vid)
        au = rng.random((n_frames, 24), dtype=np.float32)
        cache[f"au_{vid}"] = au
        valence = np.clip(rng.normal(loc=0.1 * (vid - 1), scale=0.2, size=n_frames), -1.0, 1.0)
        arousal = np.clip(rng.normal(loc=0.3, scale=0.2, size=n_frames), -1.0, 1.0)
        np.savetxt(
            rating_dir / f"{vid}_valence.csv",
            np.column_stack([np.arange(n_frames), valence]),
            fmt=["%d", "%.6f"],
            delimiter=",",
        )
        np.savetxt(
            rating_dir / f"{vid}_arousal.csv",
            np.column_stack([np.arange(n_frames), arousal]),
            fmt=["%d", "%.6f"],
            delimiter=",",
        )
    cache_path = root / "au_cache.npz"
    np.savez_compressed(cache_path, **cache)
    return cache_path, rating_dir


def _write_mock_daemon_cache(root: Path, n_sessions: int = 3, n_frames: int = 180) -> Path:
    cache: dict[str, np.ndarray] = {}
    for i in range(n_sessions):
        rng = np.random.default_rng(2000 + i)
        cache[f"au_s{i}"] = rng.random((n_frames, 20), dtype=np.float32)
    out = root / "daemon_cache.npz"
    np.savez_compressed(out, **cache)
    return out


def _build_curriculum_ramp(
    fps_keys: list[float],
    ramp_start: int,
    ramp_end: int,
    daemon_target_prob: float,
):
    daemon_bucket = min(fps_keys, key=lambda x: abs(x - 1.0))
    other_keys = [k for k in fps_keys if k != daemon_bucket]

    def _ratio(epoch: int) -> float:
        if epoch < ramp_start:
            return 0.0
        if epoch >= ramp_end:
            return daemon_target_prob
        span = max(1, ramp_end - ramp_start)
        return daemon_target_prob * ((epoch - ramp_start) / span)

    def _ramp(epoch: int) -> dict[float, float]:
        daemon_ratio = _ratio(epoch)
        weights = {k: 1.0 for k in fps_keys}
        if not other_keys or daemon_ratio <= 0.0:
            weights[daemon_bucket] = 0.0 if daemon_ratio <= 0.0 else 1.0
            return weights
        daemon_weight = (daemon_ratio / max(1e-6, 1.0 - daemon_ratio)) * len(other_keys)
        weights[daemon_bucket] = max(0.0, daemon_weight)
        return weights

    return _ramp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--au-cache")
    parser.add_argument("--rating-dir")
    parser.add_argument("--daemon-au-cache")
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup-ep", type=int, default=50)
    parser.add_argument("--ramp-start", type=int, default=40)
    parser.add_argument("--ramp-end", type=int, default=60)
    parser.add_argument("--total-ep", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-json", default="/tmp/dtaware_multifps_dryrun.json")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        if args.au_cache and args.rating_dir:
            au_cache = Path(args.au_cache)
            rating_dir = Path(args.rating_dir)
        else:
            au_cache, rating_dir = _write_mock_veatic(tmp_root)

        if args.daemon_au_cache:
            daemon_cache = Path(args.daemon_au_cache)
        else:
            daemon_cache = _write_mock_daemon_cache(tmp_root)

        veatic = VEATICDataset(au_cache, rating_dir, seq_len=args.seq_len)
        daemon = DaemonAUDataset(daemon_cache)
        # cc 修补：VEATIC 30fps × ~5s → 1Hz 仅 ~5 帧，min_seq_len 不能 >> 物理上限
        veatic_5hz = VEATICDownsampled(veatic, target_fps=5.0, min_seq_len=min(8, args.seq_len))
        veatic_1hz = VEATICDownsampled(veatic, target_fps=1.0, min_seq_len=min(4, args.seq_len))

        # cc 修补 (5/23 epoch 2 崩盘后)：sampler 按 fps 分桶但同 fps bucket 含不同 seq_len → torch.stack 失败
        # 修法：每个 fps 只用单一 seq_len 源
        # - 30fps: veatic seq_len=args.seq_len（主源）
        # - 5fps:  veatic@5Hz seq_len=8（gemini F3 钦定 L≥8 下界）
        # - 1fps:  daemon seq_len=args.seq_len（真实数据，丢 veatic@1Hz 避 stack 失败 + gemini F3 警告语义虚无）
        # 注：veatic_1hz / veatic_5hz Dataset 仍构造（保持 codex 已有 pipeline），仅 multi_ds 只挂 1 个 1Hz 源
        veatic_win = WindowedDataset(veatic, seq_len=args.seq_len)
        daemon_win = WindowedDataset(daemon, seq_len=args.seq_len)
        veatic5_win = WindowedDataset(veatic_5hz, seq_len=min(8, args.seq_len))

        multi_ds = MultiFpsDataset(
            [
                (veatic_win, 30.0),
                (daemon_win, 1.0),
                (veatic5_win, 5.0),
            ]
        )
        fps_to_indices = multi_ds.get_fps_to_indices()
        ramp = _build_curriculum_ramp(
            fps_keys=sorted(fps_to_indices.keys()),
            ramp_start=args.ramp_start,
            ramp_end=args.ramp_end,
            daemon_target_prob=0.3,
        )
        sampler = EquallyWeightedFpsBucketSampler(
            fps_to_indices=fps_to_indices,
            batch_size=args.batch_size,
            curriculum_ramp=ramp,
            seed=args.seed,
        )
        loader = DataLoader(multi_ds, batch_sampler=sampler, collate_fn=lambda x: x)

        cfg = TrainConfig(
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            hidden_dim=32,
            num_layers=1,
            num_personas=4,
            epochs=args.total_ep,
            lr=1e-3,
            freeze_f_net_steps=max(0, args.warmup_ep),
            use_ode_loss=True,
            use_vicreg=False,
            log_every_n=5,
            seed=args.seed,
            tau_upper_bound=30.0,
            invariance_weight=0.1,
        )
        trainer = TinyTrainer(cfg=cfg, dataset=veatic, device="cpu")

        train_logs: list[dict] = []
        last_eval: dict = {}
        for ep in range(args.total_ep):
            ep_out = trainer.train_one_epoch(epoch_idx=ep, dataloader=loader)
            train_logs.append(ep_out["epoch_log"])
            last_eval = run_evaluation(trainer=trainer, dataset=multi_ds, sample_per_video=1)

        summary = {
            "config": {
                "warmup_ep": args.warmup_ep,
                "ramp_start": args.ramp_start,
                "ramp_end": args.ramp_end,
                "total_ep": args.total_ep,
            },
            "n_samples": len(multi_ds),
            "fps_buckets": sorted(fps_to_indices.keys()),
            "train_epoch_logs": train_logs,
            "tau_clamp_count": int(train_logs[-1].get("tau_clamp_count", 0)) if train_logs else 0,
            "per_fps_summary": last_eval.get("per_fps_summary", {}),
            "variance_ratio": last_eval.get("variance_ratio"),
            "variance_aware_fail_reason": last_eval.get("variance_aware_fail_reason", ""),
            "gates_summary": last_eval.get("gates_summary", {}),
        }
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
