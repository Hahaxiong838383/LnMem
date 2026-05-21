"""测试 TinyTrainer 端到端 dry run（用 mock VEATIC，不依赖真 py-feat 抽取）

验证 5/22 早跑真训练前 pipeline 全栈无 bug.
"""
import tempfile
from pathlib import Path

import numpy as np
import torch

from poc.training.trainer import TinyTrainer, TrainConfig
from poc.training.dataset import VEATICDataset, build_event_tensor


def _create_mock_veatic(tmpdir: Path, n_videos: int = 5, n_frames: int = 60):
    """造 mock VEATIC 数据：AU cache + rating csv."""
    rating_dir = tmpdir / "rating_averaged"
    rating_dir.mkdir(parents=True, exist_ok=True)

    cache_data = {}
    for v in range(n_videos):
        np.random.seed(v)
        # 每视频不同 AU 模式（模拟不同面部活动）
        au = np.random.rand(n_frames, 24).astype(np.float32) * 0.5
        # 视频 0/1: 高 AU06 (smile); 视频 2/3: 高 AU04 (frown); 视频 4: AU01 (surprise)
        if v < 2:
            au[:, 4] += 0.4  # AU06 idx
            au[:, 9] += 0.3  # AU12 idx
        elif v < 4:
            au[:, 2] += 0.4  # AU04 idx
        else:
            au[:, 0] += 0.5  # AU01 idx
        au = np.clip(au, 0, 1)
        cache_data[f"au_{v}"] = au

        # 对应 V-A: 视频 0/1 偏正 valence，2/3 偏负，4 中性
        valence_base = [0.7, 0.7, -0.6, -0.6, 0.0][v]
        arousal_base = [0.5, 0.5, 0.5, 0.5, 0.7][v]
        valence = np.full(n_frames, valence_base) + np.random.randn(n_frames) * 0.1
        arousal = np.full(n_frames, arousal_base) + np.random.randn(n_frames) * 0.1

        import pandas as pd
        pd.DataFrame({
            "frame_id": np.arange(n_frames),
            "value": valence,
        }).to_csv(rating_dir / f"{v}_valence.csv", header=False, index=False)
        pd.DataFrame({
            "frame_id": np.arange(n_frames),
            "value": arousal,
        }).to_csv(rating_dir / f"{v}_arousal.csv", header=False, index=False)

    npz_path = tmpdir / "au_cache.npz"
    np.savez_compressed(npz_path, **cache_data)
    return npz_path, rating_dir


def test_trainer_dry_run_smoke():
    """端到端 5 steps mock training: 无 NaN + collapse 不触发 + grad ratio 报告"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=60)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        assert len(dataset) == 5

        cfg = TrainConfig(
            seq_len=20,
            batch_size=4,
            num_negs=4,
            hidden_dim=32,        # 小一点跑快
            num_layers=2,
            num_personas=4,
            epochs=1,
            lr=1e-3,
            persona_lr_scale=5.0,
            f_net_weight_decay=1e-3,
            freeze_f_net_steps=2,  # smoke 阶段早 unfreeze
            use_vicreg=False,
            use_breakpoint_mask=True,
            log_every_n=1,
            seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch()

        # 至少跑了几个 step
        assert result["total_steps"] > 0
        # 无 NaN
        assert all(np.isfinite(log["loss"]) for log in result["logs"])
        # 最后 loss 不是 inf
        if result["final_loss"]:
            assert np.isfinite(result["final_loss"])
        print(f"\n  dry-run: {result['total_steps']} steps, "
              f"final_loss={result['final_loss']:.4f}, "
              f"elapsed={result['elapsed_s']:.1f}s")


def test_trainer_dry_run_vicreg():
    """VICReg 路径 dry run"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=60)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20, batch_size=4, num_negs=4, hidden_dim=32,
            num_layers=2, num_personas=4, epochs=1, lr=1e-3,
            freeze_f_net_steps=2, use_vicreg=True,
            log_every_n=1, seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch()
        assert result["total_steps"] > 0
        assert all(np.isfinite(log["loss"]) for log in result["logs"])
        print(f"\n  VICReg dry-run: {result['total_steps']} steps, "
              f"final_loss={result['final_loss']:.4f}")


def test_trainer_checkpoint_under_20mb():
    """spec §1.2 ckpt ≤ 20MB"""
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=60)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20, batch_size=4, num_negs=4, hidden_dim=64,
            num_layers=2, num_personas=64,  # 真 persona table
            epochs=1, lr=1e-3, freeze_f_net_steps=2,
            log_every_n=10, seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        result = trainer.train_one_epoch()
        ckpt_path = tmp / "test_ckpt.pt"
        trainer.save_checkpoint(ckpt_path)
        size_mb = ckpt_path.stat().st_size / 1e6
        print(f"\n  ckpt size: {size_mb:.2f} MB")
        assert size_mb < 20.0


def test_full_evaluation_pipeline():
    """跑完整 evaluation: M1-V + M5 + collapse"""
    from poc.training.evaluate import run_evaluation

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        # 用 200 帧 / 5 视频 → 每视频可切 ~10 个窗口
        npz, ratings = _create_mock_veatic(tmp, n_videos=5, n_frames=200)
        dataset = VEATICDataset(npz, ratings, seq_len=20)
        cfg = TrainConfig(
            seq_len=20, batch_size=4, num_negs=4, hidden_dim=32,
            num_layers=2, num_personas=4, epochs=1, lr=1e-3,
            freeze_f_net_steps=2, log_every_n=10, seed=42,
        )
        trainer = TinyTrainer(cfg, dataset, device="cpu")
        trainer.train_one_epoch()

        # 每视频抽 10 样本 → 50 个总样本 → test 10 个 ≥ n_classes=5 ✓
        eval_result = run_evaluation(trainer, dataset, sample_per_video=10)
        print(f"\n  eval samples: {eval_result['n_samples']}")
        print(f"  M5 verdict: {eval_result['M5_video_id_shortcut']['verdict']} "
              f"(test_acc={eval_result['M5_video_id_shortcut']['test_acc']:.3f})")
        print(f"  M1 full top5: {eval_result['M1_full_retrieval']['top5_acc']:.3f}")
        print(f"  M1-V top5: {eval_result['M1V_visual_only_retrieval']['top5_acc']:.3f}")
        print(f"  M1-V pass: {eval_result['M1V_pass']}")
        print(f"  collapse var: {eval_result['collapse_var']:.4f}")

        # 至少 sanity check: 返回结构完整
        assert "M5_video_id_shortcut" in eval_result
        assert "M1V_visual_only_retrieval" in eval_result
        assert eval_result["collapse_var"] > 0  # 至少非 0
