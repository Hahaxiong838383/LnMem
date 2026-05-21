"""γ-3 完整端到端运行脚本（spec §8 γ-3 + §6.1 评估）

Usage:
    # 已有 au_cache_tiny.npz 后跑：
    HF_HUB_OFFLINE=1 PYTHONPATH=. python -m poc.training.run_tiny_training \\
        --au-cache poc/data/au_cache_tiny.npz \\
        --rating-dir /Users/fu-121/lnmem-data/veatic/veatic_extracted/rating_averaged \\
        --ckpt-out poc/data/checkpoints/base_v3_spike.pt \\
        --loss infonce  # or vicreg

输出:
    1. ckpt: poc/data/checkpoints/base_v3_spike.pt
    2. 训练 log: poc/data/checkpoints/training_logs.json
    3. evaluation 报告: poc/data/checkpoints/eval_report.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from .dataset import VEATICDataset
from .trainer import TinyTrainer, TrainConfig
from .evaluate import run_evaluation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--au-cache", required=True)
    p.add_argument("--rating-dir", required=True)
    p.add_argument("--ckpt-out", default="poc/data/checkpoints/base_v3_spike.pt")
    p.add_argument("--loss", choices=["infonce", "vicreg"], default="infonce")
    p.add_argument("--seq-len", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-breakpoint-mask", action="store_true",
                   help="禁用 gemini 终审条件 1 时序截断（默认开启 InfoNCE 才用）")
    args = p.parse_args()

    print(f"=" * 70)
    print(f"γ-3 tiny VEATIC 1 epoch 训练 + 评估")
    print(f"=" * 70)
    print(f"  AU cache: {args.au_cache}")
    print(f"  rating dir: {args.rating_dir}")
    print(f"  loss: {args.loss.upper()}")
    print(f"  seed: {args.seed}")
    print()

    # Step 1: 加载数据
    dataset = VEATICDataset(args.au_cache, args.rating_dir, seq_len=args.seq_len)
    print(f"✅ 数据集: {len(dataset)} 视频")
    for i in range(min(3, len(dataset))):
        sample = dataset[i]
        et = sample["event_tensor"]
        print(f"  video {sample['video_id']}: {et.shape}, AU mean={et[:,:24].mean():.3f}")

    # Step 2: 训练
    cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_negs=8,
        hidden_dim=args.hidden_dim,
        num_layers=2,
        num_personas=4,
        epochs=args.epochs,
        lr=args.lr,
        persona_lr_scale=1.0,         # gemini F-4.1 起点 5x → 实测 bias_grad_ratio 6.4e8 反向 broken
                                       # 改 1x 中性起点；真训练时根据实测 ratio 再调
        f_net_weight_decay=1e-3,      # gemini F-4.1
        freeze_f_net_steps=20,        # tiny spike: 20 step（spec 1k step 简化）
        use_vicreg=args.loss == "vicreg",
        use_breakpoint_mask=not args.no_breakpoint_mask,
        log_every_n=5,
        seed=args.seed,
    )
    trainer = TinyTrainer(cfg, dataset, device="cpu")
    print(f"\n=== 训练 {args.loss.upper()} ===")
    t0 = time.time()
    train_result = trainer.train_one_epoch()
    train_elapsed = time.time() - t0

    print(f"\n✅ 训练完成: {train_result['total_steps']} steps, "
          f"final_loss={train_result['final_loss']:.4f}, "
          f"elapsed={train_elapsed:.1f}s")

    # Step 3: Save ckpt
    ckpt_path = Path(args.ckpt_out)
    trainer.save_checkpoint(ckpt_path)

    # Save training logs
    logs_path = ckpt_path.parent / "training_logs.json"
    with logs_path.open("w") as f:
        json.dump({
            "config": train_result.get("config", {}),
            "logs": train_result["logs"],
            "final_loss": train_result["final_loss"],
            "total_steps": train_result["total_steps"],
            "elapsed_s": train_elapsed,
            "loss_type": args.loss,
            "seed": args.seed,
        }, f, indent=2, default=lambda o: str(o))
    print(f"   training logs: {logs_path}")

    # Step 4: Evaluation (M1 + M1-V + M5 + collapse)
    print(f"\n=== 评估 ===")
    t0 = time.time()
    eval_result = run_evaluation(trainer, dataset, sample_per_video=5)
    eval_elapsed = time.time() - t0

    eval_path = ckpt_path.parent / "eval_report.json"
    with eval_path.open("w") as f:
        json.dump({
            **eval_result,
            "eval_elapsed_s": eval_elapsed,
            "loss_type": args.loss,
            "seed": args.seed,
        }, f, indent=2, default=lambda o: str(o))
    print(f"   eval report: {eval_path}")

    print(f"\n" + "=" * 70)
    print(f"📊 γ-3 spike 报告 ({args.loss.upper()}, seed={args.seed})")
    print(f"=" * 70)
    print(f"  训练: {train_result['total_steps']} steps, loss={train_result['final_loss']:.4f}")
    print(f"  ckpt size: {ckpt_path.stat().st_size / 1e6:.2f} MB")
    print()
    print(f"  M1 KG retrieval (full): top5={eval_result['M1_full_retrieval']['top5_acc']:.3f}")
    print(f"  M1-V Visual-Only: top5={eval_result['M1V_visual_only_retrieval']['top5_acc']:.3f} "
          f"[pass={eval_result['M1V_pass']} (≥0.20 required)]")
    print(f"  M5 video_id shortcut: acc={eval_result['M5_video_id_shortcut']['test_acc']:.3f} "
          f"({eval_result['M5_video_id_shortcut']['verdict']})")
    print(f"  collapse: var={eval_result['collapse_var']:.4f} "
          f"[pass={eval_result['collapse_var_pass']} (>0.01)]")
    print(f"  pairwise dist: {eval_result['collapse_pairwise_dist']:.4f}")
    print()

    # spec §11.2 验收
    pass_count = 0
    pass_count += int(eval_result["M1V_pass"])
    pass_count += int(eval_result["M5_video_id_shortcut"]["verdict"] == "green")
    pass_count += int(eval_result["collapse_var_pass"])
    print(f"📋 spec §11.2 验收: {pass_count}/3 (M1-V / M5 green / collapse var)")
    if pass_count == 3:
        print(f"  ✅ ALL PASS — Phase γ Step 1 启动门槛达成")
    elif pass_count >= 2:
        print(f"  ⚠️ PARTIAL — 需修订后重跑")
    else:
        print(f"  ❌ FAIL — 触发 spec §9 风险表 rollback")


if __name__ == "__main__":
    main()
