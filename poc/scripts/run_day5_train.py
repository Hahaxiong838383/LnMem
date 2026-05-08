#!/usr/bin/env python3
"""Day 5 LNN training script — MPS acceleration, practical batch size.

Usage:
    cd /tmp/lnmem-fix/LnMem
    PYTHONPATH=. .venv-poc/bin/python -u poc/scripts/run_day5_train.py
"""
from poc.src.lnn.train import train
from poc.src.lnn.contracts import LNNTrainConfig

cfg = LNNTrainConfig(
    epochs=20,
    batch_size=32,
    sequence_length=60,
    sequence_stride=600,   # 10min stride → ~1K batches/epoch (~85s/epoch on MPS)
    lr=1e-3,
    device="mps",
)
print(f"[run_day5] config: epochs={cfg.epochs} batch={cfg.batch_size} "
      f"stride={cfg.sequence_stride} device={cfg.device}", flush=True)

ckpt = train(cfg)
print(f"\n[RESULT] final_state_acc={ckpt.final_state_acc:.4f}", flush=True)
print(f"[RESULT] final_boundary_f1={ckpt.final_boundary_f1:.4f}", flush=True)
print("[DONE]", flush=True)
