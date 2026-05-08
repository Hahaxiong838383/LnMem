"""Day 5 LNN end-to-end training entry point.

集成 dataset.py (gemini) + model.py (codex) + viz.py (gemini)，
由 cc 主线统一编排训练 pipeline。

Usage::

    cd /tmp/lnmem-fix/LnMem
    PYTHONPATH=. .venv-poc/bin/python -m poc.src.lnn.train
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn

from poc.src.lnn.contracts import EpochMetric, LNNCheckpoint, LNNTrainConfig
from poc.src.lnn.dataset import JsonlSignalsLoader
from poc.src.lnn.model import DefaultLNNFactory
from poc.src.lnn.viz import MatplotlibViz


def _train_one_epoch(
    state_model: nn.Module,
    boundary_model: nn.Module,
    train_dl: "torch.utils.data.DataLoader",
    state_criterion: nn.Module,
    boundary_criterion: nn.Module,
    state_optim: torch.optim.Optimizer,
    boundary_optim: torch.optim.Optimizer,
    cfg: LNNTrainConfig,
    device: str,
) -> tuple[float, float]:
    """Train both models for one epoch. Returns (avg_loss, state_accuracy)."""
    state_model.train()
    boundary_model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for batch in train_dl:
        x, state, pidx, boundary = batch
        x = x.to(device)
        state = state.to(device)
        pidx = pidx.to(device)
        boundary = boundary.to(device)

        # --- State recognition ---
        state_optim.zero_grad()
        state_logits = state_model(x, pidx)  # (B, T, 8)
        s_loss = state_criterion(
            state_logits.reshape(-1, cfg.n_state_classes),
            state.reshape(-1),
        )
        s_loss.backward()
        nn.utils.clip_grad_norm_(state_model.parameters(), cfg.grad_clip_norm)
        state_optim.step()

        # --- Boundary detection ---
        boundary_optim.zero_grad()
        b_logits = boundary_model(x, pidx)  # (B, T, 1)
        b_loss = boundary_criterion(b_logits.squeeze(-1), boundary)
        b_loss.backward()
        nn.utils.clip_grad_norm_(boundary_model.parameters(), cfg.grad_clip_norm)
        boundary_optim.step()

        # Accumulate metrics
        batch_loss = s_loss.item() + b_loss.item()
        loss_sum += batch_loss * x.shape[0]
        preds = state_logits.argmax(dim=-1)
        correct += (preds == state).sum().item()
        total += state.numel()

    avg_loss = loss_sum / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def _validate(
    state_model: nn.Module,
    boundary_model: nn.Module,
    val_dl: "torch.utils.data.DataLoader",
    state_criterion: nn.Module,
    boundary_criterion: nn.Module,
    cfg: LNNTrainConfig,
    device: str,
) -> tuple[float, float, float]:
    """Validate both models. Returns (avg_loss, state_accuracy, boundary_f1)."""
    state_model.eval()
    boundary_model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    tp = 0
    fp = 0
    fn = 0

    for batch in val_dl:
        x, state, pidx, boundary = batch
        x = x.to(device)
        state = state.to(device)
        pidx = pidx.to(device)
        boundary = boundary.to(device)

        state_logits = state_model(x, pidx)
        s_loss = state_criterion(
            state_logits.reshape(-1, cfg.n_state_classes),
            state.reshape(-1),
        )

        b_logits = boundary_model(x, pidx)
        b_loss = boundary_criterion(b_logits.squeeze(-1), boundary)

        loss_sum += (s_loss.item() + b_loss.item()) * x.shape[0]
        preds = state_logits.argmax(dim=-1)
        correct += (preds == state).sum().item()
        total += state.numel()

        # Boundary F1 components
        b_pred = (b_logits.squeeze(-1) > 0.0).float()
        tp += ((b_pred == 1) & (boundary == 1)).sum().item()
        fp += ((b_pred == 1) & (boundary == 0)).sum().item()
        fn += ((b_pred == 0) & (boundary == 1)).sum().item()

    avg_loss = loss_sum / max(total, 1)
    accuracy = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return avg_loss, accuracy, f1


def train(config: LNNTrainConfig | None = None) -> LNNCheckpoint:
    """Run full Day 5 LNN training pipeline."""
    cfg = config or LNNTrainConfig()
    device = cfg.resolved_device()
    torch.manual_seed(cfg.seed)
    print(
        f"[train] device={device} epochs={cfg.epochs} lr={cfg.lr} "
        f"batch={cfg.batch_size} seq_len={cfg.sequence_length}",
        flush=True,
    )

    # ---- Data ----
    loader = JsonlSignalsLoader()
    train_dl, val_dl, persona_to_idx = loader.build_loaders(cfg)
    print(
        f"[train] {len(train_dl)} train batches, {len(val_dl)} val batches, "
        f"{len(persona_to_idx)} personas",
        flush=True,
    )

    # ---- Models ----
    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg).to(device)
    boundary_model = factory.create_boundary_model(cfg).to(device)
    n_params = sum(p.numel() for p in state_model.parameters())
    n_params += sum(p.numel() for p in boundary_model.parameters())
    print(f"[train] total parameters: {n_params:,}", flush=True)

    # ---- Optimizers ----
    state_optim = torch.optim.Adam(
        state_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )
    boundary_optim = torch.optim.Adam(
        boundary_model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # ---- Loss ----
    state_criterion = nn.CrossEntropyLoss()
    boundary_criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([cfg.boundary_pos_weight], device=device),
    )

    # ---- Training loop ----
    metrics: list[EpochMetric] = []
    best_val_acc = 0.0

    for epoch in range(cfg.epochs):
        t0 = time.time()

        train_loss, train_acc = _train_one_epoch(
            state_model, boundary_model, train_dl,
            state_criterion, boundary_criterion,
            state_optim, boundary_optim,
            cfg, device,
        )
        val_loss, val_acc, val_f1 = _validate(
            state_model, boundary_model, val_dl,
            state_criterion, boundary_criterion,
            cfg, device,
        )

        elapsed = time.time() - t0
        metric = EpochMetric(
            epoch=epoch,
            train_loss=train_loss,
            val_loss=val_loss,
            train_state_acc=train_acc,
            val_state_acc=val_acc,
            val_boundary_f1=val_f1,
        )
        metrics.append(metric)

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(
            f"[epoch {epoch:3d}/{cfg.epochs}] "
            f"loss={train_loss:.4f}/{val_loss:.4f} "
            f"acc={train_acc:.3f}/{val_acc:.3f} "
            f"bnd_f1={val_f1:.3f} "
            f"({elapsed:.1f}s)",
            flush=True,
        )

        # ---- Incremental checkpoint (every 5 epochs or on improvement) ----
        if (epoch + 1) % 5 == 0 or val_acc >= best_val_acc:
            cfg.out_dir.mkdir(parents=True, exist_ok=True)
            _save_path = cfg.out_dir / "checkpoint_day5_latest.pt"
            torch.save(
                {
                    "config": cfg,
                    "persona_to_idx": persona_to_idx,
                    "state_model": state_model.state_dict(),
                    "boundary_model": boundary_model.state_dict(),
                    "epoch_metrics": tuple(metrics),
                    "final_state_acc": best_val_acc,
                    "final_boundary_f1": metric.val_boundary_f1,
                    "epoch": epoch,
                },
                _save_path,
            )

    # ---- Save final checkpoint ----
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = cfg.out_dir / "checkpoint_day5.pt"
    torch.save(
        {
            "config": cfg,
            "persona_to_idx": persona_to_idx,
            "state_model": state_model.state_dict(),
            "boundary_model": boundary_model.state_dict(),
            "epoch_metrics": tuple(metrics),
            "final_state_acc": best_val_acc,
            "final_boundary_f1": metrics[-1].val_boundary_f1 if metrics else 0.0,
        },
        ckpt_path,
    )
    print(f"[train] checkpoint saved: {ckpt_path}", flush=True)

    checkpoint = LNNCheckpoint(
        config=cfg,
        persona_to_idx=persona_to_idx,
        state_model_state_dict=state_model.state_dict(),
        boundary_model_state_dict=boundary_model.state_dict(),
        epoch_metrics=tuple(metrics),
        final_state_acc=best_val_acc,
        final_boundary_f1=metrics[-1].val_boundary_f1 if metrics else 0.0,
    )

    # ---- Viz ----
    viz = MatplotlibViz()
    viz.plot_loss_curve(tuple(metrics), cfg.out_dir / "loss_curve.png")
    viz.plot_tsne_personalization(
        state_model.personalization.embedding.weight,
        persona_to_idx,
        cfg.out_dir / "persona_tsne.png",
    )
    print(f"[train] visualizations saved: {cfg.out_dir}/", flush=True)

    return checkpoint


if __name__ == "__main__":
    train()
