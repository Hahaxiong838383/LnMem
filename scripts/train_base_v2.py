#!/usr/bin/env python3
"""Train base_v2 on VEATIC sequences with smooth V-A supervision."""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.fusion import DEFAULT_CONFIG, _fatigue_attention_features
from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.model import _PersonaConditionedCfC
from poc.src.lnn.personalization import DefaultPersonalizationAdapter
from poc.src.vision.au_extractor import AU_NAMES
from scripts.sweep_va_to_state import map_va_to_state

_N_STATE_CLASSES = 9
_SEQ_LENGTH = 60
_SEQ_STRIDE = 30
_MISSING_DEFAULT = DEFAULT_CONFIG.missing_default
_OUTPUT_STATE = "state_model"
_VEATIC_FACE_COLUMNS = (
    "Pitch",
    "Yaw",
    "pitch_deg",
    "yaw_deg",
    "AU61",
    "AU62",
    "AU63",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADR-007 Phase 1.4 base_v2 trainer")
    parser.add_argument("--input-parquet", required=True, type=Path)
    parser.add_argument("--veatic-ratings-dir", required=True, type=Path)
    parser.add_argument(
        "--base-v1-ckpt",
        default=Path("poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"),
        type=Path,
    )
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--r-neutral", type=float, default=0.35)
    parser.add_argument("--theta-offset-deg", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--va-loss-weight", type=float, default=0.5)
    parser.add_argument("--soft-assignment-sigma-init", type=float, default=0.35)
    parser.add_argument("--soft-assignment-sigma-min", type=float, default=0.1)
    parser.add_argument("--soft-assignment-sigma-max", type=float, default=1.0)
    parser.add_argument("--soft-assignment-sigma-prior-weight", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        output = float(value)
    except (TypeError, ValueError):
        return default
    return output if math.isfinite(output) else default


def _clamp01(value: float, default: float = _MISSING_DEFAULT) -> float:
    if not math.isfinite(value):
        return default
    return max(0.0, min(1.0, value))


def _load_rating_csv(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing VEATIC rating csv: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 2:
        raise ValueError(f"expected 2 columns in {path.name}, got {df.shape[1]}")
    out = df.rename(columns={0: "frame_idx", 1: value_name}).copy()
    out["frame_idx"] = out["frame_idx"].astype(int)
    out[value_name] = out[value_name].astype(float)
    return out


def _load_frame_table(input_parquet: Path, ratings_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_parquet)
    required = {"filename", "frame_idx", "source_dataset", "face_detected", *AU_NAMES}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"input parquet missing columns: {missing}")
    keep_cols = [
        "filename",
        "frame_idx",
        "source_dataset",
        "face_detected",
        "consecutive_nan_au",
        *AU_NAMES,
        *[col for col in _VEATIC_FACE_COLUMNS if col in df.columns],
    ]
    base = df.loc[df["source_dataset"] == "veatic", keep_cols].copy()
    if base.empty:
        raise ValueError("no rows after source_dataset==veatic filter")
    merged: list[pd.DataFrame] = []
    for filename, group in base.groupby("filename", sort=False):
        stem = Path(str(filename)).stem
        valence = _load_rating_csv(ratings_dir / f"{stem}_valence.csv", "valence")
        arousal = _load_rating_csv(ratings_dir / f"{stem}_arousal.csv", "arousal")
        joined = valence.merge(arousal, on="frame_idx", how="inner")
        part = group.merge(joined, on="frame_idx", how="inner").copy()
        if len(part) != len(group):
            dropped = len(group) - len(part)
            print(f"[warn] {filename}: dropped {dropped} frame(s) without V-A join", file=sys.stderr)
        merged.append(part)
    if not merged:
        raise ValueError("no VEATIC rows joined with rating csv")
    return pd.concat(merged, ignore_index=True).sort_values(["filename", "frame_idx"]).reset_index(drop=True)


def _face_pose_value(row: dict[str, Any], *candidates: str, default: float = 0.0) -> float:
    for key in candidates:
        value = _safe_float(row.get(key), default=math.nan)
        if math.isfinite(value):
            return value
    return default


def _gaze_stability(row: dict[str, Any]) -> float:
    values = [_safe_float(row.get(key), default=math.nan) for key in ("AU61", "AU62", "AU63")]
    finite = [max(0.0, min(1.0, value)) for value in values if math.isfinite(value)]
    return float(sum(finite) / len(finite)) if finite else _MISSING_DEFAULT


def _expression_features(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        _clamp01(_safe_float(row.get("AU12"), 0.0)),
        _clamp01(_safe_float(row.get("AU04"), 0.0)),
        _clamp01(_safe_float(row.get("AU06"), 0.0)),
        _clamp01(_safe_float(row.get("AU15"), 0.0)),
        _clamp01(_safe_float(row.get("AU26"), 0.0)),
        _clamp01(_safe_float(row.get("AU25"), 0.0)),
    )


def _has_face(row: dict[str, Any]) -> bool:
    if not bool(row.get("face_detected", False)):
        return False
    return any(math.isfinite(_safe_float(row.get(au), default=math.nan)) for au in AU_NAMES)


def _veatic_au_to_fusion_18(row: dict[str, Any]) -> tuple[float, ...]:
    if not _has_face(row):
        return (_MISSING_DEFAULT,) * 18
    pitch_deg = _face_pose_value(row, "Pitch", "pitch_deg", default=0.0)
    yaw_deg = _face_pose_value(row, "Yaw", "yaw_deg", default=0.0)
    gaze = _gaze_stability(row)
    face = {
        "eye_closure": _clamp01(_safe_float(row.get("AU43"), _MISSING_DEFAULT)),
        "head_pitch_deg": pitch_deg,
        "head_yaw_deg": yaw_deg,
        "gaze_stability": gaze,
    }
    face_feats = (
        face["eye_closure"],
        _clamp01(abs(pitch_deg) / DEFAULT_CONFIG.head_pitch_max_deg, 0.0),
        _clamp01(abs(yaw_deg) / DEFAULT_CONFIG.head_yaw_max_deg, 0.0),
        gaze,
    )
    derived = _fatigue_attention_features(face, DEFAULT_CONFIG)
    expression = _expression_features(row)
    audio = (_MISSING_DEFAULT, _MISSING_DEFAULT)
    context = (_MISSING_DEFAULT, _MISSING_DEFAULT)
    return face_feats + derived + expression + audio + context


class _WindowDataset(Dataset[tuple[Tensor, Tensor, Tensor, Tensor]]):
    """Materialized VEATIC sliding windows for base_v2 training."""

    def __init__(self, frame_table: pd.DataFrame, r_neutral: float, theta_offset_deg: float) -> None:
        self.features: list[np.ndarray] = []
        self.targets_va: list[np.ndarray] = []
        self.states: list[np.ndarray] = []
        self.sample_states: list[int] = []
        for _, video_df in frame_table.groupby("filename", sort=False):
            ordered = video_df.sort_values("frame_idx").reset_index(drop=True)
            if len(ordered) < _SEQ_LENGTH:
                continue
            feature_rows = np.array(
                [_veatic_au_to_fusion_18(row) for row in ordered.to_dict("records")],
                dtype=np.float32,
            )
            va_rows = ordered[["valence", "arousal"]].to_numpy(dtype=np.float32)
            hard_states = map_va_to_state(
                ordered["valence"].to_numpy(),
                ordered["arousal"].to_numpy(),
                r_neutral=r_neutral,
                theta_offset_deg=theta_offset_deg,
                n_states=8,
            ).astype(np.int64)
            for start in range(0, len(ordered) - _SEQ_LENGTH + 1, _SEQ_STRIDE):
                end = start + _SEQ_LENGTH
                center = start + (_SEQ_LENGTH // 2)
                self.features.append(feature_rows[start:end].copy())
                self.targets_va.append(va_rows[start:end].copy())
                self.states.append(hard_states[start:end].copy())
                self.sample_states.append(int(hard_states[center]))

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        return (
            torch.from_numpy(self.features[index]),
            torch.from_numpy(self.targets_va[index]),
            torch.from_numpy(self.states[index]),
            torch.tensor(0, dtype=torch.long),
        )


def build_class_balanced_sampler(
    sample_states: Sequence[int],
    num_classes: int = _N_STATE_CLASSES,
    seed: int = 42,
) -> WeightedRandomSampler:
    counts = np.bincount(np.asarray(sample_states, dtype=np.int64), minlength=num_classes)
    class_weights = np.zeros(num_classes, dtype=np.float64)
    nonzero = counts > 0
    class_weights[nonzero] = 1.0 / counts[nonzero]
    sample_weights = torch.tensor([class_weights[state] for state in sample_states], dtype=torch.double)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_states),
        replacement=True,
        generator=generator,
    )


def _split_train_val(frame_table: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    filenames = sorted(frame_table["filename"].astype(str).unique().tolist())
    if len(filenames) < 2:
        raise ValueError("need at least 2 VEATIC videos for train/val split")
    rng = np.random.default_rng(seed)
    rng.shuffle(filenames)
    split_idx = max(1, int(len(filenames) * 0.8))
    split_idx = min(split_idx, len(filenames) - 1)
    train_names = set(filenames[:split_idx])
    val_names = set(filenames[split_idx:])
    train_df = frame_table[frame_table["filename"].astype(str).isin(train_names)].copy()
    val_df = frame_table[frame_table["filename"].astype(str).isin(val_names)].copy()
    return train_df, val_df


def build_dataloaders(
    frame_table: pd.DataFrame,
    batch_size: int,
    seed: int,
    r_neutral: float,
    theta_offset_deg: float,
) -> tuple[DataLoader, DataLoader]:
    train_df, val_df = _split_train_val(frame_table, seed=seed)
    train_ds = _WindowDataset(train_df, r_neutral=r_neutral, theta_offset_deg=theta_offset_deg)
    val_ds = _WindowDataset(val_df, r_neutral=r_neutral, theta_offset_deg=theta_offset_deg)
    if len(train_ds) == 0 or len(val_ds) == 0:
        raise ValueError("train/val windows are empty after 60-frame sliding window build")
    sampler = build_class_balanced_sampler(train_ds.sample_states, seed=seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def _sigma_logit_for_init(sigma_init: float, sigma_min: float, sigma_max: float) -> float:
    if not sigma_min < sigma_init < sigma_max:
        raise ValueError("soft-assignment sigma init must be strictly inside [min, max]")
    normalized = (sigma_init - sigma_min) / (sigma_max - sigma_min)
    return math.log(normalized / (1.0 - normalized))


class BaseV2Model(nn.Module):
    """CfC hidden backbone + 9-state head + V-A regression head."""

    def __init__(self, sigma_init: float, sigma_min: float, sigma_max: float) -> None:
        super().__init__()
        self.config = LNNTrainConfig(
            feature_dim=18,
            hidden_dim=64,
            n_state_classes=_N_STATE_CLASSES,
            n_personas=1,
            persona_embed_dim=8,
        )
        adapter = DefaultPersonalizationAdapter(self.config)
        self.cfc_block = _PersonaConditionedCfC(self.config, adapter, out_dim=self.config.hidden_dim)
        self.cfc_block.readout = nn.Identity()
        self.state_head = nn.Linear(self.config.hidden_dim, _N_STATE_CLASSES)
        self.va_head = nn.Linear(self.config.hidden_dim, 2)
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self.sigma_init = sigma_init
        self.sigma_logit = nn.Parameter(
            torch.tensor(_sigma_logit_for_init(sigma_init, sigma_min, sigma_max), dtype=torch.float32)
        )

    @property
    def sigma(self) -> Tensor:
        scale = torch.sigmoid(self.sigma_logit)
        return self._sigma_min + (self._sigma_max - self._sigma_min) * scale

    def forward(self, x: Tensor, persona_idx: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.cfc_block(x, persona_idx)
        state_logits = self.state_head(hidden)
        va_pred = torch.tanh(self.va_head(hidden))
        return state_logits, va_pred


def load_base_v1_weights(model: BaseV2Model, ckpt_path: Path) -> dict[str, list[str]]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get(_OUTPUT_STATE) or ckpt.get("state_model_state_dict")
    if not isinstance(state_dict, dict):
        raise ValueError("base_v1 checkpoint missing state_model state_dict")
    remapped: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("readout."):
            continue
        if key == "personalization.embedding.weight":
            expected = model.cfc_block.personalization.embedding.weight.shape[0]
            if value.shape[0] == expected:
                remapped[key] = value
            else:
                remapped[key] = value.mean(dim=0, keepdim=True).to(value.dtype)
            continue
        remapped[key] = value
    result = model.cfc_block.load_state_dict(remapped, strict=False)
    for name, param in model.cfc_block.personalization.named_parameters():
        if name.startswith(("tau_projection", "gamma_projection", "beta_projection")):
            assert param.requires_grad, f"{name} unexpectedly frozen"
    return {"missing_keys": result.missing_keys, "unexpected_keys": result.unexpected_keys}


def build_soft_state_targets(
    va_targets: Tensor,
    sigma: Tensor,
    r_neutral: float,
    theta_offset_deg: float,
) -> Tensor:
    flattened = va_targets.reshape(-1, 2)
    valence = flattened[:, 0]
    arousal = flattened[:, 1]
    radius = torch.sqrt(valence.square() + arousal.square())
    theta = torch.atan2(arousal, valence)
    distances = [torch.abs(radius - r_neutral)]
    centers = torch.deg2rad(
        theta.new_tensor([theta_offset_deg + 22.5 + 45.0 * idx for idx in range(8)], dtype=theta.dtype)
    )
    for center in centers:
        wrapped = torch.atan2(torch.sin(theta - center), torch.cos(theta - center)).abs()
        distances.append(wrapped)
    dist_tensor = torch.stack(distances, dim=-1)
    denom = 2.0 * torch.clamp(sigma, min=1e-6).square()
    return torch.softmax(-(dist_tensor.square()) / denom, dim=-1)


def concordance_correlation_coefficient(pred: Tensor, target: Tensor) -> Tensor:
    pred_mean = pred.mean()
    target_mean = target.mean()
    pred_var = pred.var(unbiased=False)
    target_var = target.var(unbiased=False)
    covariance = ((pred - pred_mean) * (target - target_mean)).mean()
    numerator = 2.0 * covariance
    denominator = pred_var + target_var + (pred_mean - target_mean).square() + 1e-8
    return numerator / denominator


def va_ccc_loss(pred_va: Tensor, true_va: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pred_flat = pred_va.reshape(-1, 2)
    true_flat = true_va.reshape(-1, 2)
    ccc_v = concordance_correlation_coefficient(pred_flat[:, 0], true_flat[:, 0])
    ccc_a = concordance_correlation_coefficient(pred_flat[:, 1], true_flat[:, 1])
    loss = 1.0 - ((ccc_v + ccc_a) / 2.0)
    return loss, ccc_v, ccc_a


def _batch_losses(
    model: BaseV2Model,
    state_logits: Tensor,
    pred_va: Tensor,
    true_va: Tensor,
    va_loss_weight: float,
    r_neutral: float,
    theta_offset_deg: float,
    sigma_prior_weight: float,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    sigma = model.sigma
    soft_targets = build_soft_state_targets(true_va, sigma, r_neutral, theta_offset_deg)
    log_probs = torch.log_softmax(state_logits.reshape(-1, _N_STATE_CLASSES), dim=-1)
    soft_state_ce = -(soft_targets * log_probs).sum(dim=-1).mean()
    ccc_loss, ccc_v, ccc_a = va_ccc_loss(pred_va, true_va)
    sigma_prior = sigma_prior_weight * (sigma - model.sigma_init).square()
    total_loss = soft_state_ce + (va_loss_weight * ccc_loss) + sigma_prior
    return total_loss, soft_state_ce, ccc_v, ccc_a


def _run_epoch(
    model: BaseV2Model,
    loader: DataLoader,
    device: str,
    optimizer: torch.optim.Optimizer | None,
    va_loss_weight: float,
    r_neutral: float,
    theta_offset_deg: float,
    sigma_prior_weight: float,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(mode=training)
    totals = {"loss": 0.0, "correct": 0.0, "count": 0.0, "ccc_v": 0.0, "ccc_a": 0.0}
    grad_context = torch.enable_grad() if training else torch.no_grad()
    with grad_context:
        for features, va_targets, hard_states, persona_idx in loader:
            features = features.to(device=device, dtype=torch.float32)
            va_targets = va_targets.to(device=device, dtype=torch.float32)
            hard_states = hard_states.to(device=device, dtype=torch.long)
            persona_idx = persona_idx.to(device=device, dtype=torch.long)
            state_logits, pred_va = model(features, persona_idx)
            total_loss, _, ccc_v, ccc_a = _batch_losses(
                model=model,
                state_logits=state_logits,
                pred_va=pred_va,
                true_va=va_targets,
                va_loss_weight=va_loss_weight,
                r_neutral=r_neutral,
                theta_offset_deg=theta_offset_deg,
                sigma_prior_weight=sigma_prior_weight,
            )
            if training:
                optimizer.zero_grad()
                total_loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            n_frames = float(hard_states.numel())
            predictions = state_logits.argmax(dim=-1)
            totals["loss"] += total_loss.item() * n_frames
            totals["correct"] += float((predictions == hard_states).sum().item())
            totals["ccc_v"] += float(ccc_v.item()) * n_frames
            totals["ccc_a"] += float(ccc_a.item()) * n_frames
            totals["count"] += n_frames
    count = max(totals["count"], 1.0)
    return {
        "loss": totals["loss"] / count,
        "state_acc": totals["correct"] / count,
        "ccc_v": totals["ccc_v"] / count,
        "ccc_a": totals["ccc_a"] / count,
    }


def _log(line: str, log_path: Path) -> None:
    print(line, file=sys.stderr)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def _checkpoint_payload(
    model: BaseV2Model,
    epoch: int,
    args: argparse.Namespace,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "config": model.config,
        "config_dict": asdict(model.config),
        "model_state": model.state_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "learned_sigma": float(model.sigma.detach().cpu().item()),
        "cli_args": vars(args).copy(),
    }


def _save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(args: argparse.Namespace) -> int:
    _set_seed(args.seed)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested --device mps but torch.backends.mps.is_available() is False")
    frame_table = _load_frame_table(args.input_parquet, args.veatic_ratings_dir.expanduser())
    train_loader, val_loader = build_dataloaders(
        frame_table,
        batch_size=args.batch_size,
        seed=args.seed,
        r_neutral=args.r_neutral,
        theta_offset_deg=args.theta_offset_deg,
    )
    model = BaseV2Model(
        sigma_init=args.soft_assignment_sigma_init,
        sigma_min=args.soft_assignment_sigma_min,
        sigma_max=args.soft_assignment_sigma_max,
    ).to(args.device)
    load_info = load_base_v1_weights(model, args.base_v1_ckpt)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    output_path = args.output_ckpt
    output_dir = output_path.parent
    log_path = output_dir / "train_base_v2.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")
    best_val_loss = math.inf
    best_epoch = 0
    best_path = output_dir / "checkpoint_base_v2_best.pt"
    final_path = output_dir / "checkpoint_base_v2_final.pt"
    history: list[dict[str, Any]] = []
    patience_left = args.patience
    _log(
        f"loaded base_v1 missing={load_info['missing_keys']} unexpected={load_info['unexpected_keys']}",
        log_path,
    )
    for epoch in range(1, args.epochs + 1):
        started = time.time()
        train_metrics = _run_epoch(
            model,
            train_loader,
            args.device,
            optimizer,
            args.va_loss_weight,
            args.r_neutral,
            args.theta_offset_deg,
            args.soft_assignment_sigma_prior_weight,
        )
        val_metrics = _run_epoch(
            model,
            val_loader,
            args.device,
            None,
            args.va_loss_weight,
            args.r_neutral,
            args.theta_offset_deg,
            args.soft_assignment_sigma_prior_weight,
        )
        history.append(
            {
                "epoch": epoch,
                "elapsed_s": round(time.time() - started, 3),
                "train": train_metrics,
                "val": val_metrics,
                "sigma": float(model.sigma.detach().cpu().item()),
            }
        )
        _log(
            "epoch "
            f"{epoch} train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"val_state_acc={val_metrics['state_acc']:.4f} val_ccc_v={val_metrics['ccc_v']:.4f} "
            f"val_ccc_a={val_metrics['ccc_a']:.4f} current_σ={model.sigma.detach().cpu().item():.4f}",
            log_path,
        )
        payload = _checkpoint_payload(model, epoch, args, {"history": history})
        if epoch % 5 == 0:
            _save_checkpoint(output_dir / f"checkpoint_base_v2_epoch{epoch}.pt", payload)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            patience_left = args.patience
            _save_checkpoint(best_path, payload)
        else:
            patience_left -= 1
            if patience_left <= 0:
                _log(f"early_stop epoch={epoch} best_epoch={best_epoch} best_val_loss={best_val_loss:.4f}", log_path)
                break
    final_payload = _checkpoint_payload(model, best_epoch, args, {"history": history, "best_val_loss": best_val_loss})
    _save_checkpoint(final_path, final_payload)
    if output_path != final_path:
        _save_checkpoint(output_path, final_payload)
    _log("## Summary", log_path)
    _log(
        f"best_epoch={best_epoch} best_val_loss={best_val_loss:.4f} "
        f"learned_sigma={model.sigma.detach().cpu().item():.4f}",
        log_path,
    )
    _log(f"saved_best={best_path}", log_path)
    _log(f"saved_final={final_path}", log_path)
    if output_path != final_path:
        _log(f"saved_alias={output_path}", log_path)
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return train(args)
    except Exception as exc:  # noqa: BLE001
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
