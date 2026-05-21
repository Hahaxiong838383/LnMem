#!/usr/bin/env python3
"""Fine-tune base_v2 into base_v2.1 with frozen CfC + FiLM adaptation."""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.feedback_to_batch import _normalize_true_state
from poc.src.lnn.model import _PersonaConditionedCfC
from poc.src.lnn.personalization import DefaultPersonalizationAdapter
from poc.src.lnn.signals_loader import SignalsFrame, parse_signals_row

_N_CLASSES = 9
_PERSONA_SELF = 30
_SECONDS_30_MIN = 30.0 * 60.0


@dataclass(frozen=True)
class WindowRef:
    """Reference one sliding window within a session tensor."""

    session_id: str | None
    end_idx: int
    ts_s: float


@dataclass(frozen=True)
class SessionTensor:
    """Materialized session features and metadata."""

    features: Tensor
    frame_ids: tuple[int, ...]
    ts_s: tuple[float, ...]


@dataclass(frozen=True)
class LabeledWindow:
    """One labeled replayable feedback window."""

    event_id: str
    true_state: str
    class_index: int
    session_id: str | None
    frame_id: int
    ts: str
    window: Tensor


@dataclass(frozen=True)
class EpochStats:
    """Compact epoch metrics for markdown/report export."""

    phase: str
    epoch: int
    loss: float
    aux: dict[str, float]


class BaseV21Model(nn.Module):
    """CfC backbone + 9-way state head with configurable persona table."""

    def __init__(
        self,
        *,
        feature_dim: int,
        hidden_dim: int,
        persona_embed_dim: int,
        n_personas: int,
        cfc_mode: str,
        use_tau_bias: bool,
        use_film: bool,
        sigma_init: float,
        sigma_min: float,
        sigma_max: float,
    ) -> None:
        super().__init__()
        self.config = LNNTrainConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            n_state_classes=_N_CLASSES,
            n_personas=n_personas,
            persona_embed_dim=persona_embed_dim,
            cfc_mode=cfc_mode,
            use_tau_bias=use_tau_bias,
            use_film=use_film,
        )
        personalization = DefaultPersonalizationAdapter(self.config)
        self.cfc_block = _PersonaConditionedCfC(self.config, personalization, out_dim=hidden_dim)
        self.cfc_block.readout = nn.Identity()
        self.state_head = nn.Linear(hidden_dim, _N_CLASSES)
        self.va_head = nn.Linear(hidden_dim, 2)
        self._sigma_min = sigma_min
        self._sigma_max = sigma_max
        self.sigma_init = sigma_init
        self.sigma_logit = nn.Parameter(
            torch.tensor(_sigma_logit_for_init(sigma_init, sigma_min, sigma_max), dtype=torch.float32),
        )

    @property
    def sigma(self) -> Tensor:
        scale = torch.sigmoid(self.sigma_logit)
        return self._sigma_min + (self._sigma_max - self._sigma_min) * scale

    def encode(self, x: Tensor, persona_idx: Tensor) -> Tensor:
        """Return per-step hidden states after persona-conditioned CfC."""
        return self.cfc_block(x, persona_idx)

    def classify(self, hidden: Tensor) -> Tensor:
        """Project hidden states to 9-way state logits."""
        return self.state_head(hidden)

    def forward(self, x: Tensor, persona_idx: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.encode(x, persona_idx)
        return self.classify(hidden), torch.tanh(self.va_head(hidden))

    def film_effects(self, persona_idx: int) -> tuple[Tensor, Tensor]:
        """Return effective FiLM scale/shift for one persona row."""
        persona = torch.tensor([persona_idx], dtype=torch.long, device=self.state_head.weight.device)
        gamma, beta = self.cfc_block.personalization.get_film_params(persona)
        scale = 1.0 + 0.1 * torch.tanh(gamma)
        shift = 0.1 * torch.tanh(beta)
        return scale.squeeze(0), shift.squeeze(0)


class TemporalContrastDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Sample anchor/positive/negative windows for temporal contrast."""

    def __init__(
        self,
        sessions: dict[str | None, SessionTensor],
        *,
        sequence_length: int,
        delta: int,
        n_negatives: int,
        min_negative_gap_s: float,
        seed: int,
    ) -> None:
        self.sessions = sessions
        self.sequence_length = sequence_length
        self.delta = delta
        self.n_negatives = n_negatives
        self.min_negative_gap_s = min_negative_gap_s
        self.seed = seed
        self.refs: list[WindowRef] = []
        for session_id, session in sessions.items():
            start = sequence_length - 1
            stop = session.features.shape[0] - delta
            for end_idx in range(start, stop):
                self.refs.append(WindowRef(session_id=session_id, end_idx=end_idx, ts_s=session.ts_s[end_idx]))
        if not self.refs:
            raise ValueError("no ssl windows built from unlabeled signals")
        self.refs.sort(key=lambda item: item.ts_s)

    def __len__(self) -> int:
        return len(self.refs)

    def _slice(self, ref: WindowRef, *, offset: int = 0) -> Tensor:
        session = self.sessions[ref.session_id]
        end_idx = ref.end_idx + offset
        start_idx = end_idx - self.sequence_length + 1
        return session.features[start_idx : end_idx + 1]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        ref = self.refs[index]
        rng = random.Random(self.seed + index)
        anchor = self._slice(ref)
        positive = self._slice(ref, offset=self.delta)
        negatives: list[Tensor] = []
        tries = 0
        while len(negatives) < self.n_negatives and tries < self.n_negatives * 64:
            tries += 1
            candidate = self.refs[rng.randrange(len(self.refs))]
            if abs(candidate.ts_s - ref.ts_s) < self.min_negative_gap_s:
                continue
            negatives.append(self._slice(candidate))
        if len(negatives) < self.n_negatives:
            fallback = sorted(self.refs, key=lambda item: abs(item.ts_s - ref.ts_s), reverse=True)
            for candidate in fallback:
                if len(negatives) >= self.n_negatives:
                    break
                negatives.append(self._slice(candidate))
        return anchor, positive, torch.stack(negatives[: self.n_negatives], dim=0)


class LabeledWindowDataset(Dataset[tuple[Tensor, Tensor]]):
    """Supervised labeled windows from real mismatch corrections."""

    def __init__(self, items: Sequence[LabeledWindow]) -> None:
        self.items = list(items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        item = self.items[index]
        return item.window, torch.tensor(item.class_index, dtype=torch.long)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADR-013 DFAT fine-tune for base_v2.1")
    parser.add_argument(
        "--base-v2-ckpt",
        default=Path("poc/data/checkpoints/base_v2/checkpoint_base_v2_best.pt"),
        type=Path,
    )
    parser.add_argument(
        "--base-v1-ckpt",
        default=Path("poc/data/checkpoints/day12_with_self/checkpoint_day12.pt"),
        type=Path,
    )
    parser.add_argument(
        "--unlabeled-signals-glob",
        nargs="+",
        default=["~/.lnmem/data/signals_2026-05-*.jsonl"],
    )
    parser.add_argument(
        "--labeled-queries-glob",
        nargs="+",
        default=["~/.lnmem/feedback/queries_*.jsonl"],
    )
    parser.add_argument("--output-ckpt", required=True, type=Path)
    parser.add_argument("--output-report", type=Path, default=None)
    parser.add_argument("--ssl-epochs", type=int, default=5)
    parser.add_argument("--sup-epochs", type=int, default=30)
    parser.add_argument("--sup-patience", type=int, default=8)
    parser.add_argument("--batch-size-ssl", type=int, default=16)
    parser.add_argument("--batch-size-sup", type=int, default=8)
    parser.add_argument("--sequence-length", type=int, default=60)
    parser.add_argument("--ssl-delta", type=int, default=5)
    parser.add_argument("--ssl-negatives", type=int, default=8)
    parser.add_argument("--persona-idx", type=int, default=_PERSONA_SELF)
    parser.add_argument("--r-neutral", type=float, default=0.35)
    parser.add_argument("--theta-offset-deg", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def _sigma_logit_for_init(sigma_init: float, sigma_min: float, sigma_max: float) -> float:
    if not sigma_min < sigma_init < sigma_max:
        raise ValueError("sigma_init must be inside (sigma_min, sigma_max)")
    normalized = (sigma_init - sigma_min) / (sigma_max - sigma_min)
    return math.log(normalized / (1.0 - normalized))


def _expand_globs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p).expanduser() for p in glob.glob(str(Path(pattern).expanduser())))
        if matches:
            paths.extend(matches)
            continue
        maybe = Path(pattern).expanduser()
        if maybe.exists():
            paths.append(maybe)
    return sorted(dict.fromkeys(paths))


def _parse_ts(raw: str) -> float:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc).timestamp()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def _class_index_from_state(state: str) -> int:
    return STATE_LABELS.index(state) + 1


def _class_label(index: int) -> str:
    if index == 0:
        return "neutral_top1"
    return STATE_LABELS[index - 1]


def build_model_from_v2_ckpt(base_v2_ckpt: Path, base_v1_ckpt: Path, persona_idx: int) -> BaseV21Model:
    """Create a v2.1 model with 31-row persona table and load v2 weights."""
    ckpt_v2 = torch.load(base_v2_ckpt, map_location="cpu", weights_only=False)
    ckpt_v1 = torch.load(base_v1_ckpt, map_location="cpu", weights_only=False)
    config_v2: LNNTrainConfig = ckpt_v2["config"]
    config_v1: LNNTrainConfig = ckpt_v1["config"]
    model = BaseV21Model(
        feature_dim=config_v2.feature_dim,
        hidden_dim=config_v2.hidden_dim,
        persona_embed_dim=config_v2.persona_embed_dim,
        n_personas=max(config_v1.n_personas, persona_idx + 1),
        cfc_mode=config_v2.cfc_mode,
        use_tau_bias=config_v2.use_tau_bias,
        use_film=config_v2.use_film,
        sigma_init=float(ckpt_v2.get("learned_sigma", 0.326)),
        sigma_min=0.1,
        sigma_max=1.0,
    )
    state_dict = dict(ckpt_v2["model_state"])
    embed_key = "cfc_block.personalization.embedding.weight"
    if state_dict[embed_key].shape[0] == 1 and model.config.n_personas > 1:
        state_dict[embed_key] = state_dict[embed_key].repeat(model.config.n_personas, 1)
    result = model.load_state_dict(state_dict, strict=False)
    if result.unexpected_keys:
        raise ValueError(f"unexpected base_v2 keys: {result.unexpected_keys}")
    inject_persona_embedding_from_v1(model, base_v1_ckpt, persona_idx)
    return model


def inject_persona_embedding_from_v1(model: BaseV21Model, base_v1_ckpt: Path, persona_idx: int) -> Tensor:
    """Overwrite row `persona_idx` with base_v1 vboss_self embedding."""
    ckpt = torch.load(base_v1_ckpt, map_location="cpu", weights_only=False)
    embedding = ckpt["state_model"]["personalization.embedding.weight"][persona_idx].detach().clone()
    with torch.no_grad():
        model.cfc_block.personalization.embedding.weight[persona_idx].copy_(embedding)
    return embedding


def freeze_backbone_for_dfat(model: BaseV21Model) -> dict[str, float]:
    """Freeze all but FiLM gamma/beta and state head."""
    for name, param in model.named_parameters():
        trainable = (
            name.startswith("state_head.")
            or name.startswith("cfc_block.personalization.gamma_projection.")
            or name.startswith("cfc_block.personalization.beta_projection.")
        )
        param.requires_grad = trainable
    total = sum(param.numel() for param in model.parameters())
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    ratio = trainable / max(total, 1)
    if ratio >= 0.10:
        raise AssertionError(f"trainable ratio must be <10%, got {ratio:.4f}")
    return {"trainable_params": float(trainable), "total_params": float(total), "ratio": ratio}


def load_sessions_from_signal_paths(paths: Sequence[Path]) -> dict[str | None, SessionTensor]:
    """Parse raw daemon signals into session tensors once for SSL + supervised."""
    grouped: dict[str | None, list[SignalsFrame]] = defaultdict(list)
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                frame = parse_signals_row(raw)
                if frame is not None:
                    grouped[frame.session_id].append(frame)
    sessions: dict[str | None, SessionTensor] = {}
    for session_id, frames in grouped.items():
        frames.sort(key=lambda item: (_parse_ts(item.ts), item.frame_id))
        features = torch.tensor([list(frame.vector) for frame in frames], dtype=torch.float32)
        sessions[session_id] = SessionTensor(
            features=features,
            frame_ids=tuple(frame.frame_id for frame in frames),
            ts_s=tuple(_parse_ts(frame.ts) for frame in frames),
        )
    if not sessions:
        raise ValueError("no signal sessions loaded from unlabeled-signals-glob")
    return sessions


def load_labeled_windows(
    query_paths: Sequence[Path],
    sessions: dict[str | None, SessionTensor],
    *,
    sequence_length: int,
    default_persona_idx: int,
) -> list[LabeledWindow]:
    """Read true_state mismatch rows and materialize their 60-frame windows."""
    rows: list[LabeledWindow] = []
    for path in query_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                if record.get("reason") != "brow_burst":
                    continue
                response = record.get("user_response") or {}
                if response.get("label") not in {"match", "mismatch"}:
                    continue
                true_state = _normalize_true_state(response.get("true_state"))
                if true_state is None:
                    continue
                session_id = record.get("session_id")
                session = sessions.get(session_id)
                if session is None:
                    continue
                frame_id = int(record["frame_id"])
                try:
                    end_idx = session.frame_ids.index(frame_id)
                except ValueError:
                    continue
                start_idx = end_idx - sequence_length + 1
                if start_idx < 0:
                    continue
                window = session.features[start_idx : end_idx + 1].clone()
                event_id = str(record.get("event_uuid") or f"{path.name}:{record['ts']}:{frame_id}")
                rows.append(
                    LabeledWindow(
                        event_id=event_id,
                        true_state=true_state,
                        class_index=_class_index_from_state(true_state),
                        session_id=session_id,
                        frame_id=frame_id,
                        ts=str(record["ts"]),
                        window=window,
                    )
                )
    if not rows:
        raise ValueError("no labeled windows built from queries glob")
    return rows


def split_labeled_windows(items: Sequence[LabeledWindow], seed: int) -> tuple[list[LabeledWindow], list[LabeledWindow]]:
    """Split by unique event_id to avoid train/val leakage."""
    ids = sorted({item.event_id for item in items})
    rng = random.Random(seed)
    rng.shuffle(ids)
    split = max(1, int(len(ids) * 0.8))
    split = min(split, len(ids) - 1)
    train_ids = set(ids[:split])
    train = [item for item in items if item.event_id in train_ids]
    val = [item for item in items if item.event_id not in train_ids]
    if not train or not val:
        raise ValueError("labeled split produced empty train or val partition")
    return train, val


def build_class_weights(items: Sequence[LabeledWindow], device: str) -> Tensor:
    """Inverse-frequency weights over 9 classes for focused/distracted fine-tune."""
    counts = Counter(item.class_index for item in items)
    weights = torch.full((_N_CLASSES,), 1e-6, dtype=torch.float32, device=device)
    for class_index, count in counts.items():
        weights[class_index] = 1.0 / float(count)
    return weights


def build_cost_matrix(device: str) -> Tensor:
    """Non-symmetric confusion penalty matrix from the CC spec."""
    matrix = torch.ones((_N_CLASSES, _N_CLASSES), dtype=torch.float32, device=device)
    focused = _class_index_from_state("focused")
    distracted = _class_index_from_state("distracted")
    negative = _class_index_from_state("emotional_negative")
    transitioning = _class_index_from_state("transitioning")
    matrix[focused, negative] = 5.0
    matrix[focused, transitioning] = 5.0
    matrix[distracted, negative] = 5.0
    matrix[distracted, transitioning] = 5.0
    return matrix


def cost_sensitive_ce(logits: Tensor, labels: Tensor, class_weights: Tensor, cost_matrix: Tensor) -> Tensor:
    """Weighted CE multiplied by the current argmax confusion cost."""
    ce = F.cross_entropy(logits, labels, weight=class_weights, reduction="none")
    predicted = logits.argmax(dim=-1)
    multipliers = cost_matrix[labels, predicted]
    return (ce * multipliers).mean()


def russell_centers(r_neutral: float, theta_offset_deg: float) -> Tensor:
    """Map 9 classes into V-A centers; class 0 is the origin."""
    centers = torch.zeros((_N_CLASSES, 2), dtype=torch.float32)
    radius = (1.0 + r_neutral) / 2.0
    for index, _label in enumerate(STATE_LABELS, start=1):
        angle_deg = theta_offset_deg + ((index - 1) + 0.5) * (360.0 / len(STATE_LABELS))
        angle_rad = math.radians(angle_deg)
        centers[index, 0] = radius * math.cos(angle_rad)
        centers[index, 1] = radius * math.sin(angle_rad)
    return centers


def expected_russell_distance(logits: Tensor, labels: Tensor, centers: Tensor) -> Tensor:
    """Smooth Russell penalty using expected arc length under predicted probs."""
    probs = torch.softmax(logits, dim=-1)
    unit_centers = F.normalize(centers.to(logits.device), dim=-1)
    true_vectors = unit_centers[labels]
    dots = torch.einsum("bd,cd->bc", true_vectors, unit_centers)
    arc = torch.arccos(torch.clamp(dots, -1.0, 1.0))
    return (probs * arc).sum(dim=-1).mean()


def supervised_contrastive_loss(features: Tensor, labels: Tensor, temperature: float = 0.2) -> Tensor:
    """SupCon over batch features; returns 0 for degenerate label sets."""
    if features.shape[0] < 2:
        return features.new_tensor(0.0)
    normed = F.normalize(features, dim=-1)
    logits = torch.matmul(normed, normed.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values
    same = labels.unsqueeze(0) == labels.unsqueeze(1)
    mask = same & ~torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    if not mask.any():
        return features.new_tensor(0.0)
    exp_logits = torch.exp(logits) * ~torch.eye(labels.shape[0], dtype=torch.bool, device=labels.device)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    positive_log_prob = (mask * log_prob).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
    valid = mask.sum(dim=1) > 0
    return -positive_log_prob[valid].mean()


def film_regularization(model: BaseV21Model, persona_idx: int) -> Tensor:
    """Keep effective FiLM close to identity after warm start."""
    scale, shift = model.film_effects(persona_idx)
    return F.mse_loss(scale, torch.ones_like(scale)) + F.mse_loss(shift, torch.zeros_like(shift))


def temporal_contrast_loss(anchor: Tensor, positive: Tensor, negatives: Tensor, temperature: float = 0.1) -> Tensor:
    """InfoNCE over (anchor, positive, negatives)."""
    anchor_n = F.normalize(anchor, dim=-1)
    positive_n = F.normalize(positive, dim=-1)
    negatives_n = F.normalize(negatives, dim=-1)
    pos_logits = (anchor_n * positive_n).sum(dim=-1, keepdim=True) / temperature
    neg_logits = torch.einsum("bd,bnd->bn", anchor_n, negatives_n) / temperature
    logits = torch.cat([pos_logits, neg_logits], dim=1)
    targets = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    return F.cross_entropy(logits, targets)


def run_ssl_phase(
    model: BaseV21Model,
    loader: DataLoader,
    *,
    persona_idx: int,
    epochs: int,
    device: str,
) -> list[EpochStats]:
    """Warm up FiLM gamma/beta with temporal neighborhood contrast."""
    if epochs <= 0:
        return []
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=1e-4,
        weight_decay=1e-5,
    )
    history: list[EpochStats] = []
    persona = torch.full((1,), persona_idx, dtype=torch.long, device=device)
    for epoch in range(1, epochs + 1):
        model.train(True)
        total_loss = 0.0
        total_tnc = 0.0
        total_reg = 0.0
        steps = 0
        for anchor, positive, negatives in loader:
            steps += 1
            anchor = anchor.to(device)
            positive = positive.to(device)
            negatives = negatives.to(device)
            batch = anchor.shape[0]
            persona_batch = persona.expand(batch)
            anchor_hidden = model.encode(anchor, persona_batch)[:, -1, :]
            positive_hidden = model.encode(positive, persona_batch)[:, -1, :]
            negatives_flat = negatives.reshape(batch * negatives.shape[1], negatives.shape[2], negatives.shape[3])
            negatives_hidden = model.encode(negatives_flat, persona.expand(negatives_flat.shape[0]))[:, -1, :]
            negatives_hidden = negatives_hidden.reshape(batch, negatives.shape[1], -1)
            loss_tnc = temporal_contrast_loss(anchor_hidden, positive_hidden, negatives_hidden)
            loss_reg = 0.01 * film_regularization(model, persona_idx)
            loss = loss_tnc + loss_reg
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += float(loss.item())
            total_tnc += float(loss_tnc.item())
            total_reg += float(loss_reg.item())
        history.append(
            EpochStats(
                phase="ssl",
                epoch=epoch,
                loss=total_loss / max(steps, 1),
                aux={"tnc": total_tnc / max(steps, 1), "reg": total_reg / max(steps, 1)},
            ),
        )
    return history


def evaluate_supervised(
    model: BaseV21Model,
    loader: DataLoader,
    *,
    persona_idx: int,
    device: str,
    class_weights: Tensor,
    cost_matrix: Tensor,
    centers: Tensor,
) -> dict[str, float]:
    """Evaluate weighted supervised loss and focused/distracted accuracy."""
    model.train(False)
    totals = Counter()
    persona = torch.full((1,), persona_idx, dtype=torch.long, device=device)
    with torch.no_grad():
        for windows, labels in loader:
            windows = windows.to(device)
            labels = labels.to(device)
            persona_batch = persona.expand(windows.shape[0])
            hidden = model.encode(windows, persona_batch)[:, -1, :]
            logits = model.classify(hidden)
            loss = (
                cost_sensitive_ce(logits, labels, class_weights, cost_matrix)
                + 0.5 * expected_russell_distance(logits, labels, centers)
                + 0.3 * supervised_contrastive_loss(hidden, labels)
            )
            preds = logits.argmax(dim=-1)
            totals["loss"] += float(loss.item()) * labels.shape[0]
            totals["count"] += labels.shape[0]
            totals["correct"] += int((preds == labels).sum().item())
            for state_name in ("focused", "distracted"):
                state_index = _class_index_from_state(state_name)
                mask = labels == state_index
                totals[f"{state_name}_n"] += int(mask.sum().item())
                totals[f"{state_name}_correct"] += int(((preds == labels) & mask).sum().item())
    count = max(totals["count"], 1)
    return {
        "loss": totals["loss"] / count,
        "acc": totals["correct"] / count,
        "focused_acc": totals["focused_correct"] / max(totals["focused_n"], 1),
        "distracted_acc": totals["distracted_correct"] / max(totals["distracted_n"], 1),
    }


def run_supervised_phase(
    model: BaseV21Model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    persona_idx: int,
    epochs: int,
    patience: int,
    device: str,
    class_weights: Tensor,
    cost_matrix: Tensor,
    centers: Tensor,
    output_ckpt: Path,
    cli_args: argparse.Namespace,
    trainable_stats: dict[str, float],
    ssl_history: Sequence[EpochStats],
) -> tuple[list[EpochStats], dict[str, Any]]:
    """Fine-tune state head + FiLM and save best / periodic checkpoints."""
    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=1e-4,
        weight_decay=1e-5,
    )
    history: list[EpochStats] = []
    best: dict[str, Any] | None = None
    best_loss = float("inf")
    bad_epochs = 0
    persona = torch.full((1,), persona_idx, dtype=torch.long, device=device)
    for epoch in range(1, epochs + 1):
        model.train(True)
        train_loss_sum = 0.0
        train_count = 0
        for windows, labels in train_loader:
            windows = windows.to(device)
            labels = labels.to(device)
            persona_batch = persona.expand(windows.shape[0])
            hidden = model.encode(windows, persona_batch)[:, -1, :]
            logits = model.classify(hidden)
            loss = (
                cost_sensitive_ce(logits, labels, class_weights, cost_matrix)
                + 0.5 * expected_russell_distance(logits, labels, centers)
                + 0.3 * supervised_contrastive_loss(hidden, labels)
            )
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss_sum += float(loss.item()) * labels.shape[0]
            train_count += labels.shape[0]
        val_metrics = evaluate_supervised(
            model,
            val_loader,
            persona_idx=persona_idx,
            device=device,
            class_weights=class_weights,
            cost_matrix=cost_matrix,
            centers=centers,
        )
        train_metrics = evaluate_supervised(
            model,
            train_loader,
            persona_idx=persona_idx,
            device=device,
            class_weights=class_weights,
            cost_matrix=cost_matrix,
            centers=centers,
        )
        history.append(
            EpochStats(
                phase="sup",
                epoch=epoch,
                loss=val_metrics["loss"],
                aux={
                    "train_loss": train_loss_sum / max(train_count, 1),
                    "train_acc": train_metrics["acc"],
                    "val_acc": val_metrics["acc"],
                    "focused_acc": val_metrics["focused_acc"],
                    "distracted_acc": val_metrics["distracted_acc"],
                },
            ),
        )
        payload = checkpoint_payload(model, cli_args, trainable_stats, ssl_history, history)
        if epoch % 5 == 0:
            periodic = output_ckpt.with_name(f"{output_ckpt.stem}_ep{epoch:02d}{output_ckpt.suffix}")
            save_checkpoint(periodic, payload)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            bad_epochs = 0
            best = payload
            save_checkpoint(output_ckpt, payload)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best is None:
        raise RuntimeError("no best checkpoint captured during supervised fine-tune")
    return history, best


def checkpoint_payload(
    model: BaseV21Model,
    cli_args: argparse.Namespace,
    trainable_stats: dict[str, float],
    ssl_history: Sequence[EpochStats],
    sup_history: Sequence[EpochStats],
) -> dict[str, Any]:
    """Serialize a replay-compatible checkpoint plus fine-tune metadata."""
    return {
        "config": model.config,
        "config_dict": asdict(model.config),
        "model_state": model.state_dict(),
        "learned_sigma": float(model.sigma.detach().cpu().item()),
        "cli_args": vars(cli_args).copy(),
        "phase": "base_v2_1_finetune",
        "metrics": {
            "ssl_history": [asdict(item) for item in ssl_history],
            "sup_history": [asdict(item) for item in sup_history],
            "trainable": trainable_stats,
        },
    }


def save_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    """Persist checkpoint to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def markdown_report(
    args: argparse.Namespace,
    signal_paths: Sequence[Path],
    labeled_items: Sequence[LabeledWindow],
    trainable_stats: dict[str, float],
    ssl_history: Sequence[EpochStats],
    sup_history: Sequence[EpochStats],
    best_payload: dict[str, Any],
) -> str:
    """Render a compact markdown run report."""
    lines = [
        "# 2026-05-21 base_v2.1 finetune",
        "",
        f"- base_v2_ckpt: `{args.base_v2_ckpt}`",
        f"- base_v1_ckpt: `{args.base_v1_ckpt}`",
        f"- output_ckpt: `{args.output_ckpt}`",
        f"- unlabeled_signal_files: **{len(signal_paths)}**",
        f"- labeled_windows: **{len(labeled_items)}**",
        f"- persona_idx: **{args.persona_idx}**",
        f"- trainable_ratio: **{trainable_stats['ratio']:.4%}**",
        f"- learned_sigma: **{best_payload['learned_sigma']:.4f}**",
        "",
        "## SSL",
        "",
        "| epoch | loss | tnc | reg |",
        "|---|---:|---:|---:|",
    ]
    for item in ssl_history:
        lines.append(f"| {item.epoch} | {item.loss:.4f} | {item.aux['tnc']:.4f} | {item.aux['reg']:.4f} |")
    lines.extend(["", "## Supervised", "", "| epoch | val_loss | train_loss | val_acc | focused_acc | distracted_acc |", "|---|---:|---:|---:|---:|---:|"])
    for item in sup_history:
        lines.append(
            f"| {item.epoch} | {item.loss:.4f} | {item.aux['train_loss']:.4f} | "
            f"{item.aux['val_acc']:.4%} | {item.aux['focused_acc']:.4%} | {item.aux['distracted_acc']:.4%} |",
        )
    return "\n".join(lines) + "\n"


def train(args: argparse.Namespace) -> int:
    """Main training flow for base_v2.1."""
    _set_seed(args.seed)
    signal_paths = _expand_globs(args.unlabeled_signals_glob)
    query_paths = _expand_globs(args.labeled_queries_glob)
    if not signal_paths:
        raise ValueError("no signal files matched --unlabeled-signals-glob")
    if not query_paths:
        raise ValueError("no query files matched --labeled-queries-glob")
    sessions = load_sessions_from_signal_paths(signal_paths)
    labeled_items = load_labeled_windows(
        query_paths,
        sessions,
        sequence_length=args.sequence_length,
        default_persona_idx=args.persona_idx,
    )
    train_items, val_items = split_labeled_windows(labeled_items, seed=args.seed)
    model = build_model_from_v2_ckpt(args.base_v2_ckpt, args.base_v1_ckpt, args.persona_idx).to(args.device)
    trainable_stats = freeze_backbone_for_dfat(model)
    ssl_loader = DataLoader(
        TemporalContrastDataset(
            sessions,
            sequence_length=args.sequence_length,
            delta=args.ssl_delta,
            n_negatives=args.ssl_negatives,
            min_negative_gap_s=_SECONDS_30_MIN,
            seed=args.seed,
        ),
        batch_size=args.batch_size_ssl,
        shuffle=True,
        num_workers=0,
    )
    train_loader = DataLoader(LabeledWindowDataset(train_items), batch_size=args.batch_size_sup, shuffle=True, num_workers=0)
    val_loader = DataLoader(LabeledWindowDataset(val_items), batch_size=args.batch_size_sup, shuffle=False, num_workers=0)
    ssl_history = run_ssl_phase(model, ssl_loader, persona_idx=args.persona_idx, epochs=args.ssl_epochs, device=args.device)
    class_weights = build_class_weights(train_items, args.device)
    cost_matrix = build_cost_matrix(args.device)
    centers = russell_centers(args.r_neutral, args.theta_offset_deg).to(args.device)
    sup_history, best_payload = run_supervised_phase(
        model,
        train_loader,
        val_loader,
        persona_idx=args.persona_idx,
        epochs=args.sup_epochs,
        patience=args.sup_patience,
        device=args.device,
        class_weights=class_weights,
        cost_matrix=cost_matrix,
        centers=centers,
        output_ckpt=args.output_ckpt,
        cli_args=args,
        trainable_stats=trainable_stats,
        ssl_history=ssl_history,
    )
    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            markdown_report(args, signal_paths, labeled_items, trainable_stats, ssl_history, sup_history, best_payload),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "output_ckpt": str(args.output_ckpt),
                "labeled_windows": len(labeled_items),
                "ssl_epochs": len(ssl_history),
                "sup_epochs": len(sup_history),
                "trainable_ratio": trainable_stats["ratio"],
            },
            ensure_ascii=False,
        ),
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("requested --device mps but torch.backends.mps.is_available() is False")
    start = time.time()
    try:
        code = train(args)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(f"elapsed_s={time.time() - start:.2f}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
