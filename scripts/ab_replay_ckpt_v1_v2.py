#!/usr/bin/env python3
"""Replay base_v1/base_v2/base_v2.1 on real brow_burst feedback windows."""
from __future__ import annotations

import argparse
import bisect
import glob
import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.lnn.expression_dynamics import STATE_LABELS
from poc.src.lnn.feedback_to_batch import _normalize_true_state
from poc.src.lnn.model import DefaultLNNFactory, _PersonaConditionedCfC
from poc.src.lnn.personalization import DefaultPersonalizationAdapter
from poc.src.lnn.signals_loader import SignalsFrame, load_signals

DEFAULT_V2_LABELS = ",".join(STATE_LABELS)


@dataclass(frozen=True)
class ReplayEvent:
    """Ground-truth brow_burst event from real feedback."""

    ts: str
    frame_id: int
    session_id: str | None
    persona_idx: int
    response_label: str
    true_state: str
    signal_path: str
    vlm_label: str
    source_file: str


@dataclass(frozen=True)
class Prediction:
    """Mapped model prediction for one event window."""

    label: str
    confidence: float
    top_index: int
    selected_index: int
    probabilities: tuple[float, ...] = ()
    raw_top1_label: str = ""

    @property
    def entropy(self) -> float:
        if not self.probabilities:
            return 0.0
        probs = torch.tensor(self.probabilities, dtype=torch.float32)
        return float(-(probs * torch.log(probs.clamp_min(1e-8))).sum().item())


@dataclass(frozen=True)
class ReplayRow:
    """Per-event replay outcome across all compared checkpoints."""

    event: ReplayEvent
    predictions: Mapping[str, Prediction] = field(default_factory=dict)


class _BaseV2ReplayModel(nn.Module):
    """Minimal base_v2/base_v2.1 inference graph using the shared CfC block."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        adapter = DefaultPersonalizationAdapter(config)
        self.cfc_block = _PersonaConditionedCfC(config, adapter, out_dim=config.hidden_dim)
        self.cfc_block.readout = nn.Identity()
        self.state_head = nn.Linear(config.hidden_dim, config.n_state_classes)

    def forward(self, x: Tensor, persona_idx: Tensor) -> Tensor:
        hidden = self.cfc_block(x, persona_idx)
        return self.state_head(hidden)


class BaseV1Runner:
    """Old 8-class checkpoint wrapper."""

    def __init__(self, ckpt_path: Path, device: str) -> None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        factory = DefaultLNNFactory()
        self.model = factory.create_state_model(ckpt["config"]).to(device)
        self.model.load_state_dict(ckpt["state_model"], strict=True)
        self.model.eval()
        self.device = device
        self.n_personas = ckpt["config"].n_personas

    def predict(self, window: Tensor, persona_idx: int) -> Prediction:
        clipped = max(0, min(persona_idx, self.n_personas - 1))
        pid = torch.tensor([clipped], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(window.to(self.device), pid)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        top = int(probs.argmax().item())
        return Prediction(
            label=STATE_LABELS[top],
            confidence=float(probs[top].item()),
            top_index=top,
            selected_index=top,
            probabilities=tuple(float(item) for item in probs.tolist()),
            raw_top1_label=STATE_LABELS[top],
        )


class BaseV2Runner:
    """9-class checkpoint wrapper with raw-neutral collapse monitoring."""

    def __init__(self, ckpt_path: Path, device: str, label_order: Sequence[str]) -> None:
        if len(label_order) != 8:
            raise ValueError("v2 label order must contain exactly 8 labels")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model_state"]
        filtered = {k: v for k, v in state_dict.items() if k.startswith(("cfc_block.", "state_head."))}
        self.model = _BaseV2ReplayModel(ckpt["config"]).to(device)
        self.model.load_state_dict(filtered, strict=True)
        self.model.eval()
        self.device = device
        self.label_order = tuple(label_order)
        self.n_personas = ckpt["config"].n_personas

    def predict(self, window: Tensor, persona_idx: int) -> Prediction:
        clipped = max(0, min(persona_idx, self.n_personas - 1))
        pid = torch.tensor([clipped], dtype=torch.long, device=self.device)
        with torch.no_grad():
            logits = self.model(window.to(self.device), pid)[0, -1]
        probs = torch.softmax(logits, dim=-1)
        ranked = torch.argsort(probs, descending=True).tolist()
        top = int(ranked[0])
        selected = next((idx for idx in ranked if idx != 0), top)
        selected_label = self.label_order[max(0, min(selected - 1, len(self.label_order) - 1))]
        raw_label = "neutral_top1" if top == 0 else self.label_order[top - 1]
        return Prediction(
            label=selected_label,
            confidence=float(probs[selected].item()),
            top_index=top,
            selected_index=selected,
            probabilities=tuple(float(item) for item in probs.tolist()),
            raw_top1_label=raw_label,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ADR-013 real-data replay for v1/v2/v2.1")
    parser.add_argument(
        "--queries-glob",
        nargs="+",
        default=["~/.lnmem/feedback/queries_2026-05-*.jsonl"],
    )
    parser.add_argument("--signals-dir", type=Path, default=Path("~/.lnmem/data").expanduser())
    parser.add_argument("--ckpt-v1", type=Path, required=True)
    parser.add_argument("--ckpt-v2", type=Path, required=True)
    parser.add_argument("--ckpt-extra", nargs="*", default=[])
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--default-persona-idx", type=int, default=30)
    parser.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--v2-label-order", default=DEFAULT_V2_LABELS)
    parser.add_argument("--r-neutral", type=float, default=0.35)
    parser.add_argument("--theta-offset-deg", type=float, default=0.0)
    return parser.parse_args(argv)


def _parse_ts(raw: str) -> datetime:
    return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)


def _expand_globs(patterns: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        expanded = sorted(Path(item) for item in glob.glob(str(Path(pattern).expanduser())))
        if expanded:
            paths.extend(expanded)
            continue
        maybe = Path(pattern).expanduser()
        if maybe.exists():
            paths.append(maybe)
    return sorted(dict.fromkeys(paths))


def _normalize_vlm_label(raw: object) -> str:
    if not isinstance(raw, dict):
        return "none"
    model_used = raw.get("model_used")
    return str(model_used) if model_used else "dict"


def _signal_path(raw: dict[str, Any]) -> str:
    path = raw.get("signal_path") or (raw.get("signal") or {}).get("brow_used_path")
    return str(path) if path else "missing"


def load_evaluable_events(query_paths: Sequence[Path], default_persona_idx: int) -> list[ReplayEvent]:
    """Collect brow_burst rows with explicit ground truth."""
    events: list[ReplayEvent] = []
    for path in query_paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                user_response = row.get("user_response") or {}
                if row.get("reason") != "brow_burst":
                    continue
                if user_response.get("label") not in {"match", "mismatch"}:
                    continue
                true_state = _normalize_true_state(user_response.get("true_state"))
                if true_state is None:
                    continue
                events.append(
                    ReplayEvent(
                        ts=str(row["ts"]),
                        frame_id=int(row["frame_id"]),
                        session_id=row.get("session_id"),
                        persona_idx=int(row.get("persona_idx", default_persona_idx) or default_persona_idx),
                        response_label=str(user_response.get("label")),
                        true_state=true_state,
                        signal_path=_signal_path(row),
                        vlm_label=_normalize_vlm_label(row.get("vlm_label")),
                        source_file=path.name,
                    )
                )
    return sorted(events, key=lambda item: item.ts)


def _dates_to_load(events: Sequence[ReplayEvent]) -> list[str]:
    out: set[str] = set()
    for event in events:
        day = _parse_ts(event.ts).date()
        out.add(day.isoformat())
        out.add((day - timedelta(days=1)).isoformat())
    return sorted(out)


def load_signal_sessions(signals_dir: Path, events: Sequence[ReplayEvent]) -> dict[str | None, list[SignalsFrame]]:
    """Load only signal dates needed for event windows."""
    sessions: dict[str | None, list[SignalsFrame]] = defaultdict(list)
    for day in _dates_to_load(events):
        for frame in load_signals(signals_dir, date_prefix=day):
            sessions[frame.session_id].append(frame)
    for frames in sessions.values():
        frames.sort(key=lambda item: (_parse_ts(item.ts), item.frame_id))
    return dict(sessions)


def slice_window(
    sessions: dict[str | None, list[SignalsFrame]],
    event: ReplayEvent,
    window_size: int,
) -> Tensor | None:
    """Locate the last `window_size` frames ending at the event frame or nearest ts."""
    frames = sessions.get(event.session_id)
    if not frames:
        return None
    frame_to_index = {frame.frame_id: idx for idx, frame in enumerate(frames)}
    index = frame_to_index.get(event.frame_id)
    if index is None:
        timestamps = [_parse_ts(frame.ts) for frame in frames]
        event_ts = _parse_ts(event.ts)
        pos = bisect.bisect_left(timestamps, event_ts)
        candidates = [idx for idx in (pos - 1, pos) if 0 <= idx < len(frames)]
        if not candidates:
            return None
        index = min(candidates, key=lambda idx: abs((timestamps[idx] - event_ts).total_seconds()))
    start = index - window_size + 1
    if start < 0:
        return None
    rows = [list(frame.vector) for frame in frames[start : index + 1]]
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


def parse_extra_ckpts(values: Sequence[str]) -> dict[str, Path]:
    """Parse `alias=path` or plain path specs into ordered extra checkpoint mapping."""
    parsed: dict[str, Path] = {}
    for raw in values:
        if "=" in raw:
            alias, path = raw.split("=", 1)
            parsed[alias.strip()] = Path(path).expanduser()
            continue
        path = Path(raw).expanduser()
        parsed[path.stem] = path
    return parsed


def build_runners(args: argparse.Namespace, label_order: Sequence[str]) -> dict[str, Any]:
    """Instantiate v1/v2 plus any extra v2-style checkpoints."""
    runners: dict[str, Any] = {
        "base_v1": BaseV1Runner(args.ckpt_v1, args.device),
        "base_v2": BaseV2Runner(args.ckpt_v2, args.device, label_order),
    }
    for alias, path in parse_extra_ckpts(args.ckpt_extra).items():
        runners[alias] = BaseV2Runner(path, args.device, label_order)
    return runners


def replay_events(
    events: Sequence[ReplayEvent],
    sessions: dict[str | None, list[SignalsFrame]],
    runners: Mapping[str, Any],
    window_size: int,
) -> tuple[list[ReplayRow], Counter[str]]:
    """Replay each evaluable event against all requested model runners."""
    rows: list[ReplayRow] = []
    skipped: Counter[str] = Counter()
    for event in events:
        window = slice_window(sessions, event, window_size)
        if window is None:
            skipped["missing_window"] += 1
            continue
        predictions = {name: runner.predict(window, event.persona_idx) for name, runner in runners.items()}
        rows.append(ReplayRow(event=event, predictions=predictions))
    return rows, skipped


def _mismatch_rate(rows: Sequence[ReplayRow], model_name: str) -> float:
    return sum(row.predictions[model_name].label != row.event.true_state for row in rows) / max(len(rows), 1)


def _state_accuracy(rows: Sequence[ReplayRow], model_name: str, state_name: str) -> float:
    relevant = [row for row in rows if row.event.true_state == state_name]
    if not relevant:
        return 0.0
    return sum(row.predictions[model_name].label == state_name for row in relevant) / len(relevant)


def _confusion_table(rows: Sequence[ReplayRow], model_name: str) -> str:
    header = "| true_state | " + " | ".join(STATE_LABELS) + " |\n"
    sep = "|" + "---|" * (len(STATE_LABELS) + 1) + "\n"
    counts: dict[str, Counter[str]] = {state: Counter() for state in STATE_LABELS}
    for row in rows:
        counts[row.event.true_state][row.predictions[model_name].label] += 1
    body = []
    for state in STATE_LABELS:
        cells = " | ".join(str(counts[state][pred]) for pred in STATE_LABELS)
        body.append(f"| {state} | {cells} |")
    return header + sep + "\n".join(body) + "\n"


def _russell_centers(r_neutral: float, theta_offset_deg: float) -> dict[str, Tensor]:
    radius = (1.0 + r_neutral) / 2.0
    centers: dict[str, Tensor] = {}
    for index, label in enumerate(STATE_LABELS):
        angle_deg = theta_offset_deg + (index + 0.5) * (360.0 / len(STATE_LABELS))
        angle_rad = math.radians(angle_deg)
        centers[label] = torch.tensor([radius * math.cos(angle_rad), radius * math.sin(angle_rad)], dtype=torch.float32)
    return centers


def _arc_length(pred_label: str, true_label: str, centers: Mapping[str, Tensor]) -> float:
    pred = torch.nn.functional.normalize(centers[pred_label], dim=0)
    true = torch.nn.functional.normalize(centers[true_label], dim=0)
    dot = float(torch.clamp(torch.dot(pred, true), -1.0, 1.0).item())
    return float(math.acos(dot))


def _russell_stats(rows: Sequence[ReplayRow], model_name: str, centers: Mapping[str, Tensor]) -> tuple[float, dict[str, float]]:
    per_state: dict[str, list[float]] = defaultdict(list)
    overall: list[float] = []
    for row in rows:
        distance = _arc_length(row.predictions[model_name].label, row.event.true_state, centers)
        per_state[row.event.true_state].append(distance)
        overall.append(distance)
    means = {state: (sum(values) / len(values) if values else 0.0) for state, values in per_state.items()}
    return (sum(overall) / len(overall) if overall else 0.0), means


def _entropy_stats(rows: Sequence[ReplayRow], model_name: str) -> tuple[float, float, float]:
    values = [row.predictions[model_name].entropy for row in rows]
    if not values:
        return 0.0, 0.0, 0.0
    return sum(values) / len(values), min(values), max(values)


def _neutral_top1_count(rows: Sequence[ReplayRow], model_name: str) -> int:
    return sum(row.predictions[model_name].raw_top1_label == "neutral_top1" for row in rows)


def _summary_table(rows: Sequence[ReplayRow], model_names: Sequence[str], centers: Mapping[str, Tensor]) -> str:
    lines = [
        "| model | mismatch_rate | focused_acc | distracted_acc | mean_russell | mean_entropy | top1_neutral_count |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name in model_names:
        mean_russell, _ = _russell_stats(rows, name, centers)
        mean_entropy, _, _ = _entropy_stats(rows, name)
        lines.append(
            f"| {name} | {_mismatch_rate(rows, name):.2%} | {_state_accuracy(rows, name, 'focused'):.2%} | "
            f"{_state_accuracy(rows, name, 'distracted'):.2%} | {mean_russell:.4f} | {mean_entropy:.4f} | "
            f"{_neutral_top1_count(rows, name)} |",
        )
    return "\n".join(lines)


def build_report(
    rows: Sequence[ReplayRow],
    events_total: int,
    skipped: Counter[str],
    args: argparse.Namespace,
    model_names: Sequence[str],
) -> str:
    """Render markdown report with confusion, Russell, and entropy metrics."""
    centers = _russell_centers(args.r_neutral, args.theta_offset_deg)
    label_counts = Counter(event.response_label for event in (row.event for row in rows))
    true_counts = Counter(event.true_state for event in (row.event for row in rows))
    lines = [
        "# 2026-05-21 base_v1 / base_v2 / base_v2.1 replay",
        "",
        f"- queries_glob: `{args.queries_glob}`",
        f"- signals_dir: `{args.signals_dir}`",
        f"- evaluable_events_total: **{events_total}**",
        f"- replayed_events: **{len(rows)}**",
        f"- skipped_events: **{sum(skipped.values())}** ({dict(skipped)})",
        f"- response_label_counts: **{dict(label_counts)}**",
        f"- true_state_counts: **{dict(true_counts)}**",
        f"- compared_models: **{', '.join(model_names)}**",
        "",
        "## Summary",
        "",
        _summary_table(rows, model_names, centers),
        "",
        "## Collapse Monitor",
        "",
        "| model | mean_entropy | min_entropy | max_entropy | top1_neutral_count | status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name in model_names:
        mean_e, min_e, max_e = _entropy_stats(rows, name)
        neutral_count = _neutral_top1_count(rows, name)
        status = "WARN collapse" if mean_e < 0.5 or neutral_count == len(rows) else "OK"
        lines.append(f"| {name} | {mean_e:.4f} | {min_e:.4f} | {max_e:.4f} | {neutral_count} | {status} |")
    lines.extend(["", "## Confusion Matrix", ""])
    for name in model_names:
        lines.extend([f"### {name}", "", _confusion_table(rows, name)])
    lines.extend(["## Russell Distance", ""])
    for name in model_names:
        overall, per_state = _russell_stats(rows, name, centers)
        lines.extend([f"### {name}", "", f"- overall_mean: **{overall:.4f}**", "", "| true_state | mean_arc_length |", "|---|---:|"])
        for state in STATE_LABELS:
            lines.append(f"| {state} | {per_state.get(state, 0.0):.4f} |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    query_paths = _expand_globs(args.queries_glob)
    if not query_paths:
        print("no query files matched --queries-glob", file=sys.stderr)
        return 2
    label_order = [part.strip() for part in args.v2_label_order.split(",") if part.strip()]
    events = load_evaluable_events(query_paths, args.default_persona_idx)
    if not events:
        print("no evaluable brow_burst events found", file=sys.stderr)
        return 2
    sessions = load_signal_sessions(args.signals_dir.expanduser(), events)
    runners = build_runners(args, label_order)
    rows, skipped = replay_events(events, sessions, runners, args.window_size)
    if not rows:
        print("all evaluable events were skipped during window lookup", file=sys.stderr)
        return 2
    model_names = list(runners.keys())
    report = build_report(rows, len(events), skipped, args, model_names)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(report + "\n", encoding="utf-8")
    print(f"replayed_events={len(rows)} skipped={sum(skipped.values())}", file=sys.stderr)
    print(f"models={model_names}", file=sys.stderr)
    print(f"report={args.output_report}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
