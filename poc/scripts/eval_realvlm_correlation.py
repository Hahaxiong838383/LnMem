#!/usr/bin/env python3
"""Evaluate Day6 LTC generalization on real VLM-labeled signals."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

import numpy as np
import torch

from poc.src.fusion import FEATURE_DIM, FusionConfig, MultimodalFusion
from poc.src.lnn.contracts import LNNTrainConfig
from poc.src.lnn.dataset import _STATE_LABELS
from poc.src.lnn.model import DefaultLNNFactory

JSONL_PATH = Path("/Users/rayman.chen/.lnmem/data/signals_2026-05-09.jsonl")
CKPT_PATH = Path("poc/data/checkpoints/day6_ltc/checkpoint_day5.pt")
CSV_PATH = Path("/tmp/eval_realvlm_result.csv")
WINDOW = 60
STRIDE = 30
ANCHOR_TAIL = 20
DEFAULT_PERSONA = "vboss_001"


def _safe_float(value: Any, fallback: float) -> float:
    """Convert value to float or fallback when value is invalid."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _to_fusion_frame(row: dict[str, Any], idx: int) -> dict[str, Any]:
    """Convert real JSONL row into MultimodalFusion input schema."""
    face_detected = bool(row.get("face_detected", False))
    expression = row.get("expression") or {}
    head_pose = row.get("head_pose") or {}
    calibrated_au = row.get("calibrated_au") or {}

    face: dict[str, Any] | None = None
    if face_detected:
        face = {
            "eye_closure": _safe_float(calibrated_au.get("AU43"), 0.0),
            "head_pitch_deg": _safe_float(head_pose.get("pitch_deg"), 0.0),
            "head_yaw_deg": _safe_float(head_pose.get("yaw_deg"), 0.0),
            "gaze_stability": 0.5,  # real-time JSONL has no gaze_stability; keep neutral default
            "smile_intensity": _safe_float(expression.get("smile_intensity"), 0.0),
            "brow_furrow": _safe_float(expression.get("brow_furrow"), 0.0),
            "cheek_raise": _safe_float(expression.get("cheek_raise"), 0.0),
            "lip_corner_depress": _safe_float(expression.get("lip_corner_depress"), 0.0),
            "jaw_drop": _safe_float(expression.get("jaw_drop"), 0.0),
            "mouth_open_ratio": _safe_float(expression.get("mouth_open_ratio"), 0.0),
        }

    return {
        "ts": row.get("timestamp") or row.get("ts") or "",
        "frame_idx": int(row.get("frame_id", idx)),
        "face": face,
        "audio": None,
        "context": None,
    }


def _load_rows_sorted(path: Path) -> list[dict[str, Any]]:
    """Load and sort JSONL rows by timestamp/ts in ascending order."""
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    rows.sort(key=lambda r: (str(r.get("timestamp") or r.get("ts") or ""), int(r.get("frame_id", 0))))
    return rows


def _pearson(x: list[float], y: list[float]) -> float:
    """Compute Pearson r with numpy; returns NaN when undefined."""
    if len(x) < 2 or len(y) < 2:
        return math.nan
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) <= 1e-12 or np.std(y_arr) <= 1e-12:
        return math.nan
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def _fmt_mean_std(values: list[float]) -> str:
    """Format mean±std for terminal output."""
    if not values:
        return "n/a"
    mu = mean(values)
    sigma = pstdev(values) if len(values) > 1 else 0.0
    return f"{mu:.4f}±{sigma:.4f}"


def main() -> None:
    """Run real-data window inference and report VLM correlation."""
    if not JSONL_PATH.exists():
        raise FileNotFoundError(f"jsonl not found: {JSONL_PATH}")
    if not CKPT_PATH.exists():
        raise FileNotFoundError(f"checkpoint not found: {CKPT_PATH}")

    rows = _load_rows_sorted(JSONL_PATH)
    fusion = MultimodalFusion(FusionConfig())
    fused_vectors: list[tuple[float, ...]] = []
    for i, row in enumerate(rows):
        fused = fusion.fuse(_to_fusion_frame(row, i))
        if len(fused.vector) != FEATURE_DIM:
            raise RuntimeError(f"feature_dim mismatch: {len(fused.vector)} != {FEATURE_DIM}")
        fused_vectors.append(fused.vector)

    checkpoint = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    if not isinstance(cfg, LNNTrainConfig):
        raise TypeError(f"checkpoint config type mismatch: {type(cfg)!r}")
    if cfg.feature_dim != FEATURE_DIM:
        raise RuntimeError(f"checkpoint feature_dim={cfg.feature_dim}, expected FEATURE_DIM={FEATURE_DIM}")

    persona_to_idx: dict[str, int] = checkpoint["persona_to_idx"]
    persona_id = DEFAULT_PERSONA if DEFAULT_PERSONA in persona_to_idx else sorted(persona_to_idx.keys())[0]
    persona_idx = int(persona_to_idx[persona_id])

    factory = DefaultLNNFactory()
    state_model = factory.create_state_model(cfg)
    boundary_model = factory.create_boundary_model(cfg)
    state_model.load_state_dict(checkpoint["state_model"])
    boundary_model.load_state_dict(checkpoint["boundary_model"])
    state_model.eval()
    boundary_model.eval()

    result_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    vlm_confidences: list[float] = []

    for start in range(0, max(0, len(rows) - WINDOW + 1), STRIDE):
        end = start + WINDOW
        window_rows = rows[start:end]
        window_vecs = fused_vectors[start:end]
        tail = window_rows[-ANCHOR_TAIL:]
        anchors = [r.get("vlm_label") for r in tail if isinstance(r.get("vlm_label"), dict)]
        if not anchors:
            continue
        anchor = anchors[-1]
        x = torch.tensor(window_vecs, dtype=torch.float32).unsqueeze(0)  # (1, T, 18)
        p = torch.tensor([persona_idx], dtype=torch.long)

        with torch.no_grad():
            logits = state_model(x, p)
        last_logits = logits[0, -1, :]
        probs = torch.softmax(last_logits, dim=0)
        state_pred = int(torch.argmax(probs).item())
        lnn_score = float(torch.max(probs).item())

        vlm_valence = _safe_float(anchor.get("valence"), math.nan)
        vlm_arousal = _safe_float(anchor.get("arousal"), math.nan)
        vlm_conf = _safe_float(anchor.get("confidence"), math.nan)

        result_rows.append(
            {
                "window_idx": start // STRIDE,
                "state_pred": state_pred,
                "lnn_score": lnn_score,
                "vlm_valence": vlm_valence,
                "vlm_arousal": vlm_arousal,
                "vlm_confidence": vlm_conf,
                "vlm_au_description": str(anchor.get("au_description", "")),
            }
        )
        if not math.isnan(vlm_conf):
            scores.append(lnn_score)
            vlm_confidences.append(vlm_conf)

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "window_idx",
                "state_pred",
                "lnn_score",
                "vlm_valence",
                "vlm_arousal",
                "vlm_confidence",
                "vlm_au_description",
            ],
        )
        writer.writeheader()
        writer.writerows(result_rows)

    by_state_valence: dict[int, list[float]] = defaultdict(list)
    by_state_arousal: dict[int, list[float]] = defaultdict(list)
    for row in result_rows:
        if not math.isnan(float(row["vlm_valence"])):
            by_state_valence[int(row["state_pred"])].append(float(row["vlm_valence"]))
        if not math.isnan(float(row["vlm_arousal"])):
            by_state_arousal[int(row["state_pred"])].append(float(row["vlm_arousal"]))

    freq = Counter(int(r["state_pred"]) for r in result_rows)
    print("== State Frequency ==")
    for sid, cnt in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0])):
        label = _STATE_LABELS[sid] if 0 <= sid < len(_STATE_LABELS) else str(sid)
        print(f"state={sid:>2} ({label:<22}) count={cnt}")

    print("\n== VLM Mean±Std by Predicted State ==")
    for sid in sorted(freq.keys()):
        label = _STATE_LABELS[sid] if 0 <= sid < len(_STATE_LABELS) else str(sid)
        val_s = _fmt_mean_std(by_state_valence[sid])
        aro_s = _fmt_mean_std(by_state_arousal[sid])
        print(f"state={sid:>2} ({label:<22}) valence={val_s} arousal={aro_s}")

    pearson_r = _pearson(scores, vlm_confidences)
    print("\n== Correlation ==")
    print(f"Pearson r (lnn_score vs vlm_confidence): {pearson_r:.6f}")
    print(f"valid_windows={len(result_rows)} persona_id={persona_id} feature_dim={FEATURE_DIM}")
    print(f"csv={CSV_PATH}")


if __name__ == "__main__":
    main()
