#!/usr/bin/env python3
"""Sweep Russell V-A thresholds to discrete states; 连接 VEATIC parquet 与逐帧 V-A 标注。"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _parse_float_list(raw: str) -> list[float]:
    return [float(part.strip()) for part in raw.split(",") if part.strip()]


def _fmt_num(value: float) -> str:
    if isinstance(value, str):
        return value
    return "inf" if math.isinf(value) else f"{value:.4f}"


def _scheme_num(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text or "0"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADR-007 Phase 1.3 V-A to state sweep")
    p.add_argument("--input-parquet", required=True, type=Path)
    p.add_argument("--veatic-ratings-dir", required=True, type=Path)
    p.add_argument("--output-report", required=True, type=Path)
    p.add_argument("--r-neutral", default="0.15,0.2,0.25,0.3,0.35")
    p.add_argument("--theta-offset-deg", default="0,22.5")
    p.add_argument("--n-states", type=int, default=8)
    p.add_argument("--source-dataset-filter", default="veatic")
    return p.parse_args(argv)


def _load_rating_csv(path: Path, value_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"missing VEATIC rating csv: {path}")
    df = pd.read_csv(path, header=None)
    if df.shape[1] != 2:
        raise ValueError(f"[spec-conflict] expected 2 columns in {path.name}, got {df.shape[1]}")
    out = df.rename(columns={0: "frame_idx", 1: value_name}).copy()
    out["frame_idx"] = out["frame_idx"].astype(int)
    out[value_name] = out[value_name].astype(float)
    return out


def load_va_frame_table(input_parquet: Path, ratings_dir: Path, source_filter: str) -> pd.DataFrame:
    df = pd.read_parquet(input_parquet, columns=["filename", "frame_idx", "source_dataset"])
    need = {"filename", "frame_idx", "source_dataset"}
    if not need.issubset(df.columns):
        raise ValueError(f"[spec-conflict] parquet missing columns: {sorted(need - set(df.columns))}")
    base = df[df["source_dataset"] == source_filter][["filename", "frame_idx"]].drop_duplicates()
    if base.empty:
        raise ValueError(f"no rows after source_dataset=={source_filter!r} filter")
    merged_parts: list[pd.DataFrame] = []
    for filename, frames in base.groupby("filename", sort=False):
        stem = Path(str(filename)).stem
        valence = _load_rating_csv(ratings_dir / f"{stem}_valence.csv", "valence")
        arousal = _load_rating_csv(ratings_dir / f"{stem}_arousal.csv", "arousal")
        joined = valence.merge(arousal, on="frame_idx", how="inner")
        part = frames.merge(joined, on="frame_idx", how="inner").copy()
        part["filename"] = str(filename)
        dropped = len(frames) - len(part)
        if dropped:
            print(f"[warn] {filename}: dropped {dropped} frames without V-A join", file=sys.stderr)
        merged_parts.append(part)
    if not merged_parts:
        raise ValueError("no VEATIC frames joined with valence/arousal csv")
    out = pd.concat(merged_parts, ignore_index=True)
    return out.sort_values(["filename", "frame_idx"]).reset_index(drop=True)


def map_va_to_state(
    valence: np.ndarray | pd.Series,
    arousal: np.ndarray | pd.Series,
    r_neutral: float,
    theta_offset_deg: float,
    n_states: int = 8,
) -> np.ndarray:
    v = np.asarray(valence, dtype=float)
    a = np.asarray(arousal, dtype=float)
    radius = np.sqrt(v * v + a * a)
    states = np.zeros(len(v), dtype=np.int32)
    mask = radius >= r_neutral
    theta = (np.degrees(np.arctan2(a[mask], v[mask])) - theta_offset_deg) % 360.0
    sector = (theta // (360.0 / n_states)).astype(np.int32) + 1
    states[mask] = np.clip(sector, 1, n_states)
    return states


def _state_centers(states: np.ndarray, r_neutral: float, theta_offset_deg: float, n_states: int) -> tuple[np.ndarray, np.ndarray]:
    v_center = np.zeros(len(states), dtype=float)
    a_center = np.zeros(len(states), dtype=float)
    mask = states > 0
    if not np.any(mask):
        return v_center, a_center
    radius = (1.0 + r_neutral) / 2.0
    angle_deg = theta_offset_deg + ((states[mask] - 1) + 0.5) * (360.0 / n_states)
    angle_rad = np.radians(angle_deg)
    v_center[mask] = radius * np.cos(angle_rad)
    a_center[mask] = radius * np.sin(angle_rad)
    return v_center, a_center


def evaluate_scheme(df: pd.DataFrame, r_neutral: float, theta_offset_deg: float, n_states: int = 8) -> dict[str, Any]:
    states = map_va_to_state(df["valence"], df["arousal"], r_neutral, theta_offset_deg, n_states=n_states)
    counts = np.bincount(states, minlength=n_states + 1)
    non_neutral = counts[1:]
    coverage_unbounded = int(non_neutral.min()) == 0
    coverage: float | str = "inf" if coverage_unbounded else float(non_neutral.max() / non_neutral.min())
    switches = 0
    eval_df = df.assign(state=states)
    for _, video_df in eval_df.groupby("filename", sort=False):
        video_states = video_df["state"].to_numpy(dtype=np.int32)
        if len(video_states) > 1:
            switches += int(np.count_nonzero(video_states[1:] != video_states[:-1]))
    v_center, a_center = _state_centers(states, r_neutral, theta_offset_deg, n_states)
    mse = float(np.mean((df["valence"].to_numpy() - v_center) ** 2 + (df["arousal"].to_numpy() - a_center) ** 2))
    probs = counts[: n_states + 1].astype(float) / max(1, int(len(states)))
    nz = probs > 0
    entropy = float(-(probs[nz] * np.log2(probs[nz])).sum())
    return {
        "scheme_id": f"russell8_rn{_scheme_num(r_neutral)}_theta{_scheme_num(theta_offset_deg)}",
        "name": "Russell circumplex 8-state",
        "params": {"r_neutral": r_neutral, "theta_offset_deg": theta_offset_deg, "n_states": n_states},
        "metrics": {
            "coverage_balance": coverage,
            "coverage_unbounded": coverage_unbounded,
            "boundary_stability": float(switches / len(eval_df)),
            "va_reconstruction_mse": mse,
            "state_entropy": entropy,
        },
        "state_counts": counts.astype(int).tolist(),
    }


def _coverage_pass(candidate: dict[str, Any]) -> bool:
    cov = candidate["metrics"]["coverage_balance"]
    return not isinstance(cov, str) and cov <= 5


def pick_recommendation(candidates: list[dict[str, Any]]) -> dict[str, str]:
    pool = [c for c in candidates if _coverage_pass(c)]
    if not pool:
        best = min(candidates, key=lambda c: (c["metrics"]["boundary_stability"], -c["metrics"]["state_entropy"]))
        return {
            "scheme_id": best["scheme_id"],
            "reason": "no candidate passed coverage_balance<=5; fallback prefers boundary+entropy",
        }
    ranked = sorted(pool, key=lambda c: (c["metrics"]["boundary_stability"], c["metrics"]["va_reconstruction_mse"]))
    boundary_best = ranked[0]
    entropy_better = next(
        (c for c in ranked[1:] if c["metrics"]["state_entropy"] - boundary_best["metrics"]["state_entropy"] >= 0.3),
        None,
    )
    chosen = entropy_better or boundary_best
    mse_best = min(pool, key=lambda c: (c["metrics"]["va_reconstruction_mse"], c["metrics"]["boundary_stability"]))
    if mse_best["metrics"]["va_reconstruction_mse"] < chosen["metrics"]["va_reconstruction_mse"] * 0.8:
        return {
            "scheme_id": mse_best["scheme_id"],
            "reason": "coverage_balance 通过 + va_reconstruction_mse 优势 >20%",
        }
    if entropy_better:
        return {
            "scheme_id": entropy_better["scheme_id"],
            "reason": "coverage_balance 通过 + boundary 次优但 state_entropy 明显更高",
        }
    return {
        "scheme_id": boundary_best["scheme_id"],
        "reason": "coverage_balance 通过 + boundary_stability 最低 + va_mse 未显著更优",
    }


def build_report(df: pd.DataFrame, r_neutral_values: list[float], theta_offsets: list[float], n_states: int = 8) -> dict[str, Any]:
    candidates = [evaluate_scheme(df, rn, theta, n_states=n_states) for rn in r_neutral_values for theta in theta_offsets]
    return {
        "n_frames_evaluated": int(len(df)),
        "v_range": [float(df["valence"].min()), float(df["valence"].max())],
        "a_range": [float(df["arousal"].min()), float(df["arousal"].max())],
        "candidate_schemes": candidates,
        "recommendation": pick_recommendation(candidates),
    }


def print_preview(report: dict[str, Any]) -> None:
    print("scheme_id                     coverage   boundary   va_mse   pass", file=sys.stderr)
    for candidate in report["candidate_schemes"]:
        metrics = candidate["metrics"]
        passed = "Y" if _coverage_pass(candidate) else "N"
        print(
            f"{candidate['scheme_id']:<28} {_fmt_num(metrics['coverage_balance']):>8} "
            f"{metrics['boundary_stability']:>10.4f} {metrics['va_reconstruction_mse']:>8.4f} {passed:>6}",
            file=sys.stderr,
        )
    rec = report["recommendation"]
    print(f"recommendation: {rec['scheme_id']} | {rec['reason']}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.n_states != 8:
        print("[spec-conflict] Phase 1.3 fixes --n-states at 8", file=sys.stderr)
        return 2
    try:
        df = load_va_frame_table(args.input_parquet, args.veatic_ratings_dir, args.source_dataset_filter)
        report = build_report(df, _parse_float_list(args.r_neutral), _parse_float_list(args.theta_offset_deg), n_states=args.n_states)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 2
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print_preview(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
