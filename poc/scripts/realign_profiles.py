#!/usr/bin/env python3
"""Realign vboss expression profiles by rank-preserving target-range mapping.
按旧 attractor 排名百分位重映射到新范围，缓解训练-推理 OOD。"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from poc.src.lnn.expression_dynamics import EXPRESSION_DIMS, STATE_LABELS, ExpressionProfile, load_profile

PROFILE_DIR = Path("poc/data/personas/expression_profiles")
BACKUP_DIR = Path("poc/data/personas/expression_profiles_v1_backup")
ALIGN_DIMS = ("brow_furrow", "smile_intensity", "cheek_raise")
BASE_UPDATES = {"smile_intensity": 0.08, "brow_furrow": 0.08, "cheek_raise": 0.08}
TARGET_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "neutral": {"brow_furrow": (0.05, 0.12), "smile_intensity": (0.05, 0.15), "cheek_raise": (0.05, 0.12)},
    "focused": {"brow_furrow": (0.10, 0.20), "smile_intensity": (0.05, 0.15), "cheek_raise": (0.05, 0.15)},
    "distracted": {"brow_furrow": (0.05, 0.20), "smile_intensity": (0.05, 0.15), "cheek_raise": (0.05, 0.15)},
    "fatigued": {"brow_furrow": (0.05, 0.20), "smile_intensity": (0.00, 0.10), "cheek_raise": (0.00, 0.10)},
    "low_energy": {"brow_furrow": (0.05, 0.20), "smile_intensity": (0.00, 0.10), "cheek_raise": (0.00, 0.10)},
    "emotional_positive": {"brow_furrow": (0.00, 0.10), "smile_intensity": (0.65, 0.90), "cheek_raise": (0.50, 0.80)},
    "emotional_negative": {"brow_furrow": (0.70, 0.95), "smile_intensity": (0.00, 0.15), "cheek_raise": (0.00, 0.15)},
    "transitioning": {"brow_furrow": (0.10, 0.20), "smile_intensity": (0.10, 0.30), "cheek_raise": (0.10, 0.20)},
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _list_profile_paths(profile_dir: Path) -> list[Path]:
    paths = sorted(profile_dir.glob("vboss_*.json"))
    if len(paths) != 30:
        raise RuntimeError(f"expected 30 vboss profiles, got {len(paths)} from {profile_dir}")
    return paths


def _load_profiles(paths: list[Path]) -> dict[str, dict[str, Any]]:
    raw: dict[str, dict[str, Any]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw[ExpressionProfile.from_dict(payload).persona_id] = payload
    return raw


def _attractor(payload: dict[str, Any], state: str, dim: str) -> float:
    return _clamp(float(payload["base"][dim]) + float(payload["state_offset"][state][dim]), 0.0, 1.0)


def _compute_percentiles(raw_profiles: dict[str, dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    personae = sorted(raw_profiles)
    n = len(personae)
    if n < 2:
        raise RuntimeError("need at least 2 profiles for ranking")
    out = {pid: {s: {d: 0.0 for d in ALIGN_DIMS} for s in STATE_LABELS} for pid in personae}
    for state in STATE_LABELS:
        for dim in ALIGN_DIMS:
            ranked = sorted(personae, key=lambda pid: (_attractor(raw_profiles[pid], state, dim), pid))
            for rank, pid in enumerate(ranked):
                out[pid][state][dim] = rank / (n - 1)
    return out


def _target(state: str, dim: str, percentile: float) -> float:
    low, high = TARGET_RANGES[state][dim]
    return low + (high - low) * percentile


def _build_realigned_profile(payload: dict[str, Any], pct: dict[str, dict[str, float]]) -> dict[str, Any]:
    pid = str(payload["persona_id"])
    new_base = {dim: float(payload["base"][dim]) for dim in EXPRESSION_DIMS}
    new_base.update(BASE_UPDATES)
    new_offsets = {
        state: {dim: float(payload["state_offset"][state][dim]) for dim in EXPRESSION_DIMS}
        for state in STATE_LABELS
    }
    for state in STATE_LABELS:
        for dim in ALIGN_DIMS:
            new_offsets[state][dim] = round(_clamp(_target(state, dim, pct[state][dim]) - new_base[dim], -1.0, 1.0), 6)

    focused = _clamp(new_base["brow_furrow"] + new_offsets["focused"]["brow_furrow"], 0.0, 1.0)
    negative = _clamp(new_base["brow_furrow"] + new_offsets["emotional_negative"]["brow_furrow"], 0.0, 1.0)
    if negative - focused < 0.4:
        raise RuntimeError(f"{pid}: brow separation failed, focused={focused:.3f}, negative={negative:.3f}")

    updated = {
        "persona_id": pid,
        "base": new_base,
        "state_offset": new_offsets,
        "tau": {dim: float(payload["tau"][dim]) for dim in EXPRESSION_DIMS},
    }
    ExpressionProfile.from_dict(updated)
    return updated


def _write_profiles(paths: list[Path], updated: dict[str, dict[str, Any]], backup_dir: Path) -> None:
    backup_dir.mkdir(parents=True, exist_ok=True)
    for path in paths:
        backup_path = backup_dir / path.name
        if backup_path.exists():
            raise FileExistsError(f"backup already exists: {backup_path}")
        shutil.copy2(path, backup_path)
        path.write_text(json.dumps(updated[path.stem], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        load_profile(path)


def _print_plan(paths: list[Path], raw: dict[str, dict[str, Any]], updated: dict[str, dict[str, Any]]) -> None:
    print(f"[realign] profiles: {len(paths)}")
    print(f"[realign] source:   {PROFILE_DIR}")
    print(f"[realign] backup:   {BACKUP_DIR}")
    print("[realign] key metric deltas (focused_brow -> negative_brow, positive_smile):")
    for path in paths:
        pid = path.stem
        old, new = raw[pid], updated[pid]
        old_f, old_n = _attractor(old, "focused", "brow_furrow"), _attractor(old, "emotional_negative", "brow_furrow")
        old_p = _attractor(old, "emotional_positive", "smile_intensity")
        new_f, new_n = _attractor(new, "focused", "brow_furrow"), _attractor(new, "emotional_negative", "brow_furrow")
        new_p = _attractor(new, "emotional_positive", "smile_intensity")
        print(f"  {pid}: brow {old_f:.2f}->{new_f:.2f} / {old_n:.2f}->{new_n:.2f} (Δ={new_n-new_f:.2f}), pos_smile {old_p:.2f}->{new_p:.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Realign 30 vboss expression profiles by rank percentiles.")
    parser.add_argument("--dry-run", action="store_true", help="only print realignment plan")
    args = parser.parse_args()

    paths = _list_profile_paths(PROFILE_DIR)
    raw_profiles = _load_profiles(paths)
    percentiles = _compute_percentiles(raw_profiles)
    updated = {path.stem: _build_realigned_profile(raw_profiles[path.stem], percentiles[path.stem]) for path in paths}

    _print_plan(paths, raw_profiles, updated)
    if args.dry_run:
        print("[realign] dry-run complete (no file changed)")
        return

    _write_profiles(paths, updated, BACKUP_DIR)
    print(f"[realign] applied successfully, backup saved to: {BACKUP_DIR}")


if __name__ == "__main__":
    main()
