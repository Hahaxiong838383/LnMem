"""Tests for daemon JSONL adapter.

覆盖过滤规则、Dataset shape 与评估函数兼容性。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from poc.training.daemon_dataset import DaemonAUDataset, build_daemon_au_cache
from poc.training.evaluate import measure_input_effective_dim
from poc.training.extract_au import PYFEAT_AU_COLUMNS
from poc.training.rare_au_monitor import compute_au_activation_frequency


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write JSONL. 英文一句话 + 中文说明：把测试样本按行写盘。"""
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _au_dict(v: float = 0.2) -> dict[str, float]:
    """Build full AU dict. 英文一句话 + 中文说明：按固定 20 列生成完整 AU。"""
    return {k: float(v) for k in PYFEAT_AU_COLUMNS}


def test_build_daemon_au_cache_filters_invalid_rows(tmp_path: Path) -> None:
    """Only valid face-detected rows with full AU should survive."""
    s1 = "11111111-1111-1111-1111-111111111111"
    s2 = "22222222-2222-2222-2222-222222222222"
    good = _au_dict(0.5)
    missing = good.copy()
    missing.pop("AU43")
    rows = [
        {"frame_id": 0, "face_detected": True, "session_id": s1, "calibrated_au": good, "raw_au": good},
        {"frame_id": 1, "face_detected": False, "session_id": s1},
        {"frame_id": 2, "face_detected": True, "session_id": s1, "calibrated_au": {}, "raw_au": {}},
        {"frame_id": 3, "face_detected": True, "session_id": s1, "calibrated_au": missing, "raw_au": missing},
        {"frame_id": 4, "face_detected": True, "session_id": s2, "calibrated_au": _au_dict(0.1), "raw_au": _au_dict(0.3)},
    ]
    signal = tmp_path / "signals_2026-05-22.jsonl"
    out = tmp_path / "daemon_cache.npz"
    _write_jsonl(signal, rows)

    cache = build_daemon_au_cache([signal], out, use_calibrated=True)

    assert out.exists()
    assert len(cache) == 2
    counts = sorted(v.shape[0] for v in cache.values())
    assert counts == [1, 1]
    for arr in cache.values():
        assert arr.shape == (1, 20)
        assert arr.dtype == np.float32


def test_daemon_dataset_shapes_and_eval_compat(tmp_path: Path) -> None:
    """Daemon dataset output should match evaluation consumers."""
    s1 = "33333333-3333-3333-3333-333333333333"
    s2 = "44444444-4444-4444-4444-444444444444"
    rows = []
    for fid in range(4):
        rows.append(
            {
                "frame_id": fid,
                "face_detected": True,
                "session_id": s1,
                "calibrated_au": _au_dict(0.2 + 0.01 * fid),
                "raw_au": _au_dict(0.5),
            }
        )
    for fid in range(3):
        rows.append(
            {
                "frame_id": fid,
                "face_detected": True,
                "session_id": s2,
                "calibrated_au": _au_dict(0.3 + 0.01 * fid),
                "raw_au": _au_dict(0.4),
            }
        )

    signal = tmp_path / "signals_2026-05-23.jsonl"
    out = tmp_path / "daemon_cache.npz"
    _write_jsonl(signal, rows)
    build_daemon_au_cache([signal], out, use_calibrated=True)

    ds = DaemonAUDataset(out)
    sample = ds[0]
    T = sample["event_tensor"].shape[0]

    assert len(ds) == 2
    assert sample["event_tensor"].shape == (T, 52)
    assert sample["labels_va"].shape == (T, 2)
    assert sample["start_frame"] == 0

    freq = compute_au_activation_frequency(ds)
    eff_dim = measure_input_effective_dim(ds)
    assert freq.shape == (20,)
    assert np.all(freq >= 0.0) and np.all(freq <= 1.0)
    assert eff_dim > 0.0
