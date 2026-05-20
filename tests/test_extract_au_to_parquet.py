"""Tests for AU extraction parquet script; 覆盖批处理/失败回退/resume/视频 NaN 连续计数。"""
from __future__ import annotations

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import importlib.util

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "extract_au_to_parquet.py"
_spec = importlib.util.spec_from_file_location("extract_au_to_parquet", _SCRIPT)
assert _spec and _spec.loader
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def _make_image(path: Path) -> None:
    arr = np.full((32, 32, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(path), arr)


class FakeDetector:
    """Mock py-feat Detector; 可按文件名制造失败或 NaN。"""

    def __init__(
        self,
        fail_tokens: set[str] | None = None,
        nan_frames: set[int] | None = None,
        au_columns: tuple[str, ...] | None = None,
    ) -> None:
        self.fail_tokens = fail_tokens or set()
        self.nan_frames = nan_frames or set()
        self.au_columns = au_columns or tuple(mod.AU_NAMES)

    def detect_image(self, paths: list[str], output_size: int = 512, batch_size: int = 1) -> pd.DataFrame:
        if any(token in Path(p).name for p in paths for token in self.fail_tokens):
            if len(paths) > 1:
                raise RuntimeError("batch failure")
            raise RuntimeError("single failure")
        rows: list[dict[str, float | str]] = []
        for p in paths:
            name = Path(p).name
            frame_idx = mod.extract_frame_index_from_name(name)
            if frame_idx in self.nan_frames:
                row = {au: math.nan for au in self.au_columns}
            else:
                row = {au: 0.5 for au in self.au_columns}
            row["input"] = str(p)
            rows.append(row)
        return pd.DataFrame(rows)


def _read_df(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


def _read_audit(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_image_dir_batch_normal(tmp_path: Path, monkeypatch) -> None:
    """5 张图 + max_files=3 时应写 3 行且 face_detected=True。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    for i in range(5):
        _make_image(data_dir / f"img_{i}.jpg")
    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector())

    out = tmp_path / "au.parquet"
    rc = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--batch-size", "2",
        "--max-files", "3",
    ])
    assert rc == 0
    df = _read_df(out)
    assert len(df) == 3
    assert df["face_detected"].all()


def test_single_image_pyfeat_failure_yields_nan(tmp_path: Path, monkeypatch) -> None:
    """单图失败只影响该行：AU 全 NaN + face_detected=False。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    for i in range(5):
        _make_image(data_dir / f"img_{i}.jpg")
    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector(fail_tokens={"img_2"}))

    out = tmp_path / "au.parquet"
    rc = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--batch-size", "2",
    ])
    assert rc == 0
    df = _read_df(out)
    assert len(df) == 5
    failed = df[df["filename"].str.contains("img_2")].iloc[0]
    assert not bool(failed["face_detected"])
    assert all(pd.isna(failed[au]) for au in mod.AU_NAMES)
    ok = df[~df["filename"].str.contains("img_2")]
    assert ok["face_detected"].all()


def test_resume_skips_processed(tmp_path: Path, monkeypatch) -> None:
    """第二次 --resume 仅补新文件，结果无重复。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    for i in range(5):
        _make_image(data_dir / f"img_{i}.jpg")
    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector())

    out = tmp_path / "au.parquet"
    rc1 = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--max-files", "3",
    ])
    assert rc1 == 0

    rc2 = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--resume",
    ])
    assert rc2 == 0
    df = _read_df(out)
    assert len(df) == 5
    assert df[["filename", "frame_idx"]].drop_duplicates().shape[0] == 5


def test_video_frames_consecutive_nan_au_accumulates(tmp_path: Path, monkeypatch) -> None:
    """视频 3-5 帧全 NaN 时，frame 5=3，frame 6 重置 0。"""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "0.mp4"

    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for i in range(10):
        frame = np.full((32, 32, 3), i * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector(nan_frames={3, 4, 5}))

    out = tmp_path / "video.parquet"
    rc = mod.main([
        "--input-dir", str(video_dir),
        "--input-type", "video_frames",
        "--output-parquet", str(out),
        "--source-dataset", "veatic",
        "--batch-size", "4",
    ])
    assert rc == 0
    df = _read_df(out).sort_values("frame_idx")
    row5 = df[df["frame_idx"] == 5].iloc[0]
    row6 = df[df["frame_idx"] == 6].iloc[0]
    assert int(row5["consecutive_nan_au"]) == 3
    assert int(row6["consecutive_nan_au"]) == 0


def test_resume_preserves_consecutive_nan_au(tmp_path: Path, monkeypatch) -> None:
    """video_frames resume 后 NaN 连续计数应延续，而不是从 0/1 重新开始。"""
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "0.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (32, 32))
    for i in range(8):
        frame = np.full((32, 32, 3), i * 8, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector(nan_frames={1, 2, 3, 4}))
    monkeypatch.setattr(mod, "_sample_indices", lambda total, target: [0, 1, 2, 3])

    out = tmp_path / "resume-video.parquet"
    rc1 = mod.main([
        "--input-dir", str(video_dir),
        "--input-type", "video_frames",
        "--output-parquet", str(out),
        "--source-dataset", "veatic",
        "--batch-size", "4",
    ])
    assert rc1 == 0

    monkeypatch.setattr(mod, "_sample_indices", lambda total, target: list(range(total)))
    rc2 = mod.main([
        "--input-dir", str(video_dir),
        "--input-type", "video_frames",
        "--output-parquet", str(out),
        "--source-dataset", "veatic",
        "--batch-size", "4",
        "--resume",
    ])
    assert rc2 == 0
    df = _read_df(out).sort_values("frame_idx")
    row4 = df[df["frame_idx"] == 4].iloc[0]
    assert int(row4["consecutive_nan_au"]) == 4


def test_resume_ignores_stale_parts(tmp_path: Path, monkeypatch) -> None:
    """resume 时应忽略历史残留 parts，避免 stale 分片被重复 merge。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    for i in range(3):
        _make_image(data_dir / f"img_{i}.jpg")
    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector())

    out = tmp_path / "stale.parquet"
    rc1 = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
    ])
    assert rc1 == 0
    first = _read_df(out)
    assert len(first) == 3

    stale_root = out.with_suffix(out.suffix + ".parts")
    stale_dir = stale_root / "stale-run"
    stale_dir.mkdir(parents=True, exist_ok=True)
    stale_row = pa.Table.from_pylist([first.iloc[0].to_dict()], schema=mod.SCHEMA)
    pq.write_table(stale_row, stale_dir / "part-000000.parquet")

    rc2 = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--resume",
    ])
    assert rc2 == 0
    df = _read_df(out)
    assert len(df) == 3
    assert df[["filename", "frame_idx"]].drop_duplicates().shape[0] == 3


def test_au_schema_audit_aligned_passes(tmp_path: Path, monkeypatch) -> None:
    """20 列完全对齐时应静默通过并写 aligned audit。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    _make_image(data_dir / "img_0.jpg")
    monkeypatch.setattr(mod, "create_detector", lambda device, au_model: FakeDetector())
    monkeypatch.setattr(mod, "_get_pyfeat_version", lambda: "0.6.2")

    out = tmp_path / "aligned.parquet"
    rc = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
    ])
    assert rc == 0
    audit = _read_audit(mod._audit_path(out))
    assert audit["verdict"] == "aligned"
    assert audit["missing"] == []
    assert audit["extra"] == []
    assert audit["pyfeat_version"] == "0.6.2"


def test_au_schema_audit_missing_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    """缺 AU04 时应 SystemExit(2)，stderr 报 AU04，且 audit 仍落盘。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    _make_image(data_dir / "img_0.jpg")
    missing_columns = tuple(au for au in mod.AU_NAMES if au != "AU04")
    monkeypatch.setattr(
        mod,
        "create_detector",
        lambda device, au_model: FakeDetector(au_columns=missing_columns),
    )
    monkeypatch.setattr(mod, "_get_pyfeat_version", lambda: "0.6.2")

    out = tmp_path / "missing.parquet"
    try:
        mod.main([
            "--input-dir", str(data_dir),
            "--input-type", "image_dir",
            "--output-parquet", str(out),
            "--source-dataset", "mst_affectnet",
        ])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("expected SystemExit(2)")
    assert "AU04" in capsys.readouterr().err
    audit = _read_audit(mod._audit_path(out))
    assert audit["missing"] == ["AU04"]
    assert audit["verdict"] == "missing_fail_fast"


def test_au_schema_audit_extra_warns(tmp_path: Path, monkeypatch, capsys) -> None:
    """多 AU99 时应继续运行，stderr 告警并记录 extra_only_warning。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    _make_image(data_dir / "img_0.jpg")
    extra_columns = tuple(mod.AU_NAMES) + ("AU99",)
    monkeypatch.setattr(
        mod,
        "create_detector",
        lambda device, au_model: FakeDetector(au_columns=extra_columns),
    )
    monkeypatch.setattr(mod, "_get_pyfeat_version", lambda: "0.6.2")

    out = tmp_path / "extra.parquet"
    rc = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
    ])
    assert rc == 0
    assert "AU99" in capsys.readouterr().err
    audit = _read_audit(mod._audit_path(out))
    assert audit["verdict"] == "extra_only_warning"
    assert audit["extra"] == ["AU99"]


def test_au_schema_audit_allow_drift_skips(tmp_path: Path, monkeypatch, capsys) -> None:
    """--allow-au-drift 下缺 AU04 时不 fail-fast，并记录 skipped verdict。"""
    data_dir = tmp_path / "images"
    data_dir.mkdir()
    _make_image(data_dir / "img_0.jpg")
    missing_columns = tuple(au for au in mod.AU_NAMES if au != "AU04")
    monkeypatch.setattr(
        mod,
        "create_detector",
        lambda device, au_model: FakeDetector(au_columns=missing_columns),
    )
    monkeypatch.setattr(mod, "_get_pyfeat_version", lambda: "0.6.2")

    out = tmp_path / "allow.parquet"
    rc = mod.main([
        "--input-dir", str(data_dir),
        "--input-type", "image_dir",
        "--output-parquet", str(out),
        "--source-dataset", "mst_affectnet",
        "--allow-au-drift",
    ])
    assert rc == 0
    assert "AU04" in capsys.readouterr().err
    audit = _read_audit(mod._audit_path(out))
    assert audit["verdict"] == "missing_fail_fast_skipped_by_flag"
    assert audit["missing"] == ["AU04"]
