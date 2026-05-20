"""Tests for AU extraction parquet script; 覆盖批处理/失败回退/resume/视频 NaN 连续计数。"""
from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
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

    def __init__(self, fail_tokens: set[str] | None = None, nan_frames: set[int] | None = None) -> None:
        self.fail_tokens = fail_tokens or set()
        self.nan_frames = nan_frames or set()

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
                row = {au: math.nan for au in mod.AU_NAMES}
            else:
                row = {au: 0.5 for au in mod.AU_NAMES}
            row["input"] = str(p)
            rows.append(row)
        return pd.DataFrame(rows)


def _read_df(path: Path) -> pd.DataFrame:
    return pq.read_table(path).to_pandas()


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
