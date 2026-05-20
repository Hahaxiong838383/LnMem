#!/usr/bin/env python3
"""Extract py-feat AUs to unified parquet schema; 支持 image_dir / video_frames / resume。"""
from __future__ import annotations

import argparse
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable

import cv2
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.vision.au_extractor import AU_NAMES

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
_FLUSH_EVERY = 1000

SCHEMA = pa.schema(
    [
        ("filename", pa.string()),
        ("frame_idx", pa.int32()),
        ("source_dataset", pa.string()),
        *[(au, pa.float32()) for au in AU_NAMES],
        ("face_detected", pa.bool_()),
        ("consecutive_nan_au", pa.int32()),
    ]
)


def extract_frame_index_from_name(name: str) -> int:
    m = re.search(r"(?:frame_|img_)(\d+)", name)
    return int(m.group(1)) if m else -1


def create_detector(device: str, au_model: str = "xgb") -> Any:
    del device
    from feat import Detector  # type: ignore

    return Detector(au_model=au_model, emotion_model="resmasknet")


def _has_finite_value(values: dict[str, Any]) -> bool:
    from numbers import Real

    for value in values.values():
        if isinstance(value, bool) or not isinstance(value, Real):
            continue
        try:
            if math.isfinite(float(value)):
                return True
        except (TypeError, ValueError):
            continue
    return False


def _list_images(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in _IMAGE_EXTS])


def _list_videos(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.mp4") if p.is_file()])


def _sample_indices(total: int, target: int) -> list[int]:
    if total <= 0:
        return []
    if target <= 0 or target >= total:
        return list(range(total))
    if target == 1:
        return [0]
    return sorted({round(i * (total - 1) / (target - 1)) for i in range(target)})


def _read_processed(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    table = pq.read_table(path, columns=["filename", "frame_idx"])
    return set(zip(table.column("filename").to_pylist(), table.column("frame_idx").to_pylist()))


def _read_video_nan_streaks(path: Path) -> dict[str, int]:
    if not path.exists():
        return {}
    table = pq.read_table(path, columns=["filename", "frame_idx", "consecutive_nan_au"])
    filenames = table.column("filename").to_pylist()
    frame_indices = table.column("frame_idx").to_pylist()
    streaks = table.column("consecutive_nan_au").to_pylist()
    latest: dict[str, tuple[int, int]] = {}
    for filename, frame_idx, streak in zip(filenames, frame_indices, streaks):
        current = latest.get(filename)
        if current is None or int(frame_idx) > current[0]:
            latest[filename] = (int(frame_idx), int(streak))
    return {filename: state[1] for filename, state in latest.items()}


def _to_record(filename: str, frame_idx: int, source_dataset: str, au_values: dict[str, float], nan_count: int) -> dict[str, Any]:
    return {
        "filename": filename,
        "frame_idx": int(frame_idx),
        "source_dataset": source_dataset,
        **{au: float(au_values.get(au, math.nan)) for au in AU_NAMES},
        "face_detected": _has_finite_value(au_values),
        "consecutive_nan_au": int(nan_count),
    }


def _map_predictions(paths: list[str], fex: Any) -> dict[str, dict[str, float]]:
    if fex is None or len(fex) == 0:
        return {}
    rows: dict[str, dict[str, float]] = {}
    if "input" in fex.columns:
        grouped = fex.groupby("input", sort=False).first().reset_index()
        for _, row in grouped.iterrows():
            p = str(row["input"])
            rows[p] = {au: float(row[au]) if au in grouped.columns else math.nan for au in AU_NAMES}
        return rows
    for i, p in enumerate(paths):
        if i >= len(fex):
            break
        row = fex.iloc[i]
        rows[p] = {au: float(row[au]) if au in fex.columns else math.nan for au in AU_NAMES}
    return rows


def _detect_batch(detector: Any, paths: list[str], batch_size: int) -> dict[str, dict[str, float]]:
    try:
        fex = detector.detect_image(paths, output_size=512, batch_size=batch_size)
        return _map_predictions(paths, fex)
    except Exception:
        out: dict[str, dict[str, float]] = {}
        for p in paths:
            try:
                fex = detector.detect_image([p], output_size=512, batch_size=1)
                out[p] = _map_predictions([p], fex).get(p, {au: math.nan for au in AU_NAMES})
            except Exception:
                out[p] = {au: math.nan for au in AU_NAMES}
        return out


def _flush_parts(part_dir: Path, records: list[dict[str, Any]], part_idx: int) -> int:
    if not records:
        return part_idx
    table = pa.Table.from_pylist(records, schema=SCHEMA)
    pq.write_table(table, part_dir / f"part-{part_idx:06d}.parquet")
    records.clear()
    return part_idx + 1


def _merge_parts(output: Path, part_dir: Path, keep_existing: bool) -> None:
    part_files = sorted(part_dir.glob("part-*.parquet"))
    if not part_files and not (keep_existing and output.exists()):
        return
    tables: list[pa.Table] = []
    if keep_existing and output.exists():
        tables.append(pq.read_table(output))
    tables.extend(pq.read_table(p) for p in part_files)
    if tables:
        out = pa.concat_tables(tables)
        tmp = output.with_suffix(output.suffix + ".tmp")
        pq.write_table(out.cast(SCHEMA), tmp)
        tmp.replace(output)
    for p in part_files:
        p.unlink(missing_ok=True)
    part_dir.rmdir()
    parent = part_dir.parent
    try:
        next(parent.iterdir())
    except StopIteration:
        parent.rmdir()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ADR-007 Phase 1.2 AU extractor to parquet")
    p.add_argument("--input-dir", required=True, type=Path)
    p.add_argument("--input-type", required=True, choices=["image_dir", "video_frames", "emotic_with_bbox"])
    p.add_argument("--output-parquet", required=True, type=Path)
    p.add_argument("--source-dataset", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device", choices=["cpu", "mps"], default="cpu")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--sample-frames-per-video", type=int, default=0)
    p.add_argument("--max-files", type=int, default=0)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.input_type == "emotic_with_bbox":
        raise NotImplementedError("EMOTIC pipeline 待 .mat 解析子任务，本轨先占接口位")
    if not args.input_dir.exists():
        print(f"[skip] input dir not found: {args.input_dir}", file=sys.stderr)
        return 0

    processed = _read_processed(args.output_parquet) if args.resume else set()
    resume_nan_streaks = (
        _read_video_nan_streaks(args.output_parquet) if args.resume and args.input_type == "video_frames" else {}
    )
    detector = create_detector(args.device, "xgb")
    parts_root = args.output_parquet.with_suffix(args.output_parquet.suffix + ".parts")
    run_id = f"{int(time.time())}-{os.getpid()}"
    part_dir = parts_root / run_id
    if parts_root.exists():
        stale_entries = [p for p in parts_root.iterdir() if p.name != run_id]
        if stale_entries:
            print(
                f"[warn] stale parts found under {parts_root}; ignored {len(stale_entries)} entries from previous runs",
                file=sys.stderr,
            )
    part_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    part_idx = 0
    total = 0
    nan_rows = 0

    def emit(record: dict[str, Any]) -> None:
        nonlocal part_idx, total, nan_rows
        records.append(record)
        total += 1
        if not record["face_detected"]:
            nan_rows += 1
        if len(records) >= _FLUSH_EVERY:
            part_idx = _flush_parts(part_dir, records, part_idx)
            print(f"[progress] rows={total} nan_rate={nan_rows/max(total,1):.3%}", file=sys.stderr)

    if args.input_type == "image_dir":
        images = _list_images(args.input_dir)
        if args.max_files > 0:
            images = images[: args.max_files]
        for i in tqdm(range(0, len(images), args.batch_size), desc="image batches"):
            batch = images[i : i + args.batch_size]
            todo = [p for p in batch if (str(p.relative_to(args.input_dir)), 0) not in processed]
            preds = _detect_batch(detector, [str(p) for p in todo], args.batch_size)
            for p in todo:
                rel = str(p.relative_to(args.input_dir))
                emit(_to_record(rel, 0, args.source_dataset, preds.get(str(p), {au: math.nan for au in AU_NAMES}), 0))
    else:
        videos = _list_videos(args.input_dir)
        if args.max_files > 0:
            videos = videos[: args.max_files]
        for video in tqdm(videos, desc="videos"):
            cap = cv2.VideoCapture(str(video))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = _sample_indices(n, args.sample_frames_per_video)
            rel = str(video.relative_to(args.input_dir))
            nan_streak = int(resume_nan_streaks.get(rel, 0))
            with tempfile.TemporaryDirectory(prefix="lnmem_au_") as tdir:
                tmp = Path(tdir)
                for i in range(0, len(indices), args.batch_size):
                    batch_idx = indices[i : i + args.batch_size]
                    paths: list[str] = []
                    keys: list[tuple[int, str]] = []
                    for frame_idx in batch_idx:
                        key = (rel, frame_idx)
                        if key in processed:
                            continue
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ok, frame = cap.read()
                        if not ok or frame is None:
                            continue
                        jpg = tmp / f"{video.stem}_frame_{frame_idx}.jpg"
                        cv2.imwrite(str(jpg), frame)
                        paths.append(str(jpg))
                        keys.append((frame_idx, str(jpg)))
                    preds = _detect_batch(detector, paths, args.batch_size)
                    for frame_idx, jpg in keys:
                        au = preds.get(jpg, {au: math.nan for au in AU_NAMES})
                        nan_streak = nan_streak + 1 if not _has_finite_value(au) else 0
                        emit(_to_record(rel, frame_idx, args.source_dataset, au, nan_streak))
            cap.release()

    part_idx = _flush_parts(part_dir, records, part_idx)
    _merge_parts(args.output_parquet, part_dir, keep_existing=args.resume)
    print(f"[done] rows={total} nan_rate={nan_rows/max(total,1):.3%}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    code = main()
    print(f"[timing] elapsed_s={time.perf_counter()-t0:.2f}", file=sys.stderr)
    raise SystemExit(code)
