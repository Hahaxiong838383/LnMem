"""γ-3 Step 1: py-feat AU 抽取 + 缓存 .npz（spec §4.1 数据准备）

为 VEATIC 视频抽 24 AU per frame，落盘 .npz 避免重复抽取（10 视频 30min 一次性）.

Usage:
    # tiny spike（10 视频）
    python -m poc.training.extract_au --videos 10 --out poc/data/au_cache_tiny.npz

    # Phase γ Step 1 全量
    python -m poc.training.extract_au --videos all --out poc/data/au_cache_full.npz

输出 .npz schema：
    - au[video_id]: np.ndarray (n_frames, 24) AU 强度 [0, 1]
    - metadata: video_id → {n_frames, fps, duration_s}

参考实测发现（feedback_pyfeat_062_api_pitfalls.md）：
- py-feat 0.6.2 detect_image 只接路径不接 ndarray
- 列名是 AU01 不带 _r 后缀
- 输出范围 [0, 1] 不是 [0, 5]
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

# 12 个最 reliable 的 FACS AU（py-feat 0.6.2 默认输出，对应 LnMem spec §2.1 24 AU 子集）
PYFEAT_AU_COLUMNS = [
    "AU01", "AU02", "AU04", "AU05", "AU06", "AU07",
    "AU09", "AU10", "AU11", "AU12", "AU14", "AU15",
    "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
    "AU28", "AU43",
]


def extract_au_from_video(
    video_path: str | Path,
    detector,
    sample_every_n_frames: int = 1,
    max_frames: Optional[int] = None,
) -> tuple[np.ndarray, dict]:
    """从视频抽每帧 AU，返回 (n_frames, num_au) ndarray + metadata.

    Args:
        video_path: 视频路径
        detector: py-feat Detector 实例
        sample_every_n_frames: 抽帧间隔（>1 减少计算量，但要小心 spec §4.3 Sampler ±60 帧窗口）
        max_frames: 限制抽取帧数（spike 阶段用）

    Returns:
        au_array: (n_sampled_frames, len(PYFEAT_AU_COLUMNS))
        metadata: dict with video_id, n_frames, fps, duration_s, sampled_at
    """
    import cv2
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if max_frames:
        total_frames = min(total_frames, max_frames * sample_every_n_frames)

    # py-feat 期望文件路径而非 ndarray（实测发现）
    # detect_video 接路径
    t0 = time.time()
    out = detector.detect_video(str(video_path), skip_frames=sample_every_n_frames - 1)
    elapsed = time.time() - t0

    # 提取 AU 列（py-feat 默认输出 dataframe）
    au_data = []
    for col in PYFEAT_AU_COLUMNS:
        if col in out.columns:
            au_data.append(out[col].values)
        else:
            # py-feat 不输出某些 AU → 填 0
            au_data.append(np.zeros(len(out)))
    au_arr = np.stack(au_data, axis=1).astype(np.float32)
    # clip 到 [0, 1]
    au_arr = np.clip(au_arr, 0.0, 1.0)

    metadata = {
        "video_id": video_path.stem,
        "n_frames": total_frames,
        "n_sampled": len(au_arr),
        "fps": fps,
        "duration_s": total_frames / max(fps, 1.0),
        "sample_every_n_frames": sample_every_n_frames,
        "extract_time_s": elapsed,
    }
    return au_arr, metadata


def extract_au_batch(
    video_dir: str | Path,
    video_ids: list[str] | None = None,
    out_path: str | Path = "poc/data/au_cache_tiny.npz",
    sample_every_n_frames: int = 5,  # 30fps → 6fps，减少计算量
    max_frames_per_video: int = 300,  # spike 阶段每视频最多 300 帧（约 10s）
    device: str = "cpu",
) -> None:
    """批量抽取 → 落盘单个 .npz.

    Args:
        video_dir: VEATIC videos/ 目录
        video_ids: 指定视频 ID 列表（None = 全部）
        out_path: 输出 npz 路径
        sample_every_n_frames: 抽帧间隔（5 → 30fps 降到 6fps）
        max_frames_per_video: 每视频最大抽取帧数
    """
    from feat import Detector

    video_dir = Path(video_dir)
    if video_ids is None:
        video_ids = sorted([p.stem for p in video_dir.glob("*.mp4")], key=lambda x: int(x))

    detector = Detector(
        face_model="img2pose",
        au_model="xgb",
        emotion_model=None,  # 关掉省 5x 时间
        landmark_model=None,  # 同上
        identity_model=None,
        device=device,
    )

    cache = {}
    metadata_all = {}
    for vid in video_ids:
        video_path = video_dir / f"{vid}.mp4"
        if not video_path.exists():
            print(f"⚠️ skip {vid}: not found")
            continue
        print(f"抽取 {vid}.mp4 ...", flush=True)
        au_arr, meta = extract_au_from_video(
            video_path, detector,
            sample_every_n_frames=sample_every_n_frames,
            max_frames=max_frames_per_video,
        )
        cache[f"au_{vid}"] = au_arr
        metadata_all[vid] = meta
        print(f"  → {meta['n_sampled']} frames, {meta['extract_time_s']:.1f}s", flush=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **cache)
    meta_path = out_path.with_suffix(".meta.json")
    with meta_path.open("w") as f:
        json.dump(metadata_all, f, indent=2)
    print(f"\n✅ saved {len(cache)} videos to {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"   metadata: {meta_path}")


def load_au_cache(cache_path: str | Path) -> dict[str, np.ndarray]:
    """加载 .npz 缓存 → {video_id: au_array}"""
    data = np.load(cache_path)
    return {k[3:]: data[k] for k in data.files if k.startswith("au_")}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video-dir", default=None,
                   help="VEATIC videos/ 目录，None 则用 Studio 默认或环境变量 VEATIC_VIDEO_DIR")
    p.add_argument("--n-videos", type=int, default=10, help="抽几个视频（10=tiny spike，124=full）")
    p.add_argument("--out", default="poc/data/au_cache_tiny.npz")
    p.add_argument("--skip-frames", type=int, default=5, help="每 N 帧采样一次（默认 5: 30fps→6fps）")
    p.add_argument("--max-frames", type=int, default=300, help="每视频最大抽取帧数（spike 用）")
    args = p.parse_args()

    video_dir = args.video_dir or os.environ.get(
        "VEATIC_VIDEO_DIR",
        "/Users/fu-121/lnmem-data/veatic/veatic_extracted/videos"
    )
    video_ids = [str(i) for i in range(args.n_videos)] if args.n_videos < 124 else None
    extract_au_batch(
        video_dir=video_dir,
        video_ids=video_ids,
        out_path=args.out,
        sample_every_n_frames=args.skip_frames,
        max_frames_per_video=args.max_frames,
    )
