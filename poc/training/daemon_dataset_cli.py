"""CLI for daemon dataset cache build."""
from __future__ import annotations

import argparse
from pathlib import Path

from .daemon_dataset import build_daemon_au_cache


def _collect_signal_files(signal_dir: str | Path) -> list[Path]:
    """Collect daemon files. 英文一句话 + 中文说明：匹配 signals_*.jsonl 并排序。"""
    return sorted(Path(signal_dir).expanduser().glob("signals_*.jsonl"))


def _summarize_cache(cache: dict[str, object]) -> tuple[int, int, dict[str, int]]:
    """Build cache stats. 英文一句话 + 中文说明：生成会话数/总帧数/逐会话帧数。"""
    per_session = {k[3:]: int(v.shape[0]) for k, v in sorted(cache.items())}
    total_frames = int(sum(per_session.values()))
    return len(per_session), total_frames, per_session


def main() -> None:
    """CLI entrypoint. 英文一句话 + 中文说明：构建 daemon AU cache 并打印统计。"""
    p = argparse.ArgumentParser()
    p.add_argument("--signal-dir", default="~/.lnmem/data")
    p.add_argument("--out", required=True)
    p.add_argument("--use-raw", action="store_true", help="Use raw_au instead of calibrated_au")
    p.add_argument("--no-head-pose", action="store_true", help="Disable head pose cache build")
    p.add_argument("--no-screen", action="store_true", help="Disable screen activity cache build")
    args = p.parse_args()

    signal_files = _collect_signal_files(args.signal_dir)
    if not signal_files:
        raise FileNotFoundError(f"No signals_*.jsonl found under {args.signal_dir}")

    cache = build_daemon_au_cache(
        signal_files,
        args.out,
        use_calibrated=not args.use_raw,
        include_head_pose=not args.no_head_pose,
        include_screen=not args.no_screen,
    )
    n_sessions, total_frames, per_session = _summarize_cache(cache)
    print(f"saved: {args.out}")
    print(f"n_sessions={n_sessions}")
    print(f"total_frames={total_frames}")
    print("per_session_frames=")
    for sid, cnt in per_session.items():
        print(f"  {sid}: {cnt}")


if __name__ == "__main__":
    main()
