"""Tests for hourly signal jsonl rotation and archive behavior."""

from __future__ import annotations

import json
from pathlib import Path
import sys
import threading

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.daemon_config import StorageConfig  # noqa: E402
from poc.src.realtime.storage_manager import StorageManager  # noqa: E402


def _signal(ts: str, frame_id: int) -> dict[str, object]:
    return {"ts": ts, "frame_id": frame_id, "value": frame_id}


def test_hourly_rotation_atomic(tmp_path: Path) -> None:
    config = StorageConfig(
        data_dir=tmp_path,
        raw_video_retention_hours=24,
        signal_jsonl_retention_days=365,
        compress_codec="h264",
        compress_quality="medium",
    )
    manager = StorageManager(config=config, stop_event=threading.Event())

    manager._write_signal(_signal("2026-06-01T10:59:59+00:00", frame_id=1))
    manager._write_signal(_signal("2026-06-01T11:00:00+00:00", frame_id=2))
    manager._write_signal(_signal("2026-06-01T11:00:01+00:00", frame_id=3))
    manager._close_signal_handle()

    archive_file = tmp_path / ".archive" / "signals_2026-06-01-10.jsonl"
    live_file = tmp_path / "signals_2026-06-01.jsonl"

    assert archive_file.exists()
    assert live_file.exists()

    archived_lines = [json.loads(line) for line in archive_file.read_text(encoding="utf-8").splitlines()]
    live_lines = [json.loads(line) for line in live_file.read_text(encoding="utf-8").splitlines()]

    assert [row["frame_id"] for row in archived_lines] == [1]
    assert [row["frame_id"] for row in live_lines] == [2, 3]
