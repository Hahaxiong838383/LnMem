"""Runtime config for realtime daemon. / 实时采集 daemon 配置与启动前自检。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import toml

DEFAULT_CONFIG: dict[str, dict[str, Any]] = {
    "capture": {
        "camera_index": 0,
        "target_fps": 1.0,
        "resolution": [1280, 720],
        "device_name": "air",
    },
    "storage": {
        "data_dir": "~/.lnmem/data",
        "raw_video_retention_hours": 24,
        "signal_jsonl_retention_days": 365,
        "compress_codec": "h264",
        "compress_quality": "medium",
    },
    "pipeline": {
        "enable_pyfeat": True,
        # LNN-FIRST: mediapipe 头姿路径已由 py-feat img2pose 取代，此字段保留以兼容旧配置文件但不再生效
        "enable_mediapipe": False,
        "enable_vad": False,
    },
    "feedback": {
        # Day 9+ active learning: 每小时主动询问次数上下限
        "query_min_per_hour": 1,
        "query_max_per_hour": 3,
        # 触发询问的置信度阈值（低于此值才询问）
        "query_confidence_threshold": 0.65,
        # 勿扰模式：前台 App 在此列表时抑制询问
        "dnd_apps": ["zoom", "loom", "facetime", "obs"],
    },
}
_BYTES_PER_FRAME = {"low": 25_000, "medium": 40_000, "high": 70_000}
MAX_CPU_PCT = 30
MAX_RAM_MB = 2_048
MAX_DISK_BYTES_PER_DAY = 5 * 1024 * 1024 * 1024
HTTP_HOST = "127.0.0.1"
HTTP_PORT = 9876


@dataclass(frozen=True)
class CaptureConfig:
    camera_index: int
    target_fps: float
    resolution: tuple[int, int]
    device_name: str


@dataclass(frozen=True)
class StorageConfig:
    data_dir: Path
    raw_video_retention_hours: int
    signal_jsonl_retention_days: int
    compress_codec: str
    compress_quality: str


@dataclass(frozen=True)
class PipelineConfig:
    enable_pyfeat: bool
    enable_mediapipe: bool
    enable_vad: bool


@dataclass(frozen=True)
class FeedbackConfig:
    """Day 9+ active learning 反馈参数（V0.1 采集层预留）。"""
    query_min_per_hour: int
    query_max_per_hour: int
    query_confidence_threshold: float
    dnd_apps: tuple[str, ...]


@dataclass(frozen=True)
class DaemonConfig:
    capture: CaptureConfig
    storage: StorageConfig
    pipeline: PipelineConfig
    feedback: FeedbackConfig
    config_path: Path

    def estimate_daily_video_bytes(self) -> int:
        base = _BYTES_PER_FRAME[self.storage.compress_quality]
        pixels = self.capture.resolution[0] * self.capture.resolution[1]
        scale = pixels / float(1280 * 720)
        frames_per_day = self.capture.target_fps * 24 * 60 * 60
        return int(base * scale * frames_per_day)


def default_config_path() -> Path:
    return Path.home() / ".lnmem" / "daemon.toml"


def _dual_mode_query_max(toml_value: int) -> int:
    """V0.3.1 dual-mode: demo 模式覆盖 query_max_per_hour 到 60（1min 1 个）。

    优先级：env LNMEM_TRIGGER_MODE > ~/.lnmem/trigger_mode 文件 > toml 默认值。
    与 trigger.py 同源逻辑一致。
    """
    import os as _os
    mode = _os.environ.get("LNMEM_TRIGGER_MODE", "").strip().lower()
    if mode not in ("demo", "work"):
        mode_file = _os.path.expanduser("~/.lnmem/trigger_mode")
        try:
            if _os.path.exists(mode_file):
                with open(mode_file) as _f:
                    mode = _f.read().strip().lower()
        except Exception:
            mode = ""
    if mode == "demo":
        return 60  # 1min 1 询问，足够 demo 节奏
    return toml_value  # work: toml 配置 (默认 3)


def load_config(
    path: str | Path | None = None,
    device_override: str | None = None,
    camera_override: int | None = None,
) -> DaemonConfig:
    config_path = Path(path).expanduser() if path else default_config_path()
    if not config_path.exists():
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(toml.dumps(DEFAULT_CONFIG), encoding="utf-8")

    loaded = toml.load(config_path)
    merged = _merge_defaults(loaded)
    if device_override:
        merged["capture"]["device_name"] = device_override
    if camera_override is not None:
        merged["capture"]["camera_index"] = int(camera_override)

    config = DaemonConfig(
        capture=CaptureConfig(
            camera_index=int(merged["capture"]["camera_index"]),
            target_fps=float(merged["capture"]["target_fps"]),
            resolution=tuple(int(v) for v in merged["capture"]["resolution"]),
            device_name=str(merged["capture"]["device_name"]),
        ),
        storage=StorageConfig(
            data_dir=Path(str(merged["storage"]["data_dir"])).expanduser(),
            raw_video_retention_hours=int(merged["storage"]["raw_video_retention_hours"]),
            signal_jsonl_retention_days=int(merged["storage"]["signal_jsonl_retention_days"]),
            compress_codec=str(merged["storage"]["compress_codec"]).lower(),
            compress_quality=str(merged["storage"]["compress_quality"]).lower(),
        ),
        pipeline=PipelineConfig(
            enable_pyfeat=bool(merged["pipeline"]["enable_pyfeat"]),
            enable_mediapipe=bool(merged["pipeline"]["enable_mediapipe"]),
            enable_vad=bool(merged["pipeline"]["enable_vad"]),
        ),
        feedback=FeedbackConfig(
            query_min_per_hour=int(merged["feedback"]["query_min_per_hour"]),
            # V0.3.1 dual-mode (2026-05-16 川哥钦定): demo 时 rate-limit 放宽到 60/h（1min 1 个）
            query_max_per_hour=_dual_mode_query_max(int(merged["feedback"]["query_max_per_hour"])),
            query_confidence_threshold=float(merged["feedback"]["query_confidence_threshold"]),
            dnd_apps=tuple(str(a) for a in merged["feedback"]["dnd_apps"]),
        ),
        config_path=config_path,
    )
    config.config_path.parent.mkdir(parents=True, exist_ok=True)
    config.storage.data_dir.mkdir(parents=True, exist_ok=True)
    return config


def preflight_check(config: DaemonConfig) -> dict[str, Any]:
    width, height = config.capture.resolution
    if width <= 0 or height <= 0:
        raise ValueError("capture.resolution must be positive")
    if config.capture.target_fps <= 0:
        raise ValueError("capture.target_fps must be > 0")
    if config.capture.target_fps > 5.0:
        raise ValueError("capture.target_fps must stay <= 5.0 in V0.1")
    if config.storage.compress_quality not in _BYTES_PER_FRAME:
        raise ValueError("storage.compress_quality must be high/medium/low")
    if config.storage.raw_video_retention_hours <= 0:
        raise ValueError("storage.raw_video_retention_hours must be > 0")
    if config.storage.signal_jsonl_retention_days <= 0:
        raise ValueError("storage.signal_jsonl_retention_days must be > 0")

    estimated = config.estimate_daily_video_bytes()
    warnings: list[str] = []
    if estimated > MAX_DISK_BYTES_PER_DAY:
        raise ValueError("estimated raw video usage exceeds 5GB/day budget")
    if config.pipeline.enable_vad:
        warnings.append("VAD is reserved for Day 9+, V0.1 keeps it disabled by default")

    return {
        "config_path": str(config.config_path),
        "data_dir": str(config.storage.data_dir),
        "estimated_daily_video_gb": round(estimated / (1024 ** 3), 3),
        "resource_limits": {"cpu_pct": MAX_CPU_PCT, "ram_mb": MAX_RAM_MB, "disk_gb_per_day": 5},
        "http_bind": f"{HTTP_HOST}:{HTTP_PORT}",
        "warnings": warnings,
    }


def _merge_defaults(loaded: dict[str, Any]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for section, defaults in DEFAULT_CONFIG.items():
        current = dict(defaults)
        current.update(dict(loaded.get(section, {})))
        merged[section] = current
    return merged


__all__ = [
    "DaemonConfig",
    "FeedbackConfig",
    "HTTP_HOST",
    "HTTP_PORT",
    "MAX_CPU_PCT",
    "MAX_RAM_MB",
    "default_config_path",
    "load_config",
    "preflight_check",
]
