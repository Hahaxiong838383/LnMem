"""Realtime capture daemon package. / 跨平台真人采集 daemon 包。"""

from .daemon_config import DaemonConfig, load_config, preflight_check

__all__ = ["DaemonConfig", "load_config", "preflight_check"]
