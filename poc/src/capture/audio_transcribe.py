"""Day 1 PoC skeleton for audio transcription. Day 8+ will add real capture."""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass

try:
    import sounddevice  # type: ignore
    _SOUNDDEVICE_AVAILABLE = True
except ImportError:
    sounddevice = None  # type: ignore[assignment]
    _SOUNDDEVICE_AVAILABLE = False


@dataclass
class AudioConfig:
    """Audio config. / 音频采集与转写参数占位。"""

    whisper_cli: str = "whisper-cli"
    default_lang: str = "zh"


class WhisperCppTranscriber:
    """Wrap whisper.cpp command. / whisper.cpp 子进程包装器。"""

    def __init__(self, config: AudioConfig | None = None) -> None:
        self._config = config or AudioConfig()
        self._opened = False

    def open(self) -> None:
        """Open transcriber. / 初始化转写器。"""
        self._opened = True

    def close(self) -> None:
        """Close transcriber. / 关闭转写器。"""
        self._opened = False

    def transcribe(self, duration_s: float = 0.0) -> dict[str, str | float]:
        """Transcribe latest audio chunk. / 转写最近音频片段。"""
        duration = max(0.0, float(duration_s))
        return {
            "transcript": "",
            "lang": self._config.default_lang,
            "duration_s": duration,
        }

    def transcribe_file(self, wav_path: str, lang: str | None = None) -> dict[str, str | float]:
        """Transcribe wav file. / 调用 whisper.cpp 转写 wav 文件。"""
        if not self._opened:
            raise RuntimeError("WhisperCppTranscriber is not opened")

        binary = shutil.which(self._config.whisper_cli)
        if binary is None:
            return {
                "transcript": "",
                "lang": lang or self._config.default_lang,
                "duration_s": 0.0,
            }

        cmd = [binary, "-f", wav_path, "-l", lang or self._config.default_lang]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        transcript = result.stdout.strip() if result.returncode == 0 else ""
        return {
            "transcript": transcript,
            "lang": lang or self._config.default_lang,
            "duration_s": 0.0,
        }
