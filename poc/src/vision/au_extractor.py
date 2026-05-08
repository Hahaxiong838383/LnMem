"""py-feat AU extractor with lazy detector load. / py-feat AU 提取器（懒加载）。"""

from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

warnings.filterwarnings("ignore")

import cv2  # type: ignore

AU_NAMES = (
    "AU01",
    "AU02",
    "AU04",
    "AU05",
    "AU06",
    "AU07",
    "AU09",
    "AU10",
    "AU11",
    "AU12",
    "AU14",
    "AU15",
    "AU17",
    "AU20",
    "AU23",
    "AU24",
    "AU25",
    "AU26",
    "AU28",
    "AU43",
)

EMOTION_KEYS = ("anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral")


class AUExtractor:
    """Extract AU and emotion from frame. / 从视频帧提取 AU 与情绪分数。"""

    def __init__(self, au_model: str = "xgb", tmp_dir: str = "/tmp/lnmem_pyfeat") -> None:
        self._au_model = au_model
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._detector: Any | None = None
        self._frame_counter = 0
        self._last_au: dict[str, float] = {au: 0.0 for au in AU_NAMES}
        self._last_emo: dict[str, float] = {}

    def _ensure_detector(self) -> Any:
        """Create detector lazily. / 首次调用时懒加载 py-feat Detector。"""
        if self._detector is None:
            from feat import Detector  # type: ignore

            self._detector = Detector(au_model=self._au_model)
        return self._detector

    def extract(self, frame: Any) -> tuple[dict[str, float], dict[str, float]]:
        """Extract (au_dict, emotion_dict). / 提取 (AU 字典, 情绪字典)。"""
        if frame is None:
            return self._last_au, self._last_emo

        det = self._ensure_detector()
        self._frame_counter += 1
        tmp_path = self._tmp_dir / f"frame_{self._frame_counter % 3}.jpg"
        cv2.imwrite(str(tmp_path), frame)

        try:
            fex = det.detect_image([str(tmp_path)], output_size=512)
            if fex is None or len(fex) == 0:
                return self._last_au, self._last_emo

            row = fex.iloc[0]
            au_dict = {au: float(row[au]) for au in AU_NAMES if au in fex.columns}
            emo_dict = {key: float(row[key]) for key in EMOTION_KEYS if key in fex.columns}

            self._last_au = au_dict
            self._last_emo = emo_dict
            return au_dict, emo_dict
        except Exception:
            return self._last_au, self._last_emo


__all__ = ["AUExtractor", "AU_NAMES", "EMOTION_KEYS"]
