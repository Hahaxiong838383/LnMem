"""py-feat AU extractor with lazy detector load. / py-feat AU 提取器（懒加载）。

LNN-FIRST: py-feat Detector 是唯一面部信号来源，包含 AU / 情绪 / 头部姿态（img2pose）。
不引入任何第三方面部分析库（MediaPipe 等）。
"""

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

# py-feat FEAT_FACEPOSE_COLUMNS_3D → mapped to LNN signal keys
_POSE_COL_MAP = (
    ("Pitch", "pitch_deg"),
    ("Roll",  "roll_deg"),
    ("Yaw",   "yaw_deg"),
)
_POSE_ZERO: dict[str, float] = {"pitch_deg": 0.0, "roll_deg": 0.0, "yaw_deg": 0.0}


class AUExtractor:
    """Extract AU, emotion and head pose from frame using py-feat only.

    / 仅用 py-feat 从视频帧提取 AU 分数、情绪分数和头部姿态（img2pose）。
    LNN-FIRST 原则：禁止引入其他面部分析库作为备用路径。
    """

    def __init__(self, au_model: str = "xgb", tmp_dir: str = "/tmp/lnmem_pyfeat") -> None:
        self._au_model = au_model
        self._tmp_dir = Path(tmp_dir)
        self._tmp_dir.mkdir(parents=True, exist_ok=True)
        self._detector: Any | None = None
        self._frame_counter = 0
        self._last_au: dict[str, float] = {au: 0.0 for au in AU_NAMES}
        self._last_emo: dict[str, float] = {}
        self._last_pose: dict[str, float] = dict(_POSE_ZERO)

    def _ensure_detector(self) -> Any:
        """Create detector lazily. / 首次调用时懒加载 py-feat Detector。

        使用 py-feat 默认模型：
        - face_model = retinaface（人脸检测）
        - landmark_model = mobilefacenet（关键点）
        - au_model = xgb（FACS AU 20 通道）
        - emotion_model = resmasknet（情绪 7 类，本地 .pth）
        - facepose_model = img2pose（头部姿态 Pitch/Roll/Yaw，本地 .pth）
        - identity_model = facenet（人脸 ID，本地 .pth）
        所有模型文件均已通过 rsync 从 Air 同步到 Studio，不依赖网络。
        """
        if self._detector is None:
            from feat import Detector  # type: ignore

            self._detector = Detector(
                au_model=self._au_model,
                emotion_model="resmasknet",
            )
        return self._detector

    def extract(
        self, frame: Any
    ) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
        """Extract (au_dict, emotion_dict, pose_dict) from a BGR frame.

        / 从 BGR 视频帧提取 (AU 字典, 情绪字典, 头部姿态字典)。
        头部姿态来自 py-feat img2pose，键为 pitch_deg / roll_deg / yaw_deg。
        未检测到人脸时返回上一帧缓存值。
        """
        if frame is None:
            return self._last_au, self._last_emo, self._last_pose

        det = self._ensure_detector()
        self._frame_counter += 1
        tmp_path = self._tmp_dir / f"frame_{self._frame_counter % 3}.jpg"
        cv2.imwrite(str(tmp_path), frame)

        try:
            fex = det.detect_image([str(tmp_path)], output_size=512)
            if fex is None or len(fex) == 0:
                return self._last_au, self._last_emo, self._last_pose

            row = fex.iloc[0]

            # AU scores (20 channels)
            au_dict = {au: float(row[au]) for au in AU_NAMES if au in fex.columns}

            # Emotion scores (7 classes)
            emo_dict = {key: float(row[key]) for key in EMOTION_KEYS if key in fex.columns}

            # Head pose from img2pose (Pitch / Roll / Yaw → degrees)
            pose_dict: dict[str, float] = {}
            for col, lnn_key in _POSE_COL_MAP:
                if col in fex.columns:
                    val = float(row[col])
                    if val == val:  # skip NaN (no face detected by img2pose)
                        pose_dict[lnn_key] = val
            if len(pose_dict) == 3:
                self._last_pose = pose_dict

            self._last_au = au_dict
            self._last_emo = emo_dict
            return au_dict, emo_dict, self._last_pose
        except Exception:
            return self._last_au, self._last_emo, self._last_pose


__all__ = ["AUExtractor", "AU_NAMES", "EMOTION_KEYS"]
