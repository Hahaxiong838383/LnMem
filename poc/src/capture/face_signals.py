"""Day 1 PoC skeleton for face signals. Day 8+ will add real capture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

try:
    import mediapipe  # type: ignore
    _MEDIAPIPE_AVAILABLE = True
except ImportError:
    mediapipe = None  # type: ignore[assignment]
    _MEDIAPIPE_AVAILABLE = False


@dataclass
class FaceSignalConfig:
    """Face signal config. / 面部信号参数占位。"""

    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class FaceSignalExtractor:
    """Extract coarse face signals. / 提取 Day 1 粗粒度面部信号。"""

    def __init__(self, config: FaceSignalConfig | None = None) -> None:
        self._config = config or FaceSignalConfig()
        self._face_mesh: Any | None = None

    def open(self) -> None:
        """Initialize extractor. / 初始化面部提取器。"""
        if not _MEDIAPIPE_AVAILABLE:
            self._face_mesh = None
            return
        face_mesh = mediapipe.solutions.face_mesh  # type: ignore[union-attr]
        self._face_mesh = face_mesh.FaceMesh(
            max_num_faces=self._config.max_num_faces,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
        )

    def close(self) -> None:
        """Release extractor. / 释放提取器资源。"""
        if self._face_mesh is not None and hasattr(self._face_mesh, "close"):
            self._face_mesh.close()
        self._face_mesh = None

    def capture_signals(self, frame: Any) -> dict[str, float | int]:
        """Capture one face signal set. / 采集单帧面部信号。"""
        base = {
            "eye_closure": 0.0,
            "head_pitch_deg": 0.0,
            "head_yaw_deg": 0.0,
            "gaze_stability": 0.0,
            "landmarks_n": 0,
        }
        if frame is None or self._face_mesh is None:
            return base

        rgb_frame = frame
        if _CV2_AVAILABLE and cv2 is not None and hasattr(cv2, "cvtColor"):
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # type: ignore[arg-type]

        result = self._face_mesh.process(rgb_frame)
        faces = getattr(result, "multi_face_landmarks", None)
        if not faces:
            return base

        landmarks = faces[0].landmark
        landmarks_n = len(landmarks)
        eye_gap_l = abs(landmarks[159].y - landmarks[145].y) if landmarks_n > 160 else 0.0
        eye_gap_r = abs(landmarks[386].y - landmarks[374].y) if landmarks_n > 386 else 0.0
        eye_closure = _clamp01(1.0 - (eye_gap_l + eye_gap_r) * 8.0)

        nose = landmarks[1] if landmarks_n > 1 else landmarks[0]
        left = landmarks[234] if landmarks_n > 234 else landmarks[0]
        right = landmarks[454] if landmarks_n > 454 else landmarks[-1]
        up = landmarks[10] if landmarks_n > 10 else landmarks[0]
        down = landmarks[152] if landmarks_n > 152 else landmarks[-1]

        center_x = (left.x + right.x) / 2.0
        center_y = (up.y + down.y) / 2.0
        width = max(1e-6, abs(right.x - left.x))
        height = max(1e-6, abs(down.y - up.y))

        head_yaw_deg = (nose.x - center_x) / width * 45.0
        head_pitch_deg = (nose.y - center_y) / height * 35.0
        gaze_stability = _clamp01(1.0 - min(1.0, (abs(head_yaw_deg) + abs(head_pitch_deg)) / 120.0))

        return {
            "eye_closure": eye_closure,
            "head_pitch_deg": round(float(head_pitch_deg), 3),
            "head_yaw_deg": round(float(head_yaw_deg), 3),
            "gaze_stability": round(float(gaze_stability), 3),
            "landmarks_n": landmarks_n,
        }


def _clamp01(value: float) -> float:
    """Clamp float to [0,1]. / 将浮点裁剪到 0-1。"""
    return max(0.0, min(1.0, float(value)))
