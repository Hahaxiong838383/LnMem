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

_AU_SCALE = 12.0
_CHEEK_BASELINE_FACE_RATIO = 0.40
_BROW_DIST_BASELINE = 0.17  # 中性时归一值（mediapipe 实测参考）
_BROW_FURROW_SCALE = 5.0    # 调到让皱眉饱和 1.0、中性接近 0
_EXPRESSION_DEFAULTS = {
    "smile_intensity": 0.0,
    "brow_furrow": 0.0,
    "cheek_raise": 0.0,
    "lip_corner_depress": 0.0,
    "jaw_drop": 0.0,
    "mouth_open_ratio": 0.0,
}


@dataclass
class FaceSignalConfig:
    """Face signal config. / 面部信号参数占位。"""

    max_num_faces: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    use_pyfeat: bool = False


class FaceSignalExtractor:
    """Extract coarse face signals. / 提取 Day 1 粗粒度面部信号。"""

    def __init__(self, config: FaceSignalConfig | None = None) -> None:
        self._config = config or FaceSignalConfig()
        self._face_mesh: Any | None = None
        self._au_extractor: Any | None = None

    def open(self) -> None:
        """Initialize extractor. / 初始化面部提取器。"""
        if self._config.use_pyfeat:
            try:
                from poc.src.vision.au_extractor import AUExtractor

                self._au_extractor = AUExtractor()
            except Exception:
                self._au_extractor = None
        if not _MEDIAPIPE_AVAILABLE:
            self._face_mesh = None
            return
        face_mesh = mediapipe.solutions.face_mesh  # type: ignore[union-attr]
        self._face_mesh = face_mesh.FaceMesh(
            max_num_faces=self._config.max_num_faces,
            min_detection_confidence=self._config.min_detection_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            refine_landmarks=True,
        )

    def close(self) -> None:
        """Release extractor. / 释放提取器资源。"""
        if self._face_mesh is not None and hasattr(self._face_mesh, "close"):
            self._face_mesh.close()
        self._face_mesh = None
        self._au_extractor = None

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
        expression = _extract_expression_features(landmarks=landmarks, landmarks_n=landmarks_n)
        if self._au_extractor is not None and frame is not None:
            try:
                from poc.src.vision.feature_calibrator import calibrate_to_lnn_expression
                au_dict, _emo, pose_dict = self._au_extractor.extract(frame)
                expression = calibrate_to_lnn_expression(au_dict)
                # LNN-FIRST: 头部姿态切到 py-feat img2pose（覆盖上面 MediaPipe landmark 估算）
                if pose_dict:
                    head_yaw_deg = float(pose_dict.get("yaw_deg", head_yaw_deg))
                    head_pitch_deg = float(pose_dict.get("pitch_deg", head_pitch_deg))
                    gaze_stability = _clamp01(
                        1.0 - min(1.0, (abs(head_yaw_deg) + abs(head_pitch_deg)) / 120.0)
                    )
            except Exception:
                expression = _extract_expression_features(landmarks=landmarks, landmarks_n=landmarks_n)

        return {
            "eye_closure": eye_closure,
            "head_pitch_deg": round(float(head_pitch_deg), 3),
            "head_yaw_deg": round(float(head_yaw_deg), 3),
            "gaze_stability": round(float(gaze_stability), 3),
            **expression,
            "landmarks_n": landmarks_n,
        }


def _clamp01(value: float) -> float:
    """Clamp float to [0,1]. / 将浮点裁剪到 0-1。"""
    return max(0.0, min(1.0, float(value)))


def _extract_expression_features(landmarks: list[Any], landmarks_n: int) -> dict[str, float]:
    """Extract AU features from landmarks. / 从 landmarks 提取 AU 表情特征。"""
    required_idx = [10, 13, 14, 55, 61, 112, 152, 234, 285, 291, 341, 454]
    if landmarks_n <= max(required_idx):
        return dict(_EXPRESSION_DEFAULTS)

    left = landmarks[234]
    right = landmarks[454]
    up = landmarks[10]
    down = landmarks[152]
    mouth_up = landmarks[13]
    mouth_down = landmarks[14]
    mouth_left = landmarks[61]
    mouth_right = landmarks[291]
    brow_left = landmarks[55]
    brow_right = landmarks[285]
    lid_left = landmarks[112]
    lid_right = landmarks[341]

    face_w = max(1e-6, abs(right.x - left.x))
    face_h = max(1e-6, abs(down.y - up.y))
    mouth_center_y = float(mouth_up.y)
    corners_avg_y = (float(mouth_left.y) + float(mouth_right.y)) / 2.0

    smile_intensity = _clamp01((mouth_center_y - corners_avg_y) / face_h * _AU_SCALE)
    lip_corner_depress = _clamp01((corners_avg_y - mouth_center_y) / face_h * _AU_SCALE)
    brow_dist = abs(float(brow_right.x) - float(brow_left.x))
    brow_dist_norm = brow_dist / max(face_w, 1e-6)  # 中性时 ~0.22
    brow_furrow = _clamp01(
        (_BROW_DIST_BASELINE - brow_dist_norm) / _BROW_DIST_BASELINE * _BROW_FURROW_SCALE
    )

    lip_gap = abs(float(mouth_down.y) - float(mouth_up.y))
    jaw_drop = _clamp01(lip_gap / face_h * _AU_SCALE)
    mouth_w = max(1e-6, abs(float(mouth_right.x) - float(mouth_left.x)))
    mouth_open_ratio = _clamp01(lip_gap / mouth_w * _AU_SCALE)

    cheek_base = max(1e-6, face_h * _CHEEK_BASELINE_FACE_RATIO)
    if landmarks_n > 473:
        pupil_left_y = float(landmarks[468].y)
        pupil_right_y = float(landmarks[473].y)
    else:
        pupil_left_y = (float(landmarks[159].y) + float(landmarks[145].y)) / 2.0 if landmarks_n > 159 else float(mouth_up.y)
        pupil_right_y = (float(landmarks[386].y) + float(landmarks[374].y)) / 2.0 if landmarks_n > 386 else float(mouth_up.y)
    lower_lid_dist = (abs(float(lid_left.y) - pupil_left_y) + abs(float(lid_right.y) - pupil_right_y)) / 2.0
    cheek_raise = _clamp01(1.0 - (lower_lid_dist / cheek_base) * _AU_SCALE)

    return {
        "smile_intensity": round(float(smile_intensity), 3),
        "brow_furrow": round(float(brow_furrow), 3),
        "cheek_raise": round(float(cheek_raise), 3),
        "lip_corner_depress": round(float(lip_corner_depress), 3),
        "jaw_drop": round(float(jaw_drop), 3),
        "mouth_open_ratio": round(float(mouth_open_ratio), 3),
    }
