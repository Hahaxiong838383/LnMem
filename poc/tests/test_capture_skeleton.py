"""Day 1 PoC skeleton tests. Day 8+ will add real capture tests."""

from __future__ import annotations

from unittest import mock


def test_import_multimodal_capture_class() -> None:
    """Import MultimodalCapture. / 验证主类可导入。"""
    from poc.src.capture import MultimodalCapture

    assert MultimodalCapture.__name__ == "MultimodalCapture"


def test_import_face_signals_module() -> None:
    """Import face signal module. / 验证 face_signals 模块可导入。"""
    from poc.src.capture import face_signals

    assert hasattr(face_signals, "FaceSignalExtractor")


def test_import_streamlit_entry() -> None:
    """Import streamlit entrypoint. / 验证 streamlit 入口可导入。"""
    from poc.src.capture import streamlit_app

    assert hasattr(streamlit_app, "main")


def test_face_signal_happy_path_with_mocked_cv2_mediapipe() -> None:
    """Mock cv2/mediapipe and run one signal extraction. / mock 外部依赖跑 happy path。"""
    from poc.src.capture import face_signals

    class FakeCV2:
        """Minimal cv2 stub. / 最小 cv2 stub。"""

        COLOR_BGR2RGB = 1

        @staticmethod
        def cvtColor(frame: object, _code: int) -> object:
            return frame

    class FakeLandmark:
        """Single landmark stub. / 单个 landmark stub。"""

        def __init__(self, x: float, y: float) -> None:
            self.x = x
            self.y = y

    class FakeFace:
        """Face landmarks stub. / 人脸 landmarks stub。"""

        def __init__(self) -> None:
            self.landmark = [FakeLandmark(0.5, 0.5) for _ in range(468)]
            self.landmark[159] = FakeLandmark(0.45, 0.47)
            self.landmark[145] = FakeLandmark(0.45, 0.53)
            self.landmark[386] = FakeLandmark(0.55, 0.47)
            self.landmark[374] = FakeLandmark(0.55, 0.53)
            self.landmark[234] = FakeLandmark(0.3, 0.5)
            self.landmark[454] = FakeLandmark(0.7, 0.5)
            self.landmark[10] = FakeLandmark(0.5, 0.3)
            self.landmark[152] = FakeLandmark(0.5, 0.7)
            self.landmark[1] = FakeLandmark(0.52, 0.48)

    class FakeResult:
        """Process result stub. / process 输出 stub。"""

        def __init__(self) -> None:
            self.multi_face_landmarks = [FakeFace()]

    class FakeFaceMeshRunner:
        """FaceMesh runner stub. / FaceMesh 运行器 stub。"""

        def __init__(self, **_kwargs: object) -> None:
            self.closed = False

        def process(self, _frame: object) -> FakeResult:
            return FakeResult()

        def close(self) -> None:
            self.closed = True

    class FakeFaceMeshNamespace:
        """MediaPipe face mesh namespace stub. / face_mesh 命名空间 stub。"""

        FaceMesh = FakeFaceMeshRunner

    class FakeSolutions:
        """MediaPipe solutions stub. / solutions 容器 stub。"""

        face_mesh = FakeFaceMeshNamespace()

    class FakeMediaPipe:
        """MediaPipe module stub. / mediapipe 模块 stub。"""

        solutions = FakeSolutions()

    with (
        mock.patch.object(face_signals, "_CV2_AVAILABLE", True),
        mock.patch.object(face_signals, "_MEDIAPIPE_AVAILABLE", True),
        mock.patch.object(face_signals, "cv2", FakeCV2),
        mock.patch.object(face_signals, "mediapipe", FakeMediaPipe()),
    ):
        extractor = face_signals.FaceSignalExtractor()
        extractor.open()
        signals = extractor.capture_signals(frame=object())
        extractor.close()

    assert signals["landmarks_n"] == 468
    assert 0.0 <= float(signals["eye_closure"]) <= 1.0
    assert "head_pitch_deg" in signals
    assert "head_yaw_deg" in signals
