"""RealtimeLNNInferencer 单元测试（M0 W3 Phase 4.2）

覆盖 codex F-01/F-02/F-03/F-06 修复：
- F-01: forward 双参数（持续测）
- F-02: signals_loader helper 用法（不重新拼字典）
- F-03: OnlineLoRAAdapter wrap + tau_drift load
- F-06: 失败模式独立 fail-soft
"""
from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from poc.src.realtime.lnn_inferencer import RealtimeLNNInferencer, InferenceResult


class TestDisabledInferencer(unittest.TestCase):
    """ckpt_path=None → disabled 实例，infer 始终返 disabled"""

    def test_none_ckpt_disabled(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=None)
        self.assertFalse(inferencer.is_enabled)
        self.assertIsNone(inferencer.ckpt_sha8)

    def test_disabled_infer_returns_disabled_status(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=None)
        result = inferencer.infer({"face_detected": True})
        self.assertEqual(result.lnn_status, "disabled")
        self.assertIsNone(result.predicted_state)


class TestMissingCkptFailSoft(unittest.TestCase):
    """F-06: ckpt 路径不存在 → fail-soft 不抛"""

    def test_missing_ckpt_disabled(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=Path("/nonexistent/ckpt.pt"))
        self.assertFalse(inferencer.is_enabled)
        # infer 仍可调用
        result = inferencer.infer({"face_detected": True})
        self.assertEqual(result.lnn_status, "disabled")


class TestHealthDict(unittest.TestCase):
    """/health endpoint 暴露 inferencer 状态"""

    def test_disabled_health(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=None)
        h = inferencer.health_dict()
        self.assertFalse(h["enabled"])
        self.assertEqual(h["persona_idx"], 30)
        self.assertEqual(h["history_len"], 0)
        self.assertEqual(h["fail_count"], 0)


class TestSignalPathPropagation(unittest.TestCase):
    """F-05: signal_path 元数据带入 inference 结果"""

    def test_signal_path_in_disabled_result(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=None, signal_path="raw")
        result = inferencer.infer({"face_detected": True})
        self.assertEqual(result.signal_path_used, "raw")

    def test_signal_path_default_calibrated(self):
        inferencer = RealtimeLNNInferencer(ckpt_path=None)
        self.assertEqual(inferencer._signal_path, "calibrated")


class TestRealtimeInferReal(unittest.TestCase):
    """真实加载 m0w2_phase2 ckpt 跑 forward（如果存在）"""

    @classmethod
    def setUpClass(cls):
        repo = Path(__file__).resolve().parent.parent.parent
        ckpts = sorted(
            (repo / "poc/data/checkpoints/m0w2_phase2").glob("*.pt"),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        cls.ckpt = ckpts[0] if ckpts else None

    def setUp(self):
        if self.ckpt is None:
            self.skipTest("no m0w2_phase2 ckpt available")

    def test_load_with_online_drift(self):
        """F-03: ckpt 含 online_adapter_schema=phase2_v1 → wrap"""
        inferencer = RealtimeLNNInferencer(ckpt_path=self.ckpt, persona_idx=30)
        self.assertTrue(inferencer.is_enabled)
        self.assertIsNotNone(inferencer.ckpt_sha8)
        self.assertEqual(len(inferencer.ckpt_sha8), 8)

    def test_warmup_then_ok(self):
        """F-01/F-02: 60 帧 warmup 后 forward 出 InferenceResult"""
        inferencer = RealtimeLNNInferencer(ckpt_path=self.ckpt, persona_idx=30)
        # 构造 60 帧 dummy signal（有 face + 完整 raw_au）
        signal = {
            "ts": "2026-05-16T08:00:00Z",
            "frame_id": 0,
            "face_detected": True,
            "raw_au": {f"AU{i:02d}": 0.3 for i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]},
            "calibrated_au": {f"AU{i:02d}": 0.3 for i in [1, 2, 4, 5, 6, 7, 9, 10, 11, 12, 14, 15, 17, 20, 23, 24, 25, 26, 28, 43]},
            "expression": {"brow_furrow": 0.3, "smile_intensity": 0.2, "eye_closure": 0.1, "cheek_raise": 0.2,
                           "lip_corner_depress": 0.1, "jaw_drop": 0.0, "mouth_open_ratio": 0.0},
            "head_pose": {"pitch_deg": 5.0, "yaw_deg": 2.0, "roll_deg": -1.0},
        }
        # 头 59 帧应该 warmup
        for i in range(59):
            r = inferencer.infer(signal)
            self.assertEqual(r.lnn_status, "warmup", f"frame {i} should be warmup")
        # 第 60 帧 ok
        r = inferencer.infer(signal)
        self.assertEqual(r.lnn_status, "ok")
        self.assertIsNotNone(r.predicted_state)
        self.assertIsNotNone(r.margin)
        self.assertIsNotNone(r.entropy)
        self.assertGreater(r.margin, -0.001)
        self.assertLess(r.margin, 1.001)
        self.assertGreater(r.entropy, -0.001)


if __name__ == "__main__":
    unittest.main()
