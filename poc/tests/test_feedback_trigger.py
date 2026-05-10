"""Tests for ProactiveQueryTrigger."""
import unittest

from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger, TriggerEvent


def _frame(brow=0.0, smile=0.0, eye=0.0, vlm=None, face=True, fid=1, ts="2026-05-09T17:00:00"):
    return {
        "ts": ts,
        "frame_id": fid,
        "face_detected": face,
        "expression": {"brow_furrow": brow, "smile_intensity": smile, "eye_closure": eye, "cheek_raise": 0.0},
        "vlm_label": vlm,
    }


class TestProactiveQueryTrigger(unittest.TestCase):
    def test_no_face_returns_none(self):
        t = ProactiveQueryTrigger()
        self.assertIsNone(t.observe(_frame(face=False)))

    def test_brow_burst_requires_5_consecutive(self):
        t = ProactiveQueryTrigger(brow_window=5, brow_threshold=0.7)
        for i in range(4):
            self.assertIsNone(t.observe(_frame(brow=0.9, fid=i)))
        ev = t.observe(_frame(brow=0.9, fid=4))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "brow_burst")
        self.assertEqual(ev.frame_id, 4)

    def test_brow_below_threshold_resets(self):
        t = ProactiveQueryTrigger(brow_window=5, brow_threshold=0.7)
        for i in range(4):
            t.observe(_frame(brow=0.9, fid=i))
        # 第 5 帧低于阈值 → 不触发
        self.assertIsNone(t.observe(_frame(brow=0.5, fid=4)))

    def test_brow_burst_clears_window(self):
        t = ProactiveQueryTrigger(brow_window=5, brow_threshold=0.7)
        for i in range(5):
            ev = t.observe(_frame(brow=0.9, fid=i))
        self.assertIsNotNone(ev)  # frame 4 触发
        # 触发后窗口清空，下一帧不应连续触发
        self.assertIsNone(t.observe(_frame(brow=0.9, fid=5)))

    def test_low_vlm_conf_triggers(self):
        t = ProactiveQueryTrigger()
        ev = t.observe(_frame(vlm={"confidence": 0.4, "valence": 0.0, "arousal": 0.5, "au_description": "ambiguous"}))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "low_vlm_conf")
        self.assertIsNotNone(ev.vlm_label)

    def test_high_vlm_conf_no_trigger(self):
        t = ProactiveQueryTrigger()
        self.assertIsNone(t.observe(_frame(vlm={"confidence": 0.9})))

    def test_vlm_priority_over_brow(self):
        """Both brow_burst and low_vlm_conf candidates → low_vlm_conf wins."""
        t = ProactiveQueryTrigger(brow_window=5, brow_threshold=0.7)
        for i in range(4):
            t.observe(_frame(brow=0.9, fid=i))
        ev = t.observe(_frame(brow=0.9, vlm={"confidence": 0.3}, fid=4))
        self.assertEqual(ev.reason, "low_vlm_conf")

    def test_face_lost_clears_brow_history(self):
        t = ProactiveQueryTrigger(brow_window=5, brow_threshold=0.7)
        for i in range(4):
            t.observe(_frame(brow=0.9, fid=i))
        t.observe(_frame(face=False, fid=4))
        # Brow history 已清空，再 brow=0.9 一帧不应触发
        self.assertIsNone(t.observe(_frame(brow=0.9, fid=5)))

    def test_signal_snapshot_populated(self):
        t = ProactiveQueryTrigger()
        ev = t.observe(_frame(brow=0.5, smile=0.8, eye=0.3, vlm={"confidence": 0.3}))
        self.assertIsNotNone(ev)
        self.assertAlmostEqual(ev.signal_snapshot["brow_furrow"], 0.5)
        self.assertAlmostEqual(ev.signal_snapshot["smile_intensity"], 0.8)
        self.assertAlmostEqual(ev.signal_snapshot["eye_closure"], 0.3)

    def test_skip_low_vlm_conf_when_no_face_detected(self):
        """FB-09: VLM 自报无脸 (confidence=0.0 + au_description='未检测到人脸…') → 不弹窗。"""
        t = ProactiveQueryTrigger()
        vlm = {
            "confidence": 0.0,
            "valence": 0.0,
            "arousal": 0.0,
            "au_description": "未检测到人脸，无法识别面部动作单元。",
        }
        # 虽然 low_vlm_conf 分支判断会触发（confidence < 0.65），但被 FB-09 过滤跳过
        result = t.observe(_frame(vlm=vlm))
        self.assertIsNone(result, "VLM 自报无脸时应返回 None")

    def test_skip_low_vlm_conf_when_low_light(self):
        """FB-09: VLM 自报光线不足 (confidence=0.05 + au_description='光线…') → 不弹窗。"""
        t = ProactiveQueryTrigger()
        vlm = {
            "confidence": 0.05,
            "valence": 0.0,
            "arousal": 0.0,
            "au_description": "图片中未包含人脸，无法识别面部动作单元。",
        }
        result = t.observe(_frame(vlm=vlm))
        self.assertIsNone(result, "VLM 自报无脸/光线不足时应返回 None")

    def test_normal_low_vlm_conf_still_triggers(self):
        """FB-09: VLM 看到脸但不确定状态 (confidence=0.4, au_description 无关键词) → 正常弹窗。"""
        t = ProactiveQueryTrigger()
        vlm = {
            "confidence": 0.4,
            "valence": 0.2,
            "arousal": 0.3,
            "au_description": "AU4 和 AU7，表现出专注与轻微压力。",
        }
        ev = t.observe(_frame(vlm=vlm))
        self.assertIsNotNone(ev, "VLM 自有描述且无 no-face 关键词时应触发")
        self.assertEqual(ev.reason, "low_vlm_conf")


class TestTriggerEvent(unittest.TestCase):
    def test_immutable(self):
        ev = TriggerEvent(ts="t", frame_id=1, reason="brow_burst", signal_snapshot={}, vlm_label=None)
        with self.assertRaises((AttributeError, TypeError)):
            ev.frame_id = 2  # type: ignore


if __name__ == "__main__":
    unittest.main()
