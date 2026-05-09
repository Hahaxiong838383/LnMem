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


class TestTriggerEvent(unittest.TestCase):
    def test_immutable(self):
        ev = TriggerEvent(ts="t", frame_id=1, reason="brow_burst", signal_snapshot={}, vlm_label=None)
        with self.assertRaises((AttributeError, TypeError)):
            ev.frame_id = 2  # type: ignore


if __name__ == "__main__":
    unittest.main()
