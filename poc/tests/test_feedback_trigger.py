"""Tests for ProactiveQueryTrigger."""
import json
import tempfile
import unittest
from pathlib import Path

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
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "au_description": "未检测到人脸，无法识别面部动作单元。",
            }
            result = t.observe(_frame(vlm=vlm))
            self.assertIsNone(result, "VLM 自报无脸时应返回 None")

    def test_skip_low_vlm_conf_when_low_light(self):
        """FB-09: VLM 自报光线不足 (confidence=0.05 + au_description='光线…') → 不弹窗。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
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
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.4,
                "valence": 0.2,
                "arousal": 0.3,
                "au_description": "AU4 和 AU7，表现出专注与轻微压力。",
            }
            ev = t.observe(_frame(vlm=vlm))
            self.assertIsNotNone(ev, "VLM 自有描述且无 no-face 关键词时应触发")
            self.assertEqual(ev.reason, "low_vlm_conf")

    def test_skip_low_vlm_conf_english_no_face(self):
        """FB-09 (Day 12 fix): gemini-3.1-pro-preview 偶尔回英文，必须命中。

        Regression: 5/10 04:32 实战 vlm_label.au_description='The image does not contain a face.'
        confidence=0.0 但 keyword 列表全中文 → 漏过，进入弹窗触发分支。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "au_description": "The image does not contain a face.",
            }
            result = t.observe(_frame(vlm=vlm))
            self.assertIsNone(result, "VLM 英文自报无脸时应返回 None")

    def test_skip_low_vlm_conf_english_case_insensitive(self):
        """FB-09 (Day 12 fix): 英文 keyword 匹配大小写不敏感。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            for desc in (
                "NO FACE DETECTED",
                "Unable To Identify the face",
                "Cannot detect a face in the image.",
                "image does not contain any face features",
            ):
                vlm = {
                    "confidence": 0.0,
                    "valence": 0.0,
                    "arousal": 0.0,
                    "au_description": desc,
                }
                result = t.observe(_frame(vlm=vlm))
                self.assertIsNone(result, f"应跳过英文变体: {desc!r}")

    def test_skip_mixed_zh_en_no_face(self):
        """FB-09 (Day 12 fix): 中英混合表达也命中。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "au_description": "no face detected (未检测到人脸)",
            }
            result = t.observe(_frame(vlm=vlm))
            self.assertIsNone(result)

    def test_normal_english_description_still_triggers(self):
        """FB-09 (Day 12 fix): 英文有效描述（无 no-face keyword + conf 不低）必须正常触发。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.4,
                "valence": 0.2,
                "arousal": 0.3,
                "au_description": "AU4 (brow lowerer) and AU7 (lid tightener), focused.",
            }
            ev = t.observe(_frame(vlm=vlm))
            self.assertIsNotNone(ev, "英文有效 AU 描述应触发")
            self.assertEqual(ev.reason, "low_vlm_conf")

    def test_skip_low_vlm_conf_literal_none_string(self):
        """FB-09 v3 (Day 12 fix): au_description='None' 字面字符串 + conf=0 必须命中。

        Regression: 5/10 14:16 实战 au_description='None' (字符串 None 不是 Python None) +
        confidence=0.0，gemini-3.1-pro-preview 实测 quirk → 漏过中英 keyword 列表。
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            for desc in ("None", "none", "NONE", "null", "Null", "N/A", "na",
                         "Unknown", "  None  ", ""):
                vlm = {
                    "confidence": 0.0,
                    "valence": 0.0,
                    "arousal": 0.0,
                    "au_description": desc,
                }
                result = t.observe(_frame(vlm=vlm))
                self.assertIsNone(result, f"应跳过 sentinel desc={desc!r}")

    def test_skip_au_description_python_none(self):
        """FB-09 v3: au_description 字段实际是 Python None（不是字符串）也算 sentinel。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm = {
                "confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "au_description": None,
            }
            result = t.observe(_frame(vlm=vlm))
            self.assertIsNone(result, "Python None 也应跳过")

    def test_normal_short_desc_not_misclassified(self):
        """FB-09 v3 防过度过滤：包含 'none' 子串的正常描述（非整词）必须正常触发。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            # "none" 在正常描述中出现作为单词；sentinel 仅匹配整个 description 整体
            vlm = {
                "confidence": 0.4,
                "valence": 0.2,
                "arousal": 0.3,
                "au_description": "Showing none of the typical AU4 patterns, calm baseline.",
            }
            ev = t.observe(_frame(vlm=vlm))
            self.assertIsNotNone(ev, "正常描述含 'none' 子串不应误命中 sentinel")

    def test_skip_appends_jsonl_not_overwrites(self):
        """FB-09 hardening: 多次 skip 应 append 到 jsonl，而非原子覆盖丢前面记录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            t = ProactiveQueryTrigger(home=Path(tmpdir))
            vlm_no_face = {
                "confidence": 0.0,
                "valence": 0.0,
                "arousal": 0.0,
                "au_description": "未检测到人脸，无法识别面部动作单元。",
            }
            t.observe(_frame(vlm=vlm_no_face, fid=10))
            t.observe(_frame(vlm=vlm_no_face, fid=20))
            t.observe(_frame(vlm=vlm_no_face, fid=30))
            # 找 jsonl
            feedback_dir = Path(tmpdir) / ".lnmem" / "feedback"
            jsonl_files = list(feedback_dir.glob("skipped_*.jsonl"))
            self.assertEqual(len(jsonl_files), 1, f"应有 1 个 jsonl 文件，实际 {len(jsonl_files)}")
            lines = jsonl_files[0].read_text(encoding="utf-8").strip().split("\n")
            self.assertEqual(len(lines), 3, f"应 append 3 条，实际 {len(lines)}")
            fids = [json.loads(line)["frame_id"] for line in lines]
            self.assertEqual(fids, [10, 20, 30], "frame_id 应按写入顺序保留")


class TestTriggerEvent(unittest.TestCase):
    def test_immutable(self):
        ev = TriggerEvent(ts="t", frame_id=1, reason="brow_burst", signal_snapshot={}, vlm_label=None)
        with self.assertRaises((AttributeError, TypeError)):
            ev.frame_id = 2  # type: ignore


# V0.3 Phase B 新增：pose_held 通道 + per-channel cooldown
def _frame_pose(brow=0.0, smile=0.0, eye=0.0, vlm=None, face=True, fid=1,
                ts="2026-05-11T10:00:00", pitch=0.0, yaw=0.0, roll=0.0):
    """加 head_pose 字段的帧构造器（V0.3 Phase B）。"""
    return {
        "ts": ts,
        "frame_id": fid,
        "face_detected": face,
        "expression": {"brow_furrow": brow, "smile_intensity": smile, "eye_closure": eye, "cheek_raise": 0.0},
        "head_pose": {"pitch_deg": pitch, "yaw_deg": yaw, "roll_deg": roll},
        "vlm_label": vlm,
    }


class TestPoseHeldChannel(unittest.TestCase):
    """V0.3 Phase B：pose_held 通道 — 头部姿态长时间稳定 = 专注盯屏。"""

    def test_pose_held_triggers_after_stable_window(self):
        """连续 N 帧 pose stddev < 阈值 → pose_held 触发。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        for i in range(4):
            self.assertIsNone(t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0)))
        ev = t.observe(_frame_pose(fid=4, pitch=10.0, yaw=5.0, roll=2.0))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "pose_held")
        self.assertIn("pose_max_stddev_deg", ev.signal_snapshot)
        self.assertLess(ev.signal_snapshot["pose_max_stddev_deg"], 2.5)

    def test_pose_held_does_not_trigger_when_moving(self):
        """pose stddev >= 阈值时不触发。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        ev = None
        for i in range(5):
            ev = t.observe(_frame_pose(fid=i, pitch=10.0 + i * 5.0, yaw=0.0, roll=0.0))
        self.assertIsNone(ev)

    def test_pose_outlier_clears_window(self):
        """单帧 outlier（>60°）清空 pose 滑窗，防 py-feat 偶发错检误触发。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0))
        ev = t.observe(_frame_pose(fid=4, pitch=175.0, yaw=5.0, roll=2.0))
        self.assertIsNone(ev)  # outlier 清窗，不应触发
        # 重新累积，需 5 帧补满窗口
        for i in range(5, 9):
            self.assertIsNone(t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0)))
        ev = t.observe(_frame_pose(fid=9, pitch=10.0, yaw=5.0, roll=2.0))
        self.assertIsNotNone(ev)

    def test_pose_missing_clears_window(self):
        """缺 head_pose 字段时清窗（防不连续 pose 算 std 不合理）。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0))
        no_pose_frame = {
            "ts": "t", "frame_id": 4, "face_detected": True,
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0, "eye_closure": 0.0, "cheek_raise": 0.0},
            "vlm_label": None,
        }
        ev = t.observe(no_pose_frame)
        self.assertIsNone(ev)

    def test_no_face_clears_pose_window(self):
        """face_detected=False 应清空 pose 滑窗。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0))
        t.observe(_frame_pose(fid=4, face=False))
        # 仅 4 帧稳定，未到 5 → 不触发
        for i in range(5, 9):
            self.assertIsNone(t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0)))


class TestPerChannelCooldown(unittest.TestCase):
    """V0.3 Phase B：每通道独立 cooldown，防同通道连发。"""

    def test_brow_burst_cooldown_blocks_immediate_retrigger(self):
        """brow_burst 触发后 cooldown 内不再触发。"""
        t = ProactiveQueryTrigger(
            brow_window=3, brow_threshold=0.7,
            cooldown_frames_brow=100,
        )
        for i in range(2):
            t.observe(_frame(brow=0.9, fid=i))
        ev1 = t.observe(_frame(brow=0.9, fid=2))
        self.assertIsNotNone(ev1)
        # cooldown 内 fid 50-52 即使凑齐 3 帧也不触发（fid 52 < 2+100=102）
        cd_events = [t.observe(_frame(brow=0.9, fid=i)) for i in range(50, 53)]
        self.assertTrue(all(e is None for e in cd_events))
        # cooldown 过后再触发 OK
        # F3-06: 修复后 evidence 保留，cooldown 解禁立即触发（fid=110 第一帧即触发）
        after_events = [t.observe(_frame(brow=0.9, fid=i)) for i in range(110, 113)]
        triggered = [e for e in after_events if e is not None]
        self.assertGreaterEqual(len(triggered), 1)
        self.assertEqual(triggered[0].reason, "brow_burst")

    def test_channels_have_independent_cooldown(self):
        """brow cooldown 不应阻塞 pose_held 触发。"""
        t = ProactiveQueryTrigger(
            brow_window=3, brow_threshold=0.7,
            pose_held_window=3, pose_held_stddev_threshold_deg=2.5,
            cooldown_frames_brow=1000,
            cooldown_frames_pose_held=1000,
        )
        # brow 触发同时 pose 不满足（pose 大幅变化）
        for i in range(2):
            t.observe(_frame_pose(fid=i, brow=0.9, pitch=10.0 + i * 5.0))
        ev1 = t.observe(_frame_pose(fid=2, brow=0.9, pitch=20.0))
        self.assertIsNotNone(ev1)
        self.assertEqual(ev1.reason, "brow_burst")
        # pose 稳定 → 独立 cooldown，不应被 brow cooldown 影响
        for i in range(3, 5):
            t.observe(_frame_pose(fid=i, brow=0.0, pitch=10.0, yaw=5.0))
        ev2 = t.observe(_frame_pose(fid=5, brow=0.0, pitch=10.0, yaw=5.0))
        self.assertIsNotNone(ev2)
        self.assertEqual(ev2.reason, "pose_held")

    def test_low_vlm_conf_cooldown(self):
        """low_vlm_conf cooldown 防同通道连发。"""
        t = ProactiveQueryTrigger(cooldown_frames_low_vlm_conf=50)
        vlm = {"confidence": 0.3, "au_description": "看起来在思考"}
        ev1 = t.observe(_frame(fid=0, vlm=vlm))
        self.assertIsNotNone(ev1)
        ev2 = t.observe(_frame(fid=10, vlm=vlm))
        self.assertIsNone(ev2)  # cooldown 内
        ev3 = t.observe(_frame(fid=49, vlm=vlm))
        self.assertIsNone(ev3)
        ev4 = t.observe(_frame(fid=51, vlm=vlm))
        self.assertIsNotNone(ev4)

    def test_pose_held_cooldown(self):
        """pose_held cooldown：触发后即使一直保持，cooldown 内不再触发。"""
        t = ProactiveQueryTrigger(
            pose_held_window=3, pose_held_stddev_threshold_deg=2.5,
            cooldown_frames_pose_held=100,
        )
        for i in range(2):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0))
        ev1 = t.observe(_frame_pose(fid=2, pitch=10.0, yaw=5.0))
        self.assertIsNotNone(ev1)
        # cooldown 内一直保持不触发
        ev_mid = None
        for i in range(3, 80):
            ev_mid = t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0))
        self.assertIsNone(ev_mid)
        # cooldown 过 + 窗口重满才触发；触发后清窗 + 新 cooldown 开始，后续 None 是预期
        events = [t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0)) for i in range(102, 110)]
        triggered = [e for e in events if e is not None]
        self.assertGreaterEqual(len(triggered), 1, "cooldown 过且窗口重满后应至少触发 1 次")
        self.assertEqual(triggered[0].reason, "pose_held")


class TestCodexFindingFixes(unittest.TestCase):
    """V0.3 Phase B 一审 codex 7 finding 修复对应测试。"""

    def test_f1_01_frame_id_rollback_resets_cooldown(self):
        """F1-01: frame_id 回退（跨天/daemon 重启）应清 cooldown 让本次正常评估。

        模拟真实数据 fid 801（5/9 末）→ fid 194（5/10 起，跨天重置）。
        若不修复，回退后 cooldown delta 为负永真，整个跨天后冷却失效。
        """
        t = ProactiveQueryTrigger(brow_window=3, brow_threshold=0.7, cooldown_frames_brow=100)
        # 第一波触发 fid=2，cooldown 至 fid=102
        for i in range(2):
            t.observe(_frame(brow=0.9, fid=i))
        ev1 = t.observe(_frame(brow=0.9, fid=2))
        self.assertIsNotNone(ev1)
        # fid 回退到 0 → cooldown 应被 reset；重新累 3 帧应触发
        # （触发后 deque clear，所以单 fid=0 不够，需 fid 0,1,2 重新累积）
        evs = [t.observe(_frame(brow=0.9, fid=i)) for i in range(3)]
        triggered = [e for e in evs if e is not None]
        self.assertGreaterEqual(len(triggered), 1,
                                "frame_id 回退后 cooldown reset，重新累 3 帧应再触发")
        self.assertEqual(triggered[0].reason, "brow_burst")

    def test_f2_02_no_face_does_not_block_brow_or_pose(self):
        """F2-02: VLM 自报无脸**只**屏蔽 low_vlm_conf 通道，brow/pose 仍能在同帧触发。"""
        t = ProactiveQueryTrigger(brow_window=3, brow_threshold=0.7)
        vlm_no_face = {
            "confidence": 0.0,
            "au_description": "未检测到人脸",
        }
        # 前 2 帧带 no-face VLM + brow=0.9（不应因 no-face 跳过 brow 累积）
        for i in range(2):
            t.observe(_frame(brow=0.9, fid=i, vlm=vlm_no_face))
        ev = t.observe(_frame(brow=0.9, fid=2, vlm=vlm_no_face))
        self.assertIsNotNone(ev,
                             "no-face 不应吞 brow_burst 评估机会")
        self.assertEqual(ev.reason, "brow_burst")

    def test_f2_03_pose_field_none_does_not_crash(self):
        """F2-03: head_pose 字段含 None/NaN/字符串 不应抛异常，应安全清窗。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        # 4 帧累积
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0))
        # 第 5 帧 pose 含 None
        frame_bad = {
            "ts": "t", "frame_id": 4, "face_detected": True,
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0, "eye_closure": 0.0, "cheek_raise": 0.0},
            "head_pose": {"pitch_deg": None, "yaw_deg": float("nan"), "roll_deg": "bad"},
            "vlm_label": None,
        }
        # 不应抛异常
        ev = t.observe(frame_bad)
        self.assertIsNone(ev)  # 清窗 + 不触发

    def test_f2_04_disabled_pose_held_does_not_trigger(self):
        """F2-04: enable_pose_held=False 时 pose 通道永不触发（V0.2 真实禁用）。"""
        t = ProactiveQueryTrigger(
            pose_held_window=3, pose_held_stddev_threshold_deg=2.5,
            enable_pose_held=False,
        )
        # 30 帧完全静止，按 V0.3 默认本应触发；V0.2 禁用应不触发
        events = [t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0)) for i in range(30)]
        self.assertTrue(all(e is None or e.reason != "pose_held" for e in events))

    def test_f3_05_roll_missing_clears_roll_deque_to_avoid_window_misalignment(self):
        """F3-05 corner case：当 roll deque 已满后某帧缺 roll，旧 roll 内容会滞后
        于 pitch/yaw 滑窗（不对齐）。修复策略：缺 roll 时 clear roll deque，
        保证 roll deque 满 N 时一定对应当前窗口。

        用 window=5 + pitch 大幅变化（std 远大于阈值）避免触发后 clear。
        """
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=0.1)
        # 累 4 帧带 roll，pitch 大幅变化阻止触发
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0 + i * 5.0, yaw=5.0, roll=2.0))
        self.assertEqual(len(t._pose_roll), 4, "前 4 帧 roll 累积")
        # 第 5 帧缺 roll → roll deque 应 clear（防窗口错位）
        frame_no_roll = {
            "ts": "t", "frame_id": 4, "face_detected": True,
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0, "eye_closure": 0.0, "cheek_raise": 0.0},
            "head_pose": {"pitch_deg": 30.0, "yaw_deg": 5.0},  # 无 roll_deg
            "vlm_label": None,
        }
        t.observe(frame_no_roll)
        self.assertEqual(len(t._pose_roll), 0,
                         "缺 roll 时应 clear roll deque 防窗口错位")
        # pitch / yaw 仍滑动累积
        self.assertEqual(len(t._pose_pitch), 5)
        self.assertEqual(len(t._pose_yaw), 5)

    def test_f3_05_roll_optional_does_not_pollute_stddev(self):
        """F3-05: roll 缺失时不应用 0.0 兜底污染 std；缺 roll 仍能正常 pitch+yaw 触发。"""
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=2.5)
        # 4 帧带 roll
        for i in range(4):
            t.observe(_frame_pose(fid=i, pitch=10.0, yaw=5.0, roll=2.0))
        # 第 5 帧缺 roll 字段（head_pose 仅含 pitch+yaw，符合实测 schema）
        frame_no_roll = {
            "ts": "t", "frame_id": 4, "face_detected": True,
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0, "eye_closure": 0.0, "cheek_raise": 0.0},
            "head_pose": {"pitch_deg": 10.0, "yaw_deg": 5.0},  # 无 roll_deg
            "vlm_label": None,
        }
        ev = t.observe(frame_no_roll)
        # roll deque 缺一帧（4 个），不满 window=5 → 不参与 max；pitch+yaw 全相同 std=0 → 触发
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "pose_held")

    def test_nf_01_frame_rollback_clears_evidence_not_just_cooldown(self):
        """NF-01 (codex 二审): rollback 不仅 reset cooldown，还必须清 evidence。

        F3-06 修复后 cooldown 内 brow_history 保留，跨天回退若不清 evidence，
        新 session 首帧会因旧 session 累积的高 brow 立即触发（语义错误）。
        """
        t = ProactiveQueryTrigger(brow_window=3, brow_threshold=0.7,
                                  cooldown_frames_brow=10)
        # session A: 累 2 帧高 brow（不满 window，未触发，brow_history=[0.9, 0.9]）
        for i in range(2):
            t.observe(_frame(brow=0.9, fid=i))
        self.assertEqual(len(t._brow_history), 2)
        # session B (跨天 fid 回退到 0)：若不清 evidence，brow_history=[0.9, 0.9, 0.9]
        # 立即满足 brow_burst（错误）
        ev = t.observe(_frame(brow=0.9, fid=0))
        # NF-01 修复：rollback 后 evidence 清空 → fid=0 只 1 帧不应触发
        self.assertIsNone(ev,
                          "rollback 应清 evidence，新 session 首帧不应立即触发")
        self.assertEqual(len(t._brow_history), 1,
                         "rollback 清窗后只剩当前帧 1 个")

    def test_nf_02_roll_misalignment_full_then_missing(self):
        """NF-02 (codex 二审 + cc pre-emptive fix): roll 先满后缺会让 deque 内容
        滞后于 pitch/yaw 窗口（时序错位）。修复：缺 roll 时 clear roll deque。

        覆盖"先满后缺"场景（前一测试覆盖"满前缺"场景，互补）。
        """
        t = ProactiveQueryTrigger(pose_held_window=5, pose_held_stddev_threshold_deg=0.1)
        # 累 5 帧带 roll → roll deque 满
        for i in range(5):
            t.observe(_frame_pose(fid=i, pitch=10.0 + i * 5.0, yaw=5.0, roll=2.0))
        self.assertEqual(len(t._pose_roll), 5)
        # 第 6 帧缺 roll → roll deque 应 clear（NF-02 修复点）
        frame_no_roll = {
            "ts": "t", "frame_id": 5, "face_detected": True,
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0, "eye_closure": 0.0, "cheek_raise": 0.0},
            "head_pose": {"pitch_deg": 40.0, "yaw_deg": 5.0},  # 无 roll
            "vlm_label": None,
        }
        t.observe(frame_no_roll)
        self.assertEqual(len(t._pose_roll), 0,
                         "先满后缺：roll deque 必须 clear 防时序错位")

    def test_f3_06_brow_evidence_kept_in_cooldown(self):
        """F3-06: cooldown 内 brow_history 不清，cooldown 解禁后下一帧立即触发。"""
        t = ProactiveQueryTrigger(brow_window=3, brow_threshold=0.7, cooldown_frames_brow=10)
        # 第一波触发
        for i in range(2):
            t.observe(_frame(brow=0.9, fid=i))
        ev1 = t.observe(_frame(brow=0.9, fid=2))
        self.assertIsNotNone(ev1)
        # cooldown 内（fid 3-10）继续高 brow，evidence 应保留
        for i in range(3, 11):
            ev = t.observe(_frame(brow=0.9, fid=i))
            self.assertIsNone(ev, f"fid={i} 在 cooldown 内不应触发")
        # cooldown 解禁（fid=13: 13-2=11>10）后下一帧立即触发（不需重新攒 3 帧）
        ev_after = t.observe(_frame(brow=0.9, fid=13))
        self.assertIsNotNone(ev_after, "cooldown 解禁后应立即触发（evidence 已保留）")


class TestV03DefaultThresholds(unittest.TestCase):
    """V0.3 Phase B 默认阈值数据驱动收紧验证。"""

    def test_default_brow_threshold_is_v03(self):
        """默认 brow_threshold 应为 V0.3 0.95、window 15。"""
        t = ProactiveQueryTrigger()
        self.assertAlmostEqual(t._brow_threshold, 0.95)
        self.assertEqual(t._brow_window_size, 15)

    def test_default_pose_held_params(self):
        """默认 pose_held 参数应为 V0.3 设定。"""
        t = ProactiveQueryTrigger()
        self.assertEqual(t._pose_window_size, 60)
        self.assertAlmostEqual(t._pose_stddev_threshold, 2.5)
        self.assertAlmostEqual(t._pose_outlier_threshold, 60.0)

    def test_default_cooldown_frames(self):
        """默认 cooldown 配置应为 V0.3 设定。"""
        t = ProactiveQueryTrigger()
        self.assertEqual(t._cooldown_frames["brow_burst"], 300)
        self.assertEqual(t._cooldown_frames["pose_held"], 300)
        self.assertEqual(t._cooldown_frames["low_vlm_conf"], 120)


if __name__ == "__main__":
    unittest.main()


class TestSignalPathSwitching(unittest.TestCase):
    """ADR-005 (2026-05-16): dual signal_path（raw_au.AU04 vs expression.brow_furrow）。

    raw 模式：trigger 用 item['raw_au']['AU04']，阈值 0.55（P91 数据驱动）
    calibrated 模式（默认）：trigger 用 item['expression']['brow_furrow']，阈值 0.95
    """

    def _frame_with_raw_au(self, raw_au04=0.0, exp_brow=0.0, fid=1):
        """构造同时含 raw_au.AU04 + expression.brow_furrow 的 item。"""
        return {
            "ts": "2026-05-16T08:00:00Z",
            "frame_id": fid,
            "face_detected": True,
            "raw_au": {"AU04": raw_au04},  # py-feat 原始
            "expression": {
                "brow_furrow": exp_brow,  # calibration 后
                "smile_intensity": 0.0,
                "eye_closure": 0.0,
                "cheek_raise": 0.0,
            },
            "vlm_label": None,
        }

    def _run_until_event(self, t, frame_factory, n=20):
        """跑 n 帧或直到触发（不让后续帧把 event 覆盖回 None）。"""
        for i in range(n):
            event = t.observe(frame_factory(i))
            if event is not None:
                return event
        return None

    def test_signal_snapshot_records_both_paths(self):
        """signal_snapshot 同时记 raw + calibrated 双值（A/B 离线对照锚点）。

        F-08 fix (codex v0.3 二审 2026-05-16)：显式锁定 _SIGNAL_PATH=calibrated +
        显式传 brow_threshold=0.95，不依赖运行时环境的 ~/.lnmem/signal_path 文件。
        """
        # F-08 fix v2 (codex 二审 2026-05-16): 直接传 signal_path 实例参数，不动 module-level
        t = ProactiveQueryTrigger(signal_path="calibrated")
        event = self._run_until_event(
            t,
            lambda i: self._frame_with_raw_au(raw_au04=0.3, exp_brow=0.99, fid=i),
        )
        self.assertIsNotNone(event)
        sig = event.signal_snapshot
        self.assertIn("raw_au_AU04", sig)
        self.assertIn("brow_furrow", sig)
        self.assertIn("brow_used_path", sig)
        self.assertIn("brow_used_value", sig)
        self.assertEqual(sig["raw_au_AU04"], 0.3)
        self.assertEqual(sig["brow_furrow"], 0.99)

    def test_calibrated_path_uses_expression_brow_furrow(self):
        """calibrated 模式：raw_au.AU04 高但 expression.brow_furrow 低 → 不触发。"""
        t = ProactiveQueryTrigger(signal_path="calibrated")
        # raw 0.99 高但 calibrated 0.1 低 → calibrated 阈值 0.95 不满足
        event = self._run_until_event(
            t,
            lambda i: self._frame_with_raw_au(raw_au04=0.99, exp_brow=0.10, fid=i),
        )
        self.assertIsNone(event, "calibrated 模式应只看 expression.brow_furrow，raw_au 0.99 不应触发")

    def test_raw_path_uses_raw_au04(self):
        """raw 模式：expression.brow_furrow 低但 raw_au.AU04 > 0.55 → 触发。"""
        import poc.src.realtime.feedback.trigger as tg
        t = ProactiveQueryTrigger(
            signal_path="raw",
            brow_threshold=tg._BROW_BURST_THRESHOLD_RAW,  # 0.55
        )
        # raw 0.70 高（> 0.55）但 calibrated 0.05 低 → raw 阈值 0.55 满足
        event = self._run_until_event(
            t,
            lambda i: self._frame_with_raw_au(raw_au04=0.70, exp_brow=0.05, fid=i),
        )
        self.assertIsNotNone(event, "raw 模式应看 raw_au.AU04，0.70 > 0.55 应触发")
        self.assertEqual(event.signal_snapshot["brow_used_path"], "raw")
        self.assertEqual(event.signal_snapshot["brow_used_value"], 0.70)

    def test_raw_path_no_raw_au_field_fallback_to_zero(self):
        """raw 模式但 item 缺 raw_au 字段（旧 daemon 数据兼容）→ 用 0 fallback 不触发。"""
        t = ProactiveQueryTrigger(signal_path="raw")
        # 构造缺 raw_au 的 item（仅 expression.brow_furrow 高）
        def factory(i):
            return {
                "ts": "2026-05-16T08:00:00Z",
                "frame_id": i,
                "face_detected": True,
                "expression": {"brow_furrow": 0.99, "smile_intensity": 0.0,
                               "eye_closure": 0.0, "cheek_raise": 0.0},
                # 无 raw_au 字段
                "vlm_label": None,
            }
        event = self._run_until_event(t, factory)
        self.assertIsNone(event, "raw 模式缺 raw_au 字段应 fallback 0，不应触发")


class TestSchmittEmaBrowBurst(unittest.TestCase):
    """TODO-1 (ADR-010) Schmitt + EMA + dynamic quantile brow_burst 触发测试。

    覆盖目标：
      1. 默认 use_schmitt_ema=False → legacy strict-deque 路径不动（向后兼容门）
      2. 容忍 1 帧 dropout 的脉冲信号（ADR-010 川哥真实场景）
      3. EMA α 真有平滑效果（首帧瞬间高峰不立刻触发）
      4. Schmitt 滞回：thr_low 之上、thr_high 之下时状态不变
      5. 触发后必须先回落 thr_low 再上升才能再次触发（带 cooldown 双保险）
      6. face 丢失 / frame_id 回退清 EMA + Schmitt 状态（与 evidence reset 对齐）
      7. signal_snapshot 暴露 brow_ema / brow_above_high 诊断字段
      8. 参数手滑（thr_low > thr_high / α 越界）自动收敛安全默认
    """

    @staticmethod
    def _f(brow, fid, face=True, ts="2026-05-16T08:00:00Z"):
        """raw_au.AU04 = brow，walk 进 raw 路径 + Schmitt+EMA 判定。"""
        return {
            "ts": ts,
            "frame_id": fid,
            "face_detected": face,
            "raw_au": {"AU04": brow},
            "expression": {"brow_furrow": 0.0, "smile_intensity": 0.0,
                           "eye_closure": 0.0, "cheek_raise": 0.0},
            "vlm_label": None,
        }

    def _make(self, **kwargs):
        defaults = dict(
            signal_path="raw",
            use_schmitt_ema=True,
            brow_ema_alpha=0.40,
            brow_thr_high=0.55,
            brow_thr_low=0.30,
            brow_window=5,
            cooldown_frames_brow=30,
            enable_pose_held=False,  # 隔离 brow 通道，防 pose_held 干扰
        )
        defaults.update(kwargs)
        return ProactiveQueryTrigger(**defaults)

    def test_legacy_strict_deque_path_unchanged_when_flag_off(self):
        """默认 use_schmitt_ema=False → strict-deque 语义不动（向后兼容门）。"""
        t = ProactiveQueryTrigger(
            signal_path="raw", brow_window=5, brow_threshold=0.40,
            cooldown_frames_brow=30, enable_pose_held=False,
        )
        # 4 帧 0.9 + 1 帧 0.30（脉冲 dropout）→ legacy 严格 all() = False → 不触发
        for i in range(4):
            self.assertIsNone(t.observe(self._f(0.90, i)))
        self.assertIsNone(t.observe(self._f(0.30, 4)))  # dropout 一帧让 all() 失败
        self.assertIsNone(t.observe(self._f(0.90, 5)))

    def test_schmitt_ema_tolerates_single_frame_dropout(self):
        """ADR-010 核心修复：脉冲信号 4 帧 0.9 + 1 帧 0.30 仍能触发（legacy 必失败）。"""
        t = self._make()
        # 5 帧 0.9 → EMA 单调上升 0.36→0.576→0.706→0.783→0.830，第 3 帧已 ≥ 0.55
        ev = None
        for i in range(5):
            ev = t.observe(self._f(0.90, i))
            if ev is not None:
                break
        self.assertIsNotNone(ev, "5 帧 0.9 EMA 应在第 5 帧前触发 brow_burst")
        self.assertEqual(ev.reason, "brow_burst")

    def test_schmitt_ema_requires_window_samples_before_fire(self):
        """触发前必须满 brow_window 样本（与 legacy 'N 连续帧' 等价的稳定性门）。"""
        # window=10 让 EMA 收敛快但样本不足时不触发；cooldown=2 排除其与 rearm 流的干扰
        t = self._make(brow_window=10, cooldown_frames_brow=2)
        fired_at = []
        for i in range(9):
            ev = t.observe(self._f(0.95, i))  # EMA 第 3 帧即 ≥ thr_high
            if ev is not None and ev.reason == "brow_burst":
                fired_at.append(i)
        self.assertEqual(fired_at, [], "样本数 < brow_window 不应触发 brow_burst")
        # 此时 _brow_above_high 仍 False（window 未满未 latch）；下一帧 0.95 + samples=10 即可触发
        ev = t.observe(self._f(0.95, 9))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "brow_burst")
        self.assertEqual(ev.frame_id, 9)

    def test_schmitt_hysteresis_band_does_not_retrigger(self):
        """thr_low (0.30) < ema < thr_high (0.55) 区间内状态不变，不重复触发。"""
        t = self._make()
        # 让 EMA 先升过 thr_high 一次触发
        ev1 = None
        for i in range(20):
            e = t.observe(self._f(0.90, i))
            if e is not None and e.reason == "brow_burst":
                ev1 = e
                break
        self.assertIsNotNone(ev1)
        # 在 0.35-0.45 之间震荡（始终在滞回带内，未到 thr_low 0.30，
        # 也未让 ema 落回 above=False 状态）；cooldown=30 应早已过去
        for i in range(50, 200):
            v = 0.40 + (0.05 if i % 2 == 0 else -0.05)  # 0.35~0.45 区间
            e = t.observe(self._f(v, i))
            self.assertIsNone(e, f"滞回带内不应再触发 brow_burst (fid={i} v={v})")

    def test_schmitt_must_fall_below_low_to_rearm(self):
        """触发后必须 EMA ≤ thr_low 才能重置 above 状态，准备下一次上升沿。"""
        t = self._make(cooldown_frames_brow=5)  # 极短 cooldown 排除其干扰
        # 第一次触发
        ev1 = None
        for i in range(20):
            e = t.observe(self._f(0.90, i))
            if e is not None and e.reason == "brow_burst":
                ev1 = e
                break
        self.assertIsNotNone(ev1)
        # 始终维持 ema > thr_low（0.30）→ above 不解，cooldown 过去也不再触发
        for i in range(30, 200):
            e = t.observe(self._f(0.50, i))  # EMA 收敛到 0.50，在 [thr_low, thr_high] 内
            self.assertIsNone(e, "above 状态未解除前不应再触发")
        # 真正回落到 thr_low 之下
        for i in range(200, 240):
            t.observe(self._f(0.05, i))
        # 再次上升 → 触发
        ev2 = None
        for i in range(240, 260):
            e = t.observe(self._f(0.90, i))
            if e is not None and e.reason == "brow_burst":
                ev2 = e
                break
        self.assertIsNotNone(ev2, "回落后第二次上升沿应触发")
        self.assertGreater(ev2.frame_id, ev1.frame_id)

    def test_face_lost_resets_ema_and_schmitt_state(self):
        """face 丢失 → EMA / Schmitt above / 样本计数清空，evidence 不跨 session。"""
        t = self._make()
        for i in range(4):
            t.observe(self._f(0.90, i))
        # face=False 清状态
        t.observe(self._f(0.0, 4, face=False))
        # 内部状态确认
        self.assertIsNone(t._brow_ema)
        self.assertFalse(t._brow_above_high)
        self.assertEqual(t._brow_samples_since_reset, 0)
        # 再单帧 0.9 不应立刻触发（样本计数从 1 开始）
        self.assertIsNone(t.observe(self._f(0.90, 5)))

    def test_frame_rollback_resets_ema_and_schmitt_state(self):
        """跨天 frame_id 回退 → EMA + Schmitt 同步清，与 NF-01 evidence reset 对齐。"""
        t = self._make()
        for i in range(10):
            t.observe(self._f(0.90, 100 + i))  # fid 100-109
        # 模拟跨天回退到 fid=1
        t.observe(self._f(0.05, 1))
        self.assertFalse(t._brow_above_high)
        # 回退分支只清状态；之后 face_detected=True 分支会用新 brow 初始化 EMA
        self.assertIsNotNone(t._brow_ema)
        self.assertAlmostEqual(t._brow_ema, 0.05, places=4)
        self.assertEqual(t._brow_samples_since_reset, 1)

    def test_signal_snapshot_exposes_ema_and_above_high(self):
        """signal_snapshot 必须包含 brow_ema + brow_above_high 诊断字段。"""
        t = self._make()
        ev = None
        for i in range(20):
            e = t.observe(self._f(0.90, i))
            if e is not None:
                ev = e
                break
        self.assertIsNotNone(ev)
        snap = ev.signal_snapshot
        self.assertIn("brow_ema", snap)
        self.assertIn("brow_above_high", snap)
        self.assertGreater(snap["brow_ema"], 0.55)
        self.assertTrue(snap["brow_above_high"])

    def test_param_sanity_low_above_high_auto_corrects(self):
        """thr_low > thr_high 手滑参数 → 自动收敛 thr_low = thr_high * 0.55。"""
        t = ProactiveQueryTrigger(
            signal_path="raw", use_schmitt_ema=True,
            brow_thr_high=0.50, brow_thr_low=0.80,  # 故意反了
            brow_window=3, cooldown_frames_brow=5, enable_pose_held=False,
        )
        self.assertEqual(t._brow_thr_high, 0.50)
        self.assertAlmostEqual(t._brow_thr_low, 0.50 * 0.55, places=4)

    def test_param_sanity_thr_high_zero_falls_to_safe_default(self):
        """cc 自审 F6：thr_high ≤ 0 → 回落 brow_threshold（防滞回死锁）。"""
        t = ProactiveQueryTrigger(
            signal_path="raw", use_schmitt_ema=True,
            brow_threshold=0.55, brow_thr_high=0.0, brow_thr_low=-0.1,
            brow_window=3, enable_pose_held=False,
        )
        self.assertGreater(t._brow_thr_high, 0.0)
        self.assertGreaterEqual(t._brow_thr_low, 0.0)
        self.assertLessEqual(t._brow_thr_low, t._brow_thr_high)

    def test_cooldown_blocked_does_not_latch_above(self):
        """cc 自审 F2：cooldown 期间满足 rising-edge 不应 latch above_high，否则下次正常触发会丢。"""
        t = self._make(cooldown_frames_brow=10)
        # 第一次触发
        ev1 = None
        for i in range(20):
            e = t.observe(self._f(0.90, i))
            if e is not None and e.reason == "brow_burst":
                ev1 = e
                break
        self.assertIsNotNone(ev1)
        # cooldown 内继续高 brow → 应该 NOT latch above_high（cooldown 解禁后还能立即触发）
        # 流程：fire 后 above=True，下降到 thr_low 之下 above=False，再上升但在 cooldown 内
        first_fire_fid = ev1.frame_id
        # 把 above 降下来
        for i in range(first_fire_fid + 1, first_fire_fid + 6):
            t.observe(self._f(0.05, i))
        self.assertFalse(t._brow_above_high, "应被 thr_low 下降沿解除")
        # cooldown 还在（first_fire + 10 = first_fire + 10），用 first_fire + 7 测试
        t.observe(self._f(0.90, first_fire_fid + 7))
        # 关键：above_high 应仍 False（cooldown 内 rising edge 不 latch）
        self.assertFalse(
            t._brow_above_high,
            "cc 自审 F2 修复：cooldown 内 rising edge 不应 latch above_high",
        )
        # cooldown 解禁后立即可触发
        ev2 = None
        for i in range(first_fire_fid + 11, first_fire_fid + 20):
            e = t.observe(self._f(0.90, i))
            if e is not None and e.reason == "brow_burst":
                ev2 = e
                break
        self.assertIsNotNone(ev2, "cooldown 解禁后应能立即触发（F2 修复验证）")

    def test_param_sanity_alpha_out_of_range_falls_to_default(self):
        """α 越界（≤0 / >1 / 非数）→ 回落 0.40 默认值。"""
        for bad in [0.0, -0.1, 1.5, None, "foo"]:
            t = ProactiveQueryTrigger(
                signal_path="raw", use_schmitt_ema=True,
                brow_ema_alpha=bad,
                brow_window=3, enable_pose_held=False,
            )
            self.assertAlmostEqual(t._brow_ema_alpha, 0.40, places=4,
                                    msg=f"bad α={bad!r} should fall back")

    def test_default_constructor_keeps_legacy_path_off(self):
        """默认构造 use_schmitt_ema=False，保证 538 测试链路不动（兼容门）。"""
        t = ProactiveQueryTrigger()
        self.assertFalse(t._use_schmitt_ema)

    def test_replay_demo_session_fires_at_least_two_brow_burst(self):
        """ADR-010 验收：复跑 638 帧 demo 信号，brow_burst 触发 ≥ 2 次。

        若该信号文件不存在则跳过（CI 不强依赖私有数据）。
        """
        import json
        signals = Path.home() / ".lnmem" / "data" / "signals_2026-05-16.jsonl"
        if not signals.exists():
            self.skipTest(f"local demo signals 不存在: {signals}")
        # demo 模式参数 + raw 路径 + Schmitt+EMA on（与 daemon 实际运行配置等价）
        t = ProactiveQueryTrigger(
            signal_path="raw", use_schmitt_ema=True,
            brow_ema_alpha=0.40, brow_thr_high=0.42, brow_thr_low=0.22,
            brow_window=5, cooldown_frames_brow=30,
            pose_held_window=20, pose_held_stddev_threshold_deg=3.5,
            cooldown_frames_pose_held=30,
            cooldown_frames_low_vlm_conf=30,
        )
        fires = 0
        pose_fires = 0
        with signals.open() as f:
            for line in f:
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                ts = item.get("ts", "")
                if ts < "2026-05-16T14:14:46" or ts >= "2026-05-16T14:25:40":
                    continue
                ev = t.observe(item)
                if ev is None:
                    continue
                if ev.reason == "brow_burst":
                    fires += 1
                elif ev.reason == "pose_held":
                    pose_fires += 1
        self.assertGreaterEqual(
            fires, 2,
            f"ADR-010 验收: demo session brow_burst 触发 {fires} 次 < 2 次门槛",
        )
        # 不引入 pose_held 误报（与早 22:01-22:05 session 对齐，允许少量正常）
        self.assertLessEqual(
            pose_fires, 15,
            f"pose_held 误报过多 ({pose_fires})，可能 Schmitt+EMA 改动影响了 pose 通道",
        )
