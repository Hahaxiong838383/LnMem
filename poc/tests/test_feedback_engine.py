"""Tests for ProactiveQueryEngine — mock all external dependencies."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from poc.src.realtime.feedback.dnd import DNDConfig
from poc.src.realtime.feedback.engine import EngineConfig, ProactiveQueryEngine
from poc.src.realtime.feedback.persistence import queries_path
from poc.src.realtime.feedback.prompt import UserResponse
from poc.src.realtime.feedback.rate_limiter import RateLimiterConfig, TokenBucketRateLimiter
from poc.src.realtime.feedback.trigger import ProactiveQueryTrigger


def _engine(home: Path, *, max_per_hour: int = 3, dnd_apps: tuple[str, ...] = (), runner=None, dry_run=False):
    rate = RateLimiterConfig(min_per_hour=1, max_per_hour=max_per_hour, state_path=home / ".lnmem" / "feedback" / "rate_state.json")
    dnd = DNDConfig(apps=dnd_apps)
    cfg = EngineConfig(rate=rate, dnd=dnd, enable_prompt=not dry_run)
    rate_limiter = TokenBucketRateLimiter(rate)
    return ProactiveQueryEngine(cfg, rate_limiter=rate_limiter, prompt_runner=runner)


def _frame_brow(brow=0.96, fid=0, vlm=None):
    return {
        "ts": "2026-05-09T17:00:00",
        "frame_id": fid,
        "face_detected": True,
        "expression": {"brow_furrow": brow, "smile_intensity": 0.0, "eye_closure": 0.5, "cheek_raise": 0.0},
        "vlm_label": vlm,
    }


def _mk_runner(label="match", true_state=None, note=None):
    """Synchronous mock that returns immediately."""
    r = MagicMock()
    r.show_notification = MagicMock()
    r.ask_match = MagicMock(return_value=label)
    r.choose_state = MagicMock(return_value=true_state)
    r.input_note = MagicMock(return_value=note)
    return r


class TestEngine(unittest.TestCase):
    def test_no_trigger_no_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, dry_run=True)
            for i in range(4):
                eng.observe(_frame_brow(brow=0.9, fid=i), home=home)
            stats = eng.stats()
            self.assertEqual(stats["triggered"], 0)
            self.assertEqual(stats["queried"], 0)

    def test_brow_burst_triggers_in_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, dry_run=True)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                for i in range(15):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
            stats = eng.stats()
            self.assertEqual(stats["triggered"], 1)
            self.assertEqual(stats["queried"], 1)

    def test_dnd_blocks_query(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, dnd_apps=("zoom",), dry_run=True)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="zoom.us"):
                for i in range(15):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
            stats = eng.stats()
            self.assertEqual(stats["triggered"], 1)
            self.assertEqual(stats["dnd_blocked"], 1)
            self.assertEqual(stats["queried"], 0)

    def test_rate_limit_blocks_after_max(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, max_per_hour=1, dry_run=True)  # only 1/h
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                # First burst → triggers + queries
                for i in range(15):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
                # Second burst → triggers but rate-blocked
                # 必须在 brow cooldown 外（默认 300 帧）；第一次触发 fid=14
                for i in range(315, 330):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
            stats = eng.stats()
            self.assertEqual(stats["triggered"], 2)
            self.assertEqual(stats["queried"], 1)
            self.assertEqual(stats["rate_blocked"], 1)

    def test_prompt_runner_match_writes_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runner = _mk_runner(label="match")
            eng = _engine(home, runner=runner)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                for i in range(15):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
            # Wait for background thread to finish
            for _ in range(20):
                if (home / ".lnmem" / "feedback").exists() and any(
                    (home / ".lnmem" / "feedback").glob("queries_*.jsonl")
                ):
                    break
                time.sleep(0.05)
            files = list((home / ".lnmem" / "feedback").glob("queries_*.jsonl"))
            self.assertEqual(len(files), 1)
            with files[0].open("r", encoding="utf-8") as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["user_response"]["label"], "match")
            self.assertEqual(rec["reason"], "brow_burst")

    def test_prompt_runner_mismatch_records_true_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            runner = _mk_runner(label="mismatch", true_state="focused")
            eng = _engine(home, runner=runner)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                for i in range(15):
                    eng.observe(_frame_brow(brow=0.96, fid=i), home=home)
            for _ in range(20):
                files = list((home / ".lnmem" / "feedback").glob("queries_*.jsonl"))
                if files:
                    break
                time.sleep(0.05)
            with list((home / ".lnmem" / "feedback").glob("queries_*.jsonl"))[0].open("r", encoding="utf-8") as f:
                rec = json.loads(f.readline())
            self.assertEqual(rec["user_response"]["label"], "mismatch")
            self.assertEqual(rec["user_response"]["true_state"], "focused")

    def test_low_vlm_conf_triggers(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, dry_run=True)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                eng.observe(_frame_brow(brow=0.1, fid=0,
                                         vlm={"confidence": 0.3, "valence": 0.0,
                                              "arousal": 0.5, "au_description": "ambiguous"}),
                            home=home)
            self.assertEqual(eng.stats()["triggered"], 1)


class TestFormatQuestionAdr005UI(unittest.TestCase):
    """ADR-005 V0.3.3：弹窗 UI 皱眉显示 brow_used_value/_path（trigger 决策实际依据），
    而不是被 calibration polarization 撒谎的 brow_furrow。

    川哥 2026-05-16 22:10 实测发现：demo 模式 raw 路径下，弹窗显示"皱眉=1.00"，
    但 trigger 决策依据的 raw_au_AU04 实际只有 0.40 左右——engine.py:44 漏改。
    """

    def _mk_event(self, snapshot):
        from poc.src.realtime.feedback.trigger import TriggerEvent
        return TriggerEvent(
            ts="2026-05-16T14:03:36Z",
            frame_id=99,
            reason="brow_burst",
            signal_snapshot=snapshot,
            vlm_label=None,
        )

    def test_raw_path_shows_brow_used_with_cal_compare(self):
        """raw 路径：显示 brow_used_value(raw) + [cal=X.XX] 对照"""
        from poc.src.realtime.feedback.engine import _format_question
        ev = self._mk_event({
            "brow_furrow": 1.0,           # calibrated polarized 撒谎值
            "raw_au_AU04": 0.41,
            "brow_used_path": "raw",
            "brow_used_value": 0.41,
            "smile_intensity": 0.05,
            "eye_closure": 0.0,
        })
        _, detail = _format_question(ev)
        self.assertIn("皱眉=0.41(raw)", detail)
        self.assertIn("[cal=1.00]", detail)  # 让用户看到 polarization
        self.assertNotIn("皱眉=1.00", detail)  # 不再被 calibrated 撒谎值误导

    def test_calibrated_path_shows_brow_used_no_compare(self):
        """calibrated 路径：只显示 brow_used_value(calibrated)，无 [cal=] 对照"""
        from poc.src.realtime.feedback.engine import _format_question
        ev = self._mk_event({
            "brow_furrow": 0.95,
            "brow_used_path": "calibrated",
            "brow_used_value": 0.95,
            "smile_intensity": 0.1,
            "eye_closure": 0.2,
        })
        _, detail = _format_question(ev)
        self.assertIn("皱眉=0.95(calibrated)", detail)
        self.assertNotIn("[cal=", detail)

    def test_legacy_snapshot_falls_back_to_brow_furrow(self):
        """老 snapshot（无 brow_used_value/_path）：回落到 brow_furrow + 标 calibrated"""
        from poc.src.realtime.feedback.engine import _format_question
        ev = self._mk_event({
            "brow_furrow": 0.7,
            "smile_intensity": 0.0,
            "eye_closure": 0.0,
        })
        _, detail = _format_question(ev)
        self.assertIn("皱眉=0.70(calibrated)", detail)
        self.assertNotIn("[cal=", detail)


if __name__ == "__main__":
    unittest.main()
