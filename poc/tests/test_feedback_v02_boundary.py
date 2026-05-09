"""Boundary tests for feedback V0.2 contracts."""

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from poc.src.realtime.daemon_main import DaemonApp
from poc.src.realtime.feedback.dnd import DNDConfig
from poc.src.realtime.feedback.engine import EngineConfig, ProactiveQueryEngine
from poc.src.realtime.feedback.rate_limiter import RateLimiterConfig, TokenBucketRateLimiter
from poc.src.realtime.feedback.trigger import (
    ProactiveQueryTrigger,
    _BROW_BURST_THRESHOLD,
    _BROW_BURST_WINDOW,
)


class TestNewThresholdConstants(unittest.TestCase):
    def test_brow_window_is_10(self):
        self.assertEqual(_BROW_BURST_WINDOW, 10)

    def test_brow_threshold_is_0_85(self):
        self.assertAlmostEqual(_BROW_BURST_THRESHOLD, 0.85)

    def test_trigger_requires_10_consecutive_high_brow(self):
        t = ProactiveQueryTrigger()
        for i in range(9):
            self.assertIsNone(t.observe(_brow_frame(0.9, fid=i)))
        ev = t.observe(_brow_frame(0.9, fid=9))
        self.assertIsNotNone(ev)
        self.assertEqual(ev.reason, "brow_burst")

    def test_brow_below_85_no_trigger(self):
        t = ProactiveQueryTrigger()
        for i in range(15):
            self.assertIsNone(t.observe(_brow_frame(0.80, fid=i)))


class TestEngineStop(unittest.TestCase):
    def test_stop_joins_pending_prompt_threads(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            ready, done = threading.Event(), threading.Event()

            def slow_ask(*_a, **_kw):
                ready.set()
                time.sleep(0.5)
                done.set()
                return "match"

            runner = MagicMock()
            runner.show_notification = MagicMock()
            runner.ask_match = slow_ask
            runner.choose_state = MagicMock(return_value=None)
            runner.input_note = MagicMock(return_value=None)

            eng = _engine(home, runner=runner)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                for i in range(10):
                    eng.observe(_brow_frame(0.9, fid=i), home=home)
            self.assertTrue(ready.wait(timeout=2.0))

            t0 = time.time()
            eng.stop(timeout_s=5.0)
            self.assertTrue(done.is_set(), "stop should wait for prompt thread")
            self.assertLess(time.time() - t0, 5.5)
            self.assertEqual(len(_query_files(home)), 1)

    def test_stop_idempotent(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            eng = _engine(home, dry_run=True)
            eng.stop()
            eng.stop()


class TestSessionIdField(unittest.TestCase):
    def test_session_id_and_event_uuid_in_jsonl(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            sid = "test-session-abc-123"
            eng = _engine(home, runner=_mk_runner(label="match"), session_id=sid)
            with patch("poc.src.realtime.feedback.engine.get_frontmost_app", return_value="Cursor"):
                for i in range(10):
                    eng.observe(_brow_frame(0.9, fid=i), home=home)
            eng.stop(timeout_s=5.0)

            rec = _read_first_record(home)
            self.assertEqual(rec["session_id"], sid)
            self.assertIn("event_uuid", rec)
            self.assertEqual(len(rec["event_uuid"]), 32)


class TestDaemonFeedbackThresholdWiring(unittest.TestCase):
    def test_daemon_build_uses_configured_vlm_threshold_and_session_id(self):
        app = object.__new__(DaemonApp)
        app._session_id = "daemon-session-1"
        app._config = SimpleNamespace(
            feedback=SimpleNamespace(
                query_min_per_hour=1,
                query_max_per_hour=3,
                query_confidence_threshold=0.5,
                dnd_apps=(),
            )
        )

        eng = DaemonApp._build_feedback_engine(app)
        self.assertAlmostEqual(eng._trigger._vlm_conf_threshold, 0.5)
        self.assertEqual(eng._session_id, "daemon-session-1")


def _brow_frame(brow=0.9, fid=0, vlm=None):
    return {
        "ts": "2026-05-09T17:00:00",
        "frame_id": fid,
        "face_detected": True,
        "expression": {
            "brow_furrow": brow,
            "smile_intensity": 0.0,
            "eye_closure": 0.5,
            "cheek_raise": 0.0,
        },
        "vlm_label": vlm,
    }


def _engine(home: Path, *, max_per_hour=3, dnd_apps=(), runner=None, dry_run=False, session_id="test-session"):
    rate = RateLimiterConfig(
        min_per_hour=1,
        max_per_hour=max_per_hour,
        state_path=home / ".lnmem" / "feedback" / "rate_state.json",
    )
    dnd = DNDConfig(apps=dnd_apps)
    cfg = EngineConfig(rate=rate, dnd=dnd, enable_prompt=not dry_run)
    return ProactiveQueryEngine(
        cfg,
        rate_limiter=TokenBucketRateLimiter(rate),
        prompt_runner=runner,
        session_id=session_id,
    )


def _mk_runner(label="match", true_state=None, note=None):
    r = MagicMock()
    r.show_notification = MagicMock()
    r.ask_match = MagicMock(return_value=label)
    r.choose_state = MagicMock(return_value=true_state)
    r.input_note = MagicMock(return_value=note)
    return r


def _query_files(home: Path) -> list[Path]:
    return list((home / ".lnmem" / "feedback").glob("queries_*.jsonl"))


def _read_first_record(home: Path) -> dict:
    files = _query_files(home)
    assert files, "expected at least one query jsonl"
    with files[0].open("r", encoding="utf-8") as f:
        return json.loads(f.readline())


if __name__ == "__main__":
    unittest.main()
