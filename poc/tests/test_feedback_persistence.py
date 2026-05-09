"""Tests for feedback persistence."""
import json
import tempfile
import unittest
from pathlib import Path

from poc.src.realtime.feedback.persistence import append_query, queries_path


class TestPersistence(unittest.TestCase):
    def test_queries_path_under_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            p = queries_path(day_iso="2026-05-09", home=home)
            self.assertEqual(p, home / ".lnmem" / "feedback" / "queries_2026-05-09.jsonl")
            self.assertTrue(p.parent.exists())

    def test_append_then_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            rec = {"ts": "2026-05-09T17:00:00", "frame_id": 42, "reason": "brow_burst",
                   "signal": {"brow_furrow": 0.9}, "vlm_label": None,
                   "user_response": {"label": "match", "true_state": None, "note": None},
                   "latency_s": 1.2}
            path = append_query(rec, home=home)
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            parsed = json.loads(lines[0])
            self.assertEqual(parsed["frame_id"], 42)
            self.assertEqual(parsed["user_response"]["label"], "match")

    def test_append_multiple_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            for i in range(3):
                append_query({"frame_id": i, "reason": "x", "user_response": {"label": "skip"}}, home=home)
            path = queries_path(home=home)
            with path.open("r", encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 3)

    def test_unicode_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            home = Path(tmp)
            append_query({"note": "我刚才在专注思考", "frame_id": 1}, home=home)
            with queries_path(home=home).open("r", encoding="utf-8") as f:
                self.assertIn("我刚才", f.read())


if __name__ == "__main__":
    unittest.main()
