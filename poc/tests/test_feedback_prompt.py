"""Tests for osascript prompt module — mock subprocess, no real GUI."""
import subprocess
import unittest
from unittest.mock import MagicMock, patch

from poc.src.realtime.feedback.prompt import (
    UserResponse,
    _osascript,
    ask_match,
    choose_state,
    input_note,
    run_three_layer_query,
    show_notification,
)


class TestOsascriptHelper(unittest.TestCase):
    @patch("subprocess.run")
    def test_osascript_returns_stdout(self, mock_run):
        mock_run.return_value = MagicMock(returncode=0, stdout="hello\n")
        self.assertEqual(_osascript("anything"), "hello")

    @patch("subprocess.run", side_effect=subprocess.TimeoutExpired(cmd=["osascript"], timeout=1))
    def test_osascript_timeout_returns_none(self, _):
        self.assertIsNone(_osascript("anything"))

    @patch("subprocess.run", side_effect=FileNotFoundError())
    def test_osascript_missing_binary_returns_none(self, _):
        self.assertIsNone(_osascript("anything"))

    @patch("subprocess.run")
    def test_osascript_nonzero_exit_returns_none(self, mock_run):
        mock_run.return_value = MagicMock(returncode=1, stdout="")
        self.assertIsNone(_osascript("anything"))


class TestAskMatch(unittest.TestCase):
    @patch("poc.src.realtime.feedback.prompt._osascript", return_value="button returned:✓ 准确")
    def test_match(self, _):
        self.assertEqual(ask_match("Q?"), "match")

    @patch("poc.src.realtime.feedback.prompt._osascript", return_value="button returned:✗ 不准")
    def test_mismatch(self, _):
        self.assertEqual(ask_match("Q?"), "mismatch")

    @patch("poc.src.realtime.feedback.prompt._osascript", return_value="button returned:📝 输入")
    def test_input(self, _):
        self.assertEqual(ask_match("Q?"), "input")

    @patch("poc.src.realtime.feedback.prompt._osascript", return_value=None)
    def test_timeout_returns_none(self, _):
        self.assertIsNone(ask_match("Q?"))


class TestChooseState(unittest.TestCase):
    @patch("poc.src.realtime.feedback.prompt._osascript", return_value="emotional_positive 积极")
    def test_returns_english_label(self, _):
        self.assertEqual(choose_state(), "emotional_positive")

    @patch("poc.src.realtime.feedback.prompt._osascript", return_value="false")
    def test_cancel_returns_none(self, _):
        self.assertIsNone(choose_state())


class TestInputNote(unittest.TestCase):
    @patch("poc.src.realtime.feedback.prompt._osascript",
           return_value="button returned:OK, text returned:刚才在思考问题")
    def test_extracts_text(self, _):
        self.assertEqual(input_note(), "刚才在思考问题")

    @patch("poc.src.realtime.feedback.prompt._osascript", return_value=None)
    def test_timeout_returns_none(self, _):
        self.assertIsNone(input_note())


class TestRunThreeLayer(unittest.TestCase):
    def _runner(self, ask_result, state_result=None, note_result=None):
        r = MagicMock()
        r.show_notification = MagicMock()
        r.ask_match = MagicMock(return_value=ask_result)
        r.choose_state = MagicMock(return_value=state_result)
        r.input_note = MagicMock(return_value=note_result)
        return r

    def test_match_shortcut(self):
        runner = self._runner("match")
        resp = run_three_layer_query("T", "S", "D", runner=runner)
        self.assertEqual(resp.label, "match")
        runner.choose_state.assert_not_called()
        runner.input_note.assert_not_called()

    def test_mismatch_chains_state_picker(self):
        runner = self._runner("mismatch", state_result="emotional_negative")
        resp = run_three_layer_query("T", "S", "D", runner=runner)
        self.assertEqual(resp.label, "mismatch")
        self.assertEqual(resp.true_state, "emotional_negative")

    def test_input_chains_note(self):
        runner = self._runner("input", note_result="我有点累")
        resp = run_three_layer_query("T", "S", "D", runner=runner)
        self.assertEqual(resp.label, "input")
        self.assertEqual(resp.note, "我有点累")

    def test_skip_on_timeout(self):
        runner = self._runner(None)
        resp = run_three_layer_query("T", "S", "D", runner=runner)
        self.assertEqual(resp.label, "skip")

    def test_user_response_immutable(self):
        r = UserResponse(label="match", true_state=None, note=None)
        with self.assertRaises((AttributeError, TypeError)):
            r.label = "x"  # type: ignore


class TestShowNotification(unittest.TestCase):
    @patch("poc.src.realtime.feedback.prompt._osascript")
    def test_notification_calls_osascript(self, mock_run):
        show_notification("title", 'msg with "quote"')
        mock_run.assert_called_once()
        # Verify quote escaping
        call_arg = mock_run.call_args[0][0]
        self.assertNotIn('msg with "quote"', call_arg)


if __name__ == "__main__":
    unittest.main()
