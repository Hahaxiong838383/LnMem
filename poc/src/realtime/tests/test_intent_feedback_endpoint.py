"""Tests for /intent_feedback handler semantics."""

from __future__ import annotations

import io
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from poc.src.realtime.intent_feedback_server import (  # noqa: E402
    IntentFeedbackHandler,
    IntentFeedbackStore,
    SessionTsRateLimiter,
)


def _run_post(
    payload: dict[str, object],
    store: IntentFeedbackStore,
    limiter: SessionTsRateLimiter,
    path: str = "/intent_feedback",
) -> tuple[int, dict[str, object]]:
    raw = json.dumps(payload).encode("utf-8")
    handler = IntentFeedbackHandler.__new__(IntentFeedbackHandler)
    handler.path = path
    handler.headers = {"Content-Length": str(len(raw))}
    handler.rfile = io.BytesIO(raw)
    handler.rate_limiter = limiter
    handler.store = store
    captured: dict[str, object] = {}

    def _reply(status: int, body: dict[str, object]) -> None:
        captured["status"] = status
        captured["body"] = body

    handler._reply = _reply  # type: ignore[method-assign]
    handler.do_POST()
    return int(captured["status"]), captured["body"]  # type: ignore[return-value]


def test_intent_feedback_accept_and_store(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    status, body = _run_post(
        {
            "ts_ns": 1_717_000_000_000_000_000,
            "predicted": "positive",
            "corrected": "negative",
            "ui_session": "session-a",
            "note": "fix label",
        },
        store=store,
        limiter=limiter,
    )

    assert status == 200
    assert body["accepted"] is True
    assert "stored_at" in body

    files = list((tmp_path / "intent_feedback").glob("*.jsonl"))
    assert len(files) == 1
    lines = files[0].read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["ts_ns"] == 1_717_000_000_000_000_000
    assert record["predicted"] == "positive"
    assert record["corrected"] == "negative"


def test_intent_feedback_rate_limit_same_session_and_ts_window(tmp_path: Path) -> None:
    store = IntentFeedbackStore(tmp_path / "intent_feedback")
    limiter = SessionTsRateLimiter()
    first_ts = 1_717_000_000_000_000_000

    status1, body1 = _run_post(
        {
            "ts_ns": first_ts,
            "predicted": "none",
            "corrected": "positive",
            "ui_session": "session-a",
        },
        store=store,
        limiter=limiter,
    )
    status2, body2 = _run_post(
        {
            "ts_ns": first_ts + 500_000_000,  # +0.5s
            "predicted": "none",
            "corrected": "negative",
            "ui_session": "session-a",
        },
        store=store,
        limiter=limiter,
    )

    assert status1 == 200 and body1["accepted"] is True
    assert status2 == 429
    assert body2["accepted"] is False
    assert body2["error"] == "rate_limited"
