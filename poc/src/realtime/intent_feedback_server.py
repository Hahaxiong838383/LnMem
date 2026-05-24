"""HTTP endpoint for intent feedback write-back."""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

INTENT_LABELS = {"positive", "negative", "none"}
_ONE_SECOND_NS = 1_000_000_000


@dataclass(frozen=True)
class IntentFeedbackServerConfig:
    """Runtime config for intent feedback endpoint."""

    host: str = "127.0.0.1"
    port: int = 9877
    data_dir: Path = Path.home() / ".lnmem" / "data" / "intent_feedback"


class SessionTsRateLimiter:
    """Deduplicate by (ui_session, ts_ns ±1s)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._accepted: dict[str, list[int]] = {}

    def allow(self, ui_session: str, ts_ns: int) -> bool:
        with self._lock:
            accepted = self._accepted.setdefault(ui_session, [])
            for past in accepted:
                if abs(past - ts_ns) <= _ONE_SECOND_NS:
                    return False
            accepted.append(ts_ns)
            if len(accepted) > 1024:
                accepted[:] = accepted[-512:]
            return True


class IntentFeedbackStore:
    """Append-only jsonl storage by day with thread safety."""

    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._lock = threading.Lock()
        self._data_dir.mkdir(parents=True, exist_ok=True)

    def append(self, record: dict[str, Any]) -> Path:
        ts_ns = int(record["ts_ns"])
        day = datetime.fromtimestamp(ts_ns / 1_000_000_000).astimezone().strftime("%Y-%m-%d")
        path = self._data_dir / f"{day}.jsonl"
        line = json.dumps(record, ensure_ascii=False) + "\n"
        with self._lock:
            with path.open("a", encoding="utf-8") as handle:
                handle.write(line)
        return path


class IntentFeedbackHandler(BaseHTTPRequestHandler):
    """Serve POST /intent_feedback with strict payload validation."""

    rate_limiter: SessionTsRateLimiter
    store: IntentFeedbackStore

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/intent_feedback":
            self._reply(404, {"accepted": False, "error": "not_found"})
            return
        try:
            payload = self._read_json()
            ts_ns = self._parse_ts_ns(payload.get("ts_ns"))
            predicted = self._parse_label(payload.get("predicted"), "predicted")
            corrected = self._parse_label(payload.get("corrected"), "corrected")
            ui_session = self._parse_ui_session(payload.get("ui_session"))
            note = payload.get("note")
            if note is not None and not isinstance(note, str):
                raise ValueError("note must be a string when provided")

            if not self.rate_limiter.allow(ui_session=ui_session, ts_ns=ts_ns):
                self._reply(429, {"accepted": False, "error": "rate_limited"})
                return

            record = {
                "received_at": datetime.now().astimezone().isoformat(),
                "ts_ns": ts_ns,
                "predicted": predicted,
                "corrected": corrected,
                "ui_session": ui_session,
                "note": note,
            }
            stored_path = self.store.append(record)
            self._reply(200, {"accepted": True, "stored_at": str(stored_path)})
        except ValueError as exc:
            self._reply(400, {"accepted": False, "error": "invalid_payload", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._reply(500, {"accepted": False, "error": "internal_error", "detail": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0:
            raise ValueError("empty body")
        raw = self.rfile.read(length)
        data = json.loads(raw.decode("utf-8"))
        if not isinstance(data, dict):
            raise ValueError("request body must be an object")
        return data

    def _parse_ts_ns(self, value: Any) -> int:
        if not isinstance(value, int) or value <= 0:
            raise ValueError("ts_ns must be a positive int")
        return value

    def _parse_label(self, value: Any, field: str) -> str:
        if not isinstance(value, str) or value not in INTENT_LABELS:
            raise ValueError(f"{field} must be one of {sorted(INTENT_LABELS)}")
        return value

    def _parse_ui_session(self, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ui_session must be a non-empty string")
        return value.strip()

    def _reply(self, status: int, body: dict[str, Any]) -> None:
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class IntentFeedbackServer:
    """Lifecycle wrapper for ThreadingHTTPServer."""

    def __init__(self, config: IntentFeedbackServerConfig | None = None) -> None:
        self._config = config or IntentFeedbackServerConfig()
        self._http: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._http is not None:
            return
        IntentFeedbackHandler.rate_limiter = SessionTsRateLimiter()
        IntentFeedbackHandler.store = IntentFeedbackStore(self._config.data_dir)
        self._http = ThreadingHTTPServer((self._config.host, self._config.port), IntentFeedbackHandler)
        self._thread = threading.Thread(target=self._http.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._http is None:
            return
        self._http.shutdown()
        self._http.server_close()
        self._http = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def bound_port(self) -> int:
        if self._http is None:
            raise RuntimeError("server not started")
        return int(self._http.server_address[1])


__all__ = ["IntentFeedbackServerConfig", "IntentFeedbackServer"]
