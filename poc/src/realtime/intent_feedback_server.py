"""HTTP endpoint for intent feedback write-back and admin triggers."""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol

INTENT_LABELS = {"positive", "negative", "none"}
_ONE_SECOND_NS = 1_000_000_000


class RecentWindowProvider(Protocol):
    """Provider interface for inferencer temporal windows."""

    def get_recent_window(self, seconds: int = 10) -> list[dict[str, Any]]:
        """Return temporal snapshots with ts_ns/h/au fields."""


@dataclass(frozen=True)
class IntentFeedbackServerConfig:
    """Runtime config for intent feedback endpoint."""

    host: str = "127.0.0.1"
    port: int = 9877
    data_dir: Path = Path.home() / ".lnmem" / "data" / "intent_feedback"
    inferencer: RecentWindowProvider | None = None


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
    """Serve POST /intent_feedback and /admin/trigger_training."""

    rate_limiter: SessionTsRateLimiter
    store: IntentFeedbackStore
    inferencer: RecentWindowProvider | None = None

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/intent_feedback":
            self._handle_intent_feedback()
            return
        if self.path == "/admin/trigger_training":
            self._handle_trigger_training()
            return
        self._reply(404, {"accepted": False, "error": "not_found"})

    def _handle_intent_feedback(self) -> None:
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

            temporal_window = self._collect_temporal_window(seconds=10)
            # ADR-040 Phase1: 独立采集 10-120s 稀疏长程上下文 (不碰现有 au_window)
            # 供 Phase2 信息论互信息/oracle 验证 intent 任务是否需要 >10s 历史
            long_window = self._collect_long_context()
            record = {
                "ts_ns": ts_ns,
                "predicted": predicted,
                "corrected": corrected,
                "ui_session": ui_session,
                "note": note,
                "hidden_state_window": [
                    {"ts_ns": int(item.get("ts_ns", 0)), "h": list(item.get("h", []))}
                    for item in temporal_window
                ],
                "au_window": [
                    {"ts_ns": int(item.get("ts_ns", 0)), "au": list(item.get("au", []))}
                    for item in temporal_window
                ],
                # ADR-040 Phase1: 新增独立长程字段, 向后兼容 (旧读取忽略即可)
                "au_window_long": [
                    {"ts_ns": int(item.get("ts_ns", 0)), "au": list(item.get("au", []))}
                    for item in long_window
                ],
                "hidden_state_window_long": [
                    {"ts_ns": int(item.get("ts_ns", 0)), "h": list(item.get("h", []))}
                    for item in long_window
                ],
                "long_context_meta": {"recent_s": 10, "long_s": 120, "long_hz": 0.1},
                "recorded_at": datetime.now().astimezone().isoformat(),
            }
            stored_path = self.store.append(record)
            self._reply(200, {"accepted": True, "stored_at": str(stored_path)})
        except ValueError as exc:
            self._reply(400, {"accepted": False, "error": "invalid_payload", "detail": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._reply(500, {"accepted": False, "error": "internal_error", "detail": str(exc)})

    def _handle_trigger_training(self) -> None:
        marker_at = datetime.now().astimezone().isoformat()
        logging.getLogger(__name__).info(
            "intent admin trigger marker: trigger_training requested at %s",
            marker_at,
        )
        self._reply(200, {"triggered": True, "marker_at": marker_at})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def _collect_temporal_window(self, seconds: int) -> list[dict[str, Any]]:
        inferencer = getattr(self, "inferencer", None)
        if inferencer is None:
            return []
        try:
            raw_window = inferencer.get_recent_window(seconds=seconds)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("get_recent_window failed: %s", exc)
            return []
        if not isinstance(raw_window, list):
            return []
        normalized: list[dict[str, Any]] = []
        for item in raw_window:
            if not isinstance(item, dict):
                continue
            ts_ns = item.get("ts_ns")
            h = item.get("h")
            au = item.get("au")
            if not isinstance(ts_ns, int):
                continue
            if not isinstance(h, list) or not isinstance(au, list):
                continue
            normalized.append({"ts_ns": ts_ns, "h": h, "au": au})
        return normalized

    def _collect_long_context(self) -> list[dict[str, Any]]:
        """ADR-040 Phase1: 收集 10-120s 稀疏长程上下文 (0.1Hz 下采样)。

        inferencer.get_long_context 缓冲不足/方法缺失时 fail-soft 返回 []，
        不阻塞反馈写入 (向后兼容: 老 inferencer 无此方法时静默降级)。
        """
        inferencer = getattr(self, "inferencer", None)
        if inferencer is None or not hasattr(inferencer, "get_long_context"):
            return []
        try:
            raw = inferencer.get_long_context(recent_s=10, long_s=120, long_hz=0.1)
        except Exception as exc:  # noqa: BLE001
            logging.getLogger(__name__).warning("get_long_context failed: %s", exc)
            return []
        if not isinstance(raw, list):
            return []
        out: list[dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            ts_ns = item.get("ts_ns")
            h = item.get("h")
            au = item.get("au")
            if not isinstance(ts_ns, int) or not isinstance(h, list) or not isinstance(au, list):
                continue
            out.append({"ts_ns": ts_ns, "h": h, "au": au})
        return out

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
        IntentFeedbackHandler.inferencer = self._config.inferencer
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
