"""Tests for daemon SSE stream and static dashboard route."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.models import Response

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import poc.src.realtime.daemon_main as daemon_main  # noqa: E402
from poc.src.realtime.daemon_main import DaemonApp  # noqa: E402


class _FakeStorage:
    def __init__(self) -> None:
        self._signals = [{
            "ts": datetime.now(timezone.utc).isoformat(),
            "frame_id": 42,
            "lnn": {
                "status": "ok",
                "predicted_state": "focused",
                "state_probs": [0.1, 0.7, 0.05, 0.02, 0.03, 0.04, 0.05, 0.01],
                "intent_probs": [0.8, 0.1, 0.1],
            },
        }]

    def get_recent(self, _seconds: int) -> list[dict[str, Any]]:
        return list(self._signals)


class _FakeInferencer:
    ckpt_sha8 = "abc12345"

    def predict_future(self, steps: int) -> list[dict[str, Any]]:
        return [{"step": idx, "state_probs": [1.0]} for idx in range(steps)]


@pytest.fixture()
def app() -> DaemonApp:
    app = DaemonApp.__new__(DaemonApp)
    app._storage = _FakeStorage()
    app._lnn_inferencer = _FakeInferencer()
    app._http_server = None
    return app


class _IterableRaw:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    def stream(self, _chunk_size: int, decode_content: bool = False):
        _ = decode_content
        yield from self._chunks

    def close(self) -> None:
        return None


class _LnMemAdapter(BaseAdapter):
    def __init__(self, app: DaemonApp) -> None:
        super().__init__()
        self._app = app

    def send(self, request, stream: bool = False, **_kwargs: Any) -> Response:
        response = Response()
        response.url = request.url
        response.request = request
        path = requests.utils.urlparse(request.url).path
        if path == "/signals/stream":
            response.encoding = "utf-8"
            if daemon_main.os.environ.get("LNMEM_DISABLE_SSE") == "1":
                response.status_code = 503
                response.headers["Content-Type"] = "application/json; charset=utf-8"
                response._content = b'{"error":"sse_disabled_by_env"}'
                return response
            response.status_code = 200
            response.headers.update(self._app.sse_headers())
            frame = self._app.build_sse_frame_event()
            payloads = [frame, self._app.build_sse_prediction_event()]
            chunks = [
                f"data: {json.dumps(payload, ensure_ascii=False)}\n\n".encode("utf-8")
                for payload in payloads
                if payload is not None
            ]
            response.raw = _IterableRaw(chunks)
            response._content = False
            return response
        if path == "/dashboard":
            code, body, content_type = self._app.dashboard_response()
            response.status_code = code
            response.headers["Content-Type"] = content_type
            response._content = body
            return response
        response.status_code = 404
        response._content = b'{"error":"not_found"}'
        return response

    def close(self) -> None:
        return None


@pytest.fixture()
def session(app: DaemonApp) -> requests.Session:
    session = requests.Session()
    session.mount("http://lnmem.test", _LnMemAdapter(app))
    return session


def test_sse_response_headers_and_two_events(
    session: requests.Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LNMEM_DISABLE_SSE", raising=False)

    with session.get("http://lnmem.test/signals/stream", stream=True, timeout=3) as resp:
        assert resp.status_code == 200
        assert resp.headers["Content-Type"] == "text/event-stream"
        assert resp.headers["Cache-Control"] == "no-cache"
        assert resp.headers["X-Accel-Buffering"] == "no"
        assert resp.headers["Connection"] == "keep-alive"

        events: list[str] = []
        for line in resp.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                events.append(line)
            if len(events) >= 2:
                break

    assert len(events) >= 2
    assert '"type": "frame"' in events[0]
    assert '"type": "prediction"' in "\n".join(events)
    assert '"predictions"' in "\n".join(events)


def test_sse_env_disable_returns_503(
    session: requests.Session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LNMEM_DISABLE_SSE", "1")

    resp = session.get("http://lnmem.test/signals/stream", timeout=3)

    assert resp.status_code == 503
    assert resp.json()["error"] == "sse_disabled_by_env"


def test_dashboard_route_serves_html(
    session: requests.Session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    dashboard = tmp_path / "realtime_dashboard.html"
    dashboard.write_text("<!doctype html><title>LnMem</title>", encoding="utf-8")
    monkeypatch.setattr(daemon_main, "_DASHBOARD_PATH", dashboard)

    resp = session.get("http://lnmem.test/dashboard", timeout=3)

    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "text/html; charset=utf-8"
    assert "LnMem" in resp.text


def test_dashboard_route_missing_file_returns_404(
    session: requests.Session,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(daemon_main, "_DASHBOARD_PATH", tmp_path / "missing.html")

    resp = session.get("http://lnmem.test/dashboard", timeout=3)

    assert resp.status_code == 404
    assert resp.json()["error"] == "dashboard_not_found"
