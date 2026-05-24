"""Tests for realtime websocket inference server and daemon cache compatibility."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from poc.realtime.inference_server import InferenceServer
from poc.realtime.ws_proto import _accept_websocket
from poc.training.daemon_dataset import DaemonAUDataset, build_daemon_au_cache
from poc.training.extract_au import PYFEAT_AU_COLUMNS


class _DummyWriter:
    def __init__(self) -> None:
        self.buf = bytearray()
        self.closed = False

    def write(self, data: bytes) -> None:
        self.buf.extend(data)

    async def drain(self) -> None:
        return None

    def close(self) -> None:
        self.closed = True

    async def wait_closed(self) -> None:
        return None

    def is_closing(self) -> bool:
        return self.closed


def _frame(fid: int, au_len: int = 20) -> dict:
    return {
        "ts": "2026-05-25T10:30:00.033Z",
        "frame_id": fid,
        "au": [0.1] * au_len,
        "head_pose": [1.0, -2.0, 3.0],
        "screen": [1.0, 2.0, 1.0],
    }


class _MockRuntime:
    def step(self, frame):  # noqa: ANN001
        hidden = [0.01 * (i + 1) for i in range(8)]
        tau = {"short_tau_mean": 5.2, "long_tau_mean": 1800.0, "short_tau_decay_remaining_s": 12.5}
        return hidden, tau


class _DummyConn:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_text(self, text: str) -> None:
        self.sent.append(text)


def test_websocket_connection_handshake() -> None:
    async def _case() -> None:
        reader = asyncio.StreamReader()
        writer = _DummyWriter()
        req = (
            "GET /ws HTTP/1.1\r\n"
            "Host: 127.0.0.1\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            "Sec-WebSocket-Version: 13\r\n"
            "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n\r\n"
        )
        reader.feed_data(req.encode("utf-8"))
        reader.feed_eof()
        conn = await _accept_websocket(reader, writer)  # noqa: F841
        assert b"101 Switching Protocols" in writer.buf

    asyncio.run(_case())


def test_inbound_schema_validation() -> None:
    async def _case() -> None:
        conn = _DummyConn()
        server = InferenceServer(runtime=_MockRuntime())
        await server._on_message(conn, json.dumps(_frame(1, au_len=19)))
        msg = json.loads(conn.sent[0])
        assert "error" in msg
        assert "20 dims" in msg["error"]

    asyncio.run(_case())


def test_outbound_schema_validation() -> None:
    async def _case() -> None:
        conn = _DummyConn()
        server = InferenceServer(runtime=_MockRuntime())
        server._clients.add(conn)  # pylint: disable=protected-access
        await server._on_message(conn, json.dumps(_frame(7)))
        out = json.loads(conn.sent[-1])
        assert out["frame_id"] == 7
        assert isinstance(out["state_hidden"], list)
        assert out["intent_pred"] is None
        assert {"short_tau_mean", "long_tau_mean", "short_tau_decay_remaining_s"} <= set(out["tau_state"].keys())
        assert out["latency_ms"] < 50.0

    asyncio.run(_case())


def test_latency_under_50ms() -> None:
    async def _case() -> None:
        conn = _DummyConn()
        server = InferenceServer(runtime=_MockRuntime())
        server._clients.add(conn)  # pylint: disable=protected-access
        values = []
        for i in range(100):
            await server._on_message(conn, json.dumps(_frame(i)))
            out = json.loads(conn.sent[-1])
            values.append(float(out["latency_ms"]))
        p50 = float(np.percentile(values, 50))
        p95 = float(np.percentile(values, 95))
        assert p50 < 30.0
        assert p95 < 50.0

    asyncio.run(_case())


def test_backward_compat_old_daemon_cache_no_head_pose(tmp_path: Path) -> None:
    out = tmp_path / "old_cache.npz"
    np.savez_compressed(
        out,
        au_demo=np.ones((4, 20), dtype=np.float32),
        frame_demo=np.arange(4, dtype=np.int64),
    )
    sample = DaemonAUDataset(out)[0]
    assert sample["head_pose"].shape == (4, 3)
    assert sample["screen"].shape == (4, 3)
    assert float(sample["head_pose"].sum().item()) == 0.0
    assert float(sample["screen"].sum().item()) == 0.0


def test_daemon_dataset_with_head_pose_and_screen(tmp_path: Path) -> None:
    session = "11111111-1111-1111-1111-111111111111"
    rows = []
    for fid in range(3):
        rows.append(
            {
                "frame_id": fid,
                "face_detected": True,
                "session_id": session,
                "calibrated_au": {k: 0.2 for k in PYFEAT_AU_COLUMNS},
                "raw_au": {k: 0.2 for k in PYFEAT_AU_COLUMNS},
                "head_pose": {"yaw_deg": 1.0 + fid, "pitch_deg": 2.0, "roll_deg": 3.0},
                "screen_activity": {"active_app": "Cursor", "key_rate_per_s": 2.5, "mouse_rate_per_s": 1.5},
            }
        )
    signal = tmp_path / "signals_2026-05-25.jsonl"
    signal.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    out = tmp_path / "cache_v2.npz"
    build_daemon_au_cache([signal], out, include_head_pose=True, include_screen=True)
    sample = DaemonAUDataset(out)[0]
    assert sample["head_pose"].shape == (3, 3)
    assert sample["screen"].shape == (3, 3)
    assert sample["head_pose"][0, 0].item() == 1.0
    assert sample["screen"][0, 0].item() == 1.0
