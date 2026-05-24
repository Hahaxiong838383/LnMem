"""Realtime LNN inference websocket server (30Hz multimodal)."""
from __future__ import annotations

import asyncio
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from poc.lnn.heads import FusionConfig, FusionLayer
from poc.lnn.multilayer_cfc import MultiLayerCustomCfC

from .ws_proto import WSConnection, start_ws_server


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_ts_sec(ts: str) -> float:
    return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()


@dataclass(frozen=True)
class InboundFrame:
    ts: str
    frame_id: int
    au: np.ndarray
    head_pose: np.ndarray
    screen: np.ndarray


def _validate_inbound(payload: dict[str, Any]) -> InboundFrame:
    if not isinstance(payload.get("ts"), str):
        raise ValueError("`ts` must be string")
    if not isinstance(payload.get("frame_id"), int):
        raise ValueError("`frame_id` must be int")
    au = np.asarray(payload.get("au"), dtype=np.float32)
    head = np.asarray(payload.get("head_pose"), dtype=np.float32)
    screen = np.asarray(payload.get("screen"), dtype=np.float32)
    if au.shape != (20,):
        raise ValueError("`au` must have 20 dims")
    if head.shape != (3,):
        raise ValueError("`head_pose` must have 3 dims")
    if screen.shape != (3,):
        raise ValueError("`screen` must have 3 dims")
    if not np.isfinite(au).all() or not np.isfinite(head).all() or not np.isfinite(screen).all():
        raise ValueError("payload contains NaN/inf")
    return InboundFrame(ts=payload["ts"], frame_id=payload["frame_id"], au=au, head_pose=head, screen=screen)


class RealtimeLNNRuntime:
    """Single-frame ODE runtime using FusionLayer + MultiLayerCustomCfC."""

    def __init__(self, ckpt_path: str | Path | None, target_fps: float = 30.0, persona_idx: int = 0):
        self.target_dt = 1.0 / max(target_fps, 1.0)
        self.persona_idx = torch.tensor([persona_idx], dtype=torch.long)
        self.fusion, self.backbone = self._load_or_init(ckpt_path)
        self.hidden = [cell.init_hidden(batch_size=1, device=torch.device("cpu")) for cell in self.backbone.cells]
        self.step_idx = 0
        self.prev_ts: float | None = None

    def _load_or_init(self, ckpt_path: str | Path | None) -> tuple[FusionLayer, MultiLayerCustomCfC]:
        fusion = FusionLayer(FusionConfig(input_dim=52, fusion_dim=64)).eval()
        backbone = MultiLayerCustomCfC(input_dim=64, hidden_dim=64, num_layers=2, num_personas=64).eval()
        if ckpt_path is None:
            return fusion, backbone
        payload = torch.load(Path(ckpt_path), map_location=torch.device("cpu"), weights_only=False)
        if not isinstance(payload, dict):
            return fusion, backbone
        f_state = payload.get("fusion_state") or payload.get("fusion") or payload.get("state_fusion")
        b_state = payload.get("backbone_state") or payload.get("backbone") or payload.get("state_backbone")
        if isinstance(f_state, dict):
            fusion.load_state_dict(f_state, strict=False)
        if isinstance(b_state, dict):
            backbone.load_state_dict(b_state, strict=False)
        return fusion, backbone

    def _build_event_tensor(self, frame: InboundFrame) -> np.ndarray:
        phase = self.step_idx * self.target_dt
        au24 = np.concatenate([frame.au, frame.head_pose, frame.screen[:1]], axis=0)
        slot = np.zeros((20,), dtype=np.float32)
        time_code = np.array([math.sin(2.0 * math.pi * phase), math.cos(2.0 * math.pi * phase)], dtype=np.float32)
        audio = np.array([frame.screen[1], frame.screen[2], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return np.concatenate([au24, slot, time_code, audio], axis=0).astype(np.float32)

    def _tau_state(self) -> dict[str, float]:
        with torch.no_grad():
            taus = [cell.compute_tau(self.persona_idx)[0].detach().cpu().numpy() for cell in self.backbone.cells]
        means = [float(np.mean(t)) for t in taus] or [0.0]
        short_tau = min(means)
        long_tau = max(means)
        return {
            "short_tau_mean": short_tau,
            "long_tau_mean": long_tau,
            "short_tau_decay_remaining_s": short_tau * math.log(2.0),
        }

    def step(self, frame: InboundFrame) -> tuple[list[float], dict[str, float]]:
        ts = _parse_ts_sec(frame.ts)
        dt = self.target_dt if self.prev_ts is None else max(1e-3, min(1.0, ts - self.prev_ts))
        self.prev_ts = ts
        x = torch.from_numpy(self._build_event_tensor(frame)).reshape(1, 52)
        with torch.no_grad():
            fused = self.fusion(x)
            self.hidden = self.backbone.step(fused, self.hidden, self.persona_idx, dt=dt)
            h = self.hidden[-1][0].detach().cpu().numpy().astype(np.float32).tolist()
        self.step_idx += 1
        return h, self._tau_state()


class InferenceServer:
    """Daemon push -> realtime inference -> UI pull (broadcast)."""

    def __init__(self, runtime: RealtimeLNNRuntime, host: str = "0.0.0.0", port: int = 8765):
        self.runtime = runtime
        self.host = host
        self.port = port
        self.bound_port = port
        self._clients: set[WSConnection] = set()
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> None:
        self._server = await start_ws_server(self.host, self.port, self._on_client)
        sock = self._server.sockets[0] if self._server.sockets else None
        if sock is not None:
            self.bound_port = int(sock.getsockname()[1])

    async def serve_forever(self) -> None:
        await self.start()
        assert self._server is not None
        await self._server.serve_forever()

    async def close(self) -> None:
        for c in list(self._clients):
            await c.close()
        self._clients.clear()
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _on_client(self, conn: WSConnection) -> None:
        self._clients.add(conn)
        try:
            while True:
                raw = await conn.recv_text()
                if raw is None:
                    break
                await self._on_message(conn, raw)
        finally:
            self._clients.discard(conn)

    async def _on_message(self, conn: WSConnection, raw: str) -> None:
        t0 = time.perf_counter()
        try:
            frame = _validate_inbound(json.loads(raw))
        except Exception as exc:
            await conn.send_text(json.dumps({"error": str(exc)}, ensure_ascii=False))
            return
        hidden, tau_state = self.runtime.step(frame)
        out = {
            "ts": _utc_now_iso(),
            "frame_id": frame.frame_id,
            "state_hidden": hidden,
            "tau_state": tau_state,
            "intent_pred": None,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        }
        payload = json.dumps(out, ensure_ascii=False)
        for client in list(self._clients):
            await client.send_text(payload)
