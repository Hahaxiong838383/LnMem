"""Minimal asyncio WebSocket protocol helpers (RFC6455 subset)."""
from __future__ import annotations

import asyncio
import base64
import hashlib
from dataclasses import dataclass
from typing import Awaitable, Callable


_MAGIC = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class WebSocketProtocolError(RuntimeError):
    """Raised when websocket framing or handshake is invalid."""


@dataclass(eq=False)
class WSConnection:
    """A lightweight text websocket connection wrapper."""

    reader: asyncio.StreamReader
    writer: asyncio.StreamWriter

    async def recv_text(self) -> str | None:
        """Receive next text frame. Return None when close frame received."""
        b1, b2 = await self._read_exact(2)
        opcode = b1 & 0x0F
        masked = (b2 & 0x80) != 0
        length = b2 & 0x7F
        if length == 126:
            ext = await self._read_exact(2)
            length = int.from_bytes(ext, "big")
        elif length == 127:
            ext = await self._read_exact(8)
            length = int.from_bytes(ext, "big")
        if opcode == 0x8:
            return None
        if opcode == 0x9:
            payload = await self._read_payload(length, masked)
            await self._send_frame(payload, opcode=0xA)
            return await self.recv_text()
        if opcode != 0x1:
            raise WebSocketProtocolError(f"unsupported opcode={opcode}")
        payload = await self._read_payload(length, masked)
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WebSocketProtocolError("invalid utf-8 text frame") from exc

    async def send_text(self, text: str) -> None:
        """Send a text frame to peer."""
        await self._send_frame(text.encode("utf-8"), opcode=0x1)

    async def close(self) -> None:
        """Best-effort close frame then close transport."""
        try:
            await self._send_frame(b"", opcode=0x8)
        except Exception:
            pass
        self.writer.close()
        await self.writer.wait_closed()

    async def _read_payload(self, length: int, masked: bool) -> bytes:
        mask_key = await self._read_exact(4) if masked else b""
        payload = await self._read_exact(length) if length > 0 else b""
        if masked:
            payload = bytes(b ^ mask_key[i % 4] for i, b in enumerate(payload))
        return payload

    async def _send_frame(self, payload: bytes, opcode: int) -> None:
        header = bytearray()
        header.append(0x80 | (opcode & 0x0F))
        n = len(payload)
        if n < 126:
            header.append(n)
        elif n <= 0xFFFF:
            header.append(126)
            header.extend(n.to_bytes(2, "big"))
        else:
            header.append(127)
            header.extend(n.to_bytes(8, "big"))
        self.writer.write(bytes(header) + payload)
        await self.writer.drain()

    async def _read_exact(self, n: int) -> bytes:
        try:
            return await self.reader.readexactly(n)
        except asyncio.IncompleteReadError as exc:
            raise WebSocketProtocolError("connection closed while reading frame") from exc


def _parse_headers(raw: str) -> tuple[str, dict[str, str]]:
    lines = raw.split("\r\n")
    if not lines:
        raise WebSocketProtocolError("empty request")
    request_line = lines[0]
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        headers[k.strip().lower()] = v.strip()
    return request_line, headers


async def _accept_websocket(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> WSConnection:
    raw = await reader.readuntil(b"\r\n\r\n")
    req_line, headers = _parse_headers(raw.decode("utf-8", errors="ignore"))
    if not req_line.startswith("GET "):
        raise WebSocketProtocolError("only GET upgrade is supported")
    if headers.get("upgrade", "").lower() != "websocket":
        raise WebSocketProtocolError("missing Upgrade: websocket")
    if "upgrade" not in headers.get("connection", "").lower():
        raise WebSocketProtocolError("missing Connection: Upgrade")
    key = headers.get("sec-websocket-key")
    if not key:
        raise WebSocketProtocolError("missing Sec-WebSocket-Key")
    accept = base64.b64encode(hashlib.sha1((key + _MAGIC).encode("utf-8")).digest()).decode("ascii")
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    )
    writer.write(resp.encode("utf-8"))
    await writer.drain()
    return WSConnection(reader=reader, writer=writer)


async def start_ws_server(
    host: str,
    port: int,
    handler: Callable[[WSConnection], Awaitable[None]],
) -> asyncio.AbstractServer:
    """Start websocket server and dispatch accepted WSConnection to handler."""

    async def _on_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        conn: WSConnection | None = None
        try:
            conn = await _accept_websocket(reader, writer)
            await handler(conn)
        except Exception:
            pass
        finally:
            if conn is not None:
                await conn.close()
            elif not writer.is_closing():
                writer.close()
                await writer.wait_closed()

    return await asyncio.start_server(_on_client, host=host, port=port)
