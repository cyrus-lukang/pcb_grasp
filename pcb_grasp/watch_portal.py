"""Minimal rim watch-portal client (same source as capture_images.py).

Connects to ``host:port/ws``, opens ``camera_image``, keeps latest JPEG per cam_id.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import socket
import struct
import threading
import time
import uuid
from typing import Callable

LOG = logging.getLogger("pcb_grasp.portal")

_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_SIGNAL_ID = "camera_image"
DEFAULT_PORTAL = "10.3.8.7:8787"


def parse_endpoint(target: str) -> tuple[str, int]:
    raw = str(target or "").strip()
    if "://" in raw:
        raw = raw.split("://", 1)[1]
    host, _, port_s = raw.partition(":")
    if not host:
        raise ValueError(f"bad portal endpoint: {target!r}")
    return host, int(port_s or "8787")


def _header_value(header: bytes, name: str) -> str:
    needle = name.lower().encode() + b":"
    for line in header.split(b"\r\n"):
        if line.lower().startswith(needle):
            return line.split(b":", 1)[1].strip().decode()
    return ""


class WatchPortal:
    """WebSocket client for rim watch portal ``/ws``."""

    def __init__(self, host: str, port: int, timeout_s: float = 5.0) -> None:
        self._timeout_s = float(timeout_s)
        key = base64.b64encode(os.urandom(16)).decode()
        self._sock = socket.create_connection((host, port), timeout_s)
        request = (
            f"GET /ws HTTP/1.1\r\n"
            f"Host: {host}:{port}\r\n"
            "Upgrade: websocket\r\n"
            "Connection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\n"
            "Sec-WebSocket-Version: 13\r\n\r\n"
        )
        self._sock.sendall(request.encode())
        header = self._read_http_header()
        if b" 101 " not in header.split(b"\r\n", 1)[0]:
            status = header.split(b"\r\n", 1)[0].decode(errors="replace")
            raise ConnectionError(f"watch portal handshake failed: {status}")
        accept = _header_value(header, "sec-websocket-accept")
        expected = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()
        ).decode()
        if accept != expected:
            raise ConnectionError("watch portal websocket accept mismatch")
        self._pending = bytearray()

    def send(self, message: dict) -> None:
        self._send_frame(0x1, json.dumps(message, separators=(",", ":")).encode())

    def recv(self) -> dict | None:
        raw = self.recv_raw()
        if raw is None:
            return None
        return json.loads(raw)

    def recv_raw(self) -> bytes | None:
        parts: list[bytes] = []
        while True:
            fin, opcode, payload = self._read_frame()
            if opcode == 0x8:
                return None
            if opcode == 0x9:
                self._send_frame(0xA, payload)
                continue
            if opcode == 0xA:
                continue
            if opcode not in (0x0, 0x1):
                continue
            if opcode == 0x1:
                parts = [payload]
            else:
                parts.append(payload)
            if fin:
                return b"".join(parts)

    def close(self) -> None:
        try:
            self._send_frame(0x8, b"")
        except OSError:
            pass
        try:
            self._sock.close()
        except OSError:
            pass

    def _read_http_header(self) -> bytes:
        buffer = bytearray()
        while b"\r\n\r\n" not in buffer:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("watch portal closed during handshake")
            buffer += chunk
        header, rest = bytes(buffer).split(b"\r\n\r\n", 1)
        self._pending = bytearray(rest)
        return header

    def _read_exact(self, size: int) -> bytes:
        data = bytearray()
        while len(data) < size:
            if self._pending:
                take = min(size - len(data), len(self._pending))
                data += self._pending[:take]
                del self._pending[:take]
                continue
            self._sock.settimeout(self._timeout_s)
            chunk = self._sock.recv(max(4096, size - len(data)))
            if not chunk:
                raise ConnectionError("watch portal closed")
            data += chunk
        if len(data) > size:
            self._pending = data[size:] + self._pending
        return bytes(data[:size])

    def _read_frame(self) -> tuple[bool, int, bytes]:
        first, second = self._read_exact(2)
        fin = (first & 0x80) != 0
        opcode = first & 0x0F
        masked = (second & 0x80) != 0
        length = second & 0x7F
        if length == 126:
            length = struct.unpack("!H", self._read_exact(2))[0]
        elif length == 127:
            length = struct.unpack("!Q", self._read_exact(8))[0]
        mask = self._read_exact(4) if masked else None
        payload = self._read_exact(length)
        if mask is not None:
            payload = bytes(
                byte ^ mask[index % 4] for index, byte in enumerate(payload)
            )
        return fin, opcode, payload

    def _send_frame(self, opcode: int, payload: bytes) -> None:
        mask = os.urandom(4)
        header = bytearray([0x80 | opcode])
        length = len(payload)
        if length < 126:
            header.append(0x80 | length)
        elif length < 65536:
            header.append(0x80 | 126)
            header += struct.pack("!H", length)
        else:
            header.append(0x80 | 127)
            header += struct.pack("!Q", length)
        header += mask
        masked = bytes(byte ^ mask[index % 4] for index, byte in enumerate(payload))
        self._sock.sendall(bytes(header) + masked)


def _wait_opened(portal: WatchPortal, card_id: str) -> str:
    while True:
        message = portal.recv()
        if message is None:
            raise ConnectionError("watch portal closed before watch opened")
        if message.get("type") == "error":
            raise RuntimeError(f"{message.get('code')}: {message.get('message')}")
        if message.get("type") == "watch_opened" and message.get("card_id") == card_id:
            watch_id = str(message.get("watch_id") or "")
            if not watch_id:
                raise RuntimeError("watch_opened missing watch_id")
            return watch_id


class PortalFrameCache:
    """Background reader: latest JPEG bytes per cam_id."""

    def __init__(
        self,
        host: str,
        port: int,
        *,
        timeout_s: float = 5.0,
        max_hz: float = 15.0,
        wanted: set[int] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.timeout_s = float(timeout_s)
        self.max_hz = float(max_hz)
        self.wanted = None if wanted is None else set(int(x) for x in wanted)
        self._on_error = on_error
        self._lock = threading.Lock()
        self._jpegs: dict[int, bytes] = {}  # color JPEG
        self._depths: dict[int, bytes] = {}  # depth PNG/JPEG (format=2)
        self._seq = 0
        self._connected = False
        self._error = ""
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, name="portal-frames", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    def jpeg(self, cam_id: int) -> bytes | None:
        with self._lock:
            return self._jpegs.get(int(cam_id))

    def depth_bytes(self, cam_id: int) -> bytes | None:
        with self._lock:
            return self._depths.get(int(cam_id))

    def wait_pair(
        self, left_id: int, right_id: int, *, timeout_s: float = 5.0
    ) -> tuple[bytes, bytes]:
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._stop.is_set():
                break
            with self._lock:
                err = self._error
                left = self._jpegs.get(int(left_id))
                right = self._jpegs.get(int(right_id))
                ok = self._connected
            if err and not ok:
                raise RuntimeError(f"watch portal: {err}")
            if left and right:
                return left, right
            time.sleep(0.05)
        raise TimeoutError(
            f"portal timeout waiting cam {left_id}/{right_id} "
            f"from {self.host}:{self.port}"
        )

    def wait_color_depth(
        self, cam_id: int, *, timeout_s: float = 5.0, need_depth: bool = True
    ) -> tuple[bytes, bytes | None]:
        """等到指定 cam 的彩色（及可选深度）一帧。"""
        cid = int(cam_id)
        deadline = time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            if self._stop.is_set():
                break
            with self._lock:
                err = self._error
                color = self._jpegs.get(cid)
                depth = self._depths.get(cid)
                ok = self._connected
            if err and not ok:
                raise RuntimeError(f"watch portal: {err}")
            if color and (depth or not need_depth):
                return color, depth
            time.sleep(0.05)
        raise TimeoutError(
            f"portal timeout waiting cam {cid} color"
            f"{'+depth' if need_depth else ''} from {self.host}:{self.port}"
        )

    def _set_link(self, connected: bool, error: str = "") -> None:
        with self._lock:
            self._connected = connected
            self._error = error
        if error and self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                pass

    def _apply_sample(self, sample: dict) -> None:
        colors: dict[int, bytes] = {}
        depths: dict[int, bytes] = {}
        for image in sample.get("images") or []:
            cam_id = int(image.get("cam_id"))
            if self.wanted is not None and cam_id not in self.wanted:
                continue
            encoded = image.get("jpeg_b64") or ""
            if not encoded:
                continue
            raw = base64.b64decode(encoded)
            # format 2 = depth（PNG 16bit 常见）；其余当彩色 JPEG
            if int(image.get("format") or 0) == 2:
                depths[cam_id] = raw
            else:
                colors[cam_id] = raw
        if not colors and not depths:
            return
        with self._lock:
            if colors:
                self._jpegs.update(colors)
            if depths:
                self._depths.update(depths)
            self._seq = int(sample.get("seq") or 0)
            self._error = ""

    def _loop(self) -> None:
        while not self._stop.is_set():
            portal: WatchPortal | None = None
            watch_id = ""
            card_id = uuid.uuid4().hex
            try:
                self._set_link(False)
                portal = WatchPortal(self.host, self.port, self.timeout_s)
                portal.send(
                    {
                        "type": "open_watch",
                        "signal_id": _SIGNAL_ID,
                        "card_id": card_id,
                        "max_hz": self.max_hz,
                    }
                )
                watch_id = _wait_opened(portal, card_id)
                self._set_link(True)
                LOG.info(
                    "portal watch open %s:%d id=%s cams=%s",
                    self.host,
                    self.port,
                    watch_id,
                    sorted(self.wanted) if self.wanted else "all",
                )
                while not self._stop.is_set():
                    # 大 JPEG 帧可能数 MB；中途 Timeout 会失步，故不吞掉，外层重连
                    raw = portal.recv_raw()
                    if raw is None:
                        raise ConnectionError("watch portal closed")
                    message = json.loads(raw)
                    if message.get("type") == "error":
                        raise RuntimeError(
                            f"{message.get('code')}: {message.get('message')}"
                        )
                    if message.get("type") == "sample":
                        self._apply_sample(message.get("sample") or {})
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._set_link(False, str(exc))
                LOG.warning("portal error: %s", exc)
                self._stop.wait(1.0)
            finally:
                if portal is not None:
                    if watch_id:
                        try:
                            portal.send(
                                {
                                    "type": "close_watch",
                                    "watch_id": watch_id,
                                    "card_id": card_id,
                                }
                            )
                        except OSError:
                            pass
                    portal.close()
