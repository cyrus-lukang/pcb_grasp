"""frame_bundle 取流缓存（替代已不推 sample 的 rim watch portal WS）。

门户 ``10.3.8.7:8787`` 目前能 ``open_watch`` / ``watch_stats``，但所有信号
都收不到 ``type=sample``。改走已验证可用的 ``rim_frame_bundle``（默认
``10.3.8.7:50066``），用 perception SDK 子进程拉 JPEG/深度。

对外 API 对齐 ``PortalFrameCache``，方便头/腕相机无缝切换。
深度 cam_id：左臂彩色 1 对应 Z16 0，右臂彩色 3 对应 Z16 2；缓存里会把深度
**回写到彩色 cam_id**，以便 ``WristPortalCam.wait_color_depth(1)`` 仍可用。
"""

from __future__ import annotations

import logging
import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable

LOG = logging.getLogger("pcb_grasp.bundle")

_PIPELINES = Path("/home/ubuntu/workspace/perception/projects/yuantu/pipelines")
_SDK_PY = Path("/home/ubuntu/workspace/perception/.venv-sdk/bin/python")
_WORKER = Path(__file__).resolve().parent / "bundle_worker.py"
DEFAULT_BUNDLE = "10.3.8.7:50066"

# 彩色 → 深度（frame_bundle 实际编号）
_DEFAULT_DEPTH_OF_COLOR = {1: 0, 3: 2}


def _sdk_python() -> str:
    if _SDK_PY.is_file():
        return str(_SDK_PY)
    return sys.executable


def _env_with_pipelines() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PIPELINES) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    env.setdefault("GRPC_VERBOSITY", "ERROR")
    return env


def depth_id_for_color(color_id: int, override: dict[int, int] | None = None) -> int | None:
    table = dict(_DEFAULT_DEPTH_OF_COLOR)
    if override:
        table.update({int(k): int(v) for k, v in override.items()})
    return table.get(int(color_id))


class BundleFrameCache:
    """后台读 frame_bundle：最新 JPEG / depth-PNG（按彩色 cam_id 索引）。"""

    def __init__(
        self,
        target: str = DEFAULT_BUNDLE,
        *,
        timeout_s: float = 5.0,
        max_hz: float = 8.0,
        wanted: set[int] | None = None,
        depth_of_color: dict[int, int] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None:
        self.target = str(target or DEFAULT_BUNDLE).strip()
        self.timeout_s = float(timeout_s)
        self.max_hz = float(max_hz)
        self.wanted = None if wanted is None else set(int(x) for x in wanted)
        self.depth_of_color = {
            int(k): int(v) for k, v in (depth_of_color or _DEFAULT_DEPTH_OF_COLOR).items()
        }
        self._on_error = on_error
        self._lock = threading.Lock()
        self._jpegs: dict[int, bytes] = {}
        self._depths: dict[int, bytes] = {}  # keyed by COLOR cam_id
        self._seq = 0
        self._connected = False
        self._error = ""
        self._stop = threading.Event()
        self._proc: subprocess.Popen | None = None
        self._thread = threading.Thread(
            target=self._loop, name="bundle-frames", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
            except OSError:
                pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                try:
                    proc.kill()
                except OSError:
                    pass
        self._thread.join(timeout=3.0)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def error(self) -> str:
        with self._lock:
            return self._error

    @property
    def seq(self) -> int:
        with self._lock:
            return int(self._seq)

    def jpeg(self, cam_id: int) -> bytes | None:
        with self._lock:
            return self._jpegs.get(int(cam_id))

    def depth_bytes(self, cam_id: int) -> bytes | None:
        with self._lock:
            return self._depths.get(int(cam_id))

    def bounce_worker(self) -> None:
        """杀掉当前 worker，让 ``_loop`` 自动拉起（帧卡住时用）。"""
        proc = self._proc
        if proc is None or proc.poll() is not None:
            return
        LOG.warning("bundle worker bounce (stale frames) target=%s", self.target)
        try:
            proc.terminate()
        except OSError:
            pass

    def wait_pair(
        self,
        left_id: int,
        right_id: int,
        *,
        timeout_s: float = 5.0,
        min_seq: int | None = None,
    ) -> tuple[bytes, bytes]:
        deadline = time.monotonic() + float(timeout_s)
        stale: tuple[bytes, bytes] | None = None
        while time.monotonic() < deadline:
            if self._stop.is_set():
                break
            with self._lock:
                err = self._error
                left = self._jpegs.get(int(left_id))
                right = self._jpegs.get(int(right_id))
                ok = self._connected
                seq = int(self._seq)
            if err and not ok:
                raise RuntimeError(f"frame_bundle: {err}")
            if left and right:
                stale = (left, right)
                if min_seq is None or seq > int(min_seq):
                    return left, right
            time.sleep(0.05)
        if stale is not None and min_seq is not None:
            # 超时仍无新帧：退回旧帧，由调用方决定是否 bounce
            return stale
        raise TimeoutError(
            f"frame_bundle timeout waiting cam {left_id}/{right_id} "
            f"from {self.target}"
        )

    def wait_color_depth(
        self, cam_id: int, *, timeout_s: float = 5.0, need_depth: bool = True
    ) -> tuple[bytes, bytes | None]:
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
                raise RuntimeError(f"frame_bundle: {err}")
            if color and (depth or not need_depth):
                return color, depth
            time.sleep(0.05)
        raise TimeoutError(
            f"frame_bundle timeout waiting cam {cid} color"
            f"{'+depth' if need_depth else ''} from {self.target}"
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

    def _color_ids(self) -> list[int]:
        if self.wanted:
            return sorted(self.wanted)
        return [1, 3, 6, 7]

    def _depth_ids(self) -> list[int]:
        ids: list[int] = []
        for cid in self._color_ids():
            did = self.depth_of_color.get(int(cid))
            if did is not None:
                ids.append(int(did))
        return sorted(set(ids))

    def _spawn(self) -> subprocess.Popen:
        if not _WORKER.is_file():
            raise FileNotFoundError(f"bundle worker missing: {_WORKER}")
        color = ",".join(str(x) for x in self._color_ids())
        depth = ",".join(str(x) for x in self._depth_ids())
        cmd = [
            _sdk_python(),
            "-u",
            str(_WORKER),
            "--target",
            self.target,
            "--color",
            color,
            "--depth",
            depth,
            "--hz",
            str(max(1.0, min(float(self.max_hz), 15.0))),
            "--connect-timeout",
            str(max(3.0, float(self.timeout_s))),
        ]
        LOG.info("bundle worker start %s", " ".join(cmd))
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_env_with_pipelines(),
            bufsize=0,
        )

    def _read_exact(self, stream, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            if self._stop.is_set():
                raise ConnectionError("bundle cache stopped")
            chunk = stream.read(n - len(buf))
            if not chunk:
                raise ConnectionError("bundle worker stdout closed")
            buf += chunk
        return bytes(buf)

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        def _run() -> None:
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, b""):
                if self._stop.is_set():
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    LOG.info("bundle_worker: %s", text)

        threading.Thread(target=_run, name="bundle-stderr", daemon=True).start()

    def _consume(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        magic = self._read_exact(proc.stdout, 4)
        if magic != b"FB01":
            raise RuntimeError(f"bad bundle magic {magic!r}")
        self._set_link(True)
        # depth_id → color_id
        depth_to_color = {int(v): int(k) for k, v in self.depth_of_color.items()}
        while not self._stop.is_set():
            hdr = self._read_exact(proc.stdout, 8)
            seq, n_parts = struct.unpack("<II", hdr)
            colors: dict[int, bytes] = {}
            depths: dict[int, bytes] = {}
            for _ in range(int(n_parts)):
                meta = self._read_exact(proc.stdout, 9)
                cam_id, kind, nbytes = struct.unpack("<IBI", meta)
                payload = self._read_exact(proc.stdout, int(nbytes))
                if kind == 0:
                    if self.wanted is None or int(cam_id) in self.wanted:
                        colors[int(cam_id)] = payload
                elif kind == 1:
                    color_id = depth_to_color.get(int(cam_id), int(cam_id))
                    if self.wanted is None or color_id in self.wanted:
                        depths[color_id] = payload
            if not colors and not depths:
                continue
            with self._lock:
                if colors:
                    self._jpegs.update(colors)
                if depths:
                    self._depths.update(depths)
                self._seq = int(seq)
                self._error = ""
                self._connected = True

    def _loop(self) -> None:
        while not self._stop.is_set():
            proc: subprocess.Popen | None = None
            try:
                self._set_link(False)
                proc = self._spawn()
                self._proc = proc
                self._drain_stderr(proc)
                self._consume(proc)
            except Exception as exc:
                if self._stop.is_set():
                    break
                self._set_link(False, str(exc))
                LOG.warning("bundle error: %s", exc)
                self._stop.wait(1.0)
            finally:
                if proc is not None:
                    try:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=1.5)
                            except Exception:
                                proc.kill()
                    except Exception:
                        pass
                if self._proc is proc:
                    self._proc = None
