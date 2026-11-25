#!/usr/bin/env python3
"""SDK 进程：从 frame_bundle 拉最新帧，按长度前缀协议写到 stdout。

协议（小端）::
  magic = b"FB01"
  循环:
    u32 seq
    u32 n_parts
    每个 part:
      u32 cam_id
      u8  kind   # 0=JPEG color, 1=PNG16 depth
      u32 nbytes
      bytes payload

由 ``bundle_cache.BundleFrameCache`` 在 pcb_grasp/.venv 里拉起
（本文件必须用 perception/.venv-sdk 的 python 跑）。
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
import traceback


def _encode_jpeg_bgr(rgb, quality: int = 85) -> bytes:
    import cv2
    import numpy as np

    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _encode_depth_png(depth_mm) -> bytes:
    import cv2
    import numpy as np

    arr = np.asarray(depth_mm)
    if arr.dtype != np.uint16:
        arr = arr.astype(np.uint16)
    ok, buf = cv2.imencode(".png", arr)
    if not ok:
        raise RuntimeError("depth png encode failed")
    return buf.tobytes()


def _write_frame(out, seq: int, parts: list[tuple[int, int, bytes]]) -> None:
    out.write(struct.pack("<II", int(seq), len(parts)))
    for cam_id, kind, payload in parts:
        out.write(struct.pack("<IBI", int(cam_id), int(kind), len(payload)))
        out.write(payload)
    out.flush()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="10.3.8.7:50066")
    parser.add_argument(
        "--color",
        default="1,3,6,7",
        help="comma-separated color cam_ids",
    )
    parser.add_argument(
        "--depth",
        default="",
        help="comma-separated depth cam_ids (Z16); empty = none",
    )
    parser.add_argument("--hz", type=float, default=8.0)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    args = parser.parse_args()

    color_ids = [int(x) for x in str(args.color).split(",") if x.strip() != ""]
    depth_ids = [int(x) for x in str(args.depth).split(",") if x.strip() != ""]
    if not color_ids and not depth_ids:
        print("bundle_worker: no cameras", file=sys.stderr)
        return 2

    # pipelines 必须在 path 上（run_ui / BundleFrameCache 会设 PYTHONPATH）
    from rim_python.camera import CameraSource

    cameras = {f"c{cid}": cid for cid in color_ids} if color_ids else None
    depth_cameras = {f"d{cid}": cid for cid in depth_ids} if depth_ids else None
    # CameraSource 不允许空 dict；无 depth 时传 None
    if depth_cameras is not None and not depth_cameras:
        depth_cameras = None

    def _need_reconnect(exc: BaseException) -> bool:
        msg = str(exc).lower()
        keys = (
            "socket closed",
            "connection refused",
            "connection reset",
            "unavailable",
            "not connected",
            "statuscode.unavailable",
            "broken pipe",
            "eof",
            "timed out",
            "timeout",
        )
        return any(k in msg for k in keys)

    def _make_src() -> CameraSource:
        return CameraSource(
            str(args.target),
            cameras=cameras,
            depth_cameras=depth_cameras,
            timeout_s=max(2.0, period * 4),
            connect_timeout_s=float(args.connect_timeout),
            node_name="pcb_grasp_bundle_worker",
        )

    period = 1.0 / max(float(args.hz), 0.5)
    out = sys.stdout.buffer
    out.write(b"FB01")
    out.flush()

    src: CameraSource | None = None
    seq = 0
    fail_streak = 0
    last_err = ""
    last_err_log_t = 0.0
    backoff = 0.2
    try:
        while True:
            t0 = time.monotonic()
            if src is None:
                try:
                    src = _make_src()
                    src.connect()
                    fail_streak = 0
                    backoff = 0.2
                    print(
                        f"bundle_worker connected target={args.target} "
                        f"color={color_ids} depth={depth_ids}",
                        file=sys.stderr,
                        flush=True,
                    )
                except Exception as exc:
                    fail_streak += 1
                    msg = str(exc)
                    now = time.monotonic()
                    if msg != last_err or now - last_err_log_t >= 2.0:
                        print(
                            f"bundle_worker connect: {exc} (retry {fail_streak})",
                            file=sys.stderr,
                            flush=True,
                        )
                        last_err, last_err_log_t = msg, now
                    try:
                        if src is not None:
                            src.close()
                    except Exception:
                        pass
                    src = None
                    time.sleep(min(backoff, 5.0))
                    backoff = min(backoff * 1.5, 5.0)
                    continue
            try:
                snap = src.get()
            except Exception as exc:
                fail_streak += 1
                msg = str(exc)
                now = time.monotonic()
                if msg != last_err or now - last_err_log_t >= 2.0:
                    print(
                        f"bundle_worker get: {exc} (streak={fail_streak})",
                        file=sys.stderr,
                        flush=True,
                    )
                    last_err, last_err_log_t = msg, now
                if _need_reconnect(exc) or fail_streak >= 3:
                    try:
                        src.close()
                    except Exception:
                        pass
                    src = None
                    time.sleep(min(backoff, 5.0))
                    backoff = min(backoff * 1.5, 5.0)
                else:
                    time.sleep(0.2)
                continue
            fail_streak = 0
            backoff = 0.2
            last_err = ""
            parts: list[tuple[int, int, bytes]] = []
            for name, image in snap.images.items():
                try:
                    payload = _encode_jpeg_bgr(image.image, quality=args.jpeg_quality)
                except Exception as exc:
                    print(f"bundle_worker jpeg {name}: {exc}", file=sys.stderr, flush=True)
                    continue
                parts.append((int(image.cam_id), 0, payload))
            for name, depth in snap.depths.items():
                try:
                    payload = _encode_depth_png(depth.depth_mm)
                except Exception as exc:
                    print(f"bundle_worker depth {name}: {exc}", file=sys.stderr, flush=True)
                    continue
                parts.append((int(depth.cam_id), 1, payload))
            if parts:
                seq += 1
                _write_frame(out, seq, parts)
            elapsed = time.monotonic() - t0
            time.sleep(max(0.0, period - elapsed))
    except BrokenPipeError:
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1
    finally:
        if src is not None:
            try:
                src.close()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    # 避免 SDK/日志污染 stdout 二进制流
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    raise SystemExit(main())
