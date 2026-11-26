#!/usr/bin/env python3
"""SDK 进程：frame_bundle 取流，与 capture_camera_subscribe 同一路。

用 ``sdk.frame_bundle.get_latest_frame()``，彩色 **原样 JPEG 直通**，
深度 **原样 Z16 直通**（不再 CameraSource 解 JPEG→RGB 再压一遍）。

协议（小端）::
  magic = b"FB02"
  循环:
    u32 seq
    u32 n_parts
    每个 part:
      u32 cam_id
      u8  kind   # 0=JPEG color, 2=raw Z16 depth
      u16 width
      u16 height
      u32 nbytes
      bytes payload

由 ``bundle_cache.BundleFrameCache`` 用 perception/.venv-sdk 拉起。
"""

from __future__ import annotations

import argparse
import asyncio
import os
import struct
import sys
import time
import traceback

from rim_algorithm_sdk import EndpointConfig, SDK, SDKConfig
from robot.camera.v1 import camera_pb2

_JPEG = int(camera_pb2.IMAGE_FORMAT_JPEG)
_Z16 = int(camera_pb2.IMAGE_FORMAT_Z16)
_KIND_JPEG = 0
_KIND_Z16 = 2


def _write_frame(
    out, seq: int, parts: list[tuple[int, int, int, int, bytes]]
) -> None:
    out.write(struct.pack("<II", int(seq), len(parts)))
    for cam_id, kind, width, height, payload in parts:
        out.write(
            struct.pack(
                "<IBHHI",
                int(cam_id),
                int(kind),
                int(width),
                int(height),
                len(payload),
            )
        )
        out.write(payload)
    out.flush()


async def _pull(
    *,
    target: str,
    color_ids: set[int],
    depth_ids: set[int],
    hz: float,
    connect_timeout: float,
    out,
) -> int:
    period = 1.0 / max(float(hz), 0.5)
    wanted = set(color_ids) | set(depth_ids)
    seq = 0
    out.write(b"FB02")
    out.flush()

    while True:
        sdk = SDK(
            SDKConfig(
                frame_bundle=EndpointConfig(target=target),
                node_name="pcb_grasp_bundle_worker",
                default_timeout=max(2.0, connect_timeout),
                connect_timeout=float(connect_timeout),
            )
        )
        try:
            await sdk.connect()
            print(
                f"bundle_worker connected target={target} "
                f"color={sorted(color_ids)} depth={sorted(depth_ids)} hz={hz}",
                file=sys.stderr,
                flush=True,
            )
            while True:
                t0 = time.monotonic()
                response = await sdk.frame_bundle.get_latest_frame(
                    timeout=max(2.0, connect_timeout)
                )
                packet = response.frame.camera_image
                parts: list[tuple[int, int, int, int, bytes]] = []
                for image in packet.images:
                    cam_id = int(image.cam_id)
                    if wanted and cam_id not in wanted:
                        continue
                    payload = bytes(image.payload)
                    if not payload:
                        continue
                    fmt = int(image.format)
                    width = int(image.width)
                    height = int(image.height)
                    if fmt == _JPEG and cam_id in color_ids:
                        parts.append(
                            (cam_id, _KIND_JPEG, width, height, payload)
                        )
                    elif fmt == _Z16 and cam_id in depth_ids:
                        expect = width * height * 2
                        if width <= 0 or height <= 0 or len(payload) != expect:
                            print(
                                f"bundle_worker Z16 cam{cam_id} "
                                f"len={len(payload)} != {width}x{height}",
                                file=sys.stderr,
                                flush=True,
                            )
                            continue
                        parts.append(
                            (cam_id, _KIND_Z16, width, height, payload)
                        )
                if parts:
                    seq += 1
                    try:
                        _write_frame(out, seq, parts)
                    except BrokenPipeError:
                        return 0
                delay = period - (time.monotonic() - t0)
                if delay > 0:
                    await asyncio.sleep(delay)
        except BrokenPipeError:
            return 0
        except Exception as exc:
            print(f"bundle_worker: {exc}", file=sys.stderr, flush=True)
            await asyncio.sleep(0.5)
        finally:
            try:
                await sdk.close()
            except Exception:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="10.3.8.7:50066")
    parser.add_argument("--color", default="1,3,6,7")
    parser.add_argument("--depth", default="")
    parser.add_argument("--hz", type=float, default=15.0)
    parser.add_argument("--connect-timeout", type=float, default=10.0)
    # 兼容旧参数（已不再二次压 JPEG）
    parser.add_argument("--jpeg-quality", type=int, default=80)
    args = parser.parse_args()
    del args.jpeg_quality

    color_ids = {int(x) for x in str(args.color).split(",") if x.strip() != ""}
    depth_ids = {int(x) for x in str(args.depth).split(",") if x.strip() != ""}
    if not color_ids and not depth_ids:
        print("bundle_worker: no cameras", file=sys.stderr)
        return 2

    try:
        return asyncio.run(
            _pull(
                target=str(args.target),
                color_ids=color_ids,
                depth_ids=depth_ids,
                hz=float(args.hz),
                connect_timeout=float(args.connect_timeout),
                out=sys.stdout.buffer,
            )
        )
    except KeyboardInterrupt:
        return 0
    except Exception:
        traceback.print_exc(file=sys.stderr)
        return 1


if __name__ == "__main__":
    os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
    raise SystemExit(main())
