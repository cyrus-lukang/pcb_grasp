"""Realtime cameras: head dual fisheye (S36/CCG3) + wrist Gemini305（门户）/ D405.

Wire capture into ``FrameSource`` / ``run_live.py``. Offline file mode stays in
``run_grasp.py``.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

LOG = logging.getLogger("pcb_grasp.cameras")

DEFAULT_S36_SDK = Path("/home/ubuntu/workspace/ccg3_8m_sdk/ccg3_camera_python")


@dataclass
class FrameBundle:
    head_fisheye_bgr: np.ndarray
    wrist_bgr: np.ndarray
    wrist_depth: np.ndarray
    head_right_bgr: np.ndarray | None = None
    head_depth: np.ndarray | None = None
    stamp_s: float = 0.0


class FrameSource:
    """Thread-safe latest-frame buffer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._bundle: FrameBundle | None = None

    def update(
        self,
        head_fisheye_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        wrist_depth: np.ndarray,
        head_depth: np.ndarray | None = None,
        head_right_bgr: np.ndarray | None = None,
    ) -> None:
        with self._lock:
            self._bundle = FrameBundle(
                head_fisheye_bgr=head_fisheye_bgr.copy(),
                wrist_bgr=wrist_bgr.copy(),
                wrist_depth=wrist_depth.copy(),
                head_right_bgr=None if head_right_bgr is None else head_right_bgr.copy(),
                head_depth=None if head_depth is None else head_depth.copy(),
                stamp_s=time.time(),
            )

    def latest(self, *, max_age_s: float | None = 1.0) -> FrameBundle:
        with self._lock:
            if self._bundle is None:
                raise RuntimeError("No frames yet")
            b = self._bundle
            if max_age_s is not None and (time.time() - b.stamp_s) > max_age_s:
                raise RuntimeError(
                    f"Frame too old: {time.time() - b.stamp_s:.2f}s > {max_age_s}s"
                )
            return FrameBundle(
                head_fisheye_bgr=b.head_fisheye_bgr.copy(),
                wrist_bgr=b.wrist_bgr.copy(),
                wrist_depth=b.wrist_depth.copy(),
                head_right_bgr=None if b.head_right_bgr is None else b.head_right_bgr.copy(),
                head_depth=None if b.head_depth is None else b.head_depth.copy(),
                stamp_s=b.stamp_s,
            )


def run_once_from_source(pipeline, source: FrameSource, *, max_age_s: float = 1.0, **run_kwargs):
    """Pull latest synced frames and run ``GraspPipeline.run`` (offline-style)."""
    b = source.latest(max_age_s=max_age_s)
    return pipeline.run(
        b.head_fisheye_bgr,
        b.wrist_bgr,
        b.wrist_depth,
        head_right_bgr=b.head_right_bgr,
        head_depth=b.head_depth,
        **run_kwargs,
    )


PollFn = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]]


def start_poll_thread(source: FrameSource, poll: PollFn, hz: float = 15.0) -> threading.Thread:
    period = 1.0 / max(hz, 1e-3)
    stop = threading.Event()

    def _loop() -> None:
        while not stop.is_set():
            try:
                head, wrist, depth, hdepth = poll()
                source.update(head, wrist, depth, hdepth)
            except Exception:
                pass
            stop.wait(period)

    t = threading.Thread(target=_loop, name="pcb_grasp_cam", daemon=True)
    t.stop_event = stop  # type: ignore[attr-defined]
    t.start()
    return t


# --------------------------------------------------------------------------- live HW


def _import_s36_sdk(sdk_dir: Path) -> tuple[Any, Any]:
    if not sdk_dir.is_dir():
        raise FileNotFoundError(f"S36 SDK 不存在: {sdk_dir}")
    if str(sdk_dir) not in sys.path:
        sys.path.insert(0, str(sdk_dir))
    from camera import Camera  # type: ignore
    from processor import FrameProcessor  # type: ignore

    return Camera, FrameProcessor


class HeadFisheyeCam:
    """S36 dual fisheye via CCG3 SDK（原图，不做去畸变）."""

    def __init__(
        self,
        left_dev: str = "/dev/video6",
        right_dev: str = "/dev/video7",
        *,
        width: int = 1920,
        height: int = 1536,
        sdk_dir: str | Path = DEFAULT_S36_SDK,
    ) -> None:
        Camera, FrameProcessor = _import_s36_sdk(Path(sdk_dir))
        self._proc0 = FrameProcessor(calib_file=None)
        self._proc1 = FrameProcessor(calib_file=None)
        self._cam0 = Camera(left_dev, width=width, height=height)
        self._cam1 = Camera(right_dev, width=width, height=height)
        self._cam0.__enter__()
        self._cam1.__enter__()
        self.left_dev = left_dev
        self.right_dev = right_dev
        LOG.info("Head fisheye open left=%s right=%s %dx%d", left_dev, right_dev, width, height)

    def grab(self, *, timeout_ms: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        f0 = self._cam0.wait_frame(timeout_ms=timeout_ms)
        f1 = self._cam1.wait_frame(timeout_ms=timeout_ms)
        b0 = self._proc0.process(f0, target="bgr", undistort=False)
        b1 = self._proc1.process(f1, target="bgr", undistort=False)
        return b0, b1

    def close(self) -> None:
        for cam in (self._cam0, self._cam1):
            try:
                cam.__exit__(None, None, None)
            except Exception:
                pass


class HeadPortalCam:
    """头双目：从 rim frame_bundle 取流（cam6/7）。

    历史名 portal：8787 watch WS 已不再推 ``sample``，实际走 ``rim_frame_bundle``
    （默认 10.3.8.7:50066）。可传入共享 ``cache``，与腕摄共用一条连接。
    """

    def __init__(
        self,
        portal: str = "10.3.8.7:50066",
        *,
        left_cam_id: int = 7,
        right_cam_id: int = 6,
        max_hz: float = 8.0,
        timeout_s: float = 60.0,
        cache=None,
        own_cache: bool = True,
        wait_first: bool = True,
    ) -> None:
        from .bundle_cache import BundleFrameCache, DEFAULT_BUNDLE

        # 兼容旧配置里的 8787 portal 地址 → 改用 frame_bundle
        target = str(portal or DEFAULT_BUNDLE).strip()
        if ":8787" in target or target.endswith("8787"):
            target = DEFAULT_BUNDLE
        self.portal = target
        self.left_cam_id = int(left_cam_id)
        self.right_cam_id = int(right_cam_id)
        self._own_cache = bool(own_cache)
        if cache is not None:
            self._cache = cache
            self._own_cache = False
        else:
            self._cache = BundleFrameCache(
                target,
                timeout_s=timeout_s,
                max_hz=max_hz,
                wanted={self.left_cam_id, self.right_cam_id},
            )
            self._cache.start()
        try:
            if wait_first:
                self._cache.wait_pair(
                    self.left_cam_id,
                    self.right_cam_id,
                    timeout_s=max(float(timeout_s), 30.0),
                )
        except Exception:
            self.close()
            raise
        self._last_seq = -1
        self._stale_grabs = 0
        LOG.info(
            "Head bundle open %s cams=%d/%d own_cache=%s",
            self.portal,
            self.left_cam_id,
            self.right_cam_id,
            self._own_cache,
        )

    def grab(self, *, timeout_ms: int = 3000) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        timeout_s = max(0.5, float(timeout_ms) / 1000.0)
        min_seq = self._last_seq if self._last_seq >= 0 else None
        left_j, right_j = self._cache.wait_pair(
            self.left_cam_id,
            self.right_cam_id,
            timeout_s=timeout_s,
            min_seq=min_seq,
        )
        seq = int(getattr(self._cache, "seq", 0) or 0)
        if min_seq is not None and seq <= min_seq:
            self._stale_grabs += 1
            if self._stale_grabs >= 3:
                LOG.warning(
                    "head bundle frames stale seq=%s ×%d → bounce worker",
                    seq,
                    self._stale_grabs,
                )
                try:
                    self._cache.bounce_worker()
                except Exception as exc:
                    LOG.warning("bundle bounce failed: %s", exc)
                self._stale_grabs = 0
                self._last_seq = -1
                left_j, right_j = self._cache.wait_pair(
                    self.left_cam_id,
                    self.right_cam_id,
                    timeout_s=max(timeout_s, 8.0),
                    min_seq=seq,
                )
                seq = int(getattr(self._cache, "seq", 0) or 0)
        else:
            self._stale_grabs = 0
        self._last_seq = seq
        left = cv2.imdecode(np.frombuffer(left_j, dtype=np.uint8), cv2.IMREAD_COLOR)
        right = cv2.imdecode(np.frombuffer(right_j, dtype=np.uint8), cv2.IMREAD_COLOR)
        if left is None or right is None:
            raise RuntimeError("portal JPEG decode failed")
        return left, right

    def close(self) -> None:
        if not self._own_cache:
            return
        try:
            self._cache.close()
        except Exception:
            pass


def open_head_camera(cfg) -> HeadFisheyeCam | HeadPortalCam:
    """按配置打开头摄：``cameras.head_source`` = portal/bundle | v4l2。"""
    source = str(getattr(cfg, "cam_head_source", "portal") or "portal").strip().lower()
    if source in ("portal", "watch", "watch_portal", "rim", "bundle", "frame_bundle"):
        return HeadPortalCam(
            getattr(cfg, "rim_frame_bundle", None)
            or getattr(cfg, "cam_portal", "10.3.8.7:50066"),
            left_cam_id=int(getattr(cfg, "cam_portal_head_left_id", 7)),
            right_cam_id=int(getattr(cfg, "cam_portal_head_right_id", 6)),
            max_hz=float(getattr(cfg, "cam_portal_max_hz", 15.0)),
            timeout_s=float(getattr(cfg, "cam_portal_timeout_s", 5.0)),
        )
    return HeadFisheyeCam(
        getattr(cfg, "cam_head_left", "/dev/video6"),
        getattr(cfg, "cam_head_right", "/dev/video7"),
        width=int(getattr(cfg, "cam_head_width", 1920)),
        height=int(getattr(cfg, "cam_head_height", 1536)),
        sdk_dir=getattr(cfg, "cam_s36_sdk", DEFAULT_S36_SDK),
    )


def open_portal_head_and_wrist(cfg) -> tuple[HeadPortalCam, "WristPortalCam"]:
    """头+腕共用一条 frame_bundle 连接（头左=7 / 右=6，左腕彩色 1 / 深度 0）。"""
    from .bundle_cache import BundleFrameCache, DEFAULT_BUNDLE

    target = str(
        getattr(cfg, "rim_frame_bundle", None)
        or getattr(cfg, "cam_portal", DEFAULT_BUNDLE)
        or DEFAULT_BUNDLE
    ).strip()
    if ":8787" in target or target.endswith("8787"):
        target = DEFAULT_BUNDLE
    left_id = int(getattr(cfg, "cam_portal_head_left_id", 7))
    right_id = int(getattr(cfg, "cam_portal_head_right_id", 6))
    wrist_id = int(getattr(cfg, "cam_portal_wrist_id", 1))
    max_hz = float(getattr(cfg, "cam_portal_max_hz", 8.0))
    timeout_s = float(getattr(cfg, "cam_portal_timeout_s", 60.0))
    depth_of_color = {1: 0, 3: 2}
    wrist_depth = getattr(cfg, "cam_portal_wrist_depth_id", None)
    if wrist_depth is not None:
        depth_of_color[int(wrist_id)] = int(wrist_depth)
    cache = BundleFrameCache(
        target,
        timeout_s=timeout_s,
        max_hz=max_hz,
        wanted={left_id, right_id, wrist_id},
        depth_of_color=depth_of_color,
    )
    cache.start()
    try:
        # 头摄必须有；腕深度可稍后，先等彩色
        cache.wait_pair(left_id, right_id, timeout_s=max(float(timeout_s), 30.0))
        head = HeadPortalCam(
            target,
            left_cam_id=left_id,
            right_cam_id=right_id,
            max_hz=max_hz,
            timeout_s=timeout_s,
            cache=cache,
            own_cache=False,
            wait_first=False,
        )
        dist = getattr(cfg, "wrist_dist_coeffs", None)
        wrist = WristPortalCam(
            target,
            cam_id=wrist_id,
            max_hz=max_hz,
            timeout_s=timeout_s,
            need_depth=bool(getattr(cfg, "cam_wrist_need_depth", True)),
            fx=float(getattr(cfg, "wrist_fx", 615.0)),
            fy=float(getattr(cfg, "wrist_fy", 615.0)),
            cx=float(getattr(cfg, "wrist_cx", 630.0)),
            cy=float(getattr(cfg, "wrist_cy", 401.0)),
            dist_coeffs=list(dist) if dist else None,
            undistort=bool(getattr(cfg, "cam_wrist_undistort", True)),
            width=int(getattr(cfg, "cam_wrist_width", 1280)),
            height=int(getattr(cfg, "cam_wrist_height", 800)),
            cache=cache,
            own_cache=True,  # 关腕时关掉共享 cache
            wait_first=False,
        )
        LOG.info(
            "Bundle shared stream %s cams=%s/%s/%s",
            target,
            left_id,
            right_id,
            wrist_id,
        )
        return head, wrist
    except Exception:
        try:
            cache.close()
        except Exception:
            pass
        raise


class WristPortalCam:
    """腕部 Gemini 305：frame_bundle 取流（cam1 左 / cam3 右），彩色 + 深度。

    深度在 bundle 里是 cam0/2，缓存会映射回彩色 cam_id。
    305 畸变很大（BROWN_CONRADY_K6）；默认按配置内参去畸变后再给检测/匹配。
    """

    def __init__(
        self,
        portal: str = "10.3.8.7:50066",
        *,
        cam_id: int = 1,
        max_hz: float = 8.0,
        timeout_s: float = 60.0,
        need_depth: bool = True,
        fx: float = 615.0,
        fy: float = 615.0,
        cx: float = 630.0,
        cy: float = 401.0,
        dist_coeffs: list[float] | None = None,
        undistort: bool = True,
        width: int = 1280,
        height: int = 800,
        cache=None,
        own_cache: bool = True,
        wait_first: bool = True,
        depth_cam_id: int | None = None,
    ) -> None:
        import cv2

        from .bundle_cache import BundleFrameCache, DEFAULT_BUNDLE

        target = str(portal or DEFAULT_BUNDLE).strip()
        if ":8787" in target or target.endswith("8787"):
            target = DEFAULT_BUNDLE
        self.portal = target
        self.cam_id = int(cam_id)
        self.need_depth = bool(need_depth)
        self.undistort = bool(undistort)
        self._own_cache = bool(own_cache)
        depth_of_color = {1: 0, 3: 2}
        if depth_cam_id is not None:
            depth_of_color[self.cam_id] = int(depth_cam_id)
        if cache is not None:
            self._cache = cache
            self._own_cache = bool(own_cache)
        else:
            self._cache = BundleFrameCache(
                target,
                timeout_s=timeout_s,
                max_hz=max_hz,
                wanted={self.cam_id},
                depth_of_color=depth_of_color,
            )
            self._cache.start()
            self._own_cache = True
        try:
            if wait_first:
                # 首连时深度可能晚到：先要彩色，深度可选
                self._cache.wait_color_depth(
                    self.cam_id,
                    timeout_s=max(float(timeout_s), 30.0),
                    need_depth=False,
                )
        except Exception:
            self.close()
            raise
        self._map1 = None
        self._map2 = None
        if self.undistort:
            K = np.array(
                [[float(fx), 0.0, float(cx)], [0.0, float(fy), float(cy)], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            )
            D = np.zeros((8, 1), dtype=np.float64)
            if dist_coeffs:
                vals = [float(x) for x in dist_coeffs]
                for i, v in enumerate(vals[:8]):
                    D[i, 0] = v
            self._map1, self._map2 = cv2.initUndistortRectifyMap(
                K, D, None, K, (int(width), int(height)), cv2.CV_16SC2
            )
        LOG.info(
            "Wrist bundle open %s cam=%d depth=%s undistort=%s own_cache=%s",
            self.portal,
            self.cam_id,
            self.need_depth,
            self.undistort,
            self._own_cache,
        )

    def grab(self, *, timeout_ms: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        import cv2

        try:
            color_b, depth_b = self._cache.wait_color_depth(
                self.cam_id,
                timeout_s=max(0.5, float(timeout_ms) / 1000.0),
                need_depth=bool(self.need_depth),
            )
        except TimeoutError:
            # 深度偶发晚到：至少拿彩色，深度置零
            color_b, depth_b = self._cache.wait_color_depth(
                self.cam_id,
                timeout_s=max(0.5, float(timeout_ms) / 1000.0),
                need_depth=False,
            )
        color = cv2.imdecode(np.frombuffer(color_b, dtype=np.uint8), cv2.IMREAD_COLOR)
        if color is None:
            raise RuntimeError(f"portal wrist cam{self.cam_id} color decode failed")
        if depth_b is None:
            depth = np.zeros(color.shape[:2], dtype=np.uint16)
        else:
            depth = cv2.imdecode(
                np.frombuffer(depth_b, dtype=np.uint8), cv2.IMREAD_UNCHANGED
            )
            if depth is None:
                raise RuntimeError(f"portal wrist cam{self.cam_id} depth decode failed")
            if depth.ndim == 3:
                depth = depth[:, :, 0]
            if depth.dtype != np.uint16:
                depth = depth.astype(np.uint16)
        if self._map1 is not None and self._map2 is not None:
            color = cv2.remap(color, self._map1, self._map2, cv2.INTER_LINEAR)
            depth = cv2.remap(depth, self._map1, self._map2, cv2.INTER_NEAREST)
        return color, depth

    def close(self) -> None:
        if not getattr(self, "_own_cache", True):
            return
        try:
            self._cache.close()
        except Exception:
            pass


class WristD405Cam:
    """腕部 RealSense D405：彩色 + 对齐深度（uint16）。"""

    def __init__(
        self,
        serial: str | None = "262422274334",
        *,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        import pyrealsense2 as rs

        self._rs = rs
        device = self._find_d405(serial)
        self.serial = device.get_info(rs.camera_info.serial_number)
        color_mode, depth_mode = self._resolve_streams(device, width, height, fps)

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        cw, ch, cf = color_mode
        config.enable_stream(rs.stream.color, cw, ch, rs.format.bgr8, cf)
        dw, dh, df = depth_mode
        config.enable_stream(rs.stream.depth, dw, dh, rs.format.z16, df)
        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        # warm-up
        for _ in range(8):
            self.grab()
        LOG.info(
            "Wrist D405 open SN=%s color=%dx%d@%d depth=%dx%d@%d",
            self.serial,
            cw,
            ch,
            cf,
            dw,
            dh,
            df,
        )

    def _find_d405(self, serial: str | None):
        rs = self._rs
        devices = list(rs.context().query_devices())
        names = [
            f"{d.get_info(rs.camera_info.name)}(SN={d.get_info(rs.camera_info.serial_number)})"
            for d in devices
        ]
        if not devices:
            raise RuntimeError("未检测到 RealSense 设备（lsusb 有相机也请检查权限/占用）")
        if serial:
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == serial:
                    return dev
            raise RuntimeError(
                f"未找到 D405 SN={serial}；当前设备: " + (", ".join(names) if names else "无")
            )
        for dev in devices:
            if "D405" in dev.get_info(rs.camera_info.name).upper():
                return dev
        raise RuntimeError("未找到 D405: " + ", ".join(names))

    def _video_modes(self, device, stream, fmt):
        rs = self._rs
        modes = []
        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if not profile.is_video_stream_profile():
                    continue
                vp = profile.as_video_stream_profile()
                if vp.stream_type() != stream or vp.format() != fmt:
                    continue
                modes.append((vp.width(), vp.height(), vp.fps()))
        return modes

    def _pick_mode(self, modes, width, height, fps):
        if not modes:
            raise RuntimeError("无可用视频模式")
        exact = [m for m in modes if m == (width, height, fps)]
        if exact:
            return exact[0]
        same = [m for m in modes if m[0] == width and m[1] == height]
        if same:
            return min(same, key=lambda m: abs(m[2] - fps))
        return min(modes, key=lambda m: (abs(m[0] * m[1] - width * height), abs(m[2] - fps)))

    def _resolve_streams(self, device, width, height, fps):
        rs = self._rs
        color = self._pick_mode(
            self._video_modes(device, rs.stream.color, rs.format.bgr8), width, height, fps
        )
        depth_modes = self._video_modes(device, rs.stream.depth, rs.format.z16)
        same_fps = [m for m in depth_modes if m[2] == color[2]]
        pool = same_fps or depth_modes
        same_wh = [m for m in pool if m[0] == color[0] and m[1] == color[1]]
        depth = same_wh[0] if same_wh else max(pool, key=lambda m: m[0] * m[1])
        return color, depth

    def grab(self, *, timeout_ms: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        frames = self.pipeline.wait_for_frames(timeout_ms=timeout_ms)
        frames = self.align.process(frames)
        color_f = frames.get_color_frame()
        depth_f = frames.get_depth_frame()
        if not color_f or not depth_f:
            raise RuntimeError("D405 未拿到 color/depth 帧")
        color = np.asanyarray(color_f.get_data()).copy()
        depth = np.asanyarray(depth_f.get_data()).copy()
        return color, depth

    def close(self) -> None:
        try:
            self.pipeline.stop()
        except Exception:
            pass


def open_wrist_camera(cfg):
    """按配置打开腕摄：``cameras.wrist_source`` = portal/bundle | d405。

    注意：与头摄同时开时请用 ``open_portal_head_and_wrist``（单连接）。
    """
    source = str(getattr(cfg, "cam_wrist_source", "portal") or "portal").strip().lower()
    if source in (
        "portal",
        "watch",
        "watch_portal",
        "rim",
        "305",
        "gemini",
        "bundle",
        "frame_bundle",
    ):
        dist = getattr(cfg, "wrist_dist_coeffs", None)
        depth_id = getattr(cfg, "cam_portal_wrist_depth_id", None)
        return WristPortalCam(
            getattr(cfg, "rim_frame_bundle", None)
            or getattr(cfg, "cam_portal", "10.3.8.7:50066"),
            cam_id=int(getattr(cfg, "cam_portal_wrist_id", 1)),
            max_hz=float(getattr(cfg, "cam_portal_max_hz", 8.0)),
            timeout_s=float(getattr(cfg, "cam_portal_timeout_s", 60.0)),
            need_depth=bool(getattr(cfg, "cam_wrist_need_depth", True)),
            fx=float(getattr(cfg, "wrist_fx", 615.0)),
            fy=float(getattr(cfg, "wrist_fy", 615.0)),
            cx=float(getattr(cfg, "wrist_cx", 630.0)),
            cy=float(getattr(cfg, "wrist_cy", 401.0)),
            dist_coeffs=list(dist) if dist else None,
            undistort=bool(getattr(cfg, "cam_wrist_undistort", True)),
            width=int(getattr(cfg, "cam_wrist_width", 1280)),
            height=int(getattr(cfg, "cam_wrist_height", 800)),
            depth_cam_id=int(depth_id) if depth_id is not None else None,
        )
    serial = getattr(cfg, "cam_wrist_serial", "262422274334")
    if serial in ("", "null", "None"):
        serial = None
    return WristD405Cam(
        serial,
        width=int(getattr(cfg, "cam_wrist_width", 1280)),
        height=int(getattr(cfg, "cam_wrist_height", 720)),
        fps=int(getattr(cfg, "cam_wrist_fps", 30)),
    )


class LiveCameras:
    """头 + 腕一起开；live 流程按阶段分别 grab。"""

    def __init__(self, cfg) -> None:
        head_src = str(getattr(cfg, "cam_head_source", "portal") or "portal").lower()
        wrist_src = str(getattr(cfg, "cam_wrist_source", "portal") or "portal").lower()
        if head_src in (
            "portal",
            "watch",
            "watch_portal",
            "rim",
            "bundle",
            "frame_bundle",
        ) and wrist_src in (
            "portal",
            "watch",
            "watch_portal",
            "rim",
            "305",
            "gemini",
            "bundle",
            "frame_bundle",
        ):
            self.head, self.wrist = open_portal_head_and_wrist(cfg)
        else:
            self.head = open_head_camera(cfg)
            self.wrist = open_wrist_camera(cfg)

    def grab_head(self) -> tuple[np.ndarray, np.ndarray]:
        return self.head.grab()

    def grab_wrist(self) -> tuple[np.ndarray, np.ndarray]:
        return self.wrist.grab()

    def close(self) -> None:
        self.head.close()
        self.wrist.close()

    def __enter__(self) -> "LiveCameras":
        return self

    def __exit__(self, *args) -> None:
        self.close()
