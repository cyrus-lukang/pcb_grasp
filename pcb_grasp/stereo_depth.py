"""Head stereo depth via Fast-FoundationStereo（本项目自带封装）。

Head is dual fisheye → rectify → FFS disparity → Z = fx * baseline / disp.
Do **not** treat this as RealSense RGB-D; always check valid-pixel / scale first.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .ffs_stereo import DEFAULT_FFS_ROOT, load_stereo_model, stereo_depth as ffs_stereo_depth
from .undistort import HeadRectifyMaps, build_maps_from_stereo_json

LOG = logging.getLogger("pcb_grasp.stereo_depth")

_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = _PKG_ROOT / "models" / "ffs" / "model_best_bp2_serialize.pth"
DEFAULT_CFG = _PKG_ROOT / "models" / "ffs" / "cfg.yaml"


@dataclass
class DepthStats:
    valid_ratio: float
    median_m: float
    p10_m: float
    p90_m: float
    shape: tuple[int, int]

    def ok(self, min_valid: float = 0.15, z_lo: float = 0.2, z_hi: float = 2.5) -> bool:
        return (
            self.valid_ratio >= min_valid
            and z_lo <= self.median_m <= z_hi
            and np.isfinite(self.median_m)
        )

    def summary(self) -> str:
        return (
            f"valid={self.valid_ratio:.1%}  "
            f"z_m p10/p50/p90={self.p10_m:.3f}/{self.median_m:.3f}/{self.p90_m:.3f}  "
            f"hw={self.shape}"
        )


def depth_stats(depth_u16_mm: np.ndarray) -> DepthStats:
    valid = depth_u16_mm > 0
    if not np.any(valid):
        return DepthStats(0.0, float("nan"), float("nan"), float("nan"), depth_u16_mm.shape[:2])
    z = depth_u16_mm[valid].astype(np.float64) * 0.001
    return DepthStats(
        valid_ratio=float(valid.mean()),
        median_m=float(np.median(z)),
        p10_m=float(np.percentile(z, 10)),
        p90_m=float(np.percentile(z, 90)),
        shape=depth_u16_mm.shape[:2],
    )


def colorize_depth(depth_u16_mm: np.ndarray, max_mm: float = 2000.0) -> np.ndarray:
    vis = np.zeros((*depth_u16_mm.shape[:2], 3), dtype=np.uint8)
    valid = depth_u16_mm > 0
    if not np.any(valid):
        return vis
    norm = np.clip(depth_u16_mm.astype(np.float32) / max_mm, 0, 1)
    gray = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    vis[valid] = color[valid]
    return vis


class HeadStereoDepth:
    """Lazy-load FFS; estimate depth on rectified left/right BGR."""

    def __init__(
        self,
        *,
        weights: str | Path | None = None,
        cfg: str | Path | None = None,
        ffs_root: str | Path | None = None,
        iters: int = 8,
        max_disp: int = 192,
        maps: HeadRectifyMaps | None = None,
        stereo_calib_json: str | Path | None = None,
        fx: float | None = None,
        baseline_mm: float | None = None,
    ) -> None:
        self.weights = Path(weights or DEFAULT_WEIGHTS)
        self.cfg = Path(cfg or DEFAULT_CFG)
        self.ffs_root = Path(ffs_root or DEFAULT_FFS_ROOT)
        self.iters = iters
        self.max_disp = max_disp
        self._model: Any = None
        if maps is not None:
            self.maps = maps
        elif stereo_calib_json:
            self.maps = build_maps_from_stereo_json(stereo_calib_json)
        else:
            self.maps = None
        self.fx = fx
        self.baseline_mm = baseline_mm
        self.flip_horizontal = False
        if self.maps is not None and self.fx is None:
            self.fx = float(self.maps.K_rect[0, 0])
        if stereo_calib_json:
            baseline_mm, _tx_neg = _baseline_from_calib(stereo_calib_json)
            if self.baseline_mm is None:
                self.baseline_mm = baseline_mm
        # 左图已是 7 号走 CAM0、右图是 6 号走 CAM1，视差为正，不再镜像。
        self.flip_horizontal = False

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        if not self.weights.is_file():
            raise FileNotFoundError(f"FFS weights missing: {self.weights}")
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        self._model = load_stereo_model(
            self.weights,
            self.cfg,
            self.iters,
            self.max_disp,
            ffs_root=self.ffs_root,
        )
        return self._model

    def estimate_rectified(
        self, left_rect_bgr: np.ndarray, right_rect_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, DepthStats]:
        """Return (depth_u16_mm, valid_mask, stats) on already-rectified pair."""
        if self.fx is None or self.baseline_mm is None:
            raise RuntimeError("fx / baseline_mm required for stereo depth")
        model = self._ensure_model()
        left_in, right_in = left_rect_bgr, right_rect_bgr
        if self.flip_horizontal:
            # 右相机在左相机 −X 侧，对应点在右图更靠右（视差为负）。
            # FFS 只搜 0..max_disp。水平翻转后匹配落到正视差，深度再翻回左图。
            left_in = np.ascontiguousarray(left_rect_bgr[:, ::-1])
            right_in = np.ascontiguousarray(right_rect_bgr[:, ::-1])
        depth, valid = ffs_stereo_depth(
            left_in,
            right_in,
            float(self.fx),
            float(self.baseline_mm),
            iters=self.iters,
            max_disp=self.max_disp,
            model=model,
            ffs_root=self.ffs_root,
        )
        if self.flip_horizontal:
            depth = np.ascontiguousarray(depth[:, ::-1])
            valid = np.ascontiguousarray(valid[:, ::-1])
        return depth, valid, depth_stats(depth)

    def estimate_raw(
        self, left_raw_bgr: np.ndarray, right_raw_bgr: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, DepthStats]:
        """Rectify raw fisheye pair then estimate depth.

        Returns left_rect, right_rect, depth_u16_mm, valid, stats.
        """
        if self.maps is None:
            raise RuntimeError("HeadStereoDepth needs maps / stereo_calib_json for raw input")
        left = self.maps.undistort_left(left_raw_bgr)
        right = self.maps.undistort_right(right_raw_bgr)
        depth, valid, stats = self.estimate_rectified(left, right)
        return left, right, depth, valid, stats


def _baseline_from_calib(path: str | Path) -> tuple[float, bool]:
    """Return (baseline_mm, flip_horizontal).

    ``T`` is millimetres, first camera to second. ``T_x < 0`` means the
    camera labeled right sits on the left, so rectified disparity is negative
    and Fast-FoundationStereo never sees the match.
    """
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    T = np.asarray(data["T"], dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(T[:3])), bool(T[0] < 0)


def epipolar_sad_disp_px(
    left_gray: np.ndarray,
    right_gray: np.ndarray,
    u: float,
    v: float,
    *,
    half: int = 16,
    max_disp: int = 350,
) -> tuple[float | None, float]:
    """校正后极线 SAD：在右图同一行找左模板，返回 (disp_px, sad)。

    disp = u_left - u_right（正常双目 >0）。找不到则 (None, inf)。
    """
    ui, vi = int(round(u)), int(round(v))
    h, w = left_gray.shape[:2]
    if ui < half or vi < half or ui + half >= w or vi + half >= h:
        return None, float("inf")
    templ = left_gray[vi - half : vi + half, ui - half : ui + half].astype(np.float32)
    best_sad = float("inf")
    best_d: int | None = None
    for d in range(1, int(max_disp) + 1):
        x0 = ui - d - half
        if x0 < 0:
            break
        patch = right_gray[vi - half : vi + half, x0 : ui - d + half].astype(np.float32)
        if patch.shape != templ.shape:
            continue
        sad = float(np.mean(np.abs(templ - patch)))
        if sad < best_sad:
            best_sad = sad
            best_d = d
    if best_d is None:
        return None, best_sad
    return float(best_d), best_sad


def epipolar_board_plane_z_m(
    left_rect_bgr: np.ndarray,
    right_rect_bgr: np.ndarray,
    corners_uv: list | np.ndarray,
    *,
    fx: float,
    baseline_mm: float,
    T_base_cam: np.ndarray,
    K,
    half: int = 16,
    max_disp: int = 350,
) -> float | None:
    """板心极线 SAD → 相机深 → 抬到基座，得到水平板面 Z（米）。

    角点纹理重复，逐角 SAD 不稳；用四角均值像素做一次匹配更稳。
    FFS 在本机头双目上视差系统性偏大，不用于绝对深。
    """
    from .localize import pixel_to_cam
    from .geometry import transform_point

    pts = np.asarray(corners_uv, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] < 1:
        return None
    u, v = float(pts[:, 0].mean()), float(pts[:, 1].mean())
    gL = cv2.cvtColor(left_rect_bgr, cv2.COLOR_BGR2GRAY)
    gR = cv2.cvtColor(right_rect_bgr, cv2.COLOR_BGR2GRAY)
    disp, sad = epipolar_sad_disp_px(
        gL, gR, u, v, half=half, max_disp=max_disp
    )
    if disp is None or disp < 1.0:
        LOG.warning("epipolar board-center SAD fail uv=(%.1f,%.1f) sad=%.1f", u, v, sad)
        return None
    z_mm = float(fx) * float(baseline_mm) / float(disp)
    z_cam_m = z_mm * 0.001
    p_cam = pixel_to_cam(u, v, z_cam_m, K)
    p_base = transform_point(np.asarray(T_base_cam, dtype=np.float64), p_cam)
    z_plane = float(p_base[2])
    LOG.info(
        "epipolar board-center uv=(%.1f,%.1f) disp=%.1fpx sad=%.1f → "
        "camZ=%.1fmm → base_plane_Z=%.4fm",
        u,
        v,
        disp,
        sad,
        z_mm,
        z_plane,
    )
    return z_plane
