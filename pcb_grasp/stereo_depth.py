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
        if self.maps is not None and self.fx is None:
            self.fx = float(self.maps.K_rect[0, 0])
        if self.baseline_mm is None and stereo_calib_json:
            self.baseline_mm = _baseline_mm_from_calib(stereo_calib_json)

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
        depth, valid = ffs_stereo_depth(
            left_rect_bgr,
            right_rect_bgr,
            float(self.fx),
            float(self.baseline_mm),
            iters=self.iters,
            max_disp=self.max_disp,
            model=model,
            ffs_root=self.ffs_root,
        )
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


def _baseline_mm_from_calib(path: str | Path) -> float:
    import json

    data = json.loads(Path(path).read_text(encoding="utf-8"))
    T = np.asarray(data["T"], dtype=np.float64).reshape(-1)
    # rimbot stereo_calib T is in mm
    return float(np.linalg.norm(T[:3]))
