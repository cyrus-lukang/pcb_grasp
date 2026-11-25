"""Head fisheye undistort (+ stereo rectify) via OpenCV fisheye maps."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass
class HeadRectifyMaps:
    map1x: np.ndarray
    map1y: np.ndarray
    map2x: np.ndarray | None
    map2y: np.ndarray | None
    K_rect: np.ndarray  # 3x3 for left after rectify (dist=0)
    image_size: tuple[int, int]  # (w, h)
    # 左目校正旋转：p_rect = R1 @ p_cam。手眼在 7 号原相机，抬点要乘 R1.T。
    R1: np.ndarray | None = None

    def undistort_left(self, bgr: np.ndarray) -> np.ndarray:
        return cv2.remap(bgr, self.map1x, self.map1y, cv2.INTER_LINEAR)

    def undistort_right(self, bgr: np.ndarray) -> np.ndarray:
        if self.map2x is None or self.map2y is None:
            raise RuntimeError("Right remap maps not loaded")
        return cv2.remap(bgr, self.map2x, self.map2y, cv2.INTER_LINEAR)


def _as_K(arr: Any) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64).reshape(3, 3)


def _as_D_fisheye(arr: Any) -> np.ndarray:
    d = np.asarray(arr, dtype=np.float64).reshape(-1)
    if d.size < 4:
        raise ValueError(f"fisheye D needs 4 coeffs, got {d.size}")
    return d[:4].reshape(4, 1)


def load_stereo_calib_json(path: str | Path) -> dict:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("model") not in (None, "fisheye"):
        # still allow if keys present
        pass
    return data


def build_maps_from_stereo_json(
    path: str | Path,
    *,
    balance: float = 0.0,
    fov_scale: float = 1.0,
) -> HeadRectifyMaps:
    """Build left/right remap from rimbot-style ``stereo_calib.json``."""
    data = load_stereo_calib_json(path)
    w, h = int(data["image_size"][0]), int(data["image_size"][1])
    size = (w, h)
    K1 = _as_K(data["left"]["K"])
    D1 = _as_D_fisheye(data["left"]["D"])
    K2 = _as_K(data["right"]["K"])
    D2 = _as_D_fisheye(data["right"]["D"])
    R = np.asarray(data["R"], dtype=np.float64).reshape(3, 3)
    T = np.asarray(data["T"], dtype=np.float64).reshape(3, 1)

    r1, r2, p1, p2, _q = cv2.fisheye.stereoRectify(
        K1,
        D1,
        K2,
        D2,
        size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        newImageSize=size,
        balance=balance,
        fov_scale=fov_scale,
    )
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K1, D1, r1, p1, size, cv2.CV_16SC2
    )
    map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
        K2, D2, r2, p2, size, cv2.CV_16SC2
    )
    K_rect = p1[:3, :3].copy()
    return HeadRectifyMaps(
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        K_rect=K_rect,
        image_size=size,
        R1=np.asarray(r1, dtype=np.float64).reshape(3, 3).copy(),
    )


def T_base_from_rectified(T_base_cam: np.ndarray, R1: np.ndarray | None) -> np.ndarray:
    """手眼 ``T_base_cam`` 是 7 号原相机。校正点 ``p_rect = R1 @ p_cam7``。"""
    T_cam = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    if R1 is None:
        return T_cam.copy()
    T_rect_to_cam = np.eye(4, dtype=np.float64)
    T_rect_to_cam[:3, :3] = np.asarray(R1, dtype=np.float64).reshape(3, 3).T
    return T_cam @ T_rect_to_cam


def load_maps_from_yml(path: str | Path, K_rect: np.ndarray | None = None) -> HeadRectifyMaps:
    """Load precomputed ``map1x/y``, ``map2x/y`` from OpenCV FileStorage yml."""
    fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError(f"Cannot open stereo maps: {path}")
    map1x = fs.getNode("map1x").mat()
    map1y = fs.getNode("map1y").mat()
    map2x_n = fs.getNode("map2x")
    map2y_n = fs.getNode("map2y")
    map2x = map2x_n.mat() if not map2x_n.empty() else None
    map2y = map2y_n.mat() if not map2y_n.empty() else None
    fs.release()
    h, w = map1x.shape[:2]
    if K_rect is None:
        K_rect = np.eye(3, dtype=np.float64)
    return HeadRectifyMaps(
        map1x=map1x,
        map1y=map1y,
        map2x=map2x,
        map2y=map2y,
        K_rect=np.asarray(K_rect, dtype=np.float64).reshape(3, 3),
        image_size=(w, h),
    )


def build_single_fisheye_maps(
    K: np.ndarray,
    D: np.ndarray,
    image_size: tuple[int, int],
    *,
    balance: float = 0.0,
) -> HeadRectifyMaps:
    """Mono fisheye undistort (no stereo)."""
    w, h = image_size
    Knew = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K, D, (w, h), np.eye(3), balance=balance
    )
    map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), Knew, (w, h), cv2.CV_16SC2
    )
    return HeadRectifyMaps(
        map1x=map1x,
        map1y=map1y,
        map2x=None,
        map2y=None,
        K_rect=Knew,
        image_size=(w, h),
    )
