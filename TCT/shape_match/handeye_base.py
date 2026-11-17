#!/usr/bin/env python3
"""Wrist eye-in-hand → base-frame helpers for planar PCB (x, y, yaw).

Uses production hand-eye:
  T_base_cam = T_base_ee @ T_tool_cam
  (camera2_eye_in_hand / wrist serial 260322274901)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

DEFAULT_HANDEYE = (
    Path(__file__).resolve().parent / "calib" / "wrist_d405_T_tool_cam.yaml"
)
DEFAULT_INTRINSICS = (
    Path(__file__).resolve().parent / "calib" / "wrist_camera_01.yaml"
)


@dataclass
class CameraK:
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float]
    width: int = 1280
    height: int = 720


def pose_xyz_rpy_to_mat4(
    x: float, y: float, z: float, rx: float, ry: float, rz: float
) -> np.ndarray:
    """RPY = Rz @ Ry @ Rx (RealMan / yuantu convention)."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    R = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def _parse_yaml_matrix(text: str, keys: Sequence[str]) -> np.ndarray | None:
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception:
        data = None

    if isinstance(data, dict):
        for key in keys:
            if key in data:
                val = data[key]
                if isinstance(val, dict) and "matrix" in val:
                    return np.asarray(val["matrix"], dtype=np.float64)
                if isinstance(val, list):
                    return np.asarray(val, dtype=np.float64)
        block = data.get("camera2_eye_in_hand") or data.get("camera2")
        if isinstance(block, dict) and "matrix" in block:
            return np.asarray(block["matrix"], dtype=np.float64)

    for key in keys:
        token = f"{key}:"
        idx = text.find(token)
        if idx < 0:
            continue
        chunk = text[idx + len(token) :]
        rows: list[list[float]] = []
        for line in chunk.splitlines():
            line = line.strip()
            if not line.startswith("-"):
                if rows:
                    break
                continue
            if "[" in line:
                inner = line[line.index("[") : line.rindex("]") + 1]
                rows.append([float(x) for x in inner.strip("[]").split(",")])
                if len(rows) == 4:
                    return np.asarray(rows, dtype=np.float64)
    return None


def load_T_tool_cam(path: str | Path | None = None) -> np.ndarray:
    path = Path(path) if path else DEFAULT_HANDEYE
    text = path.read_text(encoding="utf-8")
    M = _parse_yaml_matrix(text, ("T_tool_cam", "matrix"))
    if M is None or M.shape != (4, 4):
        raise RuntimeError(f"Could not parse T_tool_cam from {path}")
    return M


def load_intrinsics(path: str | Path | None = None) -> CameraK:
    path = Path(path) if path else DEFAULT_INTRINSICS
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
    except Exception as e:
        raise RuntimeError(f"Need PyYAML to load intrinsics: {e}") from e

    cam = data.get("camera", {})
    K = data["intrinsic"]["camera_matrix"]
    dist = data["intrinsic"].get("distortion_coefficients", [[]])
    if isinstance(dist[0], list):
        dist = dist[0]
    return CameraK(
        fx=float(K[0][0]),
        fy=float(K[1][1]),
        cx=float(K[0][2]),
        cy=float(K[1][2]),
        dist=[float(x) for x in dist],
        width=int(cam.get("image_width", 1280)),
        height=int(cam.get("image_height", 720)),
    )


def wrap_angle_deg(a: float) -> float:
    return float(((a + 180.0) % 360.0) - 180.0)


def angle_diff_deg(a: float, b: float) -> float:
    """Signed shortest difference a - b in (-180, 180]."""
    return wrap_angle_deg(a - b)


def pixel_to_cam(u: float, v: float, z_m: float, K: CameraK) -> np.ndarray:
    """Pinhole back-project (ignore distortion for planar PoC)."""
    x = (u - K.cx) * z_m / K.fx
    y = (v - K.cy) * z_m / K.fy
    return np.array([x, y, z_m], dtype=np.float64)


def transform_point(T: np.ndarray, p: np.ndarray) -> np.ndarray:
    return T[:3, :3] @ np.asarray(p, dtype=np.float64) + T[:3, 3]


def yaw_from_edge_xy(p0: np.ndarray, p1: np.ndarray) -> float:
    """Yaw of directed edge p0→p1 in base XY, degrees, atan2(dy, dx)."""
    d = p1[:2] - p0[:2]
    n = float(np.hypot(d[0], d[1]))
    if n < 1e-9:
        return 0.0
    return float(np.degrees(np.arctan2(d[1], d[0])))


@dataclass
class BasePose2D:
    x: float
    y: float
    z: float
    yaw_deg: float
    frame: str
    T_base_cam: list[list[float]]


def match_to_base(
    u: float,
    v: float,
    angle_deg: float,
    *,
    T_tool_cam: np.ndarray,
    K: CameraK,
    depth_m: float,
    ee_xyz_rpy: Sequence[float] | None = None,
    tip_len_px: float = 40.0,
) -> BasePose2D:
    """
    Convert image match (centroid u,v + silhouette angle) to base XY yaw.

    ee_xyz_rpy: [x,y,z,rx,ry,rz] meters/radians. If None, EE = I
    (result is in tool/flange frame, labeled tool_as_base).
    """
    if ee_xyz_rpy is None:
        T_base_ee = np.eye(4, dtype=np.float64)
        frame = "tool(=base@EE=I)"
    else:
        if len(ee_xyz_rpy) < 6:
            raise ValueError("ee_xyz_rpy needs 6 values: x y z rx ry rz")
        T_base_ee = pose_xyz_rpy_to_mat4(*[float(x) for x in ee_xyz_rpy[:6]])
        frame = "base"

    T_base_cam = T_base_ee @ T_tool_cam

    a = math.radians(float(angle_deg))
    # Image coords: +x right, +y down; match angle rotates model in image plane.
    tip_u = u + tip_len_px * math.cos(a)
    tip_v = v + tip_len_px * math.sin(a)

    p0_c = pixel_to_cam(u, v, depth_m, K)
    p1_c = pixel_to_cam(tip_u, tip_v, depth_m, K)
    p0_b = transform_point(T_base_cam, p0_c)
    p1_b = transform_point(T_base_cam, p1_c)

    return BasePose2D(
        x=float(p0_b[0]),
        y=float(p0_b[1]),
        z=float(p0_b[2]),
        yaw_deg=wrap_angle_deg(yaw_from_edge_xy(p0_b, p1_b)),
        frame=frame,
        T_base_cam=T_base_cam.tolist(),
    )


def silhouette_ref_angle_deg(
    edges: np.ndarray, cx: float, cy: float
) -> float | None:
    """
    Reference in-plane angle from silhouette PCA around (cx,cy).
    Returns degrees in image frame (same convention as MatchResult.angle_deg),
    or None if too few points.
    """
    ys, xs = np.where(edges > 0)
    if len(xs) < 30:
        return None
    pts = np.stack([xs.astype(np.float64) - cx, ys.astype(np.float64) - cy], axis=1)
    cov = pts.T @ pts / max(1, len(pts) - 1)
    w, V = np.linalg.eigh(cov)
    axis = V[:, int(np.argmax(w))]
    return wrap_angle_deg(float(np.degrees(np.arctan2(axis[1], axis[0]))))
