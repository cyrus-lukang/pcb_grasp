"""Pixel corners + depth/hand-eye → PCB ObjectPose in base."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .detect import PcbDetection
from .geometry import (
    ObjectPose,
    base_y_angle_to_gripper_rz,
    pose_xyz_rpy_to_mat4,
    rz_from_line_xy,
    transform_point,
    wrap_deg,
)

LOG = logging.getLogger("pcb_grasp.localize")


@dataclass
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_K(cls, K: np.ndarray) -> "Intrinsics":
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        return cls(fx=float(K[0, 0]), fy=float(K[1, 1]), cx=float(K[0, 2]), cy=float(K[1, 2]))

    def as_K(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


def pixel_to_cam(u: float, v: float, z_m: float, K: Intrinsics) -> np.ndarray:
    return np.array(
        [(u - K.cx) * z_m / K.fx, (v - K.cy) * z_m / K.fy, z_m],
        dtype=np.float64,
    )


def base_xyz_to_cam(
    xyz_base: Sequence[float],
    *,
    T_base_cam: np.ndarray,
) -> np.ndarray:
    """基座 XYZ → 头相机坐标系（``T_base_cam`` 为 cam→base 外参，取逆）。"""
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    p = np.asarray(xyz_base[:3], dtype=np.float64)
    return R.T @ (p - t)


def base_xyz_to_pixel(
    xyz_base: Sequence[float],
    *,
    K: Intrinsics,
    T_base_cam: np.ndarray,
) -> tuple[float, float] | None:
    """反向手眼：基座抓取点 → 头相机校正图像素。相机后方则返回 None。"""
    p_cam = base_xyz_to_cam(xyz_base, T_base_cam=T_base_cam)
    z = float(p_cam[2])
    if z <= 1e-4:
        return None
    u = float(K.fx) * float(p_cam[0]) / z + float(K.cx)
    v = float(K.fy) * float(p_cam[1]) / z + float(K.cy)
    return u, v


def pixel_to_base_on_plane(
    u: float,
    v: float,
    *,
    K: Intrinsics,
    T_base_cam: np.ndarray,
    z_plane_m: float,
) -> np.ndarray:
    """倾斜头目：像素射线与水平托盘平面 Z=z_plane 求交 → base XYZ。"""
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    d = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0], dtype=np.float64)
    rd = R @ d
    if abs(float(rd[2])) < 1e-9:
        raise RuntimeError("head ray parallel to tray plane")
    s = (float(z_plane_m) - float(t[2])) / float(rd[2])
    if s <= 0.05:
        raise RuntimeError(f"head ray-plane hit invalid s={s:.3f}")
    return t + s * rd


def cam_depth_to_plane_m(T_base_cam: np.ndarray, z_plane_m: float) -> float:
    """相机光轴打到水平面 Z=z_plane 的深度（米）。由当前光心位置算出。"""
    T = np.asarray(T_base_cam, dtype=np.float64).reshape(4, 4)
    origin_z = float(T[2, 3])
    axis_z = float(T[2, 2])
    if abs(axis_z) < 1e-6:
        raise RuntimeError("camera optical axis parallel to grasp plane")
    depth = (float(z_plane_m) - origin_z) / axis_z
    if depth <= 0.05:
        raise RuntimeError(
            f"camera depth to z={z_plane_m:.3f} invalid: {depth:.3f} "
            f"(camZ={origin_z:.3f})"
        )
    return float(depth)


def sample_depth_m(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    depth_scale: float = 0.001,
    radius: int = 4,
    z_min: float = 0.12,
    z_max: float = 3.0,
) -> float | None:
    """Sample depth image (uint16 mm or float meters) around (u,v)."""
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    if ui < 0 or vi < 0 or ui >= w or vi >= h:
        return None
    vals: list[float] = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            x, y = ui + dx, vi + dy
            if x < 0 or y < 0 or x >= w or y >= h:
                continue
            raw = depth[y, x]
            if depth.dtype == np.float32 or depth.dtype == np.float64:
                z = float(raw)
            else:
                z = float(raw) * depth_scale
            # 腕部悬停可近到 ~0.15m；头目托盘约 0.7–1.1m
            if float(z_min) < z < float(z_max):
                vals.append(z)
    if not vals:
        return None
    vals.sort()
    return vals[len(vals) // 2]


def corner_uv_to_base(
    u: float,
    v: float,
    *,
    T_base_cam: np.ndarray,
    K: Intrinsics,
    fixed_depth_cam_m: float | None = None,
    tray_z_m: float | None = None,
) -> list[float]:
    """单像素角点 → 基座 XYZ（大板腕精：V2·220e76 单角模型）。"""
    if tray_z_m is not None:
        return pixel_to_base_on_plane(
            float(u),
            float(v),
            K=K,
            T_base_cam=T_base_cam,
            z_plane_m=float(tray_z_m),
        ).tolist()
    if fixed_depth_cam_m is None:
        raise ValueError("Need tray_z_m or fixed_depth_cam_m")
    p_cam = pixel_to_cam(float(u), float(v), float(fixed_depth_cam_m), K)
    return transform_point(T_base_cam, p_cam).tolist()


def corners_to_object_pose(
    det: PcbDetection,
    *,
    T_base_cam: np.ndarray,
    K: Intrinsics,
    depth_m_per_corner: Sequence[float] | None = None,
    fixed_depth_cam_m: float | None = None,
    tray_z_m: float | None = None,
    kp_i: int = 0,
    kp_j: int = 1,
) -> ObjectPose:
    """Lift corners to base and compute center + flange rz from corner kp_i→kp_j.

    优先 ``tray_z_m``（水平托盘平面求交，抗倾斜头目 + 散斑飞点）；
    否则用每角深度 / 固定相机深度。支持 1 角（单角模型）或 4 角。
    """
    n = len(det.corners)
    if n < 1:
        raise ValueError("det.corners empty")
    if tray_z_m is not None:
        corners_base = [
            pixel_to_base_on_plane(
                float(u), float(v), K=K, T_base_cam=T_base_cam, z_plane_m=float(tray_z_m)
            ).tolist()
            for u, v in det.corners
        ]
    else:
        if depth_m_per_corner is None:
            if fixed_depth_cam_m is None:
                raise ValueError("Need tray_z_m, depth_m_per_corner, or fixed_depth_cam_m")
            depth_m_per_corner = [fixed_depth_cam_m] * n

        corners_cam = [
            pixel_to_cam(u, v, float(z), K)
            for (u, v), z in zip(det.corners, depth_m_per_corner)
        ]
        corners_base = [transform_point(T_base_cam, p).tolist() for p in corners_cam]
    center = np.mean(np.asarray(corners_base, dtype=np.float64), axis=0)

    i0 = int(kp_i) if 0 <= int(kp_i) < n else 0
    i1 = int(kp_j) if 0 <= int(kp_j) < n else min(1, n - 1)
    pa = corners_base[i0]
    pb = corners_base[i1]
    if i0 == i1:
        rz = 0.0
        edge = 0.0
    else:
        rz = base_y_angle_to_gripper_rz(pa, pb)
        edge = rz_from_line_xy(pa, pb)

    return ObjectPose(
        x=float(center[0]),
        y=float(center[1]),
        z=float(center[2]),
        rz=float(rz),
        edge_yaw=float(edge),
        corners_base=corners_base,
        conf=det.conf,
    )


def head_localize(
    det: PcbDetection,
    *,
    T_base_cam: np.ndarray,
    K: Intrinsics,
    depth: np.ndarray | None = None,
    depth_scale: float = 0.001,
    fixed_depth_cam_m: float | None = None,
    tray_z_m: float | None = None,
    kp_i: int = 0,
    kp_j: int = 1,
    depth_m_per_corner: Sequence[float] | None = None,
    corner_z_max: float = 1.0,
    required_corner_indices: Sequence[int] | None = None,
) -> ObjectPose:
    # 托盘平面优先：头目倾斜时固定相机深度会把近/远板 Z·XY 一起拉歪
    if tray_z_m is not None:
        pose = corners_to_object_pose(
            det,
            T_base_cam=T_base_cam,
            K=K,
            tray_z_m=float(tray_z_m),
            kp_i=kp_i,
            kp_j=kp_j,
        )
        LOG.info(
            "head tray-plane z=%.3f → xyz=(%.3f,%.3f,%.3f)",
            float(tray_z_m),
            pose.x,
            pose.y,
            pose.z,
        )
        return pose

    depths: list[float] | None = None
    if depth_m_per_corner is not None:
        depths = [float(z) for z in depth_m_per_corner]
        LOG.info("head corner depths (per-corner frame median)=%s", [round(z, 3) for z in depths])
    elif depth is not None:
        # 只留 1m 以内：更远是打到地面/背景，不用。四个角互不替换。
        # required_corner_indices 之外的角（大板的 P0/P1）没有合格深度也不丢帧。
        n_c = len(det.corners)
        required = (
            set(range(n_c))
            if required_corner_indices is None
            else {int(i) for i in required_corner_indices}
        )
        raw: list[float | None] = []
        for u, v in det.corners:
            raw.append(
                sample_depth_m(
                    depth, u, v, depth_scale=depth_scale, z_max=float(corner_z_max)
                )
            )
        need = [raw[i] for i in range(n_c) if i in required]
        if any(z is None for z in need):
            missing = [
                i
                for i in range(n_c)
                if i in required and raw[i] is None
            ]
            uv = det.corners[missing[0]]
            raise RuntimeError(
                f"No depth <{corner_z_max:.2f}m at head corner P{missing[0]} "
                f"({uv[0]:.1f},{uv[1]:.1f})"
            )
        fill = float(sorted(float(z) for z in need)[len(need) // 2])
        skipped = [i for i in range(n_c) if i not in required and raw[i] is None]
        depths = [fill if z is None else float(z) for z in raw]
        if skipped:
            LOG.info(
                "head skip depth on P%s (not required); fill=%.3f",
                skipped,
                fill,
            )
        LOG.info("head corner depths=%s", [round(z, 3) for z in depths])
    return corners_to_object_pose(
        det,
        T_base_cam=T_base_cam,
        K=K,
        depth_m_per_corner=depths,
        fixed_depth_cam_m=fixed_depth_cam_m,
        kp_i=kp_i,
        kp_j=kp_j,
    )


def wrist_localize(
    det: PcbDetection,
    *,
    T_tool_cam: np.ndarray,
    ee_xyz_rpy: Sequence[float],
    K: Intrinsics,
    depth: np.ndarray,
    depth_scale: float = 0.001,
    fallback_depth_m: float | None = None,
    force_fixed_depth: bool = False,
    kp_i: int = 0,
    kp_j: int = 1,
) -> ObjectPose:
    """Eye-in-hand: T_base_cam = T_base_tool @ T_tool_cam."""
    T_base_tool = pose_xyz_rpy_to_mat4(*[float(x) for x in ee_xyz_rpy[:6]])
    T_base_cam = T_base_tool @ T_tool_cam
    depths: list[float] = []
    used_fb = False
    if force_fixed_depth and fallback_depth_m is not None and float(fallback_depth_m) > 0.05:
        z_fix = float(fallback_depth_m)
        depths = [z_fix] * len(det.corners)
        LOG.info("wrist_localize camZ fixed=%.3fm (hover)", z_fix)
    else:
        for u, v in det.corners:
            z = sample_depth_m(depth, u, v, depth_scale=depth_scale)
            if z is None:
                if fallback_depth_m is not None and float(fallback_depth_m) > 0.05:
                    z = float(fallback_depth_m)
                    used_fb = True
                else:
                    raise RuntimeError(f"No depth at wrist corner ({u:.1f},{v:.1f})")
            depths.append(z)
        if used_fb:
            LOG.warning(
                "wrist_localize used fallback camZ=%.3fm for missing depth",
                float(fallback_depth_m),
            )
    return corners_to_object_pose(
        det,
        T_base_cam=T_base_cam,
        K=K,
        depth_m_per_corner=depths,
        kp_i=kp_i,
        kp_j=kp_j,
    )


def format_pose(p: ObjectPose) -> str:
    return (
        f"xyz=({p.x:.4f},{p.y:.4f},{p.z:.4f}) "
        f"rz={wrap_deg(p.rz * 180 / 3.141592653589793):.1f}deg "
        f"edge={wrap_deg(p.edge_yaw * 180 / 3.141592653589793):.1f}deg "
        f"conf={p.conf:.2f}"
    )
