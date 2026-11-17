"""Pose / rz math aligned with rimbot_poc perception.

Head flange rz: ``baseYAngleToGripperRz`` (corner0 → corner1 in base XY).
Wrist edge yaw: ``rzFromLineXY`` then optional fold to (-90, 90].
Flange command: ``ComputeFlangeTarget`` (rx=π, ry=0, tcp_offset, tool_rz_offset).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np

PI = math.pi
# 现场确认的「末端竖直向下」：工具 Y ≈ base −Z；不是 tip Z 朝下 (rx=π)
END_DOWN_RX = -0.5 * math.pi
END_DOWN_RY = 0.0


def pose_xyz_rpy_to_mat4(
    x: float, y: float, z: float, rx: float, ry: float, rz: float
) -> np.ndarray:
    """RPY = Rz @ Ry @ Rx (same as rimbot / RealMan)."""
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


def mat4_to_xyz_rpy(T: np.ndarray) -> list[float]:
    R = T[:3, :3]
    sy = -R[2, 0]
    cy = math.sqrt(max(0.0, 1.0 - sy * sy))
    if cy > 1e-8:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(sy, cy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(sy, cy)
        rz = 0.0
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), rx, ry, rz]


def transform_point(T: np.ndarray, p: Sequence[float]) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64).reshape(3)
    return T[:3, :3] @ p + T[:3, 3]


def wrap_pi(a: float) -> float:
    a = math.fmod(a + PI, 2.0 * PI)
    if a < 0:
        a += 2.0 * PI
    return a - PI


def wrap_deg(a: float) -> float:
    return float(((a + 180.0) % 360.0) - 180.0)


def angle_with_base_y(pa: Sequence[float], pb: Sequence[float]) -> float:
    """atan2(vx, vy): angle of edge vs base +Y. See rimbot geometry.cpp."""
    vx = float(pb[0] - pa[0])
    vy = float(pb[1] - pa[1])
    n = math.hypot(vx, vy)
    if n < 1e-9:
        return 0.0
    return math.atan2(vx / n, vy / n)


def base_y_angle_to_gripper_rz(pa: Sequence[float], pb: Sequence[float]) -> float:
    """Corner0→1 → flange rz (rad). Port of rimbot ``baseYAngleToGripperRz``."""
    yaw_deg = -(angle_with_base_y(pa, pb) * 180.0 / PI) - 90.0
    if (float(pb[1]) - float(pa[1])) < 0.0:
        yaw_deg += 180.0
    return wrap_deg(yaw_deg) * PI / 180.0


def rz_from_line_xy(pa: Sequence[float], pb: Sequence[float]) -> float:
    """Edge vs base +X (rad). Port of rimbot ``rzFromLineXY``."""
    return math.atan2(float(pb[1] - pa[1]), float(pb[0] - pa[0]))


def fold_deg_180(deg: float) -> float:
    """Fold undirected edge into (-90, 90]."""
    while deg > 90.0:
        deg -= 180.0
    while deg <= -90.0:
        deg += 180.0
    return deg


def nearest_rz_rad(ref: float, cand: float) -> float:
    """把 ``cand`` 折到与 ``ref`` 最近的 ±180° 等价角（无向边防跳变）。"""
    best = float(cand)
    best_d = abs(wrap_pi(best - float(ref)))
    for k in (-2, -1, 1, 2):
        a = float(cand) + k * PI
        d = abs(wrap_pi(a - float(ref)))
        if d < best_d:
            best, best_d = a, d
    return wrap_pi(best)


def euler_xyz_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    """3x3 row-major R = Rz @ Ry @ Rx (matches rimbot EulerXyzToMatrix usage)."""
    return pose_xyz_rpy_to_mat4(0, 0, 0, rx, ry, rz)[:3, :3]


def compute_flange_target(
    p_base_m: Sequence[float],
    rz_rad: float,
    tcp_offset: Sequence[float] = (0.0, 0.0, 0.28),
    tool_rz_offset_rad: float = PI / 2.0,
) -> list[float]:
    """Object center + gripper rz → flange xyzrpy. Port of ``ComputeFlangeTarget``.

    ``rz_rad`` is already flange-oriented yaw from ``base_y_angle_to_gripper_rz``.
    With default ``tool_rz_offset_rad = π/2``, flange_rz ≈ rz_rad.
    """
    flange_rx = PI
    flange_ry = 0.0
    flange_rz = wrap_pi(rz_rad + (tool_rz_offset_rad - PI / 2.0))
    R = euler_xyz_to_matrix(flange_rx, flange_ry, flange_rz)
    ox, oy, oz = (float(x) for x in tcp_offset[:3])
    off = R @ np.array([ox, oy, oz], dtype=np.float64)
    return [
        float(p_base_m[0]) - float(off[0]),
        float(p_base_m[1]) - float(off[1]),
        float(p_base_m[2]) - float(off[2]),
        flange_rx,
        flange_ry,
        flange_rz,
    ]


def camera_xyz_from_flange(
    flange_xyzrpy: Sequence[float],
    *,
    T_tool_cam: np.ndarray,
) -> list[float]:
    """由法兰位姿反推腕相机光心在基座系的 XYZ（米）。"""
    T_bt = pose_xyz_rpy_to_mat4(*[float(x) for x in flange_xyzrpy[:6]])
    T_tc = np.asarray(T_tool_cam, dtype=np.float64).reshape(4, 4)
    T_bc = T_bt @ T_tc
    return [float(T_bc[0, 3]), float(T_bc[1, 3]), float(T_bc[2, 3])]


def flange_pose_camera_optical_down(
    cam_xyz_base: Sequence[float],
    *,
    T_tool_cam: np.ndarray,
    cam_yaw_rad: float = 0.0,
) -> list[float]:
    """法兰 xyzrpy：腕相机光轴竖直向下（base −Z），相机光心在 ``cam_xyz_base``。

    ``T_tool_cam``：眼在手上，``p_tool = R p_cam + t``（与手眼标定一致）。
    ``cam_yaw_rad``：相机 +X 在水平面相对 base +X 的朝向（常用板边方向）。

    注意：必须整姿态一起用；单独改法兰 rz 会破坏「光心对准」关系。
    """
    T = np.asarray(T_tool_cam, dtype=np.float64).reshape(4, 4)
    R_tc = T[:3, :3]
    t_tc = T[:3, 3]
    z_c = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    x_h = np.array(
        [math.cos(float(cam_yaw_rad)), math.sin(float(cam_yaw_rad)), 0.0],
        dtype=np.float64,
    )
    y_c = np.cross(z_c, x_h)
    n = float(np.linalg.norm(y_c))
    if n < 1e-9:
        x_h = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        y_c = np.cross(z_c, x_h)
        n = float(np.linalg.norm(y_c))
    y_c /= n
    x_c = np.cross(y_c, z_c)
    x_c /= float(np.linalg.norm(x_c))
    R_base_cam = np.column_stack((x_c, y_c, z_c))
    R_base_tool = R_base_cam @ R_tc.T
    p_cam = np.asarray(cam_xyz_base[:3], dtype=np.float64)
    p_tool = p_cam - R_base_tool @ t_tc
    T_bt = np.eye(4, dtype=np.float64)
    T_bt[:3, :3] = R_base_tool
    T_bt[:3, 3] = p_tool
    return mat4_to_xyz_rpy(T_bt)


def _ori_err_deg_pose(a: Sequence[float], b: Sequence[float]) -> float:
    Ra = pose_xyz_rpy_to_mat4(*[float(x) for x in a[:6]])[:3, :3]
    Rb = pose_xyz_rpy_to_mat4(*[float(x) for x in b[:6]])[:3, :3]
    R = Ra.T @ Rb
    c = float(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def flange_pose_camera_down_nearest(
    cam_xyz_base: Sequence[float],
    *,
    T_tool_cam: np.ndarray,
    current_xyzrpy: Sequence[float] | None = None,
    n_yaw: int = 72,
    max_reach_m: float = 0.95,
) -> tuple[list[float], float]:
    """相机光心对准 ``cam_xyz_base`` 且光轴朝下；选与当前姿态最接近的 cam_yaw。

    ``cam_yaw=0`` 常导致法兰 rz 跳 ~90°，原地拧易奇异/限位。优先最小转角解。
    返回 ``(flange_xyzrpy, cam_yaw_rad)``。
    """
    cur = (
        None
        if current_xyzrpy is None
        else [float(x) for x in current_xyzrpy[:6]]
    )
    best_fl: list[float] | None = None
    best_yaw = 0.0
    best_score = float("inf")
    for i in range(max(8, int(n_yaw))):
        yaw = -math.pi + (2.0 * math.pi) * (i / float(n_yaw))
        fl = flange_pose_camera_optical_down(
            cam_xyz_base, T_tool_cam=T_tool_cam, cam_yaw_rad=yaw
        )
        reach = float(math.hypot(fl[0], fl[1]))
        if reach > float(max_reach_m):
            continue
        if abs(float(fl[1])) > 0.85 or float(fl[2]) > 1.05 or float(fl[2]) < 0.35:
            continue
        ori = 0.0 if cur is None else _ori_err_deg_pose(cur, fl)
        # 略惩罚远伸，避免贴工作区边缘
        score = ori + 15.0 * max(0.0, reach - 0.70)
        if score < best_score:
            best_score = score
            best_fl = fl
            best_yaw = yaw
    if best_fl is None:
        # 回退：不筛 reach，仍取最小转角
        for i in range(max(8, int(n_yaw))):
            yaw = -math.pi + (2.0 * math.pi) * (i / float(n_yaw))
            fl = flange_pose_camera_optical_down(
                cam_xyz_base, T_tool_cam=T_tool_cam, cam_yaw_rad=yaw
            )
            ori = 0.0 if cur is None else _ori_err_deg_pose(cur, fl)
            if ori < best_score:
                best_score = ori
                best_fl = fl
                best_yaw = yaw
    assert best_fl is not None
    return best_fl, float(best_yaw)


@dataclass
class ObjectPose:
    """PCB in base frame (meters, radians)."""

    x: float
    y: float
    z: float
    rz: float  # flange-oriented rz from corner0→1
    edge_yaw: float  # raw atan2 edge vs +X
    corners_base: list[list[float]]
    conf: float = 0.0
    board_id: int | None = None  # 1=nearest to base, …
    cls_id: int = 0  # 头模 0=pcb 1=nopcb；腕部默认 0


    @property
    def xyz(self) -> list[float]:
        return [self.x, self.y, self.z]

    def as_xyzrpy(
        self, rx: float = END_DOWN_RX, ry: float = END_DOWN_RY
    ) -> list[float]:
        return [self.x, self.y, self.z, rx, ry, self.rz]
