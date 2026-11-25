"""Large PCB dual-arm squeeze.

精定位后：用腕部精 P2/P3。
- rz：夹爪中间线 **∥ P2→P3**
- 接近 tip（相对各自角点）：
  左 tip = (P3.x+dx, P3.y+dy, z)；右 tip = (P2.x+dx, P2.y−dy, z)
  默认 dx=0.10、dy=0.08（基座：左 +Y 外、右 −Y 外）
- Z：标定 grasp_z_m
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .geometry import (
    ObjectPose,
    END_DOWN_RX,
    END_DOWN_RY,
    base_y_angle_to_gripper_rz,
    wrap_pi,
)


@dataclass
class LargeBoardPlan:
    """Dual-arm tip + flange targets for a large board (meters / radians)."""

    tip_left: list[float]
    tip_right: list[float]
    yaw_rad: float  # 左臂夹手 rz；右臂为 yaw+π
    width_m: float
    left_approach: list[float]
    right_approach: list[float]
    left_squeeze: list[float]
    right_squeeze: list[float]
    left_grasp: list[float] = field(default_factory=list)
    right_grasp: list[float] = field(default_factory=list)
    edge_yaw_rad: float = 0.0
    grasp_ux: float = 0.0
    grasp_uy: float = 0.0
    p0_xy: list[float] = field(default_factory=list)
    p2_xy: list[float] = field(default_factory=list)
    p3_xy: list[float] = field(default_factory=list)
    mid_mode: str = "corner_p2p3_xy_offset"
    grasp_z_m: float | None = None
    tip_span_m: float | None = None
    open_margin_m: float = 0.0
    approach_dx_m: float = 0.10
    approach_dy_m: float = 0.08
    calib_pending: bool = True

    def as_dict(self) -> dict:
        return {
            "mid_mode": self.mid_mode,
            "p0_xy": [round(float(v), 5) for v in self.p0_xy[:2]] if self.p0_xy else [],
            "p2_xy": [round(float(v), 5) for v in self.p2_xy[:2]],
            "p3_xy": [round(float(v), 5) for v in self.p3_xy[:2]],
            "tip_left_xy": [round(float(self.tip_left[0]), 5), round(float(self.tip_left[1]), 5)],
            "tip_right_xy": [
                round(float(self.tip_right[0]), 5),
                round(float(self.tip_right[1]), 5),
            ],
            "tip_left": [round(float(v), 5) for v in self.tip_left],
            "tip_right": [round(float(v), 5) for v in self.tip_right],
            "approach_left_xy": [
                round(float(self.left_approach[0]), 5),
                round(float(self.left_approach[1]), 5),
            ],
            "approach_right_xy": [
                round(float(self.right_approach[0]), 5),
                round(float(self.right_approach[1]), 5),
            ],
            "grasp_z_m": None
            if self.grasp_z_m is None
            else round(float(self.grasp_z_m), 5),
            "tip_span_m": None
            if self.tip_span_m is None
            else round(float(self.tip_span_m), 5),
            "open_margin_m": round(float(self.open_margin_m), 4),
            "approach_dx_m": round(float(self.approach_dx_m), 4),
            "approach_dy_m": round(float(self.approach_dy_m), 4),
            "width_m": round(float(self.width_m), 4),
            "calib_pending": bool(self.calib_pending),
            "gripper_rz_deg": round(float(self.yaw_rad) * 180.0 / math.pi, 2),
            "edge_yaw_deg": round(float(self.edge_yaw_rad) * 180.0 / math.pi, 2),
            "grasp_ux": round(float(self.grasp_ux), 6),
            "grasp_uy": round(float(self.grasp_uy), 6),
            "left_approach": [round(float(v), 5) for v in self.left_approach],
            "right_approach": [round(float(v), 5) for v in self.right_approach],
            "left_squeeze": [round(float(v), 5) for v in self.left_squeeze],
            "right_squeeze": [round(float(v), 5) for v in self.right_squeeze],
            "tip_rule": (
                "左 tip=(P3.x+dx, P3.y+dy)；右 tip=(P2.x+dx, P2.y−dy)；"
                "默认 dx=15cm、dy=8cm（外开）；接近 Z=grasp_z+hover"
            ),
            "depth_rule": (
                "头定位：四角射线 ∩ 固定板面 Z=grasp_z_m；"
                "精定位：左腕看 P3、右腕看 P2，与头四角匹配后用精 2/3"
            ),
            "approach_rule": (
                "接近=抓取 tip（固定 dx/±dy）；合拢靠双臂位移 + 遇阻停"
            ),
            "center_xy": [
                round(0.5 * (float(self.tip_left[0]) + float(self.tip_right[0])), 5),
                round(0.5 * (float(self.tip_left[1]) + float(self.tip_right[1])), 5),
            ],
            "orient_rule": (
                "左 rx=-90 ry=0 rz；右 rx=+90 ry=0 rz_right=rz+180°；"
                "rz：夹爪中间线 ∥ P2→P3（23⊥03；旧法是 ⊥P0→P3）"
            ),
            "gripper_rz_right_deg": round(
                wrap_pi(float(self.yaw_rad) + math.pi) * 180.0 / math.pi, 2
            ),
        }


def _mid(a: Sequence[float], b: Sequence[float]) -> list[float]:
    return [
        0.5 * (float(a[0]) + float(b[0])),
        0.5 * (float(a[1]) + float(b[1])),
        0.5 * (float(a[2]) + float(b[2])),
    ]


def _unit_xy(dx: float, dy: float) -> tuple[float, float]:
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return 1.0, 0.0
    return dx / n, dy / n


def gripper_rz_large_parallel_p2p3(
    p2: Sequence[float],
    p3: Sequence[float],
    *,
    orient_rz_offset_deg: float = 0.0,
) -> float:
    """大板 rz：夹爪中间线 **∥ P2→P3**（与小板「∥边」同一套公式）。

    因 P2→P3 ⊥ P0→P3，等价于旧「中间线 ⊥ P0→P3」的几何关系，但直接用精定位 2/3。
    """
    rz = float(base_y_angle_to_gripper_rz(p2, p3))
    return wrap_pi(rz + math.radians(float(orient_rz_offset_deg or 0.0)))


def gripper_rz_large_midline_perp_p0p3(
    ux: float,
    uy: float,
    *,
    orient_rz_offset_deg: float = 0.0,
) -> float:
    """兼容旧接口：中间线 ∥ 法向 u（曾用于 ⊥P0→P3）。"""
    pa = (0.0, 0.0, 0.0)
    pb = (float(ux), float(uy), 0.0)
    rz = float(base_y_angle_to_gripper_rz(pa, pb))
    return wrap_pi(rz + math.radians(float(orient_rz_offset_deg or 0.0)))


def tips_from_p2_p3(
    corners_base: Sequence[Sequence[float]],
    *,
    grasp_z_m: float | None = None,
    approach_dx_m: float = 0.15,
    approach_dy_m: float = 0.08,
    orient_rz_offset_deg: float = 0.0,
    # 兼容旧调用（忽略）
    half_width_m: float | None = None,
    tip_span_m: float | None = None,
    orient_edge: str = "parallel_p2p3",
) -> tuple[list[float], list[float], float, float, float, float, float, dict]:
    """由精/粗 P2、P3 算左右 tip。

    左 tip = (P3.x+dx, P3.y+dy, z)；右 tip = (P2.x+dx, P2.y−dy, z)。
    基座系：左臂偏 +Y（外）、右臂偏 −Y（外）。
    rz = ∥ P2→P3。

    Returns tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta.
    """
    del half_width_m, tip_span_m, orient_edge
    if corners_base is None or len(corners_base) < 4:
        raise RuntimeError("large board needs corners P0..P3")
    p0, p1, p2, p3 = (corners_base[i] for i in range(4))
    mid = _mid(p2, p3)
    dx = float(approach_dx_m)
    dy = float(approach_dy_m)
    z = float(grasp_z_m) if grasp_z_m is not None else float(mid[2])
    tip_l = [float(p3[0]) + dx, float(p3[1]) + dy, z]
    tip_r = [float(p2[0]) + dx, float(p2[1]) - dy, z]
    width = float(math.hypot(tip_r[0] - tip_l[0], tip_r[1] - tip_l[1]))
    # 外开方向：左→右（约 −Y）
    ux, uy = _unit_xy(tip_r[0] - tip_l[0], tip_r[1] - tip_l[1])
    edge_yaw = math.atan2(
        float(p3[1]) - float(p2[1]), float(p3[0]) - float(p2[0])
    )
    gripper_rz = gripper_rz_large_parallel_p2p3(
        p2, p3, orient_rz_offset_deg=orient_rz_offset_deg
    )
    meta = {
        "mid_mode": "corner_p2p3_xy_offset",
        "p0_xy": [float(p0[0]), float(p0[1])],
        "p1_xy": [float(p1[0]), float(p1[1])],
        "p2_xy": [float(p2[0]), float(p2[1])],
        "p3_xy": [float(p3[0]), float(p3[1])],
        "mid23_xy": [float(mid[0]), float(mid[1])],
        "approach_dx_m": dx,
        "approach_dy_m": dy,
        "center_xy": [
            round(0.5 * (tip_l[0] + tip_r[0]), 5),
            round(0.5 * (tip_l[1] + tip_r[1]), 5),
        ],
        "span_source": "corner_xy_offset",
        "z_source": "calib_grasp_z_m" if grasp_z_m is not None else "vision_corner_z",
        "orient_source": "parallel_p2p3",
        "calib_pending": grasp_z_m is None,
        "grasp_z_m": None if grasp_z_m is None else float(grasp_z_m),
        "tip_span_m": float(width),
    }
    return tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta


# 旧名兼容
def tips_from_p0_p3(*args, **kwargs):
    return tips_from_p2_p3(*args, **kwargs)


def tip_to_flange(
    tip_xyz: Sequence[float],
    *,
    gripper_rz: float,
    grasp_ux: float,
    grasp_uy: float,
    side: str,
    tool_y_m: float,
    tool_z_m: float,
) -> list[float]:
    """夹手 tip → 末端位姿；左右从两侧合拢。

    - 左臂：``rpy = (-90°, 0, rz)``
    - 右臂：``rpy = (+90°, 0, rz+180°)``
    """
    side_l = str(side).lower().startswith("l")
    rz_l = wrap_pi(float(gripper_rz))
    if side_l:
        rx, rz = END_DOWN_RX, rz_l
    else:
        rx, rz = -END_DOWN_RX, wrap_pi(rz_l + math.pi)
    outward = -1.0 if side_l else 1.0
    fx = float(tip_xyz[0]) + outward * float(tool_y_m) * float(grasp_ux)
    fy = float(tip_xyz[1]) + outward * float(tool_y_m) * float(grasp_uy)
    fz = float(tip_xyz[2]) + float(tool_z_m)
    return [fx, fy, fz, float(rx), END_DOWN_RY, rz]


def plan_large_board_grasp(
    pose: ObjectPose,
    *,
    tool_y_m: float = 0.01,
    tool_z_m: float = 0.27,
    open_margin_m: float = 0.0,
    squeeze_close_m: float = 0.0,
    hover_z_m: float | None = None,
    half_width_m: float | None = None,
    grasp_z_m: float | None = None,
    tip_span_m: float | None = None,
    orient_edge: str = "parallel_p2p3",
    orient_rz_offset_deg: float = 0.0,
    approach_dx_m: float = 0.15,
    approach_dy_m: float = 0.08,
    approach_z_m: float | None = None,
) -> LargeBoardPlan:
    """精/粗 P2P3 → 双臂接近（相对角点 dx / ±dy）。

    ``approach_z_m``：接近停靠 tip Z（默认与抓取 tip 同高）；抓取 tip 仍用 ``grasp_z_m``。
    """
    tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta = tips_from_p2_p3(
        pose.corners_base,
        half_width_m=half_width_m,
        tip_span_m=tip_span_m,
        grasp_z_m=grasp_z_m,
        approach_dx_m=approach_dx_m,
        approach_dy_m=approach_dy_m,
        orient_edge=orient_edge,
        orient_rz_offset_deg=orient_rz_offset_deg,
    )
    if grasp_z_m is None and hover_z_m is not None:
        tip_z = float(hover_z_m) - float(tool_z_m)
        tip_l[2] = tip_z
        tip_r[2] = tip_z
        meta["z_source"] = "hover_z_minus_tool_z"
        meta["calib_pending"] = True

    # 新策略：接近位即 tip XY（dx/±dy 已含外开）；Z 可抬高停靠
    open_m = max(0.0, float(open_margin_m))
    close_m = max(0.0, float(squeeze_close_m))
    z = tip_l[2]
    z_app = float(approach_z_m) if approach_z_m is not None else z
    meta["approach_z_m"] = float(z_app)
    tip_l_app = [float(tip_l[0]) - ux * open_m, float(tip_l[1]) - uy * open_m, z_app]
    tip_r_app = [float(tip_r[0]) + ux * open_m, float(tip_r[1]) + uy * open_m, z_app]
    half_close = 0.5 * close_m
    tip_l_sq = [tip_l[0] + ux * half_close, tip_l[1] + uy * half_close, z]
    tip_r_sq = [tip_r[0] - ux * half_close, tip_r[1] - uy * half_close, z]

    def fl(tip: Sequence[float], side: str) -> list[float]:
        return tip_to_flange(
            tip,
            gripper_rz=gripper_rz,
            grasp_ux=ux,
            grasp_uy=uy,
            side=side,
            tool_y_m=tool_y_m,
            tool_z_m=tool_z_m,
        )

    return LargeBoardPlan(
        tip_left=list(tip_l),
        tip_right=list(tip_r),
        yaw_rad=wrap_pi(gripper_rz),
        width_m=float(width),
        left_approach=fl(tip_l_app, "left"),
        right_approach=fl(tip_r_app, "right"),
        left_squeeze=fl(tip_l_sq, "left"),
        right_squeeze=fl(tip_r_sq, "right"),
        left_grasp=fl(tip_l, "left"),
        right_grasp=fl(tip_r, "right"),
        edge_yaw_rad=float(edge_yaw),
        grasp_ux=float(ux),
        grasp_uy=float(uy),
        p0_xy=list(meta.get("p0_xy") or []),
        p2_xy=list(meta["p2_xy"]),
        p3_xy=list(meta["p3_xy"]),
        mid_mode=str(meta["mid_mode"]),
        grasp_z_m=meta.get("grasp_z_m"),
        tip_span_m=meta.get("tip_span_m"),
        open_margin_m=float(open_m),
        approach_dx_m=float(meta["approach_dx_m"]),
        approach_dy_m=float(meta["approach_dy_m"]),
        calib_pending=bool(meta.get("calib_pending", True)),
    )


def draw_large_board_plan(
    image_bgr: "np.ndarray",
    corners_uv: Sequence[Sequence[float]] | None,
    *,
    title: str = "LARGE dual-arm P2/P3",
    plan: LargeBoardPlan | None = None,
) -> "np.ndarray":
    import cv2
    import numpy as np

    vis = image_bgr.copy()
    if corners_uv is not None and len(corners_uv) >= 4:
        pts = [(int(round(float(u))), int(round(float(v)))) for u, v in corners_uv[:4]]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(vis, a, b, (0, 255, 0), 2)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 0, 255), -1)
            cv2.putText(
                vis,
                str(i),
                (p[0] + 4, p[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2,
            )
        # 高亮 2–3 边
        cv2.line(vis, pts[2], pts[3], (0, 165, 255), 3)
    cv2.putText(
        vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (50, 220, 255), 2
    )
    if plan is not None:
        cv2.putText(
            vis,
            f"rz={plan.yaw_rad * 180 / math.pi:.1f}  dx={plan.approach_dx_m:.2f} "
            f"dy=±{plan.approach_dy_m:.2f}",
            (12, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (200, 200, 50),
            2,
        )
    return vis
