"""Large PCB dual-arm squeeze.

左 tip=基座 XY mid(P0,P3)（视觉左边中点）；右 tip=左+tip_span×法向。
竖直向下：左 ``rx=-90°``（同小板），右 ``rx=+90°``。
rz：夹爪中间线 ⊥ P0→P3；右臂 rz = 左 rz + 180°（保持 toolZ 平行，勿取反）。
小板逻辑不动。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence

from .geometry import (
    ObjectPose,
    PI,
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
    yaw_rad: float  # 左臂夹手 rz；右臂为 -yaw_rad
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
    p3_xy: list[float] = field(default_factory=list)
    mid_mode: str = "base_xy"  # 基座 XY 中点，非像素
    grasp_z_m: float | None = None
    tip_span_m: float | None = None
    open_margin_m: float = 0.05
    calib_pending: bool = True

    def as_dict(self) -> dict:
        return {
            "mid_mode": self.mid_mode,
            "p0_xy": [round(float(v), 5) for v in self.p0_xy[:2]],
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
                "左 tip=基座XY mid(P0,P3)（视觉左边中点）；"
                "右 tip=左+tip_span×法向（默认 47cm，右相对左固定间距）"
            ),
            "depth_rule": (
                "头定位：四角射线 ∩ 固定板面 Z=grasp_z_m（标定 tip/板面高）；"
                "不用 FFS 角点深度；未标定则暂用 head_tray_z_m"
            ),
            "approach_rule": (
                "同高 tipZ；左 tip 外开 open_margin、右 tip 外开 open_margin"
                "（默认各 5cm）；合拢间距仍为 tip_span（右=左+span）"
            ),
            "center_xy": [
                round(0.5 * (float(self.tip_left[0]) + float(self.tip_right[0])), 5),
                round(0.5 * (float(self.tip_left[1]) + float(self.tip_right[1])), 5),
            ],
            "approach_dist_to_center_m": round(
                0.5 * float(self.width_m) + float(self.open_margin_m), 5
            ),
            "orient_rule": (
                "左 rx=-90 ry=0 rz；右 rx=+90 ry=0 rz_right=rz+180° "
                "（板倾斜时左右 toolZ 仍平行；旧 rz=-左rz 只在近正放时碰巧对）；"
                "rz：夹爪中间线 ⊥ P0→P3（小板则 ∥ P0→P1）；"
                "合拢靠双臂位移 + squeeze_force_n 遇阻停（不收夹爪）"
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


def gripper_rz_large_midline_perp_p0p3(
    ux: float,
    uy: float,
    *,
    orient_rz_offset_deg: float = 0.0,
) -> float:
    """大板对夹 rz。末端与小板相同：``rx=-90°, ry=0``（竖直向下）。

    差别只在 rz：
    - 小板：夹爪中间线 **∥ P0→P1** → ``base_y_angle_to_gripper_rz(P0,P1)``
    - 大板：夹爪中间线 **⊥ P0→P3**（∥ 插入法向 u）
      → 同一套函数作用在方向 u 上，**不要**直接对 P0→P1 算（01/03 差 90°）
    """
    # 用两点表示「中间线应平行的方向」= 板内法向 u
    pa = (0.0, 0.0, 0.0)
    pb = (float(ux), float(uy), 0.0)
    rz = float(base_y_angle_to_gripper_rz(pa, pb))
    return wrap_pi(rz + math.radians(float(orient_rz_offset_deg or 0.0)))


def tips_from_p0_p3(
    corners_base: Sequence[Sequence[float]],
    *,
    half_width_m: float | None = None,
    tip_span_m: float | None = None,
    grasp_z_m: float | None = None,
    orient_edge: str = "perp_p0p3",
    orient_rz_offset_deg: float = 0.0,
) -> tuple[list[float], list[float], float, float, float, float, float, dict]:
    """左 tip 钉在视觉左边，右 tip = 左 + tip_span 沿法向。

    左 tip = 基座 XY ``mid(P0, P3)``（视觉左边中点）。
    右 tip = 左 tip + ``tip_span_m`` × 法向（指向右边 mid(P1,P2)）。
    实测夹持宽约 43cm 时，全流程目标间距用 47cm，给右臂留外开余量避免撞边。

    ``tip_span_m``：两 tip 固定间距（右相对左）；优先于视觉宽度 / half_width。
    ``grasp_z_m``：固定 tip/板面高度；头定位平面与 tip Z 同源。
    rz：夹爪中间线 ⊥ P0→P3。

    Returns tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta.
    """
    del orient_edge  # 大板 rz 固定按「中间线⊥03」；保留参数兼容旧调用
    if corners_base is None or len(corners_base) < 4:
        raise RuntimeError("large board needs corners P0..P3")
    p0, p1, p2, p3 = (corners_base[i] for i in range(4))
    # 视觉左/右边中点（基座 XY）
    edge_l = _mid(p0, p3)
    edge_r = _mid(p1, p2)
    edge_r = [edge_r[0], edge_r[1], edge_l[2]]

    ex, ey = _unit_xy(float(p3[0]) - float(p0[0]), float(p3[1]) - float(p0[1]))
    edge_yaw = math.atan2(ey, ex)
    nx, ny = -ey, ex
    # 法向指向右（朝对边）
    to_r = (
        float(edge_r[0]) - float(edge_l[0]),
        float(edge_r[1]) - float(edge_l[1]),
    )
    if nx * to_r[0] + ny * to_r[1] < 0.0:
        nx, ny = -nx, -ny

    vis_width = float(math.hypot(to_r[0], to_r[1]))
    span_src = "vision_mid_p1p2"
    width = vis_width

    if tip_span_m is not None and float(tip_span_m) > 0.05:
        width = float(tip_span_m)
        span_src = "calib_tip_span_m"
    elif half_width_m is not None and float(half_width_m) > 0.02:
        width = 2.0 * float(half_width_m)
        span_src = "calib_half_width_m"
    elif width < 0.05:
        raise RuntimeError(f"large board tip span too small: {width:.3f}m")

    # 左 tip = 视觉左边中点；右 tip = 左 + span×法向（间距固定 tip_span）
    tip_l = [float(edge_l[0]), float(edge_l[1]), float(edge_l[2])]
    tip_r = [
        float(tip_l[0]) + float(width) * nx,
        float(tip_l[1]) + float(width) * ny,
        float(tip_l[2]),
    ]
    ux, uy = nx, ny
    cx = 0.5 * (float(edge_l[0]) + float(edge_r[0]))
    cy = 0.5 * (float(edge_l[1]) + float(edge_r[1]))

    # 到视觉边的法向距离：左 tip 贴左边 → d_l≈0；右 tip 相对右边
    d_l = (float(edge_l[0]) - tip_l[0]) * nx + (float(edge_l[1]) - tip_l[1]) * ny
    d_r = (tip_r[0] - float(edge_r[0])) * nx + (tip_r[1] - float(edge_r[1])) * ny

    gripper_rz = gripper_rz_large_midline_perp_p0p3(
        ux, uy, orient_rz_offset_deg=orient_rz_offset_deg
    )
    orient_src = "midline_perp_p0p3"

    z_src = "vision_corner_z"
    if grasp_z_m is not None:
        tip_l[2] = float(grasp_z_m)
        tip_r[2] = float(grasp_z_m)
        z_src = "calib_grasp_z_m"

    meta = {
        "mid_mode": "base_xy_left_edge_plus_span",
        "p0_xy": [float(p0[0]), float(p0[1])],
        "p3_xy": [float(p3[0]), float(p3[1])],
        "p1_xy": [float(p1[0]), float(p1[1])],
        "edge_left_xy": [float(edge_l[0]), float(edge_l[1])],
        "edge_right_xy": [float(edge_r[0]), float(edge_r[1])],
        "center_xy": [round(cx, 5), round(cy, 5)],
        "vis_width_m": round(vis_width, 5),
        "tip_to_edge_left_m": round(float(d_l), 5),
        "tip_to_edge_right_m": round(float(d_r), 5),
        "span_source": span_src,
        "z_source": z_src,
        "orient_source": orient_src,
        "calib_pending": span_src.startswith("vision") or z_src.startswith("vision"),
        "grasp_z_m": None if grasp_z_m is None else float(grasp_z_m),
        "tip_span_m": None if tip_span_m is None else float(tip_span_m),
    }
    return tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta


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

    - 左臂竖直向下：``rpy = (-90°, 0, rz)``（同小板）
    - 右臂竖直向下：``rpy = (+90°, 0, rz+180°)``
      （rx 相反；rz 相对左臂 **+180°**，不是取反——取反只在板近正放时碰巧平行，
      板一斜 toolZ 就叉开，看起来右臂还停在旧方向）
    rz 基准：夹爪中间线 ⊥ P0→P3（左右同一 ``gripper_rz``）。
    """
    side_l = str(side).lower().startswith("l")
    rz_l = wrap_pi(float(gripper_rz))
    if side_l:
        rx, rz = END_DOWN_RX, rz_l
    else:
        # 右臂：rx=+90°；rz = 左 rz + π → 任意倾斜下 toolZ 与左平行
        rx, rz = -END_DOWN_RX, wrap_pi(rz_l + math.pi)
    outward = -1.0 if side_l else 1.0
    fx = float(tip_xyz[0]) + outward * float(tool_y_m) * float(grasp_ux)
    fy = float(tip_xyz[1]) + outward * float(tool_y_m) * float(grasp_uy)
    # tipZ + tool_z → TCP Z（TCP=tip 时 tool_z=0）
    fz = float(tip_xyz[2]) + float(tool_z_m)
    return [fx, fy, fz, float(rx), END_DOWN_RY, rz]


def plan_large_board_grasp(
    pose: ObjectPose,
    *,
    tool_y_m: float = 0.01,
    tool_z_m: float = 0.27,
    open_margin_m: float = 0.05,
    squeeze_close_m: float = 0.0,
    hover_z_m: float | None = None,
    half_width_m: float | None = None,
    grasp_z_m: float | None = None,
    tip_span_m: float | None = None,
    orient_edge: str = "p0p1",
    orient_rz_offset_deg: float = 0.0,
) -> LargeBoardPlan:
    """头定位 → 双臂平行；固定高度 / 固定 tip 间距停下；对挤靠阻抗遇阻停。"""
    tip_l, tip_r, width, gripper_rz, edge_yaw, ux, uy, meta = tips_from_p0_p3(
        pose.corners_base,
        half_width_m=half_width_m,
        tip_span_m=tip_span_m,
        grasp_z_m=grasp_z_m,
        orient_edge=orient_edge,
        orient_rz_offset_deg=orient_rz_offset_deg,
    )
    # 兼容旧 hover_z（法兰高度语义）→ tip Z
    if grasp_z_m is None and hover_z_m is not None:
        tip_z = float(hover_z_m) - float(tool_z_m)
        tip_l[2] = tip_z
        tip_r[2] = tip_z
        meta["z_source"] = "hover_z_minus_tool_z"
        meta["calib_pending"] = True

    open_m = max(0.0, float(open_margin_m))
    # 已标定 tip_span 时默认不再内收（避免挤板）；未标定才允许 squeeze_close
    close_m = max(0.0, float(squeeze_close_m))
    if tip_span_m is not None and float(tip_span_m) > 0.05:
        close_m = 0.0
    z = tip_l[2]
    # 接近：相对 tip 各外开 open（左 tip 在左边上，外开后左臂离边有空；右同理）
    tip_l_app = [float(tip_l[0]) - ux * open_m, float(tip_l[1]) - uy * open_m, z]
    tip_r_app = [float(tip_r[0]) + ux * open_m, float(tip_r[1]) + uy * open_m, z]
    meta["approach_from"] = "left_edge_plus_span_then_open"
    meta["approach_clearance_m"] = float(open_m)
    tip_mid = [
        0.5 * (float(tip_l[0]) + float(tip_r[0])),
        0.5 * (float(tip_l[1]) + float(tip_r[1])),
    ]
    meta["approach_dist_to_center_m"] = float(
        math.hypot(tip_l_app[0] - tip_mid[0], tip_l_app[1] - tip_mid[1])
    )
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
        p0_xy=list(meta["p0_xy"]),
        p3_xy=list(meta["p3_xy"]),
        mid_mode=str(meta["mid_mode"]),
        grasp_z_m=meta.get("grasp_z_m"),
        tip_span_m=meta.get("tip_span_m"),
        open_margin_m=float(open_m),
        calib_pending=bool(meta.get("calib_pending", True)),
    )


def draw_large_board_plan(
    image_bgr: "np.ndarray",
    corners_uv: Sequence[Sequence[float]] | None,
    *,
    title: str = "LARGE dual-arm parallel insert",
) -> "np.ndarray":
    """画 03 边、对边 tip、平行姿态箭头、直线插入方向。"""
    import cv2
    import numpy as np

    vis = image_bgr.copy()
    cv2.putText(
        vis, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )
    if corners_uv is None or len(corners_uv) < 4:
        return vis
    pts = [(int(round(float(u))), int(round(float(v)))) for u, v in corners_uv[:4]]
    # 四边
    for a, b in zip(pts, pts[1:] + pts[:1]):
        cv2.line(vis, a, b, (0, 180, 180), 1)
    # 加粗 03 边 + 对边 12
    cv2.line(vis, pts[0], pts[3], (0, 255, 255), 3)
    cv2.line(vis, pts[1], pts[2], (0, 200, 100), 3)
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 6, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"P{i}",
            (p[0] + 4, p[1] - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
        )

    tip_l = (
        0.5 * (pts[0][0] + pts[3][0]),
        0.5 * (pts[0][1] + pts[3][1]),
    )
    tip_r = (
        0.5 * (pts[1][0] + pts[2][0]),
        0.5 * (pts[1][1] + pts[2][1]),
    )
    li = (int(round(tip_l[0])), int(round(tip_l[1])))
    ri = (int(round(tip_r[0])), int(round(tip_r[1])))

    dx = float(ri[0] - li[0])
    dy = float(ri[1] - li[1])
    n = math.hypot(dx, dy) or 1.0
    ux, uy = dx / n, dy / n
    # 边向（03）单位，用于画「平行姿态」箭头（同方向）
    ex = float(pts[3][0] - pts[0][0])
    ey = float(pts[3][1] - pts[0][1])
    en = math.hypot(ex, ey) or 1.0
    ex, ey = ex / en, ey / en

    # 插入轴
    cv2.line(vis, li, ri, (0, 128, 255), 2)
    # 直线插入：左沿 +u 进 03，右沿 -u 进对边（短箭头）
    ins = 36.0
    l_in = (int(round(li[0] + ux * ins)), int(round(li[1] + uy * ins)))
    r_in = (int(round(ri[0] - ux * ins)), int(round(ri[1] - uy * ins)))
    cv2.arrowedLine(vis, li, l_in, (0, 0, 255), 3, tipLength=0.35)
    cv2.arrowedLine(vis, ri, r_in, (0, 0, 255), 3, tipLength=0.35)

    # 平行姿态：两 tip 上画同向边切向箭头（表示两臂 rz 相同）
    tang = 40.0
    for c in (li, ri):
        a0 = (int(round(c[0] - ex * tang)), int(round(c[1] - ey * tang)))
        a1 = (int(round(c[0] + ex * tang)), int(round(c[1] + ey * tang)))
        cv2.arrowedLine(vis, a0, a1, (255, 128, 0), 2, tipLength=0.3)

    cv2.circle(vis, li, 9, (0, 0, 255), 2)
    cv2.circle(vis, ri, 9, (0, 0, 255), 2)
    cv2.putText(
        vis, "tipL insert03", (li[0] - 40, li[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
    )
    cv2.putText(
        vis, "tipR insert||", (ri[0] + 8, ri[1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2,
    )
    cv2.putText(
        vis, "orange=parallel EE", (8, 54),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 128, 0), 2,
    )
    return vis
