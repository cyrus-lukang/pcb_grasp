"""Wrist fine pose: outer-contour Chamfer gives (x,y,θ); rz 取轮廓方向.

YOLO 角点只作匹配初值 / 可视化；抓取 rz 一律用轮廓 ``match.angle_deg``。
"""

from __future__ import annotations

import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .detect import PcbDetection
from .geometry import (
    ObjectPose,
    base_y_angle_to_gripper_rz,
    pose_xyz_rpy_to_mat4,
    rz_from_line_xy,
    transform_point,
)
from .localize import Intrinsics, pixel_to_cam, sample_depth_m

LOG = logging.getLogger("pcb_grasp.shape_refine")

_TCT = Path(__file__).resolve().parents[1] / "TCT"
if str(_TCT) not in sys.path:
    sys.path.insert(0, str(_TCT))

from pose_outer_refine import (  # noqa: E402
    angle_from_kpt10,
    draw_pose_and_match,
    kpt_center,
    normalize_angle_deg,
    refine_outer_near_pose,
)
from shape_matcher import ShapeModel, silhouette_edges_for_viz  # noqa: E402


@dataclass
class WristMatchInfo:
    x: float
    y: float
    angle_deg: float
    scale: float
    score: float
    ok: bool
    angle_kpt_deg: float
    depth_m: float
    vis: np.ndarray | None = None


def load_shape_model(path: str | Path) -> ShapeModel:
    model = ShapeModel.load(path)
    # pose_outer_refine expects ShapeModel.outer(); shim adds it when imported via TCT
    if not hasattr(model, "outer"):
        model.outer = lambda: np.asarray(model.points, dtype=np.float32)  # type: ignore[attr-defined]
    return model


def _det_to_box(det: PcbDetection) -> dict:
    if det.box is None:
        xs = [c[0] for c in det.corners]
        ys = [c[1] for c in det.corners]
        x1, x2 = min(xs), max(xs)
        y1, y2 = min(ys), max(ys)
    else:
        x1, y1, x2, y2 = (float(v) for v in det.box[:4])
    kpts = np.asarray(det.corners, dtype=np.float64)
    cx, cy = kpt_center(kpts)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "w": max(1.0, x2 - x1),
        "h": max(1.0, y2 - y1),
        "cx": cx,
        "cy": cy,
        "conf": float(det.conf),
    }


def _depth_roi_median_m(
    depth: np.ndarray,
    *,
    depth_scale: float,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    z_min: float = 0.12,
    z_max: float = 2.5,
    min_count: int = 5,
) -> float | None:
    h, w = depth.shape[:2]
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(w, int(x2))
    y2 = min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth[y1:y2, x1:x2]
    if depth.dtype == np.float32 or depth.dtype == np.float64:
        vals = roi[(roi > z_min) & (roi < z_max)]
    else:
        vals = roi.astype(np.float64) * float(depth_scale)
        vals = vals[(vals > z_min) & (vals < z_max)]
    if vals.size < int(min_count):
        return None
    return float(np.median(vals))


def _depth_at_uv(
    depth: np.ndarray,
    u: float,
    v: float,
    *,
    depth_scale: float,
    box: dict | None = None,
    fallback_depth_m: float | None = None,
    force_fixed: bool = False,
) -> float:
    # 约定：粗定位悬停在板上方 hover_height，精定位 camZ 直接用该固定高度（不采腕深图）
    if force_fixed and fallback_depth_m is not None and float(fallback_depth_m) > 0.05:
        return float(fallback_depth_m)

    # 可选：读腕深图；空洞时再回退悬停高度
    for radius in (4, 8, 16, 28):
        z = sample_depth_m(
            depth,
            u,
            v,
            depth_scale=depth_scale,
            radius=radius,
            z_min=0.12,
            z_max=2.5,
        )
        if z is not None:
            return float(z)
    h, w = depth.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    for rad in (24, 48, 80):
        z = _depth_roi_median_m(
            depth,
            depth_scale=depth_scale,
            x1=ui - rad,
            y1=vi - rad,
            x2=ui + rad + 1,
            y2=vi + rad + 1,
            min_count=5,
        )
        if z is not None:
            return float(z)
    if box is not None:
        z = _depth_roi_median_m(
            depth,
            depth_scale=depth_scale,
            x1=int(box["x1"]),
            y1=int(box["y1"]),
            x2=int(box["x2"]),
            y2=int(box["y2"]),
            min_count=5,
        )
        if z is not None:
            return float(z)
    if fallback_depth_m is not None and float(fallback_depth_m) > 0.05:
        LOG.warning(
            "No wrist depth at uv=(%.1f,%.1f) → fallback camZ=%.3fm (hover)",
            u,
            v,
            float(fallback_depth_m),
        )
        return float(fallback_depth_m)
    raise RuntimeError(f"No wrist depth at match uv=({u:.1f},{v:.1f})")


def match_to_object_pose(
    *,
    u: float,
    v: float,
    angle_deg: float,
    T_tool_cam: np.ndarray,
    ee_xyz_rpy: Sequence[float],
    K: Intrinsics,
    depth_m: float,
    conf: float,
    tip_len_px: float = 40.0,
    det: object | None = None,
    kp_i: int = 0,
    kp_j: int = 1,
) -> ObjectPose:
    """Match 中心 (u,v) → 板心 XY；**rz 一律用轮廓 match 角** ``angle_deg``。

    YOLO 角点只写入 ``corners_base`` 供可视化 / ID，不再决定抓取 rz。
    """
    del kp_i, kp_j  # 保留签名兼容调用方；rz 不再走角点
    T_base_tool = pose_xyz_rpy_to_mat4(*[float(x) for x in ee_xyz_rpy[:6]])
    T_base_cam = T_base_tool @ np.asarray(T_tool_cam, dtype=np.float64)

    p_c = pixel_to_cam(float(u), float(v), float(depth_m), K)
    p_b = transform_point(T_base_cam, p_c)

    # 轮廓方向：图像角 → 像素 tip → base 边向 → gripper rz
    a = math.radians(float(angle_deg))
    tip_u = float(u) + tip_len_px * math.cos(a)
    tip_v = float(v) + tip_len_px * math.sin(a)
    p1_c = pixel_to_cam(tip_u, tip_v, float(depth_m), K)
    p1_b = transform_point(T_base_cam, p1_c)
    rz = float(base_y_angle_to_gripper_rz(p_b, p1_b))
    edge = float(rz_from_line_xy(p_b, p1_b))

    corners_base: list[list[float]] = []
    corners = getattr(det, "corners", None) if det is not None else None
    if corners is not None and len(corners) >= 2:
        for cu, cv in corners:
            pc = pixel_to_cam(float(cu), float(cv), float(depth_m), K)
            corners_base.append(transform_point(T_base_cam, pc).tolist())
    else:
        corners_base = [p_b.tolist(), p1_b.tolist(), p1_b.tolist(), p_b.tolist()]

    return ObjectPose(
        x=float(p_b[0]),
        y=float(p_b[1]),
        z=float(p_b[2]),
        rz=float(rz),
        edge_yaw=float(edge),
        corners_base=corners_base,
        conf=float(conf),
    )


def wrist_localize_shape_match(
    scene_bgr: np.ndarray,
    depth: np.ndarray,
    det: PcbDetection,
    *,
    shape_model: ShapeModel,
    T_tool_cam: np.ndarray,
    ee_xyz_rpy: Sequence[float],
    K: Intrinsics,
    depth_scale: float = 0.0001,
    fallback_depth_m: float | None = None,
    force_fixed_depth: bool = False,
    angle_offset_deg: float = 0.0,
    angle_half_range: float = 30.0,
    angle_step_coarse: float = 3.0,
    angle_step_fine: float = 0.5,
    xy_step_coarse: int = 4,
    xy_step_fine: int = 1,
    xy_radius: int = 12,
    score_thresh: float = 5.2,
    edge_mode: str = "outer-ring",
    roi_margin: float = 0.25,
    scale_rel: float = 0.2,
    scale_step: float = 0.04,
    max_coarse_scales: int = 8,
    coarse_points: int = 80,
    canny1: int = 40,
    canny2: int = 120,
    ring_width: int = 18,
    min_edge_len: float = 40.0,
    blur: int = 5,
    make_vis: bool = True,
    kp_i: int = 0,
    kp_j: int = 1,
) -> tuple[ObjectPose, WristMatchInfo]:
    """YOLO + outer Chamfer：XY / rz 都用轮廓 match（角点仅作搜索初值）。"""
    if len(det.corners) < 2:
        raise RuntimeError("Wrist shape-match needs ≥2 keypoints")

    kpts = np.asarray(det.corners, dtype=np.float64)
    box = _det_to_box(det)
    angle_kpt = angle_from_kpt10(kpts)
    # 角点角只作 Chamfer 搜索中心；最终 rz 用 match.angle_deg
    angle0 = normalize_angle_deg(angle_kpt + float(angle_offset_deg))

    match = refine_outer_near_pose(
        scene_bgr,
        shape_model,
        box,
        angle0_deg=angle0,
        angle_half_range=float(angle_half_range),
        angle_step_coarse=float(angle_step_coarse),
        angle_step_fine=float(angle_step_fine),
        xy_step_coarse=int(xy_step_coarse),
        xy_step_fine=int(xy_step_fine),
        xy_radius=int(xy_radius),
        score_thresh=float(score_thresh),
        edge_mode=str(edge_mode),
        roi_margin=float(roi_margin),
        scale_rel=float(scale_rel),
        scale_step=float(scale_step),
        max_coarse_scales=int(max_coarse_scales),
        coarse_points=int(coarse_points),
        canny1=int(canny1),
        canny2=int(canny2),
        ring_width=int(ring_width),
        min_edge_len=float(min_edge_len),
        blur=int(blur),
    )
    if match.score >= 1e8:
        raise RuntimeError(
            f"Wrist outer match failed score={match.score} "
            f"kpt_ang={angle_kpt:.1f} init={angle0:.1f}"
        )

    depth_m = _depth_at_uv(
        depth,
        match.x,
        match.y,
        depth_scale=depth_scale,
        box=box,
        fallback_depth_m=fallback_depth_m,
        force_fixed=bool(force_fixed_depth),
    )
    if force_fixed_depth and fallback_depth_m is not None:
        LOG.info(
            "wrist shape-match camZ fixed=%.3fm (hover) uv=(%.1f,%.1f)",
            float(depth_m),
            float(match.x),
            float(match.y),
        )
    tip = 0.35 * max(float(box["w"]), float(box["h"]))
    tip = float(np.clip(tip, 24.0, 80.0))
    pose = match_to_object_pose(
        u=match.x,
        v=match.y,
        angle_deg=match.angle_deg,
        T_tool_cam=T_tool_cam,
        ee_xyz_rpy=ee_xyz_rpy,
        K=K,
        depth_m=depth_m,
        conf=det.conf,
        tip_len_px=tip,
        det=det,
        kp_i=int(kp_i),
        kp_j=int(kp_j),
    )

    vis = None
    if make_vis:
        scene_e = silhouette_edges_for_viz(
            scene_bgr,
            det_box=box,
            roi_margin=roi_margin,
            canny1=canny1,
            canny2=canny2,
            blur=blur,
        )
        vis = draw_pose_and_match(
            scene_bgr, box, kpts, angle_kpt, shape_model, match, scene_e
        )

    info = WristMatchInfo(
        x=float(match.x),
        y=float(match.y),
        angle_deg=float(match.angle_deg),
        scale=float(match.scale),
        score=float(match.score),
        ok=bool(match.ok),
        angle_kpt_deg=float(angle_kpt),
        depth_m=float(depth_m),
        vis=vis,
    )
    LOG.info(
        "WRIST shape-match kpt=%.1f → contour_ang=%.1f → rz=%.1f° "
        "s=%.3f score=%.2f ok=%s uv=(%.1f,%.1f) depth=%.3fm (rz←contour)",
        info.angle_kpt_deg,
        info.angle_deg,
        float(pose.rz) * 180.0 / math.pi,
        info.scale,
        info.score,
        info.ok,
        info.x,
        info.y,
        info.depth_m,
    )
    return pose, info
