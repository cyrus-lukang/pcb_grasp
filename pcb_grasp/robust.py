"""Multi-frame detection filter: drop MAD outliers, then median."""

from __future__ import annotations

import logging
import math
from typing import Sequence

import numpy as np

from .detect import PcbDetection
from .geometry import ObjectPose, wrap_deg

LOG = logging.getLogger("pcb_grasp.robust")


def _mad_keep(vals: Sequence[float], mad_k: float = 3.0) -> np.ndarray:
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        return np.array([], dtype=bool)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad < 1e-12:
        return np.ones(a.shape, dtype=bool)
    return np.abs(a - med) <= (mad_k * 1.4826 * mad)


def robust_median(vals: Sequence[float], mad_k: float = 3.0) -> tuple[float, int, int]:
    """Return (median_after_reject, n_kept, n_total)."""
    a = np.asarray(vals, dtype=np.float64)
    if a.size == 0:
        raise ValueError("empty values")
    keep = _mad_keep(a, mad_k)
    kept = a[keep]
    if kept.size == 0:
        kept = a
        keep = np.ones_like(a, dtype=bool)
    return float(np.median(kept)), int(keep.sum()), int(a.size)


def _circular_median_rad(angles: Sequence[float], mad_k: float = 3.0) -> float:
    """Median angle with unwrap around the first-pass circular mean."""
    a = np.asarray(angles, dtype=np.float64)
    # unit-vector mean as reference
    ref = math.atan2(float(np.mean(np.sin(a))), float(np.mean(np.cos(a))))
    unwrapped = np.arctan2(np.sin(a - ref), np.cos(a - ref))  # [-pi,pi] about ref
    med_off, _, _ = robust_median(unwrapped.tolist(), mad_k)
    return float(ref + med_off)


def median_detection(
    dets: Sequence[PcbDetection],
    *,
    mad_k: float = 3.0,
    n_corners: int = 4,
) -> PcbDetection:
    """Reject frame-level center outliers, then per-corner median u/v."""
    if not dets:
        raise ValueError("no detections to aggregate")
    if len(dets) == 1:
        return dets[0]

    cx = [d.center_uv[0] for d in dets]
    cy = [d.center_uv[1] for d in dets]
    keep = _mad_keep(cx, mad_k) & _mad_keep(cy, mad_k)
    kept = [d for d, ok in zip(dets, keep) if ok]
    if not kept:
        kept = list(dets)
    n_drop = len(dets) - len(kept)
    if n_drop:
        LOG.info("detect filter: drop %d/%d frame(s) as center outliers", n_drop, len(dets))

    corners: list[tuple[float, float]] = []
    for j in range(n_corners):
        us = [d.corners[j][0] for d in kept]
        vs = [d.corners[j][1] for d in kept]
        u, _, _ = robust_median(us, mad_k)
        v, _, _ = robust_median(vs, mad_k)
        corners.append((u, v))
    conf = float(np.median([d.conf for d in kept]))
    return PcbDetection(corners=corners, box=None, conf=conf)


def median_object_pose(
    poses: Sequence[ObjectPose],
    *,
    mad_k: float = 3.0,
) -> ObjectPose:
    """Reject XYZ outliers (and rz), then median pose for localization."""
    if not poses:
        raise ValueError("no poses to aggregate")
    if len(poses) == 1:
        return poses[0]

    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    zs = [p.z for p in poses]
    keep = _mad_keep(xs, mad_k) & _mad_keep(ys, mad_k) & _mad_keep(zs, mad_k)
    # also drop wild rz
    rzs = [p.rz for p in poses]
    ref = math.atan2(
        float(np.mean([math.sin(r) for r in rzs])),
        float(np.mean([math.cos(r) for r in rzs])),
    )
    rz_off = [math.atan2(math.sin(r - ref), math.cos(r - ref)) for r in rzs]
    keep = keep & _mad_keep(rz_off, mad_k)
    kept = [p for p, ok in zip(poses, keep) if ok]
    if not kept:
        kept = list(poses)
    n_drop = len(poses) - len(kept)
    if n_drop:
        LOG.info("pose filter: drop %d/%d sample(s) as outliers", n_drop, len(poses))

    x, _, _ = robust_median([p.x for p in kept], mad_k)
    y, _, _ = robust_median([p.y for p in kept], mad_k)
    z, _, _ = robust_median([p.z for p in kept], mad_k)
    rz = _circular_median_rad([p.rz for p in kept], mad_k)
    edge = _circular_median_rad([p.edge_yaw for p in kept], mad_k)
    conf = float(np.median([p.conf for p in kept]))

    # corners_base: median each corner coordinate
    n_c = len(kept[0].corners_base)
    corners_base: list[list[float]] = []
    for j in range(n_c):
        cj = []
        for dim in range(3):
            v, _, _ = robust_median([p.corners_base[j][dim] for p in kept], mad_k)
            cj.append(v)
        corners_base.append(cj)

    LOG.info(
        "pose median from %d/%d: xyz=(%.4f,%.4f,%.4f) rz=%.1fdeg",
        len(kept),
        len(poses),
        x,
        y,
        z,
        wrap_deg(rz * 180.0 / math.pi),
    )
    return ObjectPose(
        x=x,
        y=y,
        z=z,
        rz=rz,
        edge_yaw=edge,
        corners_base=corners_base,
        conf=conf,
        board_id=kept[0].board_id,
        cls_id=int(getattr(kept[0], "cls_id", 0) or 0),
    )


def pose_keep_mask(poses: Sequence[ObjectPose], mad_k: float = 3.0) -> list[bool]:
    """XYZ(+rz) MAD keep mask；全灭则全保留。"""
    if len(poses) <= 1:
        return [True] * len(poses)
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    zs = [p.z for p in poses]
    keep = _mad_keep(xs, mad_k) & _mad_keep(ys, mad_k) & _mad_keep(zs, mad_k)
    rzs = [p.rz for p in poses]
    ref = math.atan2(
        float(np.mean([math.sin(r) for r in rzs])),
        float(np.mean([math.cos(r) for r in rzs])),
    )
    rz_off = [math.atan2(math.sin(r - ref), math.cos(r - ref)) for r in rzs]
    keep = keep & _mad_keep(rz_off, mad_k)
    if not bool(np.any(keep)):
        return [True] * len(poses)
    return [bool(x) for x in keep]


def mean_pose_xyz(poses: Sequence[ObjectPose]) -> tuple[float, float, float]:
    if not poses:
        raise ValueError("empty poses")
    return (
        float(np.mean([p.x for p in poses])),
        float(np.mean([p.y for p in poses])),
        float(np.mean([p.z for p in poses])),
    )


def pick_closest_to_mean_pose(
    poses: Sequence[ObjectPose],
    *,
    mad_k: float = 3.0,
) -> tuple[ObjectPose, int, list[bool], tuple[float, float, float]]:
    """过滤异常值后，取最接近**平均位置**的那一帧真实样本（非合成中值）。

    Returns ``(pose, index, keep_mask, mean_xyz)``.
    """
    if not poses:
        raise ValueError("no poses")
    if len(poses) == 1:
        p0 = poses[0]
        return p0, 0, [True], (p0.x, p0.y, p0.z)

    keep = pose_keep_mask(poses, mad_k)
    kept = [p for p, ok in zip(poses, keep) if ok]
    if not kept:
        kept = list(poses)
        keep = [True] * len(poses)
    mx, my, mz = mean_pose_xyz(kept)
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(poses):
        if not keep[i]:
            continue
        d = (p.x - mx) ** 2 + (p.y - my) ** 2 + (p.z - mz) ** 2
        if d < best_d:
            best_i, best_d = i, d
    n_drop = len(poses) - sum(1 for ok in keep if ok)
    LOG.info(
        "pick closest-to-mean: keep %d/%d (drop %d) mean_xy=(%.4f,%.4f) "
        "→ frame#%d Δxy=%.1fmm",
        sum(1 for ok in keep if ok),
        len(poses),
        n_drop,
        mx,
        my,
        best_i + 1,
        math.hypot(poses[best_i].x - mx, poses[best_i].y - my) * 1000.0,
    )
    return poses[best_i], best_i, keep, (mx, my, mz)


def pick_closest_to_mean_xyz(
    points: Sequence[Sequence[float]],
    *,
    mad_k: float = 3.0,
) -> tuple[list[float], int, list[bool]]:
    """角点 XYZ 多帧：MAD 滤异常 → 取最接近均值的那次真实测量。"""
    if not points:
        raise ValueError("no points")
    if len(points) == 1:
        return [float(x) for x in points[0][:3]], 0, [True]
    xs = [float(p[0]) for p in points]
    ys = [float(p[1]) for p in points]
    zs = [float(p[2]) for p in points]
    keep = _mad_keep(xs, mad_k) & _mad_keep(ys, mad_k) & _mad_keep(zs, mad_k)
    if not bool(np.any(keep)):
        keep = np.ones(len(points), dtype=bool)
    kept_idx = [i for i, ok in enumerate(keep) if ok]
    mx = float(np.mean([xs[i] for i in kept_idx]))
    my = float(np.mean([ys[i] for i in kept_idx]))
    mz = float(np.mean([zs[i] for i in kept_idx]))
    best_i, best_d = kept_idx[0], float("inf")
    for i in kept_idx:
        d = (xs[i] - mx) ** 2 + (ys[i] - my) ** 2 + (zs[i] - mz) ** 2
        if d < best_d:
            best_i, best_d = i, d
    return [xs[best_i], ys[best_i], zs[best_i]], best_i, [bool(x) for x in keep]
