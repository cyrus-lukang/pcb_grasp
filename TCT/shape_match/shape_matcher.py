#!/usr/bin/env python3
"""
2D shape (silhouette) template matching for planar PCB pose (x, y, theta).

OpenCV chamfer-matching style PoC (no Halcon). RGB / masked crop / binary mask.

Prefer the local venv (Python 3.12). System `python3` on macOS is often 3.9
and will fail against packages built for 3.12:

  ./run.sh build --template ... --from-mask --out models/d435_pcb.npz
  ./run.sh match --model models/d435_pcb.npz --image ... --edge-mode canny --viz out.jpg

Or:  .venv/bin/python shape_matcher.py ...
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

if sys.version_info < (3, 10):
    raise SystemExit(
        f"Need Python >= 3.10, got {sys.version.split()[0]} ({sys.executable}).\n"
        "Use:  ./run.sh ...   or   .venv/bin/python shape_matcher.py ..."
    )

import cv2
import numpy as np

from handeye_base import (
    DEFAULT_HANDEYE,
    DEFAULT_INTRINSICS,
    angle_diff_deg,
    load_T_tool_cam,
    load_intrinsics,
    match_to_base,
    silhouette_ref_angle_deg,
    wrap_angle_deg,
)


@dataclass
class ShapeModel:
    """Edge-point shape model in template coordinates (origin = silhouette centroid)."""

    points: np.ndarray  # (N, 2) float32, relative to centroid
    angle_step_deg: float
    size: tuple[int, int]  # (w, h) of template image used to build

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            points=self.points.astype(np.float32),
            angle_step_deg=np.float32(self.angle_step_deg),
            size=np.array(self.size, dtype=np.int32),
        )

    @staticmethod
    def load(path: str | Path) -> "ShapeModel":
        data = np.load(path)
        return ShapeModel(
            points=data["points"].astype(np.float32),
            angle_step_deg=float(data["angle_step_deg"]),
            size=(int(data["size"][0]), int(data["size"][1])),
        )


@dataclass
class MatchResult:
    x: float  # image x of template centroid
    y: float  # image y of template centroid
    angle_deg: float  # rotation applied to model points
    score: float  # lower is better (mean chamfer distance, px)
    ok: bool
    scale: float = 1.0


def load_bgr(path: str | Path) -> np.ndarray:
    """Load image; composite alpha onto black so silhouette is recoverable."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        bgr = img[:, :, :3].astype(np.float32)
        a = img[:, :, 3:4].astype(np.float32) / 255.0
        return (bgr * a).astype(np.uint8)
    return img


def load_mask(path: str | Path) -> np.ndarray:
    """Load binary / soft mask as uint8 0/255 single-channel."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(path)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = img[:, :, 3]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    return bw


def foreground_mask(bgr: np.ndarray) -> np.ndarray:
    """
    Binary FG mask for masked PCB crops.
    Supports: alpha-composited black bg, near-black jpg mask, else Otsu.
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    _, fg = cv2.threshold(gray, 12, 255, cv2.THRESH_BINARY)
    ratio = float((fg > 0).mean())
    if ratio < 0.02 or ratio > 0.98:
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if (fg[0, 0] > 0) and (fg > 0).mean() > 0.5:
            fg = 255 - fg
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return fg


def extract_silhouette_edges(bgr: np.ndarray) -> np.ndarray:
    """Largest external contour as 1px edge map (stable PCB outline)."""
    fg = foreground_mask(bgr)
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    edges = np.zeros(fg.shape, dtype=np.uint8)
    if not contours:
        return edges
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:
        return edges
    cv2.drawContours(edges, [largest], -1, 255, 1)
    return edges


def extract_canny_edges(
    bgr: np.ndarray, canny1: int = 40, canny2: int = 120
) -> np.ndarray:
    """Canny edges for full RGB scenes (tray / clutter)."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.Canny(blur, canny1, canny2)


def extract_scene_edges(bgr: np.ndarray, mode: str = "auto") -> np.ndarray:
    """
    auto: silhouette if a clear masked FG exists, else Canny.
    silhouette / canny: force that mode.
    """
    if mode == "canny":
        return extract_canny_edges(bgr)
    if mode == "silhouette":
        return extract_silhouette_edges(bgr)
    fg = foreground_mask(bgr)
    ratio = float((fg > 0).mean())
    if 0.005 < ratio < 0.5:
        edges = extract_silhouette_edges(bgr)
        if np.any(edges):
            return edges
    return extract_canny_edges(bgr)


def edge_points(edge_map: np.ndarray, max_points: int = 600) -> np.ndarray:
    ys, xs = np.where(edge_map > 0)
    if len(xs) == 0:
        return np.zeros((0, 2), dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    if len(pts) > max_points:
        idx = np.linspace(0, len(pts) - 1, max_points).astype(int)
        pts = pts[idx]
    return pts


def build_model(
    template_bgr: np.ndarray,
    angle_step_deg: float = 2.0,
    max_points: int = 600,
) -> ShapeModel:
    edges = extract_silhouette_edges(template_bgr)
    pts = edge_points(edges, max_points=max_points)
    if len(pts) < 30:
        raise ValueError(
            f"Too few silhouette points ({len(pts)}). "
            "Check that the template is a masked crop (alpha or black bg)."
        )
    fg = foreground_mask(template_bgr)
    m = cv2.moments(fg, binaryImage=True)
    if m["m00"] > 1e-3:
        centroid = np.array(
            [m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32
        )
    else:
        centroid = pts.mean(axis=0)
    rel = pts - centroid
    h, w = template_bgr.shape[:2]
    return ShapeModel(
        points=rel.astype(np.float32),
        angle_step_deg=angle_step_deg,
        size=(w, h),
    )


def build_model_from_mask(
    mask: np.ndarray,
    angle_step_deg: float = 2.0,
    max_points: int = 600,
) -> ShapeModel:
    """Build shape model from a binary PCB mask (full-frame or crop)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("No contour in mask")
    largest = max(contours, key=cv2.contourArea)
    edges = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(edges, [largest], -1, 255, 1)
    pts = edge_points(edges, max_points=max_points)
    if len(pts) < 30:
        raise ValueError(f"Too few silhouette points ({len(pts)}) in mask")
    m = cv2.moments(mask, binaryImage=True)
    if m["m00"] > 1e-3:
        centroid = np.array(
            [m["m10"] / m["m00"], m["m01"] / m["m00"]], dtype=np.float32
        )
    else:
        centroid = pts.mean(axis=0)
    rel = pts - centroid
    h, w = mask.shape[:2]
    return ShapeModel(
        points=rel.astype(np.float32),
        angle_step_deg=angle_step_deg,
        size=(w, h),
    )


def _rotate_points(points: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    r = np.array([[c, -s], [s, c]], dtype=np.float32)
    return points @ r.T


def _chamfer_score(points: np.ndarray, dist: np.ndarray, h: int, w: int) -> float:
    if len(points) == 0:
        return 1e9
    xy = np.round(points).astype(np.int32)
    inside = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    if inside.sum() < max(20, len(points) // 4):
        return 1e9
    xy = xy[inside]
    return float(dist[xy[:, 1], xy[:, 0]].mean())


def model_aabb_size(model: ShapeModel) -> tuple[float, float]:
    """Axis-aligned width/height of template silhouette points."""
    pts = model.points
    return float(pts[:, 0].max() - pts[:, 0].min()), float(
        pts[:, 1].max() - pts[:, 1].min()
    )


def scale_from_det_box(model: ShapeModel, box_w: float, box_h: float) -> float:
    """
    Nominal scale from detection AABB vs model AABB (axis-aligned diagonal ratio).
    For rotated boards the YOLO box is larger; use scales_from_det_box() to search.
    """
    mw, mh = model_aabb_size(model)
    model_diag = float(np.hypot(mw, mh))
    det_diag = float(np.hypot(box_w, box_h))
    if model_diag < 1e-3:
        return 1.0
    return float(np.clip(det_diag / model_diag, 0.2, 4.0))


def scales_from_det_box(
    model: ShapeModel,
    box_w: float,
    box_h: float,
    rel: float = 0.08,
    step: float = 0.04,
) -> tuple[float, ...]:
    """
    Tight scale band around det-box size.

    Uses mean of (box_w/model_w, box_h/model_h). YOLO AABB grows when the
    board is rotated, so keep only a small ±rel search — do NOT widen down
    toward "45° correction", or chamfer will latch onto internal chip edges
    with an undersized template.
    """
    mw, mh = model_aabb_size(model)
    if mw < 1e-3 or mh < 1e-3:
        return (1.0,)
    s0 = 0.5 * (box_w / mw + box_h / mh)
    s0 = float(np.clip(s0, 0.2, 4.0))
    lo = max(0.2, s0 * (1.0 - rel))
    hi = s0 * (1.0 + rel)
    # `step` is absolute scale step (e.g. 0.04 → …, 1.66, 1.70, 1.74, …)
    step = max(1e-4, float(step))
    return tuple(np.round(np.arange(lo, hi + 1e-6, step), 4))


def parse_yolo_labels(
    path: str | Path, img_w: int, img_h: int
) -> list[dict]:
    """
    Parse YOLO-format labels: class cx cy w h [conf] (normalized).
    Returns pixel boxes: {cls, x1, y1, x2, y2, cx, cy, w, h, conf}.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    boxes: list[dict] = []
    for line in path.read_text().strip().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        cx_n, cy_n, w_n, h_n = map(float, parts[1:5])
        conf = float(parts[5]) if len(parts) >= 6 else 1.0
        bw, bh = w_n * img_w, h_n * img_h
        cx, cy = cx_n * img_w, cy_n * img_h
        x1 = int(round(cx - bw / 2))
        y1 = int(round(cy - bh / 2))
        x2 = int(round(cx + bw / 2))
        y2 = int(round(cy + bh / 2))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)
        boxes.append(
            {
                "cls": cls,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "cx": cx,
                "cy": cy,
                "w": bw,
                "h": bh,
                "conf": conf,
            }
        )
    return boxes


def expand_box(
    x1: int, y1: int, x2: int, y2: int, img_w: int, img_h: int, margin: float = 0.25
) -> tuple[int, int, int, int]:
    """Expand AABB by relative margin (of box size), clamped to image."""
    # YOLO / float boxes must become int pixel indices before slicing images.
    x1_i, y1_i = int(round(float(x1))), int(round(float(y1)))
    x2_i, y2_i = int(round(float(x2))), int(round(float(y2)))
    bw, bh = x2_i - x1_i, y2_i - y1_i
    dx, dy = int(bw * margin), int(bh * margin)
    return (
        max(0, x1_i - dx),
        max(0, y1_i - dy),
        min(int(img_w), x2_i + dx),
        min(int(img_h), y2_i + dy),
    )


def scales_around(scale0: float, rel: float = 0.12, step: float = 0.04) -> tuple[float, ...]:
    """Small scale band around det-estimated scale0."""
    lo = max(0.2, scale0 * (1.0 - rel))
    hi = scale0 * (1.0 + rel)
    return tuple(np.round(np.arange(lo, hi + 1e-6, step), 4))


def _prepare_dist(edges: np.ndarray, pad: int) -> tuple[np.ndarray, np.ndarray, int, int]:
    edges_p = cv2.copyMakeBorder(edges, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
    dist = cv2.distanceTransform(255 - edges_p, cv2.DIST_L2, 5)
    hp, wp = edges_p.shape[:2]
    return dist, edges_p, hp, wp


def _score_pose(
    model_pts: np.ndarray,
    scale: float,
    ang: float,
    cx: float,
    cy: float,
    dist: np.ndarray,
    hp: int,
    wp: int,
) -> float:
    rot = _rotate_points(model_pts * float(scale), float(ang))
    world = rot + np.array([cx, cy], dtype=np.float32)
    return _chamfer_score(world, dist, hp, wp)


def match_with_det_box(
    scene_bgr: np.ndarray,
    model: ShapeModel,
    box: dict,
    angle_min: float = -180.0,
    angle_max: float = 180.0,
    score_thresh: float = 3.0,
    edge_mode: str = "silhouette",
    roi_margin: float = 0.25,
    scale_rel: float = 0.08,
    scale_step: float = 0.04,
    angle_step_coarse: float = 10.0,
    angle_step_fine: float = 2.0,
) -> MatchResult:
    """
    Fast det-box guided match:
      1) ROI + silhouette edges
      2) coarse angle (default 10°) at det-box center, few scales
      3) fine angle/scale/xy around best — no full matchTemplate grid
    """
    h, w = scene_bgr.shape[:2]
    scales = scales_from_det_box(
        model, box["w"], box["h"], rel=scale_rel, step=scale_step
    )
    rx1, ry1, rx2, ry2 = expand_box(
        box["x1"], box["y1"], box["x2"], box["y2"], w, h, margin=roi_margin
    )
    work = scene_bgr[ry1:ry2, rx1:rx2]
    mode = "silhouette" if edge_mode == "auto" else edge_mode
    edges = extract_scene_edges(work, mode=mode)
    if not np.any(edges) and mode == "silhouette":
        edges = extract_scene_edges(work, mode="canny")
    if not np.any(edges):
        return MatchResult(0, 0, 0, 1e9, False, 1.0)

    # Subsample points for speed; keep enough for stable chamfer
    pts_full = model.points
    if len(pts_full) > 200:
        idx = np.linspace(0, len(pts_full) - 1, 200).astype(int)
        pts_fast = pts_full[idx]
    else:
        pts_fast = pts_full

    rad = float(np.linalg.norm(pts_full, axis=1).max()) * float(max(scales))
    pad = int(rad + 8)
    dist, _, hp, wp = _prepare_dist(edges, pad)

    # Det center in padded ROI coords
    cx0 = float(box["cx"] - rx1 + pad)
    cy0 = float(box["cy"] - ry1 + pad)

    best = MatchResult(0, 0, 0, 1e9, False, float(scales[len(scales) // 2]))

    # --- coarse: angle × scale at box center (+ small xy) ---
    coarse_angles = np.arange(angle_min, angle_max + 1e-6, angle_step_coarse)
    for scale in scales:
        for ang in coarse_angles:
            for dx, dy in ((0, 0), (-4, 0), (4, 0), (0, -4), (0, 4)):
                cx, cy = cx0 + dx, cy0 + dy
                score = _score_pose(pts_fast, scale, ang, cx, cy, dist, hp, wp)
                if score < best.score:
                    best = MatchResult(
                        float(cx - pad),
                        float(cy - pad),
                        float(ang),
                        float(score),
                        False,
                        float(scale),
                    )

    # --- fine: ±coarse around best angle, local scale/xy ---
    if best.score < 1e8:
        fine_lo = best.angle_deg - angle_step_coarse
        fine_hi = best.angle_deg + angle_step_coarse
        fine_angles = np.arange(fine_lo, fine_hi + 1e-6, angle_step_fine)
        # also try 180° flip (near-rect boards)
        extra = [best.angle_deg + 180.0, best.angle_deg - 180.0]
        fine_angles = np.unique(np.concatenate([fine_angles, extra]))
        s_lo = best.scale * (1.0 - min(0.1, scale_rel))
        s_hi = best.scale * (1.0 + min(0.1, scale_rel))
        fine_scales = tuple(
            np.round(np.arange(s_lo, s_hi + 1e-6, max(0.02, scale_step * 0.5)), 4)
        )
        seed = best
        for scale in fine_scales:
            for ang in fine_angles:
                for dx in (-6, -3, 0, 3, 6):
                    for dy in (-6, -3, 0, 3, 6):
                        cx = seed.x + pad + dx
                        cy = seed.y + pad + dy
                        score = _score_pose(
                            pts_fast, scale, ang, cx, cy, dist, hp, wp
                        )
                        if score < best.score:
                            best = MatchResult(
                                float(cx - pad),
                                float(cy - pad),
                                float(ang),
                                float(score),
                                False,
                                float(scale),
                            )
        # final score with full point set
        best.score = _score_pose(
            pts_full,
            best.scale,
            best.angle_deg,
            best.x + pad,
            best.y + pad,
            dist,
            hp,
            wp,
        )

    # Map pose back to full image
    best.x += float(rx1)
    best.y += float(ry1)

    best.ok = best.score <= score_thresh
    return best


def match_model(
    scene_bgr: np.ndarray,
    model: ShapeModel,
    angle_min: float = -180.0,
    angle_max: float = 180.0,
    score_thresh: float = 3.0,
    scales: tuple[float, ...] | None = None,
    edge_mode: str = "auto",
    angle_step: float | None = None,
) -> MatchResult:
    """
    Chamfer matching over scale + angle + translation (coarse-to-fine).
    """
    edges = extract_scene_edges(scene_bgr, mode=edge_mode)
    if not np.any(edges):
        return MatchResult(0, 0, 0, 1e9, False, 1.0)

    if scales is None:
        scales = tuple(np.round(np.arange(0.5, 2.0 + 1e-6, 0.1), 3))

    coarse = float(angle_step if angle_step is not None else max(8.0, model.angle_step_deg * 4))
    fine = float(model.angle_step_deg)

    pts_full = model.points
    if len(pts_full) > 200:
        idx = np.linspace(0, len(pts_full) - 1, 200).astype(int)
        pts_fast = pts_full[idx]
    else:
        pts_fast = pts_full

    rad0 = float(np.linalg.norm(pts_full, axis=1).max())
    max_rad = rad0 * max(scales)
    pad = int(max_rad + 8)
    dist, _, hp, wp = _prepare_dist(edges, pad)

    # Use matchTemplate only on coarse angles (fewer); refine locally
    inv = np.exp(-dist / 2.5).astype(np.float32)
    coarse_angles = np.arange(angle_min, angle_max + 1e-6, coarse)
    best = MatchResult(0, 0, 0, 1e9, False, 1.0)

    for scale in scales:
        pts = pts_fast * float(scale)
        rad = float(np.linalg.norm(pts, axis=1).max())
        tw = th = int(2 * rad + 16)
        shift = np.array([tw / 2.0, th / 2.0], dtype=np.float32)
        if hp < th or wp < tw:
            continue
        for ang in coarse_angles:
            rot = _rotate_points(pts, float(ang))
            patch = np.zeros((th, tw), dtype=np.float32)
            draw = np.round(rot + shift).astype(np.int32)
            valid = (
                (draw[:, 0] >= 0)
                & (draw[:, 0] < tw)
                & (draw[:, 1] >= 0)
                & (draw[:, 1] < th)
            )
            draw = draw[valid]
            if len(draw) < 15:
                continue
            patch[draw[:, 1], draw[:, 0]] = 1.0
            resp = cv2.matchTemplate(inv, patch, cv2.TM_CCORR)
            _, _, _, max_loc = cv2.minMaxLoc(resp)
            cx = max_loc[0] + tw / 2.0
            cy = max_loc[1] + th / 2.0
            score = _chamfer_score(
                rot + np.array([cx, cy], dtype=np.float32), dist, hp, wp
            )
            if score < best.score:
                best = MatchResult(
                    float(cx - pad),
                    float(cy - pad),
                    float(ang),
                    float(score),
                    False,
                    float(scale),
                )

    if best.score < 1e8:
        # fine angle around coarse best
        seed = best
        for ang in np.arange(
            seed.angle_deg - coarse, seed.angle_deg + coarse + 1e-6, fine
        ):
            for ds in (-0.04, 0.0, 0.04):
                scale = seed.scale + ds
                if scale <= 0.2:
                    continue
                for dx in (-4, 0, 4):
                    for dy in (-4, 0, 4):
                        cx = seed.x + pad + dx
                        cy = seed.y + pad + dy
                        score = _score_pose(
                            pts_fast, scale, ang, cx, cy, dist, hp, wp
                        )
                        if score < best.score:
                            best = MatchResult(
                                float(cx - pad),
                                float(cy - pad),
                                float(ang),
                                float(score),
                                False,
                                float(scale),
                            )
        best = _refine(best, model, dist, hp, wp, pad)
        best.score = _score_pose(
            pts_full,
            best.scale,
            best.angle_deg,
            best.x + pad,
            best.y + pad,
            dist,
            hp,
            wp,
        )

    best.ok = best.score <= score_thresh
    return best


def _refine(
    seed: MatchResult,
    model: ShapeModel,
    dist: np.ndarray,
    hp: int,
    wp: int,
    pad: int,
) -> MatchResult:
    best = MatchResult(seed.x, seed.y, seed.angle_deg, seed.score, False, seed.scale)
    fine = max(0.25, model.angle_step_deg / 4.0)
    for ds in (-0.04, -0.02, 0.0, 0.02, 0.04):
        scale = seed.scale + ds
        if scale <= 0.2:
            continue
        pts = model.points * float(scale)
        for dang in (-fine, 0.0, fine, -2 * fine, 2 * fine):
            for dx in (-3, -1, 0, 1, 3):
                for dy in (-3, -1, 0, 1, 3):
                    ang = seed.angle_deg + dang
                    cx = seed.x + pad + dx
                    cy = seed.y + pad + dy
                    rot = _rotate_points(pts, ang)
                    world = rot + np.array([cx, cy], dtype=np.float32)
                    score = _chamfer_score(world, dist, hp, wp)
                    if score < best.score:
                        best = MatchResult(
                            float(cx - pad),
                            float(cy - pad),
                            float(ang),
                            float(score),
                            False,
                            float(scale),
                        )
    return best


def board_center_from_det_box(det_box: dict) -> tuple[float, float]:
    """
    Full-board geometric center from detection box:
    midpoint of left-edge mid and right-edge mid = box center.
    """
    return (
        0.5 * (float(det_box["x1"]) + float(det_box["x2"])),
        0.5 * (float(det_box["y1"]) + float(det_box["y2"])),
    )


def draw_result(
    image: np.ndarray,
    model: ShapeModel,
    result: MatchResult,
    color=(0, 255, 0),
    det_box: dict | None = None,
) -> np.ndarray:
    vis = image.copy()
    if image.ndim == 2:
        vis = cv2.cvtColor(vis, cv2.COLOR_GRAY2BGR)
    board_xy = None
    if det_box is not None:
        x1, y1 = int(det_box["x1"]), int(det_box["y1"])
        x2, y2 = int(det_box["x2"]), int(det_box["y2"])
        # BGR orange, thick — easy to tell from green match contour
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 3)
        tag = "det"
        if "conf" in det_box:
            tag = f"det {det_box['conf']:.2f}"
        cv2.putText(
            vis,
            tag,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 140, 255),
            2,
        )
        mid_y = int(0.5 * (y1 + y2))
        left_mid = (x1, mid_y)
        right_mid = (x2, mid_y)
        board_xy = board_center_from_det_box(det_box)
        cv2.circle(vis, left_mid, 4, (255, 0, 255), -1)
        cv2.circle(vis, right_mid, 4, (255, 0, 255), -1)
        cv2.line(vis, left_mid, right_mid, (255, 0, 255), 1)

    rot = _rotate_points(model.points * float(result.scale), result.angle_deg)
    world = rot + np.array([result.x, result.y], dtype=np.float32)
    for p in np.round(world).astype(np.int32):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < vis.shape[1] and 0 <= y < vis.shape[0]:
            cv2.circle(vis, (x, y), 1, color, -1)

    # yellow ring: partial-mesh centroid (may be offset)
    cv2.circle(vis, (int(result.x), int(result.y)), 4, (0, 255, 255), 1)
    # red: board center for localization (det L/R midpoints)
    cx, cy = board_xy if board_xy is not None else (result.x, result.y)
    tip = _rotate_points(np.array([[40.0, 0.0]], dtype=np.float32), result.angle_deg)[0]
    cv2.circle(vis, (int(cx), int(cy)), 6, (0, 0, 255), 2)
    cv2.arrowedLine(
        vis,
        (int(cx), int(cy)),
        (int(cx + tip[0]), int(cy + tip[1])),
        (0, 0, 255),
        2,
        tipLength=0.3,
    )
    label = (
        f"xy=({result.x:.1f},{result.y:.1f}) ang={result.angle_deg:.1f} "
        f"s={result.scale:.2f} score={result.score:.2f} ok={result.ok}"
    )
    cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    if board_xy is not None:
        cv2.putText(
            vis,
            f"board_c=({board_xy[0]:.1f},{board_xy[1]:.1f}) "
            f"yellow=mesh质心 red=板心(左右边中点)",
            (8, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 255),
            2,
        )
    return vis


def cmd_build(args: argparse.Namespace) -> None:
    if args.from_mask:
        mask = load_mask(args.template)
        model = build_model_from_mask(
            mask, angle_step_deg=args.angle_step, max_points=args.max_points
        )
        edges = np.zeros(mask.shape, dtype=np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cv2.drawContours(edges, [max(contours, key=cv2.contourArea)], -1, 255, 1)
    else:
        img = load_bgr(args.template)
        model = build_model(
            img, angle_step_deg=args.angle_step, max_points=args.max_points
        )
        edges = extract_silhouette_edges(img)
    model.save(args.out)
    preview = Path(args.out).with_suffix(".edges.png")
    cv2.imwrite(str(preview), edges)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "n_points": int(len(model.points)),
                "edges_preview": str(preview),
                "angle_step_deg": model.angle_step_deg,
                "from_mask": bool(args.from_mask),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def cmd_match(args: argparse.Namespace) -> None:
    model = ShapeModel.load(args.model)
    img = load_bgr(args.image)
    h, w = img.shape[:2]
    det_box = None
    scale0 = None

    if args.label:
        boxes = parse_yolo_labels(args.label, w, h)
        if not boxes:
            raise ValueError(f"No boxes in {args.label}")
        det_box = max(boxes, key=lambda b: b["w"] * b["h"])
        scale0 = scale_from_det_box(model, det_box["w"], det_box["h"])
        result = match_with_det_box(
            img,
            model,
            det_box,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            score_thresh=args.score_thresh,
            edge_mode=args.edge_mode,
            roi_margin=args.roi_margin,
            scale_rel=args.scale_rel,
            scale_step=args.scale_step,
            angle_step_coarse=args.angle_step_coarse,
            angle_step_fine=args.angle_step_fine,
        )
        work = img
    else:
        offset = np.array([0.0, 0.0], dtype=np.float32)
        if args.roi:
            x1, y1, x2, y2 = [int(v) for v in args.roi]
            work = img[y1:y2, x1:x2]
            offset = np.array([x1, y1], dtype=np.float32)
        else:
            work = img
        scales = None
        if args.scale_min is not None and args.scale_max is not None:
            scales = tuple(
                np.round(
                    np.arange(args.scale_min, args.scale_max + 1e-6, args.scale_step),
                    4,
                )
            )
        result = match_model(
            work,
            model,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            score_thresh=args.score_thresh,
            scales=scales,
            edge_mode=args.edge_mode,
        )
        result.x += float(offset[0])
        result.y += float(offset[1])

    out = {
        "x": result.x,
        "y": result.y,
        "angle_deg": result.angle_deg,
        "scale": result.scale,
        "score_px": result.score,
        "ok": result.ok,
        "edge_mode": args.edge_mode,
        "note": "score_px = mean edge distance; lower is better",
    }
    if det_box is not None:
        out["det_box"] = {
            "x1": det_box["x1"],
            "y1": det_box["y1"],
            "x2": det_box["x2"],
            "y2": det_box["y2"],
            "conf": det_box["conf"],
        }
        out["scale_from_det"] = scale0
    print(json.dumps(out, indent=2, ensure_ascii=False))

    if args.viz:
        vis = draw_result(img, model, result, det_box=det_box)
        Path(args.viz).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.viz, vis)
        print(f"wrote {args.viz}")


def _load_ee_pose_map(path: str | Path | None) -> dict[str, list[float]]:
    """Optional JSON: {\"color_0000.jpg\": [x,y,z,rx,ry,rz], ...} or {\"default\": [...]}."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--ee-poses must be a JSON object")
    return {str(k): [float(x) for x in v] for k, v in data.items()}


def _resolve_ee(
    ee_map: dict[str, list[float]],
    image_name: str,
    ee_pose: list[float] | None,
) -> list[float] | None:
    if image_name in ee_map:
        return ee_map[image_name]
    stem = Path(image_name).stem
    if stem in ee_map:
        return ee_map[stem]
    if "default" in ee_map:
        return ee_map["default"]
    return ee_pose


def load_depth_m(path: str | Path) -> np.ndarray | None:
    """Load aligned depth in meters from .npy / *_mm.png / .png(u16 raw*scale unknown→mm)."""
    path = Path(path)
    if path.suffix == ".npy":
        d = np.load(str(path))
        return d.astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim != 2:
        img = img[:, :, 0]
    if path.name.endswith("_mm.png") or path.stem.endswith("_mm"):
        return img.astype(np.float32) / 1000.0
    # uint16 aligned depth from capture_wrist: already scaled to meters in .npy;
    # .png is raw sensor units — prefer .npy. If only .png, treat as millimeters
    # when max > 20 (typical mm range), else meters.
    d = img.astype(np.float32)
    if float(np.nanmax(d)) > 20.0:
        d = d / 1000.0
    return d


def depth_roi_median(
    depth_m: np.ndarray,
    box: dict,
    z_min: float = 0.05,
    z_max: float = 2.0,
) -> float | None:
    """Median depth (m) inside detection box; None if no valid pixels."""
    h, w = depth_m.shape[:2]
    x1 = max(0, int(box["x1"]))
    y1 = max(0, int(box["y1"]))
    x2 = min(w, int(box["x2"]))
    y2 = min(h, int(box["y2"]))
    if x2 <= x1 or y2 <= y1:
        return None
    roi = depth_m[y1:y2, x1:x2]
    valid = roi[(roi > z_min) & (roi < z_max)]
    if valid.size < 20:
        return None
    return float(np.median(valid))


def resolve_depth_path(depth_dir: Path, image_name: str) -> Path | None:
    stem = Path(image_name).stem
    for cand in (
        depth_dir / f"{stem}.npy",
        depth_dir / f"{stem}_mm.png",
        depth_dir / f"{stem}.png",
    ):
        if cand.is_file():
            return cand
    return None


def _attach_base_and_angle_metrics(
    row: dict,
    img: np.ndarray,
    r: MatchResult,
    det_box: dict,
    args: argparse.Namespace,
    T_tool_cam: np.ndarray | None,
    K,
    ee_pose: list[float] | None,
    depth_m: float | None = None,
) -> None:
    """Fill base-frame pose + silhouette reference angle error into row."""
    # Silhouette PCA angle in image (independent of template) for error check.
    # Prefer ROI silhouette so tray edges do not dominate.
    h, w = img.shape[:2]
    rx1, ry1, rx2, ry2 = expand_box(
        det_box["x1"],
        det_box["y1"],
        det_box["x2"],
        det_box["y2"],
        w,
        h,
        margin=args.roi_margin,
    )
    work = img[ry1:ry2, rx1:rx2]
    edges = extract_scene_edges(
        work, mode="silhouette" if args.edge_mode == "auto" else args.edge_mode
    )
    if not np.any(edges):
        edges = extract_scene_edges(work, mode="canny")
    ref = silhouette_ref_angle_deg(edges, r.x - rx1, r.y - ry1)
    ref_aligned = None
    if ref is not None:
        # PCA axis is undirected → choose flip closest to match angle
        cand = [ref, wrap_angle_deg(ref + 180.0)]
        ref_aligned = min(cand, key=lambda a: abs(angle_diff_deg(r.angle_deg, a)))
        err = angle_diff_deg(r.angle_deg, ref_aligned)
        row["ref_angle_deg"] = round(ref_aligned, 2)
        row["angle_error_deg"] = round(err, 2)
        row["angle_error_abs_deg"] = round(abs(err), 2)
    else:
        row["ref_angle_deg"] = None
        row["angle_error_deg"] = None
        row["angle_error_abs_deg"] = None

    if T_tool_cam is None or K is None:
        return
    if depth_m is None:
        depth_m = float(args.depth_m)
    depth_m = float(depth_m)

    # Localization center: full-board center from det box (mid of L/R edges),
    # NOT the partial-mesh centroid (r.x, r.y), which is biased by notches.
    center_mode = getattr(args, "loc_center", "det_box")
    if center_mode == "det_box":
        u, v = board_center_from_det_box(det_box)
    else:
        u, v = float(r.x), float(r.y)
    row["board_cx"] = round(float(u), 2)
    row["board_cy"] = round(float(v), 2)
    row["match_cx"] = round(float(r.x), 2)
    row["match_cy"] = round(float(r.y), 2)

    base = match_to_base(
        u,
        v,
        r.angle_deg,
        T_tool_cam=T_tool_cam,
        K=K,
        depth_m=depth_m,
        ee_xyz_rpy=ee_pose,
    )
    row["base"] = {
        "x_m": round(base.x, 5),
        "y_m": round(base.y, 5),
        "z_m": round(base.z, 5),
        "yaw_deg": round(base.yaw_deg, 2),
        "frame": base.frame,
        "depth_m_used": round(depth_m, 5),
        "loc_center": center_mode,
    }
    if ref_aligned is not None:
        base_ref = match_to_base(
            u,
            v,
            ref_aligned,
            T_tool_cam=T_tool_cam,
            K=K,
            depth_m=depth_m,
            ee_xyz_rpy=ee_pose,
        )
        yaw_err = angle_diff_deg(base.yaw_deg, base_ref.yaw_deg)
        row["base"]["ref_yaw_deg"] = round(base_ref.yaw_deg, 2)
        row["base"]["yaw_error_deg"] = round(yaw_err, 2)
        row["base"]["yaw_error_abs_deg"] = round(abs(yaw_err), 2)


def cmd_match_dir(args: argparse.Namespace) -> None:
    """Batch match: images + YOLO labels dir → viz + summary.json."""
    model = ShapeModel.load(args.model)
    img_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    T_tool_cam = None
    K = None
    if not args.no_base:
        T_tool_cam = load_T_tool_cam(args.handeye)
        K = load_intrinsics(args.intrinsics)
        print(
            f"handeye={args.handeye}\n"
            f"T_tool_cam=\n{np.array2string(T_tool_cam, precision=6)}\n"
            f"intrinsics fx={K.fx:.2f} fy={K.fy:.2f} cx={K.cx:.2f} cy={K.cy:.2f}",
            flush=True,
        )

    ee_map = _load_ee_pose_map(args.ee_poses)
    ee_default = list(args.ee_pose) if args.ee_pose is not None else None

    depth_dir: Path | None = None
    if getattr(args, "depth_dir", ""):
        depth_dir = Path(args.depth_dir)
    else:
        # default: sibling depth_aligned next to color/
        cand = img_dir.parent / "depth_aligned"
        if cand.is_dir():
            depth_dir = cand
    if depth_dir is not None:
        print(f"depth_dir={depth_dir} (ROI median; fallback --depth-m={args.depth_m})", flush=True)
    else:
        print(f"depth_dir=None → using fixed --depth-m={args.depth_m}", flush=True)

    images = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    # de-dup while keeping order
    seen = set()
    uniq = []
    for p in images:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    images = uniq

    results = []
    for img_path in images:
        label_path = label_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            print(f"skip {img_path.name}: no label", flush=True)
            continue
        img = load_bgr(img_path)
        h, w = img.shape[:2]
        boxes = parse_yolo_labels(label_path, w, h)
        if not boxes:
            print(f"skip {img_path.name}: empty label", flush=True)
            continue
        det_box = max(boxes, key=lambda b: b["w"] * b["h"])
        scale0 = scale_from_det_box(model, det_box["w"], det_box["h"])
        r = match_with_det_box(
            img,
            model,
            det_box,
            angle_min=args.angle_min,
            angle_max=args.angle_max,
            score_thresh=args.score_thresh,
            edge_mode=args.edge_mode,
            roi_margin=args.roi_margin,
            scale_rel=args.scale_rel,
            scale_step=args.scale_step,
            angle_step_coarse=args.angle_step_coarse,
            angle_step_fine=args.angle_step_fine,
        )
        row = {
            "image": img_path.name,
            "x": round(r.x, 2),
            "y": round(r.y, 2),
            "angle_deg": round(r.angle_deg, 2),
            "scale": round(r.scale, 4),
            "scale_from_det": round(scale0, 4),
            "score_px": round(r.score, 4),
            "ok": bool(r.ok),
            "det_box": [det_box["x1"], det_box["y1"], det_box["x2"], det_box["y2"]],
            "det_conf": det_box["conf"],
        }
        ee_pose = _resolve_ee(ee_map, img_path.name, ee_default)
        depth_m = None
        if depth_dir is not None:
            dp = resolve_depth_path(depth_dir, img_path.name)
            if dp is not None:
                depth_map = load_depth_m(dp)
                if depth_map is not None:
                    depth_m = depth_roi_median(depth_map, det_box)
        _attach_base_and_angle_metrics(
            row, img, r, det_box, args, T_tool_cam, K, ee_pose, depth_m=depth_m
        )
        results.append(row)
        viz = draw_result(img, model, r, det_box=det_box)
        if row.get("base"):
            b = row["base"]
            extra = (
                f"base xy=({b['x_m']:.3f},{b['y_m']:.3f}) "
                f"yaw={b['yaw_deg']:.1f} d={b['depth_m_used']:.3f} [{b['frame']}]"
            )
            cv2.putText(
                viz,
                extra,
                (8, 72),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 200, 0),
                2,
            )
            if row.get("angle_error_abs_deg") is not None:
                cv2.putText(
                    viz,
                    f"ang_err={row['angle_error_abs_deg']:.1f}deg "
                    f"(vs silhouette PCA)",
                    (8, 96),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 200, 0),
                    2,
                )
        cv2.imwrite(str(out_dir / f"{img_path.stem}_match.jpg"), viz)
        ang_err = row.get("angle_error_abs_deg")
        ang_err_s = f" ang_err={ang_err:.1f}" if ang_err is not None else ""
        base_s = ""
        if row.get("base"):
            b = row["base"]
            base_s = (
                f" | base=({b['x_m']:.3f},{b['y_m']:.3f},{b['z_m']:.3f}) "
                f"yaw={b['yaw_deg']:.1f}"
            )
        print(
            f"{img_path.name}: xy=({r.x:.1f},{r.y:.1f}) ang={r.angle_deg:.1f} "
            f"s={r.scale:.2f}(det={scale0:.2f}) score={r.score:.2f} ok={r.ok}"
            f"{ang_err_s}{base_s}",
            flush=True,
        )

    errs = [
        float(r["angle_error_abs_deg"])
        for r in results
        if r.get("angle_error_abs_deg") is not None
    ]
    yaw_errs = [
        float(r["base"]["yaw_error_abs_deg"])
        for r in results
        if r.get("base") and r["base"].get("yaw_error_abs_deg") is not None
    ]
    angle_stats = None
    if errs:
        angle_stats = {
            "n": len(errs),
            "mean_abs_deg": round(float(np.mean(errs)), 3),
            "max_abs_deg": round(float(np.max(errs)), 3),
            "std_abs_deg": round(float(np.std(errs)), 3),
            "note": "match angle vs silhouette PCA (180° ambiguity resolved toward match)",
        }
    yaw_stats = None
    if yaw_errs:
        yaw_stats = {
            "n": len(yaw_errs),
            "mean_abs_deg": round(float(np.mean(yaw_errs)), 3),
            "max_abs_deg": round(float(np.max(yaw_errs)), 3),
            "std_abs_deg": round(float(np.std(yaw_errs)), 3),
            "note": "base-frame yaw error (match vs PCA), same depth/EE",
        }

    summary = {
        "model": str(args.model),
        "mode": "det_box_scale",
        "n_images": len(results),
        "n_ok": sum(1 for r in results if r["ok"]),
        "handeye": None if args.no_base else str(args.handeye),
        "intrinsics": None if args.no_base else str(args.intrinsics),
        "depth_m_fallback": None if args.no_base else float(args.depth_m),
        "depth_dir": str(depth_dir) if depth_dir is not None else None,
        "ee_pose_default": ee_default,
        "angle_error_stats": angle_stats,
        "base_yaw_error_stats": yaw_stats,
        "results": results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    # Human-readable angle comparison table
    lines = [
        "image\tangle_deg\tref_angle_deg\tangle_error_abs\tbase_yaw\tbase_yaw_err\tok\tscore",
    ]
    for r in results:
        b = r.get("base") or {}
        lines.append(
            f"{r['image']}\t{r['angle_deg']}\t{r.get('ref_angle_deg')}\t"
            f"{r.get('angle_error_abs_deg')}\t{b.get('yaw_deg')}\t"
            f"{b.get('yaw_error_abs_deg')}\t{r['ok']}\t{r['score_px']}"
        )
    if angle_stats:
        lines.append(
            f"\n# image-angle error: mean={angle_stats['mean_abs_deg']} "
            f"max={angle_stats['max_abs_deg']} std={angle_stats['std_abs_deg']}"
        )
    if yaw_stats:
        lines.append(
            f"# base-yaw error: mean={yaw_stats['mean_abs_deg']} "
            f"max={yaw_stats['max_abs_deg']} std={yaw_stats['std_abs_deg']}"
        )
    (out_dir / "angle_errors.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        f"\nwrote {out_dir}/summary.json  ok={summary['n_ok']}/{summary['n_images']}",
        flush=True,
    )
    if angle_stats:
        print(
            f"angle error (vs silhouette PCA): "
            f"mean={angle_stats['mean_abs_deg']}° "
            f"max={angle_stats['max_abs_deg']}° "
            f"std={angle_stats['std_abs_deg']}°",
            flush=True,
        )
    print(f"wrote {out_dir}/angle_errors.tsv", flush=True)


def cmd_selftest(args: argparse.Namespace) -> None:
    img = load_bgr(args.template)
    model = build_model(img, angle_step_deg=args.angle_step)
    h, w = img.shape[:2]
    pad = int(0.35 * max(h, w))
    canvas = cv2.copyMakeBorder(
        img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    ch, cw = canvas.shape[:2]
    # warpAffine maps dst->src; use -angle so content rotates by +test_angle
    M = cv2.getRotationMatrix2D((cw / 2, ch / 2), -args.test_angle, 1.0)
    rotated = cv2.warpAffine(canvas, M, (cw, ch), borderValue=(0, 0, 0))
    result = match_model(
        rotated,
        model,
        angle_min=-180,
        angle_max=180,
        score_thresh=args.score_thresh,
    )
    err = abs(((result.angle_deg - args.test_angle + 180) % 360) - 180)
    print(
        json.dumps(
            {
                "injected_angle_deg": args.test_angle,
                "estimated_angle_deg": result.angle_deg,
                "angle_error_deg": err,
                "xy": [result.x, result.y],
                "scale": result.scale,
                "score_px": result.score,
                "ok": result.ok,
                "n_model_points": int(len(model.points)),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if args.viz:
        vis = draw_result(rotated, model, result)
        cv2.imwrite(args.viz, vis)
        print(f"wrote {args.viz}")


def main() -> None:
    p = argparse.ArgumentParser(description="PCB 2D shape (silhouette) matching")
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build shape model from masked template")
    b.add_argument("--template", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--from-mask", action="store_true", help="Template is a binary mask")
    b.add_argument("--angle-step", type=float, default=2.0)
    b.add_argument("--max-points", type=int, default=600)
    b.set_defaults(func=cmd_build)

    m = sub.add_parser("match", help="Match model in a scene/crop image")
    m.add_argument("--model", required=True)
    m.add_argument("--image", required=True)
    m.add_argument(
        "--label",
        default="",
        help="YOLO label txt; use box size for scale + ROI",
    )
    m.add_argument(
        "--roi", nargs=4, type=int, default=None, metavar=("X1", "Y1", "X2", "Y2")
    )
    m.add_argument("--roi-margin", type=float, default=0.25)
    m.add_argument("--angle-min", type=float, default=-180.0)
    m.add_argument("--angle-max", type=float, default=180.0)
    m.add_argument(
        "--scale-rel",
        type=float,
        default=0.08,
        help="With --label: search scale in s0*(1±rel), default 0.08 (±8%%)",
    )
    m.add_argument(
        "--scale-step",
        type=float,
        default=0.04,
        help="Scale search step (absolute), default 0.04",
    )
    m.add_argument("--angle-step-coarse", type=float, default=10.0)
    m.add_argument("--angle-step-fine", type=float, default=2.0)
    m.add_argument(
        "--scale-min",
        type=float,
        default=None,
        help="Without --label: absolute scale range min (e.g. 0.5)",
    )
    m.add_argument(
        "--scale-max",
        type=float,
        default=None,
        help="Without --label: absolute scale range max (e.g. 2.0)",
    )
    m.add_argument("--score-thresh", type=float, default=3.0)
    m.add_argument(
        "--edge-mode",
        choices=("auto", "silhouette", "canny"),
        default="auto",
        help="Scene edge extraction: canny for full RGB scenes",
    )
    m.add_argument("--viz", default="")
    m.set_defaults(func=cmd_match)

    md = sub.add_parser(
        "match-dir",
        help="Batch match with YOLO labels (scale from det box)",
    )
    md.add_argument("--model", required=True)
    md.add_argument("--images", required=True, help="Directory of images")
    md.add_argument("--labels", required=True, help="Directory of YOLO .txt labels")
    md.add_argument("--out", required=True, help="Output directory for viz + summary")
    md.add_argument("--roi-margin", type=float, default=0.25)
    md.add_argument("--angle-min", type=float, default=-180.0)
    md.add_argument("--angle-max", type=float, default=180.0)
    md.add_argument(
        "--scale-rel",
        type=float,
        default=0.08,
        help="Search scale in s0*(1±rel); s0 from det box. Default 0.08",
    )
    md.add_argument(
        "--scale-step",
        type=float,
        default=0.04,
        help="Scale search step (absolute). Default 0.04",
    )
    md.add_argument(
        "--angle-step-coarse",
        type=float,
        default=10.0,
        help="Coarse angle step (deg). Default 10",
    )
    md.add_argument(
        "--angle-step-fine",
        type=float,
        default=2.0,
        help="Fine angle step (deg). Default 2",
    )
    md.add_argument("--score-thresh", type=float, default=3.0)
    md.add_argument(
        "--edge-mode",
        choices=("auto", "silhouette", "canny"),
        default="silhouette",
        help="With det boxes, silhouette (outer contour) is preferred",
    )
    md.add_argument(
        "--handeye",
        default=str(DEFAULT_HANDEYE),
        help="Wrist eye-in-hand YAML (T_tool_cam / camera2_eye_in_hand)",
    )
    md.add_argument(
        "--intrinsics",
        default=str(DEFAULT_INTRINSICS),
        help="Wrist camera intrinsics YAML",
    )
    md.add_argument(
        "--depth-m",
        type=float,
        default=0.35,
        help="Fallback camera Z depth (m) when depth map / ROI median unavailable",
    )
    md.add_argument(
        "--depth-dir",
        default="",
        help="Aligned depth dir (prefers *.npy meters). Default: <images>/../depth_aligned",
    )
    md.add_argument(
        "--loc-center",
        choices=("det_box", "match"),
        default="det_box",
        help="Pixel used for base xyz: det_box=full-board center (L/R mid); match=mesh centroid",
    )
    md.add_argument(
        "--ee-pose",
        nargs=6,
        type=float,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="EE pose in base: xyz(m) + rpy(rad). Omit → EE=I (tool frame as base)",
    )
    md.add_argument(
        "--ee-poses",
        default="",
        help='JSON map of per-image EE poses, optional key "default"',
    )
    md.add_argument(
        "--no-base",
        action="store_true",
        help="Skip hand-eye / base-frame conversion",
    )
    md.set_defaults(func=cmd_match_dir)

    t = sub.add_parser("selftest", help="Rotate template and recover angle")
    t.add_argument("--template", required=True)
    t.add_argument("--test-angle", type=float, default=35.0)
    t.add_argument("--angle-step", type=float, default=2.0)
    t.add_argument("--score-thresh", type=float, default=3.0)
    t.add_argument("--viz", default="")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
