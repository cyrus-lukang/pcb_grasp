"""Compatibility shim: re-export TCT shape_matcher + outer-ring helpers for pose_outer_refine."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import cv2
import numpy as np

_TCT_DIR = Path(__file__).resolve().parents[1] / "TCT" / "shape_match"
_TCT_FILE = _TCT_DIR / "shape_matcher.py"
if str(_TCT_DIR) not in sys.path:
    sys.path.insert(0, str(_TCT_DIR))
_spec = importlib.util.spec_from_file_location("_tct_shape_matcher", _TCT_FILE)
assert _spec and _spec.loader
_tct = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _tct
_spec.loader.exec_module(_tct)

# re-export commonly used symbols
MatchResult = _tct.MatchResult
ShapeModel = _tct.ShapeModel
_prepare_dist = _tct._prepare_dist
_score_pose = _tct._score_pose
expand_box = _tct.expand_box
extract_canny_edges = _tct.extract_canny_edges
extract_silhouette_edges = _tct.extract_silhouette_edges
load_bgr = _tct.load_bgr
scale_from_det_box = _tct.scale_from_det_box
scales_from_det_box = _tct.scales_from_det_box
_draw_result_core = _tct.draw_result
_extract_scene_edges_core = _tct.extract_scene_edges


def _outer(self) -> np.ndarray:
    return np.asarray(self.points, dtype=np.float32)


ShapeModel.outer = _outer  # type: ignore[attr-defined]


def extract_coarse_outer_edges(
    bgr: np.ndarray,
    canny1: int = 40,
    canny2: int = 120,
    blur: int = 5,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    k = blur if blur % 2 == 1 else blur + 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)
    edges = cv2.Canny(gray, canny1, canny2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(edges)
    h, w = edges.shape[:2]
    min_len = 0.15 * min(h, w)
    for c in contours:
        if cv2.arcLength(c, False) >= min_len:
            cv2.drawContours(out, [c], -1, 255, 1)
    return out if np.any(out) else edges


def extract_outer_ring_edges(
    bgr: np.ndarray,
    canny1: int = 40,
    canny2: int = 120,
    ring_width: int = 18,
    min_edge_len: float = 40.0,
    blur: int = 5,
) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    k = blur if blur % 2 == 1 else blur + 1
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV) if bgr.ndim == 3 else None
    if hsv is not None:
        hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        fg = (((hh > 35) & (hh < 95) & (ss > 40) & (vv > 40)) * 255).astype(np.uint8)
    else:
        _, fg = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros_like(gray)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) > 50:
            cv2.drawContours(mask, [largest], -1, 255, -1)

    if not np.any(mask):
        return extract_canny_edges(bgr, canny1=canny1, canny2=canny2)

    rw = max(2, int(ring_width))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rw * 2 + 1, rw * 2 + 1))
    ring = cv2.subtract(cv2.dilate(mask, ker), cv2.erode(mask, ker))
    canny = cv2.Canny(gray, canny1, canny2)
    edged = cv2.bitwise_and(canny, ring)
    n, labels, stats, _ = cv2.connectedComponentsWithStats((edged > 0).astype(np.uint8), 8)
    out = np.zeros_like(edged)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= float(min_edge_len):
            out[labels == i] = 255
    return out if np.any(out) else extract_silhouette_edges(bgr)


def extract_scene_edges(bgr: np.ndarray, mode: str = "auto", **kwargs) -> np.ndarray:
    canny1 = int(kwargs.get("canny1", 40))
    canny2 = int(kwargs.get("canny2", 120))
    blur = int(kwargs.get("blur", 5))
    ring_width = int(kwargs.get("ring_width", 18))
    min_edge_len = float(kwargs.get("min_edge_len", 40.0))
    if mode == "outer-ring":
        e = extract_outer_ring_edges(
            bgr,
            canny1=canny1,
            canny2=canny2,
            ring_width=ring_width,
            min_edge_len=min_edge_len,
            blur=blur,
        )
        return e if np.any(e) else extract_canny_edges(bgr, canny1=canny1, canny2=canny2)
    if mode in ("canny", "silhouette", "auto"):
        return _extract_scene_edges_core(bgr, mode=mode)
    return _extract_scene_edges_core(bgr, mode="auto")


def silhouette_edges_for_viz(
    img: np.ndarray,
    det_box: dict | None = None,
    roi_margin: float = 0.25,
    canny1: int = 40,
    canny2: int = 120,
    blur: int = 5,
) -> np.ndarray:
    h, w = img.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    if det_box is None:
        return extract_silhouette_edges(img)
    rx1, ry1, rx2, ry2 = expand_box(
        int(det_box["x1"]), int(det_box["y1"]), int(det_box["x2"]), int(det_box["y2"]),
        w, h, margin=roi_margin,
    )
    crop = img[ry1:ry2, rx1:rx2]
    e = extract_coarse_outer_edges(crop, canny1=canny1, canny2=canny2, blur=blur)
    if not np.any(e):
        e = extract_silhouette_edges(crop)
    full[ry1:ry2, rx1:rx2] = e
    return full


def scene_edges_for_viz(
    img: np.ndarray,
    edge_mode: str = "outer-ring",
    det_box: dict | None = None,
    roi_margin: float = 0.25,
    canny1: int = 40,
    canny2: int = 120,
    ring_width: int = 18,
    min_edge_len: float = 40.0,
    blur: int = 5,
) -> np.ndarray:
    h, w = img.shape[:2]
    full = np.zeros((h, w), dtype=np.uint8)
    kwargs = dict(
        canny1=canny1, canny2=canny2, ring_width=ring_width,
        min_edge_len=min_edge_len, blur=blur,
    )
    if det_box is None:
        return extract_scene_edges(img, mode=edge_mode, **kwargs)
    rx1, ry1, rx2, ry2 = expand_box(
        int(det_box["x1"]), int(det_box["y1"]), int(det_box["x2"]), int(det_box["y2"]),
        w, h, margin=roi_margin,
    )
    e = extract_scene_edges(img[ry1:ry2, rx1:rx2], mode=edge_mode, **kwargs)
    full[ry1:ry2, rx1:rx2] = e
    return full


def draw_result(
    image: np.ndarray,
    model: ShapeModel,
    result: MatchResult,
    color=(0, 255, 0),
    det_box: dict | None = None,
    scene_edges: np.ndarray | None = None,
    points: np.ndarray | None = None,
    stage_title: str = "",
    legend: str = "",
    **_kwargs,
) -> np.ndarray:
    old_pts = model.points
    if points is not None:
        model.points = np.asarray(points, dtype=np.float32)
    try:
        vis = _draw_result_core(image, model, result, color=color, det_box=det_box)
    finally:
        model.points = old_pts

    if scene_edges is not None and np.any(scene_edges):
        ys, xs = np.where(scene_edges > 0)
        for x, y in zip(xs[::2], ys[::2]):
            cv2.circle(vis, (int(x), int(y)), 1, (0, 255, 255), -1)

    if stage_title:
        cv2.putText(
            vis, stage_title, (8, vis.shape[0] - 36),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
    if legend:
        cv2.putText(
            vis, legend, (8, vis.shape[0] - 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
        )
    return vis
