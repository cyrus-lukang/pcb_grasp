#!/usr/bin/env python3
"""
YOLO pose 推理 + 特征点 1→0 定方向 + 外轮廓匹配微调旋转/平移。

流程：
  1. best.pt 对图像做 pose 推理（参考 ultralytics/script/test.py）
  2. 用关键点 P1→P0 向量得到初始朝向角
  3. 在该角 ±10° 内，用 shape_match 外轮廓 chamfer 微调角度与平移

用法：
  python pose_outer_refine.py
  python pose_outer_refine.py --angle-offset 0 --edge-mode outer-ring
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# local shape_matcher
sys.path.insert(0, str(Path(__file__).resolve().parent))
from shape_matcher import (  # noqa: E402
    MatchResult,
    ShapeModel,
    _prepare_dist,
    _score_pose,
    draw_result,
    expand_box,
    extract_coarse_outer_edges,
    extract_outer_ring_edges,
    extract_scene_edges,
    load_bgr,
    scale_from_det_box,
    scales_from_det_box,
    silhouette_edges_for_viz,
    scene_edges_for_viz,
)


CLASS_NAMES = {0: "pcb", 1: "nopcb"}
CLASS_COLORS = {
    0: (0, 255, 0),
    1: (0, 255, 255),
}
DEFAULT_COLOR = (255, 255, 255)


def normalize_angle_deg(a: float) -> float:
    return float((a + 180.0) % 360.0 - 180.0)


def angle_from_kpt10(kpts: np.ndarray) -> float:
    """方向角（度）：特征点 1 → 0，图像坐标系，逆时针为正（与 shape_match 一致）。"""
    p0 = kpts[0, :2].astype(np.float64)
    p1 = kpts[1, :2].astype(np.float64)
    d = p0 - p1
    return float(np.degrees(np.arctan2(d[1], d[0])))


def kpt_center(kpts: np.ndarray) -> tuple[float, float]:
    pts = kpts[:, :2].astype(np.float64)
    return float(pts[:, 0].mean()), float(pts[:, 1].mean())


def _xy_offs(radius: int, step: int) -> list[int]:
    offs = list(range(-radius, radius + 1, max(1, step)))
    if 0 not in offs:
        offs.append(0)
    return offs


def _subsample_pts(pts: np.ndarray, n: int) -> np.ndarray:
    if len(pts) <= n:
        return pts
    return pts[np.linspace(0, len(pts) - 1, n).astype(int)]


def _limit_scales(
    scales_list: tuple[float, ...] | list[float] | np.ndarray,
    max_n: int = 8,
) -> np.ndarray:
    """粗搜只用均匀抽样的少量 scale，避免候选爆炸。"""
    scales = np.asarray(
        [float(s) for s in scales_list if float(s) >= 0.2], dtype=np.float32
    )
    if scales.size == 0:
        return scales
    scales = np.unique(np.round(scales, 4))
    if len(scales) <= max_n:
        return scales.astype(np.float32)
    idx = np.linspace(0, len(scales) - 1, max_n).astype(int)
    return scales[idx].astype(np.float32)


def _best_pose_vectorized(
    pts: np.ndarray,
    scales_list: tuple[float, ...] | list[float] | np.ndarray,
    angs: np.ndarray,
    dxs: list[int] | np.ndarray,
    dys: list[int] | np.ndarray,
    cx_base: float,
    cy_base: float,
    dist: np.ndarray,
    hp: int,
    wp: int,
    pad: int,
    fallback: MatchResult,
) -> MatchResult:
    """
    向量化评分：按 scale 分块，每块一次算完 (ang, dx, dy)。
    比一次性堆 (S,A,T,N) 大张量更省带宽。
    """
    pts = np.asarray(pts, dtype=np.float32)
    scales = np.asarray(
        [s for s in scales_list if float(s) >= 0.2], dtype=np.float32
    )
    angs = np.asarray(angs, dtype=np.float32)
    dxs = np.asarray(dxs, dtype=np.float32)
    dys = np.asarray(dys, dtype=np.float32)
    if scales.size == 0 or angs.size == 0 or dxs.size == 0 or dys.size == 0:
        return fallback

    off_x = np.repeat(dxs, len(dys))
    off_y = np.tile(dys, len(dxs))
    cx = cx_base + off_x
    cy = cy_base + off_y
    n_pts = len(pts)
    min_inside = max(20, n_pts // 4)

    rad = np.deg2rad(angs)
    cos_a = np.cos(rad)[:, None]  # (A, 1)
    sin_a = np.sin(rad)[:, None]

    best = fallback
    best_score = float(fallback.score)

    for scale in scales:
        sp = pts * float(scale)  # (N, 2)
        x = sp[:, 0]
        y = sp[:, 1]
        # (A, N)
        xr = x[None, :] * cos_a - y[None, :] * sin_a
        yr = x[None, :] * sin_a + y[None, :] * cos_a
        # (A, T, N)
        wx = xr[:, None, :] + cx[None, :, None]
        wy = yr[:, None, :] + cy[None, :, None]

        ix = np.rint(wx).astype(np.int32)
        iy = np.rint(wy).astype(np.int32)
        inside = (ix >= 0) & (ix < wp) & (iy >= 0) & (iy < hp)
        vals = dist[np.clip(iy, 0, hp - 1), np.clip(ix, 0, wp - 1)]
        vals = np.where(inside, vals, 0.0).astype(np.float32)
        n_in = inside.sum(axis=-1)
        scores = np.where(
            n_in >= min_inside,
            vals.sum(axis=-1) / np.maximum(n_in, 1).astype(np.float32),
            np.float32(1e9),
        )
        flat_idx = int(np.argmin(scores))
        sc = float(scores.ravel()[flat_idx])
        if sc < best_score:
            best_score = sc
            ai, ti = np.unravel_index(flat_idx, scores.shape)
            best = MatchResult(
                float(cx[ti] - pad),
                float(cy[ti] - pad),
                normalize_angle_deg(float(angs[ai])),
                sc,
                False,
                float(scale),
            )

    return best


def refine_outer_near_pose(
    scene_bgr: np.ndarray,
    model: ShapeModel,
    box: dict,
    angle0_deg: float,
    angle_half_range: float = 30.0,
    angle_step_coarse: float = 3.0,
    angle_step_fine: float = 0.5,
    xy_step_coarse: int = 4,
    xy_step_fine: int = 1,
    xy_radius: int = 12,
    score_thresh: float = 5.0,
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
) -> MatchResult:
    """
    以外轮廓为唯一证据，在 angle0±range 内粗→细微调旋转与平移（不做 180° 翻转）。

    评分向量化：每个 stage 把 (scale, ang, dx, dy) 一次 batch 算完。
    """
    h, w = scene_bgr.shape[:2]
    scales = scales_from_det_box(
        model, box["w"], box["h"], rel=scale_rel, step=scale_step
    )
    scale_lo, scale_hi = float(min(scales)), float(max(scales))
    rx1, ry1, rx2, ry2 = expand_box(
        box["x1"], box["y1"], box["x2"], box["y2"], w, h, margin=roi_margin
    )
    work = scene_bgr[ry1:ry2, rx1:rx2]

    edges = extract_coarse_outer_edges(work, canny1=canny1, canny2=canny2, blur=blur)
    if not np.any(edges):
        edges = extract_outer_ring_edges(
            work,
            canny1=canny1,
            canny2=canny2,
            ring_width=ring_width,
            min_edge_len=min_edge_len,
            blur=blur,
        )
    if not np.any(edges):
        edges = extract_scene_edges(
            work,
            mode="canny" if edge_mode == "outer-ring" else edge_mode,
            canny1=canny1,
            canny2=canny2,
            ring_width=ring_width,
            min_edge_len=min_edge_len,
            blur=blur,
        )
    if not np.any(edges):
        return MatchResult(0, 0, angle0_deg, 1e9, False, float(scales[len(scales) // 2]))

    outer_pts = model.outer()
    outer_fast = _subsample_pts(outer_pts, coarse_points)
    outer_mid = _subsample_pts(outer_pts, min(160, max(coarse_points, len(outer_pts))))
    coarse_scales = _limit_scales(scales, max_n=max_coarse_scales)

    rad = float(np.linalg.norm(outer_pts, axis=1).max()) * float(max(scales))
    pad = int(rad + 8)
    dist, _, hp, wp = _prepare_dist(edges, pad)

    cx0 = float(box["cx"] - rx1 + pad)
    cy0 = float(box["cy"] - ry1 + pad)
    a0 = normalize_angle_deg(angle0_deg)
    init = MatchResult(
        float(box["cx"] - rx1),
        float(box["cy"] - ry1),
        a0,
        1e9,
        False,
        float(scales[len(scales) // 2]),
    )

    # --- stage 1: coarse（少 scale / 少点 / 大角度步长）---
    coarse_angs = np.arange(
        a0 - angle_half_range,
        a0 + angle_half_range + 1e-9,
        angle_step_coarse,
    )
    coarse_xy = _xy_offs(xy_radius, xy_step_coarse)
    best = _best_pose_vectorized(
        outer_fast,
        coarse_scales,
        coarse_angs,
        coarse_xy,
        coarse_xy,
        cx0,
        cy0,
        dist,
        hp,
        wp,
        pad,
        init,
    )

    if best.score >= 1e8:
        best.ok = False
        return MatchResult(
            best.x + float(rx1),
            best.y + float(ry1),
            best.angle_deg,
            best.score,
            False,
            best.scale,
        )

    # --- stage 2: medium ---
    mid_angs = np.arange(
        best.angle_deg - angle_step_coarse,
        best.angle_deg + angle_step_coarse + 1e-9,
        angle_step_fine,
    )
    mid_xy = _xy_offs(xy_step_coarse, xy_step_fine)
    s_lo = max(scale_lo, best.scale * 0.96)
    s_hi = min(scale_hi, best.scale * 1.04)
    mid_scales = tuple(
        np.round(np.arange(s_lo, s_hi + 1e-9, max(0.015, scale_step * 0.5)), 4)
    ) or (round(best.scale, 4),)
    best = _best_pose_vectorized(
        outer_mid,
        mid_scales,
        mid_angs,
        mid_xy,
        mid_xy,
        best.x + pad,
        best.y + pad,
        dist,
        hp,
        wp,
        pad,
        best,
    )

    # --- stage 3: fine ---
    fine_angs = np.arange(
        best.angle_deg - angle_step_fine,
        best.angle_deg + angle_step_fine + 1e-9,
        max(0.25, angle_step_fine * 0.5),
    )
    fine_scales = (best.scale * 0.985, best.scale, best.scale * 1.015)
    fine_xy = [-2, -1, 0, 1, 2]
    best = _best_pose_vectorized(
        outer_pts,
        fine_scales,
        fine_angs,
        fine_xy,
        fine_xy,
        best.x + pad,
        best.y + pad,
        dist,
        hp,
        wp,
        pad,
        best,
    )

    best.score = _score_pose(
        outer_pts,
        best.scale,
        best.angle_deg,
        best.x + pad,
        best.y + pad,
        dist,
        hp,
        wp,
    )
    best.ok = best.score <= score_thresh
    return MatchResult(
        best.x + float(rx1),
        best.y + float(ry1),
        best.angle_deg,
        best.score,
        best.ok,
        best.scale,
    )


def draw_pose_and_match(
    img: np.ndarray,
    box: dict,
    kpts: np.ndarray,
    angle_kpt: float,
    model: ShapeModel,
    match: MatchResult,
    scene_edges: np.ndarray | None,
) -> np.ndarray:
    vis = draw_result(
        img,
        model,
        match,
        color=(0, 255, 0),
        det_box=box,
        scene_edges=scene_edges,
        points=model.outer(),
        stage_title="OUTER refine (±30° from kpt1→0)",
        legend="yellow=scene  green=outer  cyan=kpt1→0  magenta=pose",
    )

    # 关键点 + 1→0 方向
    pts = kpts[:, :2]
    for k, (px, py) in enumerate(pts):
        cv2.circle(vis, (int(px), int(py)), 5, (0, 0, 255), -1)
        cv2.putText(
            vis,
            f"P{k}",
            (int(px) + 5, int(py) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )
    if len(pts) >= 2:
        p0 = (int(pts[0, 0]), int(pts[0, 1]))
        p1 = (int(pts[1, 0]), int(pts[1, 1]))
        cv2.arrowedLine(vis, p1, p0, (255, 255, 0), 2, tipLength=0.25)
        mid = ((p0[0] + p1[0]) // 2, (p0[1] + p1[1]) // 2)
        cv2.putText(
            vis,
            f"kpt ang={angle_kpt:.1f}",
            (mid[0] + 6, mid[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )

    delta = normalize_angle_deg(match.angle_deg - angle_kpt)
    cv2.putText(
        vis,
        f"delta_ang={delta:+.2f}  match_ang={match.angle_deg:.2f}",
        (8, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 220, 255),
        2,
        cv2.LINE_AA,
    )
    return vis


def pick_target(dets: list[dict]) -> dict | None:
    """优先选 pcb(cls=0)，否则选置信度最高。"""
    if not dets:
        return None
    pcbs = [d for d in dets if d["cls_id"] == 0]
    pool = pcbs if pcbs else dets
    return max(pool, key=lambda d: d["conf"] * d["box"]["w"] * d["box"]["h"])


def run(args: argparse.Namespace) -> None:
    model_yolo = YOLO(str(args.weights))
    shape = ShapeModel.load(args.shape_model)

    src = Path(args.source)
    images = sorted(src.glob("*.jpg")) + sorted(src.glob("*.png"))
    # de-dup
    seen = set()
    uniq = []
    for p in images:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    images = uniq

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    viz_dir = out_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    print(f"YOLO: {args.weights}")
    print(f"shape model: {args.shape_model}")
    print(f"images: {len(images)} from {src}")
    print(f"angle_offset={args.angle_offset}  half_range=±{args.angle_range}")

    results_yolo = model_yolo.predict(
        source=str(src),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=False,
        verbose=True,
    )

    rows: list[dict] = []
    summary_results: list[dict] = []

    by_name = {Path(r.path).name: r for r in results_yolo}

    for img_path in images:
        result = by_name.get(img_path.name)
        img = load_bgr(img_path)
        h, w = img.shape[:2]
        image_name = img_path.name
        image_stem = img_path.stem

        dets: list[dict] = []
        if result is not None and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            clss = result.boxes.cls.cpu().numpy()
            kpts_all = None
            if result.keypoints is not None:
                kpts_all = result.keypoints.data.cpu().numpy()

            order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1], boxes[i][0]))
            for det_id, det_idx in enumerate(order):
                x1, y1, x2, y2 = boxes[det_idx]
                conf = float(confs[det_idx])
                cls_id = int(clss[det_idx])
                box = {
                    "cls": cls_id,
                    "x1": int(round(x1)),
                    "y1": int(round(y1)),
                    "x2": int(round(x2)),
                    "y2": int(round(y2)),
                    "cx": float((x1 + x2) / 2.0),
                    "cy": float((y1 + y2) / 2.0),
                    "w": float(x2 - x1),
                    "h": float(y2 - y1),
                    "conf": conf,
                }
                kpts = None
                if kpts_all is not None and det_idx < len(kpts_all):
                    kpts = kpts_all[det_idx]
                dets.append(
                    {
                        "det_index": det_id,
                        "cls_id": cls_id,
                        "cls_name": CLASS_NAMES.get(cls_id, f"unknown_{cls_id}"),
                        "conf": conf,
                        "box": box,
                        "kpts": kpts,
                    }
                )

        # 保存纯 pose 可视化
        pose_vis = img.copy()
        for d in dets:
            color = CLASS_COLORS.get(d["cls_id"], DEFAULT_COLOR)
            b = d["box"]
            if d["kpts"] is not None and d["kpts"].shape[0] >= 4:
                pts = d["kpts"][:4, :2].astype(np.int32)
                cv2.polylines(pose_vis, [pts], True, color, 2)
                for k in range(4):
                    px, py = int(d["kpts"][k, 0]), int(d["kpts"][k, 1])
                    cv2.circle(pose_vis, (px, py), 5, (0, 0, 255), -1)
                    cv2.putText(
                        pose_vis,
                        f"P{k}",
                        (px + 5, py - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
            cv2.putText(
                pose_vis,
                f"{d['cls_id']}:{d['conf']:.2f}",
                (b["x1"], max(20, b["y1"] - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
        cv2.imwrite(str(viz_dir / f"{image_stem}_pose.jpg"), pose_vis)

        target = pick_target(dets)
        if target is None or target["kpts"] is None or target["kpts"].shape[0] < 2:
            print(f"[跳过] {image_name}: 无可用关键点目标")
            summary_results.append(
                {"image": image_name, "ok": False, "reason": "no_kpts"}
            )
            continue

        kpts = target["kpts"]
        box = target["box"]
        # 用关键点中心作为匹配初值中心（通常比 AABB 中心更准）
        kc_x, kc_y = kpt_center(kpts)
        box_for_match = dict(box)
        box_for_match["cx"] = kc_x
        box_for_match["cy"] = kc_y

        angle_kpt = angle_from_kpt10(kpts)
        angle0 = normalize_angle_deg(angle_kpt + args.angle_offset)

        t0 = time.perf_counter()
        match = refine_outer_near_pose(
            img,
            shape,
            box_for_match,
            angle0_deg=angle0,
            angle_half_range=args.angle_range,
            angle_step_coarse=args.angle_step_coarse,
            angle_step_fine=args.angle_step_fine,
            xy_step_coarse=args.xy_step_coarse,
            xy_step_fine=args.xy_step_fine,
            xy_radius=args.xy_radius,
            score_thresh=args.score_thresh,
            edge_mode=args.edge_mode,
            roi_margin=args.roi_margin,
            scale_rel=args.scale_rel,
            scale_step=args.scale_step,
            max_coarse_scales=args.max_coarse_scales,
            coarse_points=args.coarse_points,
            canny1=args.canny1,
            canny2=args.canny2,
            ring_width=args.ring_width,
            min_edge_len=args.min_edge_len,
            blur=args.blur,
        )
        match_ms = (time.perf_counter() - t0) * 1000.0

        scale0 = scale_from_det_box(shape, box["w"], box["h"])
        delta_ang = normalize_angle_deg(match.angle_deg - angle0)
        delta_xy = (match.x - kc_x, match.y - kc_y)

        scene_e = silhouette_edges_for_viz(
            img,
            det_box=box_for_match,
            roi_margin=args.roi_margin,
            canny1=args.canny1,
            canny2=args.canny2,
            blur=args.blur,
        )
        if not np.any(scene_e):
            scene_e = scene_edges_for_viz(
                img,
                edge_mode=args.edge_mode,
                det_box=box_for_match,
                roi_margin=args.roi_margin,
                canny1=args.canny1,
                canny2=args.canny2,
                ring_width=args.ring_width,
                min_edge_len=args.min_edge_len,
                blur=args.blur,
            )

        match_vis = draw_pose_and_match(
            img, box_for_match, kpts, angle_kpt, shape, match, scene_e
        )
        cv2.imwrite(str(viz_dir / f"{image_stem}_match.jpg"), match_vis)

        row = {
            "target_id": f"{image_stem}_{target['det_index']}",
            "image_name": image_name,
            "det_index": target["det_index"],
            "cls_id": target["cls_id"],
            "cls_name": target["cls_name"],
            "conf": target["conf"],
            "bbox_x1": box["x1"],
            "bbox_y1": box["y1"],
            "bbox_x2": box["x2"],
            "bbox_y2": box["y2"],
            "kpt_center_x": kc_x,
            "kpt_center_y": kc_y,
            "angle_kpt_deg": round(angle_kpt, 4),
            "angle_init_deg": round(angle0, 4),
            "angle_offset_deg": args.angle_offset,
            "match_x": round(match.x, 4),
            "match_y": round(match.y, 4),
            "match_angle_deg": round(match.angle_deg, 4),
            "match_scale": round(match.scale, 6),
            "match_score_px": round(match.score, 4),
            "match_ok": bool(match.ok),
            "delta_angle_deg": round(delta_ang, 4),
            "delta_x": round(delta_xy[0], 4),
            "delta_y": round(delta_xy[1], 4),
            "scale_from_det": round(scale0, 6),
            "match_ms": round(match_ms, 2),
        }
        for k in range(min(4, kpts.shape[0])):
            row[f"p{k}_x"] = float(kpts[k, 0])
            row[f"p{k}_y"] = float(kpts[k, 1])
            row[f"p{k}_conf"] = float(kpts[k, 2]) if kpts.shape[1] >= 3 else -1.0

        rows.append(row)
        summary_results.append(row)

        print(
            f"{image_name}: kpt_ang={angle_kpt:.1f} → match_ang={match.angle_deg:.1f} "
            f"(Δ{delta_ang:+.2f}°)  xy=({match.x:.1f},{match.y:.1f}) "
            f"Δxy=({delta_xy[0]:+.1f},{delta_xy[1]:+.1f}) "
            f"s={match.scale:.3f} score={match.score:.2f} ok={match.ok} "
            f"{match_ms:.0f}ms",
            flush=True,
        )

    # CSV
    csv_path = out_dir / "pose_outer_refine.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        all_keys = set()
        for r in rows:
            all_keys.update(r.keys())
        for k in sorted(all_keys):
            if k not in fieldnames:
                fieldnames.append(k)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    summary = {
        "weights": str(args.weights),
        "shape_model": str(args.shape_model),
        "source": str(src),
        "angle_offset_deg": args.angle_offset,
        "angle_half_range_deg": args.angle_range,
        "n_images": len(images),
        "n_matched": len(rows),
        "n_ok": sum(1 for r in rows if r.get("match_ok")),
        "avg_match_ms": round(
            sum(r["match_ms"] for r in rows) / len(rows), 2
        )
        if rows
        else 0.0,
        "results": summary_results,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(
        f"\n完成: ok={summary['n_ok']}/{summary['n_matched']} "
        f"avg={summary['avg_match_ms']:.1f}ms\n"
        f"CSV: {csv_path}\n"
        f"summary: {out_dir / 'summary.json'}\n"
        f"viz: {viz_dir}"
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="YOLO pose + outer contour refine")
    p.add_argument(
        "--weights",
        default="/media/nas_users/wangzixiang/best.pt",
    )
    p.add_argument(
        "--source",
        default="/media/nas_users/wangzixiang/260322274901",
    )
    p.add_argument(
        "--shape-model",
        default="/media/nas_users/wangzixiang/shape_match/models/pcb_labelme_12.npz",
        help="外轮廓模板；默认 pcb_labelme_12.npz",
    )
    p.add_argument(
        "--out",
        default="/media/nas_users/wangzixiang/260322274901/pose_outer_refine",
    )
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--conf", type=float, default=0.5)
    p.add_argument("--device", default="0")
    p.add_argument(
        "--angle-offset",
        type=float,
        default=0.0,
        help="模板 0° 相对 P1→P0 的固定偏置（度）",
    )
    p.add_argument(
        "--angle-range",
        type=float,
        default=30.0,
        help="外轮廓微调半幅（度），默认 ±30",
    )
    p.add_argument(
        "--angle-step-coarse",
        type=float,
        default=3.0,
        help="粗搜角度步长（度）",
    )
    p.add_argument(
        "--max-coarse-scales",
        type=int,
        default=8,
        help="粗搜最多使用的 scale 数",
    )
    p.add_argument(
        "--coarse-points",
        type=int,
        default=80,
        help="粗搜外轮廓采样点数",
    )
    p.add_argument(
        "--angle-step-fine",
        type=float,
        default=0.5,
        help="细搜角度步长（度）",
    )
    p.add_argument("--xy-step-coarse", type=int, default=4, help="粗搜平移步长（px）")
    p.add_argument("--xy-step-fine", type=int, default=1, help="细搜平移步长（px）")
    p.add_argument("--xy-radius", type=int, default=12)
    p.add_argument("--score-thresh", type=float, default=5.2)
    p.add_argument(
        "--edge-mode",
        choices=("outer-ring", "silhouette", "canny", "auto"),
        default="outer-ring",
    )
    p.add_argument("--roi-margin", type=float, default=0.25)
    p.add_argument("--scale-rel", type=float, default=0.2)
    p.add_argument(
        "--scale-step",
        type=float,
        default=0.04,
        help="粗搜 scale 步长（细搜在最优附近收紧）",
    )
    p.add_argument("--canny1", type=int, default=40)
    p.add_argument("--canny2", type=int, default=120)
    p.add_argument("--blur", type=int, default=5)
    p.add_argument("--ring-width", type=int, default=18)
    p.add_argument("--min-edge-len", type=float, default=40.0)
    return p


if __name__ == "__main__":
    run(build_parser().parse_args())
