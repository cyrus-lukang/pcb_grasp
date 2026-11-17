#!/usr/bin/env python3
"""
Visualize matched mesh contour vs scene-segmentation silhouette contour.

The Chamfer `score` compares:
  - model edge points after (x,y,angle,scale)  →  "匹配 mesh 轮廓"
  - scene edges in the det ROI (silhouette/canny) → "分割轮廓"
NOT a separately saved SAM mask file.

  .venv/bin/python viz_contour_compare.py \
    --set ../captures/260322274901_20261010_merged_平行 \
    --match-tag match_rot180 \
    --model models/d435_pcb.npz \
    --out ../captures/260322274901_20261010_merged_平行/contour_compare_B
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from shape_matcher import (
    ShapeModel,
    expand_box,
    extract_scene_edges,
    load_bgr,
    parse_yolo_labels,
    _rotate_points,
)


def model_from_edges_png(path: Path) -> ShapeModel:
    """Rebuild a ShapeModel from a saved *.edges.png preview."""
    edges = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if edges is None:
        raise FileNotFoundError(path)
    _, bw = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)
    ys, xs = np.where(bw > 0)
    if len(xs) < 30:
        raise ValueError(f"too few edge pixels in {path}")
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    # fill contour for centroid
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    fill = np.zeros_like(bw)
    if cnts:
        cv2.drawContours(fill, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
        m = cv2.moments(fill, binaryImage=True)
        if m["m00"] > 1e-3:
            c = np.array([m["m10"] / m["m00"], m["m01"] / m["m00"]], np.float32)
        else:
            c = pts.mean(axis=0)
    else:
        c = pts.mean(axis=0)
    if len(pts) > 600:
        idx = np.linspace(0, len(pts) - 1, 600).astype(int)
        pts = pts[idx]
    h, w = bw.shape[:2]
    return ShapeModel(points=(pts - c).astype(np.float32), angle_step_deg=2.0, size=(w, h))


def draw_overlay(
    bgr: np.ndarray,
    model: ShapeModel,
    row: dict,
    edge_mode: str = "silhouette",
    roi_margin: float = 0.25,
) -> np.ndarray:
    h, w = bgr.shape[:2]
    box = {
        "x1": row["det_box"][0],
        "y1": row["det_box"][1],
        "x2": row["det_box"][2],
        "y2": row["det_box"][3],
        "cx": 0.5 * (row["det_box"][0] + row["det_box"][2]),
        "cy": 0.5 * (row["det_box"][1] + row["det_box"][3]),
        "w": row["det_box"][2] - row["det_box"][0],
        "h": row["det_box"][3] - row["det_box"][1],
    }
    rx1, ry1, rx2, ry2 = expand_box(
        box["x1"], box["y1"], box["x2"], box["y2"], w, h, margin=roi_margin
    )
    work = bgr[ry1:ry2, rx1:rx2]
    mode = "silhouette" if edge_mode == "auto" else edge_mode
    edges = extract_scene_edges(work, mode=mode)
    if not np.any(edges) and mode == "silhouette":
        edges = extract_scene_edges(work, mode="canny")

    vis = bgr.copy()
    # det box
    cv2.rectangle(
        vis,
        (int(box["x1"]), int(box["y1"])),
        (int(box["x2"]), int(box["y2"])),
        (0, 140, 255),
        2,
    )
    # ROI
    cv2.rectangle(vis, (rx1, ry1), (rx2, ry2), (90, 90, 90), 1)

    # scene silhouette / seg contour (cyan) — what Chamfer compares against
    if np.any(edges):
        ys, xs = np.where(edges > 0)
        for x, y in zip(xs[::2], ys[::2]):
            cv2.circle(vis, (int(x + rx1), int(y + ry1)), 1, (255, 220, 0), -1)  # cyan-ish BGR
        # also draw outer contour polyline thicker
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            c = max(cnts, key=cv2.contourArea)
            c = c + np.array([[[rx1, ry1]]])
            cv2.drawContours(vis, [c], -1, (255, 220, 0), 2)

    # matched mesh contour (green)
    rot = _rotate_points(model.points * float(row["scale"]), float(row["angle_deg"]))
    world = rot + np.array([row["x"], row["y"]], np.float32)
    for p in np.round(world).astype(np.int32):
        x, y = int(p[0]), int(p[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis, (x, y), 1, (0, 255, 0), -1)

    # center + arrow
    tip = _rotate_points(np.array([[40.0, 0.0]], np.float32), float(row["angle_deg"]))[0]
    cv2.circle(vis, (int(row["x"]), int(row["y"])), 5, (0, 0, 255), 2)
    cv2.arrowedLine(
        vis,
        (int(row["x"]), int(row["y"])),
        (int(row["x"] + tip[0]), int(row["y"] + tip[1])),
        (0, 0, 255),
        2,
        tipLength=0.3,
    )

    score = row.get("score_px", row.get("score"))
    ok = row.get("ok")
    label1 = (
        f"{row['image']}  ang={row['angle_deg']:.1f} s={row['scale']:.2f} "
        f"score={score:.2f} ok={ok}"
    )
    label2 = "cyan=分割轮廓(silhouette)  green=匹配mesh轮廓  score=两者Chamfer"
    cv2.putText(vis, label1, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
    cv2.putText(vis, label2, (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 220, 0), 2)
    return vis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", required=True, help="capture set dir")
    ap.add_argument("--match-tag", default="match_rot180")
    ap.add_argument("--model", default="", help="npz model; empty → rebuild from --edges")
    ap.add_argument("--edges", default="", help="edges.png to rebuild model (for old Mesh A)")
    ap.add_argument("--labels", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edge-mode", default="silhouette")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    set_dir = Path(args.set)
    summary = json.loads((set_dir / args.match_tag / "summary.json").read_text())
    rows = summary.get("results") or []
    if args.limit > 0:
        rows = rows[: args.limit]

    if args.model:
        model = ShapeModel.load(args.model)
    elif args.edges:
        model = model_from_edges_png(Path(args.edges))
    else:
        raise SystemExit("need --model or --edges")

    label_dir = Path(args.labels) if args.labels else set_dir / "labels_pcb" / "labels"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    n = 0
    for row in rows:
        img_path = set_dir / "color" / row["image"]
        if not img_path.is_file():
            continue
        # prefer summary det_box; refresh from label if missing
        img = load_bgr(img_path)
        if "det_box" not in row:
            h, w = img.shape[:2]
            boxes = parse_yolo_labels(label_dir / f"{img_path.stem}.txt", w, h)
            if not boxes:
                continue
            b = max(boxes, key=lambda x: x["w"] * x["h"])
            row = dict(row)
            row["det_box"] = [b["x1"], b["y1"], b["x2"], b["y2"]]
        vis = draw_overlay(img, model, row, edge_mode=args.edge_mode)
        cv2.imwrite(str(out / f"{img_path.stem}_contour.jpg"), vis)
        n += 1
        if n % 20 == 0:
            print(f"... {n}/{len(rows)}", flush=True)

    meta = {
        "set": str(set_dir),
        "match_tag": args.match_tag,
        "model": args.model or args.edges,
        "n": n,
        "note": "cyan=scene silhouette contour used by Chamfer; green=matched mesh contour",
    }
    (out / "README.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(json.dumps(meta, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
