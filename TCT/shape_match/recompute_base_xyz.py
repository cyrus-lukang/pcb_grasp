#!/usr/bin/env python3
"""
Recompute summary.json base xyz using real depth_aligned ROI median.
Does not re-run matching — only reprojects existing (x,y,angle) with depth_roi.

  .venv/bin/python recompute_base_xyz.py --captures ../captures --tags match,match_rot180,match_partial_0000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from handeye_base import (
    DEFAULT_HANDEYE,
    DEFAULT_INTRINSICS,
    angle_diff_deg,
    load_T_tool_cam,
    load_intrinsics,
    match_to_base,
)
from shape_matcher import (
    _load_ee_pose_map,
    _resolve_ee,
    board_center_from_det_box,
    depth_roi_median,
    load_depth_m,
    resolve_depth_path,
)


def recompute_summary(
    summary_path: Path,
    set_dir: Path,
    T_tool_cam: np.ndarray,
    K,
    fallback_depth: float = 0.35,
    loc_center: str = "det_box",
) -> dict:
    data = json.loads(summary_path.read_text())
    rows = data.get("results") or []
    depth_dir = set_dir / "depth_aligned"
    ee_map = _load_ee_pose_map(str(set_dir / "ee_poses.json")) if (set_dir / "ee_poses.json").is_file() else {}

    n_real = 0
    n_fallback = 0
    for row in rows:
        box = {
            "x1": row["det_box"][0],
            "y1": row["det_box"][1],
            "x2": row["det_box"][2],
            "y2": row["det_box"][3],
        }
        depth_m = None
        if depth_dir.is_dir():
            dp = resolve_depth_path(depth_dir, row["image"])
            if dp is not None:
                dm = load_depth_m(dp)
                if dm is not None:
                    depth_m = depth_roi_median(dm, box)
        if depth_m is None:
            depth_m = fallback_depth
            n_fallback += 1
        else:
            n_real += 1

        if loc_center == "det_box":
            u, v = board_center_from_det_box(box)
        else:
            u, v = float(row["x"]), float(row["y"])
        row["board_cx"] = round(float(u), 2)
        row["board_cy"] = round(float(v), 2)
        row["match_cx"] = round(float(row["x"]), 2)
        row["match_cy"] = round(float(row["y"]), 2)

        ee = _resolve_ee(ee_map, row["image"], None)
        base = match_to_base(
            float(u),
            float(v),
            float(row["angle_deg"]),
            T_tool_cam=T_tool_cam,
            K=K,
            depth_m=float(depth_m),
            ee_xyz_rpy=ee,
        )
        row["base"] = {
            "x_m": round(base.x, 5),
            "y_m": round(base.y, 5),
            "z_m": round(base.z, 5),
            "yaw_deg": round(base.yaw_deg, 2),
            "frame": base.frame,
            "depth_m_used": round(float(depth_m), 5),
            "loc_center": loc_center,
        }
        # keep PCA yaw error if we still have ref_angle_deg
        if row.get("ref_angle_deg") is not None:
            base_ref = match_to_base(
                float(u),
                float(v),
                float(row["ref_angle_deg"]),
                T_tool_cam=T_tool_cam,
                K=K,
                depth_m=float(depth_m),
                ee_xyz_rpy=ee,
            )
            yaw_err = angle_diff_deg(base.yaw_deg, base_ref.yaw_deg)
            row["base"]["ref_yaw_deg"] = round(base_ref.yaw_deg, 2)
            row["base"]["yaw_error_deg"] = round(yaw_err, 2)
            row["base"]["yaw_error_abs_deg"] = round(abs(yaw_err), 2)

    data["results"] = rows
    data["depth_dir"] = str(depth_dir) if depth_dir.is_dir() else None
    data["depth_m_fallback"] = fallback_depth
    data["loc_center"] = loc_center
    data["depth_recomputed"] = {"n_real": n_real, "n_fallback": n_fallback}
    summary_path.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    return data["depth_recomputed"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="../captures")
    ap.add_argument("--tags", default="match,match_rot180,match_partial_0000")
    ap.add_argument("--handeye", default=str(DEFAULT_HANDEYE))
    ap.add_argument("--intrinsics", default=str(DEFAULT_INTRINSICS))
    ap.add_argument("--depth-m", type=float, default=0.35)
    ap.add_argument(
        "--loc-center",
        choices=("det_box", "match"),
        default="det_box",
        help="det_box=full board center from L/R edge mids; match=mesh centroid",
    )
    args = ap.parse_args()

    captures = Path(args.captures)
    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    T = load_T_tool_cam(args.handeye)
    K = load_intrinsics(args.intrinsics)

    for set_dir in sorted(captures.glob("260322274901*")):
        if not set_dir.is_dir():
            continue
        for tag in tags:
            sp = set_dir / tag / "summary.json"
            if not sp.is_file():
                continue
            stats = recompute_summary(
                sp, set_dir, T, K, args.depth_m, loc_center=args.loc_center
            )
            print(
                f"{set_dir.name}/{tag}: real={stats['n_real']} "
                f"fallback={stats['n_fallback']} loc={args.loc_center}"
            )


if __name__ == "__main__":
    main()
