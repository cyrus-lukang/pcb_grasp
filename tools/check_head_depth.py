#!/usr/bin/env python3
"""先检查头部双目 FFS 深度是否靠谱，再谈粗定位。

用法（建议用带 torch/cuda 的环境）:
  python tools/check_head_depth.py \\
    --left raw_left.jpg --right raw_right.jpg \\
    --out /tmp/head_depth_check

输出:
  left_rect.jpg / right_rect.jpg / depth_mm.png / depth_color.jpg / stats.json
  以及 valid 比例、深度分位数。valid 太低或中位深度离谱 → 先修标定/FFS，别直接抓。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb_grasp.stereo_depth import HeadStereoDepth, colorize_depth  # noqa: E402
from pcb_grasp.undistort import build_maps_from_stereo_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument(
        "--calib",
        type=Path,
        default=ROOT / "calib" / "stereo" / "stereo_calib.json",
    )
    ap.add_argument("--out", type=Path, default=Path("/tmp/head_depth_check"))
    ap.add_argument("--already-rectified", action="store_true")
    ap.add_argument("--min-valid", type=float, default=0.15)
    ap.add_argument("--z-lo", type=float, default=0.2)
    ap.add_argument("--z-hi", type=float, default=2.5)
    args = ap.parse_args()

    left = cv2.imread(str(args.left), cv2.IMREAD_COLOR)
    right = cv2.imread(str(args.right), cv2.IMREAD_COLOR)
    if left is None or right is None:
        print("cannot read images", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    maps = build_maps_from_stereo_json(args.calib)
    est = HeadStereoDepth(maps=maps, stereo_calib_json=args.calib)

    if args.already_rectified:
        depth, valid, stats = est.estimate_rectified(left, right)
        left_r, right_r = left, right
    else:
        left_r, right_r, depth, valid, stats = est.estimate_raw(left, right)

    cv2.imwrite(str(args.out / "left_rect.jpg"), left_r)
    cv2.imwrite(str(args.out / "right_rect.jpg"), right_r)
    cv2.imwrite(str(args.out / "depth_mm.png"), depth)
    cv2.imwrite(str(args.out / "depth_color.jpg"), colorize_depth(depth))
    cv2.imwrite(str(args.out / "valid_mask.png"), (valid.astype("uint8") * 255))

    ok = bool(stats.ok(args.min_valid, args.z_lo, args.z_hi))
    payload = {
        "valid_ratio": float(stats.valid_ratio),
        "median_m": float(stats.median_m),
        "p10_m": float(stats.p10_m),
        "p90_m": float(stats.p90_m),
        "fx": float(est.fx) if est.fx is not None else None,
        "baseline_mm": float(est.baseline_mm) if est.baseline_mm is not None else None,
        "ok": ok,
        "criteria": {
            "min_valid": float(args.min_valid),
            "z_lo": float(args.z_lo),
            "z_hi": float(args.z_hi),
        },
        "left": str(args.left),
        "right": str(args.right),
    }
    (args.out / "stats.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(stats.summary())
    print("ok=" + str(ok))
    print("wrote", args.out)
    if not ok:
        print(
            "DEPTH NOT TRUSTWORTHY yet — fix rectify/calib/FFS before using head for grasp XY/Z.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
