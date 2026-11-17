#!/usr/bin/env python3
"""CLI: 头鱼眼粗定位 → 上方 30cm → 腕部精定位(只动 XY + rz)。

建议顺序:
  1) 先查头部双目深度是否靠谱
       python tools/check_head_depth.py --left L.jpg --right R.jpg --out /tmp/hd
  2) 深度 OK 后再跑本脚本（默认 dry-run）
  3) 接真机: --real（走 TJ_demo 笛卡尔阻抗 + TipFT 碰触急停）
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_grasp.config import load_config
from pcb_grasp.motion import DryRunMotion
from pcb_grasp.pipeline import GraspPipeline

LOG = logging.getLogger("run_grasp")


def _read_bgr(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def _read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(path)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    return depth


def main() -> int:
    ap = argparse.ArgumentParser(description="PCB coarse+fine grasp (XY fine, Z held)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "grasp_config.yaml")
    ap.add_argument("--head", type=Path, required=True, help="头部左鱼眼原图")
    ap.add_argument("--head-right", type=Path, default=None, help="头部右鱼眼原图（FFS 深度）")
    ap.add_argument("--wrist", type=Path, required=True, help="腕部彩色图")
    ap.add_argument("--wrist-depth", type=Path, required=True, help="腕部深度图 uint16/float")
    ap.add_argument("--head-depth", type=Path, default=None, help="可选：已有头部深度图(mm)")
    ap.add_argument("--already-undistorted", action="store_true")
    ap.add_argument("--vis", type=Path, default=None)
    ap.add_argument("--do-grasp", action="store_true", help="预留：闭合夹爪（暂不降 Z）")
    ap.add_argument(
        "--real",
        action="store_true",
        help="接 TJ_demo 真机（阻抗+碰触急停）；默认 dry-run",
    )
    ap.add_argument("--arm", choices=("A", "B"), default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.arm:
        cfg.tj_arm = args.arm

    motion = None
    if args.real:
        from pcb_grasp.tj_motion import TJDemoMotion

        cfg.dry_run = False
        motion = TJDemoMotion(
            robot_ip=cfg.tj_robot_ip,
            arm=cfg.tj_arm,
            use_impedance=cfg.tj_use_impedance,
            contact_threshold_n=cfg.tj_contact_threshold_n,
            cart_vel_mm_s=cfg.cart_vel_mm_s,
            cart_acc_mm_s2=cfg.cart_acc_mm_s2,
            vel_ratio=cfg.tj_vel_ratio,
            acc_ratio=cfg.tj_acc_ratio,
            dry_run=False,
        )
        motion.connect()
        arm = str(cfg.tj_arm).upper()
        ready = cfg.ready_joints_A if arm == "A" else cfg.ready_joints_B
        if ready:
            motion.preferred_ik_seed = list(ready)
        motion.safe_z_m = float(cfg.safe_z_m)
        LOG.info(
            "REAL robot connected via TJ_demo arm=%s impedance=%s safe_z=%.0fmm",
            cfg.tj_arm,
            cfg.tj_use_impedance,
            cfg.safe_z_m * 1000.0,
        )
    else:
        motion = DryRunMotion(initial_pose=cfg.home_pose)

    pipe = GraspPipeline(cfg=cfg, motion=motion)

    head = _read_bgr(args.head)
    head_r = _read_bgr(args.head_right) if args.head_right else None
    wrist = _read_bgr(args.wrist)
    wdepth = _read_depth(args.wrist_depth)
    hdepth = _read_depth(args.head_depth) if args.head_depth else None

    try:
        result = pipe.run(
            head,
            wrist,
            wdepth,
            head_right_bgr=head_r,
            head_depth=hdepth,
            already_undistorted=args.already_undistorted,
            do_grasp=args.do_grasp,
            save_vis_dir=args.vis,
        )
    finally:
        if args.real and motion is not None:
            motion.close()

    payload = {
        "ok": result.ok,
        "message": result.message,
        "coarse": None
        if result.coarse is None
        else {
            "xyz": result.coarse.xyz,
            "rz_rad": result.coarse.rz,
            "edge_yaw_rad": result.coarse.edge_yaw,
            "conf": result.coarse.conf,
            "corners_base": result.coarse.corners_base,
        },
        "fine": None
        if result.fine is None
        else {
            "xyz": result.fine.xyz,
            "rz_rad": result.fine.rz,
            "edge_yaw_rad": result.fine.edge_yaw,
            "conf": result.fine.conf,
            "corners_base": result.fine.corners_base,
        },
        "hover_flange": result.hover_flange,
        "fine_flange": result.fine_flange,
        "motion_history": getattr(motion, "history", []),
        "debug": result.debug,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
