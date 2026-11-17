#!/usr/bin/env python3
"""实时一套流程：开相机 → 头粗定 → 按 cls 选抓取方式（小板腕精收夹 / 大板双臂对挤）。

只用本仓库环境，推荐::

  cd /home/ubuntu/workspace/pcb_grasp
  ./run_live.sh --real --record-grasp-z          # 示教小板夹取高度
  ./run_live.sh --real --vis output/live_vis     # 全流程（自动按识别选方式）
  ./run_live.sh --force-class large --vis ...    # 无大板权重时强制大板几何
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_grasp.cameras import LiveCameras
from pcb_grasp.config import load_config
from pcb_grasp.motion import DryRunMotion
from pcb_grasp.pipeline import GraspPipeline

LOG = logging.getLogger("run_live")


def main() -> int:
    ap = argparse.ArgumentParser(description="PCB live grasp: cameras → localize → move")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "grasp_config.yaml")
    ap.add_argument(
        "--vis",
        type=Path,
        default=ROOT / "output" / "live_vis",
        help="实验根目录；每次启动新建 batch_NNN_时间戳/ 存本轮全部图与深度",
    )
    ap.add_argument(
        "--real",
        action="store_true",
        help="接真机（默认 rim_python.move_to_pose；配置 motion_backend=tj 可回退）；默认只算位姿",
    )
    ap.add_argument("--arm", choices=("A", "B"), default=None)
    ap.add_argument(
        "--no-grasp",
        action="store_true",
        help="小板：只到示教高度不收夹；大板：只接近不对挤",
    )
    ap.add_argument(
        "--record-grasp-z",
        action="store_true",
        help="读取当前法兰 Z 写入 config/grasp_z.json（示教夹取高度），然后退出",
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=None,
        help="每次运动结束后额外等待秒数，再判静止并识别（默认读配置）",
    )
    ap.add_argument(
        "--stage",
        choices=("all", "coarse"),
        default="all",
        help="all=头粗+腕精；coarse=只头部粗定位并悬停（不跑腕）；大板忽略腕",
    )
    ap.add_argument(
        "--force-class",
        choices=("large", "small"),
        default=None,
        help="强制按大板/小板流程（模型未训好时可测大板几何）",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    if args.arm:
        cfg.tj_arm = args.arm
    settle = float(args.settle) if args.settle is not None else cfg.live_settle_s

    motion = None
    dual = None
    need_real = bool(args.real or args.record_grasp_z)
    if need_real:
        from pcb_grasp.robot_factory import connect_real_motion

        motion, dual = connect_real_motion(cfg, record_grasp_z=bool(args.record_grasp_z))
    else:
        from pcb_grasp.dual_tj_motion import DryRunDualMotion

        motion = DryRunMotion(initial_pose=cfg.home_pose)
        dual = DryRunDualMotion()
        dual.connect()
        LOG.warning("DRY-RUN：不算真机动臂。确认位姿后加 --real")

    if args.record_grasp_z:
        from pcb_grasp.grasp_z import save_grasp_z_m

        assert motion is not None
        pose = motion.get_flange_pose()
        out = save_grasp_z_m(
            float(pose[2]),
            cfg.grasp_z_file or None,
            meta={
                "xyz_m": [round(float(x), 5) for x in pose[:3]],
                "rpy_rad": [round(float(x), 5) for x in pose[3:6]],
                "arm": cfg.tj_arm,
            },
        )
        print(
            json.dumps(
                {
                    "ok": True,
                    "grasp_z_m": float(pose[2]),
                    "grasp_z_mm": round(float(pose[2]) * 1000.0, 2),
                    "file": str(out),
                    "pose_m_rad": [round(float(x), 5) for x in pose[:6]],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        if dual is not None:
            dual.close()
        elif motion is not None and hasattr(motion, "close"):
            motion.close()
        return 0

    if cfg.grasp_z_m is not None:
        LOG.info("grasp_z_m=%.1fmm (fixed grasp height)", cfg.grasp_z_m * 1000.0)
    else:
        LOG.warning(
            "grasp_z_m not set — teach with: ./run_live.sh --real --record-grasp-z "
            "(仅小板需要；大板对夹不依赖)"
        )

    from pcb_grasp.experiment_log import next_batch_dir

    batch_dir = next_batch_dir(args.vis)
    LOG.info("this experiment batch: %s", batch_dir)
    if args.force_class:
        LOG.info("force_class=%s", args.force_class)

    pipe = GraspPipeline(
        cfg=cfg, motion=motion, dual_motion=dual, force_class=args.force_class
    )

    try:
        with LiveCameras(cfg) as cams:
            result = pipe.run_live(
                cams.grab_head,
                cams.grab_wrist,
                settle_s=settle,
                do_grasp=(not args.no_grasp),
                save_vis_dir=batch_dir,
                stage=args.stage,
            )
    finally:
        if dual is not None:
            dual.close()
        elif motion is not None and hasattr(motion, "close"):
            motion.close()

    if result.debug is not None:
        result.debug["batch_dir"] = str(batch_dir)
        if args.force_class:
            result.debug["force_class"] = args.force_class
    payload = {
        "ok": result.ok,
        "message": result.message,
        "batch_dir": str(batch_dir),
        "hover_flange": result.hover_flange,
        "fine_flange": result.fine_flange,
        "coarse": None
        if result.coarse is None
        else {
            "xyz": list(result.coarse.xyz),
            "rz": float(result.coarse.rz),
            "cls_id": int(result.coarse.cls_id),
            "board_id": result.coarse.board_id,
        },
        "fine": None
        if result.fine is None
        else {
            "xyz": list(result.fine.xyz),
            "rz": float(result.fine.rz),
        },
        "debug": result.debug,
    }
    try:
        (batch_dir / "result.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception:
        pass
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
