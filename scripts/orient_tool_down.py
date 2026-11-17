#!/usr/bin/env python3
"""只改末端姿态（允许小幅 XYZ），不定位、不开相机。

用来对齐「竖直向下」到底是哪组 rpy：

  rx90   → (-90°,  0°, 当前 rz)   ← 你觉得更像竖直的候选
  tip180 → (180°,  0°, 当前 rz)   ← 旧版 tip-down / ComputeFlangeTarget
  custom → --rx-deg --ry-deg [--rz-deg]

默认：先抬 ``--lift-mm``（默认 30mm）再转到目标姿态（同一高度），
XYZ 可小动，避免纯转姿态时 MC arrive timeout、臂不动。

默认只打印目标，加 ``--execute`` 才下发。

用法::

  cd /home/ubuntu/workspace/pcb_grasp
  ./orient_tool_down.sh --arm A --mode rx90
  ./orient_tool_down.sh --arm A --mode rx90 --execute
  ./orient_tool_down.sh --arm A --mode tip180 --execute --lift-mm 40
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_grasp.config import load_config
from pcb_grasp.geometry import pose_xyz_rpy_to_mat4
from pcb_grasp.rim_motion import _ori_err_deg

LOG = logging.getLogger("orient_tool_down")


def _rpy_deg(pose6: list[float]) -> tuple[float, float, float]:
    return (
        math.degrees(float(pose6[3])),
        math.degrees(float(pose6[4])),
        math.degrees(float(pose6[5])),
    )


def _axis_report(pose6: list[float]) -> dict:
    R = pose_xyz_rpy_to_mat4(*[float(x) for x in pose6[:6]])[:3, :3]
    down = np.array([0.0, 0.0, -1.0], dtype=np.float64)

    def ang(axis: np.ndarray) -> float:
        c = float(np.clip(float(axis @ down), -1.0, 1.0))
        return float(math.degrees(math.acos(c)))

    return {
        "tool_X_base": [round(float(v), 4) for v in R[:, 0]],
        "tool_Y_base": [round(float(v), 4) for v in R[:, 1]],
        "tool_Z_base": [round(float(v), 4) for v in R[:, 2]],
        "tool_X_vs_world_down_deg": round(ang(R[:, 0]), 2),
        "tool_Y_vs_world_down_deg": round(ang(R[:, 1]), 2),
        "tool_Z_vs_world_down_deg": round(ang(R[:, 2]), 2),
        "note": "哪个轴 vs_world_down 最接近 0，就是当前朝地的工具轴",
    }


def _goal_rpy(
    cur: list[float],
    *,
    mode: str,
    rx_deg: float | None,
    ry_deg: float | None,
    rz_deg: float | None,
) -> tuple[float, float, float]:
    rz_keep = float(cur[5]) if rz_deg is None else math.radians(float(rz_deg))
    if mode == "rx90":
        return math.radians(-90.0), 0.0, rz_keep
    if mode == "tip180":
        return math.pi, 0.0, rz_keep
    if mode == "custom":
        if rx_deg is None or ry_deg is None:
            raise SystemExit("custom 模式必须给 --rx-deg 和 --ry-deg")
        return (
            math.radians(float(rx_deg)),
            math.radians(float(ry_deg)),
            rz_keep,
        )
    raise SystemExit(f"未知 mode={mode}")


def _pose_brief(pose6: list[float]) -> dict:
    return {
        "xyz_m": [round(v, 4) for v in pose6[:3]],
        "rpy_deg": [round(v, 2) for v in _rpy_deg(pose6)],
        "axes": _axis_report(pose6),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="小幅 XYZ + 改姿态到候选竖直向下")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "grasp_config.yaml")
    ap.add_argument("--arm", choices=("A", "B"), default="A")
    ap.add_argument(
        "--mode",
        choices=("rx90", "tip180", "custom"),
        default="rx90",
        help="rx90=(-90,0,rz); tip180=(180,0,rz); custom=--rx-deg/--ry-deg",
    )
    ap.add_argument("--rx-deg", type=float, default=None)
    ap.add_argument("--ry-deg", type=float, default=None)
    ap.add_argument(
        "--rz-deg",
        type=float,
        default=None,
        help="默认保留当前 rz；给了就覆盖",
    )
    ap.add_argument(
        "--lift-mm",
        type=float,
        default=30.0,
        help="先抬这么多毫米再转姿态（给 MC 一点平移，避免纯转不动）；0=不抬",
    )
    ap.add_argument(
        "--execute",
        action="store_true",
        help="真下发；默认只打印当前/目标",
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=0.35,
        help="move 速度缩放（纯姿态建议别太慢）",
    )
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = load_config(args.config)
    cfg.tj_arm = args.arm

    from pcb_grasp.robot_factory import connect_real_motion

    motion, _dual = connect_real_motion(cfg, record_grasp_z=False)
    cur = [float(x) for x in motion.get_flange_pose()[:6]]
    rx, ry, rz = _goal_rpy(
        cur,
        mode=args.mode,
        rx_deg=args.rx_deg,
        ry_deg=args.ry_deg,
        rz_deg=args.rz_deg,
    )
    lift_m = max(0.0, float(args.lift_mm) / 1000.0)
    # 目标：当前 XY，Z 抬一点，姿态换成候选竖直
    goal = [float(cur[0]), float(cur[1]), float(cur[2]) + lift_m, rx, ry, rz]
    # 中间点：先抬、保当前姿态（小平移），再转
    via = [
        float(cur[0]),
        float(cur[1]),
        float(cur[2]) + lift_m,
        float(cur[3]),
        float(cur[4]),
        float(cur[5]),
    ]

    ori_now = _ori_err_deg(cur, goal)
    out = {
        "arm": args.arm,
        "mode": args.mode,
        "execute": bool(args.execute),
        "lift_mm": float(args.lift_mm),
        "ori_err_to_goal_deg": round(ori_now, 2),
        "current": _pose_brief(cur),
        "via_lift_keep_rpy": _pose_brief(via) if lift_m > 1e-4 else None,
        "goal": _pose_brief(goal),
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))

    if not args.execute:
        LOG.info(
            "未下发（加 --execute 才会动）。将抬 %.0fmm 再转到 rpy_deg=(%.1f,%.1f,%.1f)",
            float(args.lift_mm),
            *_rpy_deg(goal),
        )
        return 0

    scale = float(args.scale)
    if lift_m > 1e-4:
        LOG.info(
            "EXECUTE lift +%.0fmm keep rpy → then orient rpy_deg=(%.1f,%.1f,%.1f)",
            float(args.lift_mm),
            *_rpy_deg(goal),
        )
        motion.move_line(
            via,
            velocity_scale=scale,
            keep_orientation=False,
            direct=True,
        )
    else:
        LOG.info(
            "EXECUTE orient (no lift) → rpy_deg=(%.1f,%.1f,%.1f)",
            *_rpy_deg(goal),
        )

    after = motion.move_line(
        goal,
        velocity_scale=scale,
        keep_orientation=False,
        direct=True,
    )
    after6 = [float(x) for x in after[:6]]
    err = _ori_err_deg(after6, goal)
    dxyz_mm = float(
        np.linalg.norm(np.asarray(after6[:3]) - np.asarray(cur[:3])) * 1000.0
    )
    result = {
        "after": _pose_brief(after6),
        "ori_err_deg": round(err, 2),
        "xyz_moved_mm": round(dxyz_mm, 1),
        "ok": err <= 8.0,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if err > 8.0:
        LOG.error("姿态未到位 ori_err=%.1f°（看 after.rpy_deg / axes）", err)
        return 2
    LOG.info("到位 ori_err=%.1f° xyz_moved=%.0fmm", err, dxyz_mm)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
