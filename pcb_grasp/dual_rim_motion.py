"""双臂 RIM 后端（大板）：左右各走 ``rim_python.move_to_pose --execute``（不切 POSITION）。"""

from __future__ import annotations

import logging
from typing import Sequence

from .dual_tj_motion import DualMoveResult
from .rim_motion import RimMotion

LOG = logging.getLogger("pcb_grasp.dual_rim")


class DualRimMotion:
    def __init__(
        self,
        *,
        frame_bundle: str = "10.3.8.7:50066",
        mc: str = "",
        max_linear_vel: float = 0.06,
        max_angular_vel: float = 0.25,
        dry_run: bool = False,
        ready_joints_A: Sequence[float] | None = None,
        ready_joints_B: Sequence[float] | None = None,
        connect_timeout_s: float = 10.0,
        plan_timeout_s: float = 120.0,
    ) -> None:
        common = dict(
            frame_bundle=frame_bundle,
            mc=mc,
            max_linear_vel=max_linear_vel,
            max_angular_vel=max_angular_vel,
            dry_run=dry_run,
            connect_timeout_s=connect_timeout_s,
            plan_timeout_s=plan_timeout_s,
        )
        self.left = RimMotion(arm="A", **common)
        self.right = RimMotion(arm="B", **common)
        if ready_joints_A:
            self.left.preferred_ik_seed = list(ready_joints_A)
        if ready_joints_B:
            self.right.preferred_ik_seed = list(ready_joints_B)
        self.dry_run = bool(dry_run)

    def connect(self) -> None:
        self.left.connect()
        self.right.connect()
        LOG.info(
            "DualRim connected dry_run=%s @ %s",
            self.dry_run,
            self.left.frame_bundle,
        )

    def close(self) -> None:
        self.right.close()
        self.left.close()

    def get_flange_poses(self) -> tuple[list[float], list[float]]:
        return self.left.get_flange_pose(), self.right.get_flange_pose()

    def move_both(
        self,
        left6: Sequence[float],
        right6: Sequence[float],
        *,
        velocity_scale: float = 0.08,
        parallel: bool = False,
    ) -> DualMoveResult:
        """双臂到位。默认串行：MC 并行 PLAN 会把另一侧标成 superseded。"""
        del parallel  # 本机 MC 不支持双臂同时 PLAN，强制先左后右
        LOG.info(
            "DUAL_RIM move_both scale=%.3f sequential L→R "
            "L=(%.3f,%.3f,%.3f) R=(%.3f,%.3f,%.3f)",
            velocity_scale,
            float(left6[0]),
            float(left6[1]),
            float(left6[2]),
            float(right6[0]),
            float(right6[1]),
            float(right6[2]),
        )
        try:
            # 左臂结束后立刻发右臂：跳过右臂 move 前的 CLI 读 TCP
            pose_l = self.left.move_line(left6, velocity_scale=velocity_scale)
            pose_r = self.right.move_line(
                right6, velocity_scale=velocity_scale, refresh_start=False
            )
            return DualMoveResult(left=pose_l, right=pose_r, ok=True, message="ok")
        except Exception as exc:  # noqa: BLE001
            return DualMoveResult(
                left=self.left.get_flange_pose(),
                right=self.right.get_flange_pose(),
                ok=False,
                message=str(exc),
            )

    def jog_both(
        self,
        left6: Sequence[float],
        right6: Sequence[float],
        *,
        max_linear_vel: float = 0.15,
        timeout_s: float = 8.0,
    ) -> DualMoveResult:
        """短距点动（内收）：先左后右串行 jog（避免 MC superseded）。"""
        LOG.info(
            "DUAL_RIM jog_both vmax=%.3f sequential L→R "
            "L=(%.3f,%.3f,%.3f) R=(%.3f,%.3f,%.3f)",
            max_linear_vel,
            float(left6[0]),
            float(left6[1]),
            float(left6[2]),
            float(right6[0]),
            float(right6[1]),
            float(right6[2]),
        )
        pose_l = list(left6[:6])
        pose_r = list(right6[:6])
        errs: list[str] = []
        try:
            pose_l = self.left.move_jog(
                left6, max_linear_vel=max_linear_vel, timeout_s=timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            errs.append(f"left:{exc}")
        try:
            pose_r = self.right.move_jog(
                right6, max_linear_vel=max_linear_vel, timeout_s=timeout_s
            )
        except Exception as exc:  # noqa: BLE001
            errs.append(f"right:{exc}")
        if errs:
            return DualMoveResult(
                left=pose_l, right=pose_r, ok=False, message="; ".join(errs)
            )
        return DualMoveResult(left=pose_l, right=pose_r, ok=True, message="ok")

    def squeeze(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        force_n: float = 5.0,
        velocity_scale: float = 0.01,
    ) -> DualMoveResult:
        LOG.warning("DualRim.squeeze 无力控，退化为慢速合拢 force_n=%.1f", force_n)
        return self.move_both(
            left_target, right_target, velocity_scale=velocity_scale, parallel=False
        )
