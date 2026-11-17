"""Dual-arm TJ backend for large-board squeeze grasp (A=left, B=right)."""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Sequence

from .tj_motion import TJDemoMotion

LOG = logging.getLogger("pcb_grasp.dual_tj")


@dataclass
class DualMoveResult:
    left: list[float]
    right: list[float]
    contact: bool = False
    contact_side: str = ""
    ok: bool = True
    message: str = ""


class DualTJMotion:
    """Left=A / right=B. Shares one robot connection; squeeze via TipFT contact stop."""

    def __init__(
        self,
        *,
        robot_ip: str = "192.168.1.190",
        use_impedance: bool = True,
        cart_vel_mm_s: float = 40.0,
        cart_acc_mm_s2: float = 80.0,
        vel_ratio: int = 18,
        acc_ratio: int = 18,
        contact_threshold_n: float = 10.0,
        dry_run: bool = True,
        ready_joints_A: Sequence[float] | None = None,
        ready_joints_B: Sequence[float] | None = None,
    ) -> None:
        self.dry_run = bool(dry_run)
        common = dict(
            robot_ip=robot_ip,
            use_impedance=use_impedance,
            cart_vel_mm_s=cart_vel_mm_s,
            cart_acc_mm_s2=cart_acc_mm_s2,
            vel_ratio=vel_ratio,
            acc_ratio=acc_ratio,
            contact_threshold_n=contact_threshold_n,
            dry_run=dry_run,
        )
        self.left = TJDemoMotion(arm="A", **common)
        self.right = TJDemoMotion(arm="B", **common)
        if ready_joints_A:
            self.left.preferred_ik_seed = list(ready_joints_A)
        if ready_joints_B:
            self.right.preferred_ik_seed = list(ready_joints_B)
        self._owns_robot = False

    def connect(self) -> None:
        self.left.connect()
        self._owns_robot = not self.left.dry_run
        self.right.connect_as_peer(self.left)
        LOG.info(
            "DualTJ connected dry_run=%s left=A right=B",
            self.left.dry_run or self.dry_run,
        )

    def close(self) -> None:
        # peer must not release shared robot twice
        self.right.robot = None
        self.right.dcss = None
        self.right.close()
        if self._owns_robot:
            self.left.close()
        else:
            self.left.robot = None
            self.left.close()

    def __enter__(self) -> "DualTJMotion":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get_flange_poses(self) -> tuple[list[float], list[float]]:
        return self.left.get_flange_pose(), self.right.get_flange_pose()

    def move_both(
        self,
        left6: Sequence[float],
        right6: Sequence[float],
        *,
        velocity_scale: float = 0.08,
        parallel: bool = True,
    ) -> DualMoveResult:
        """Move both flanges to targets (m/rad)."""
        LOG.info(
            "DUAL move_both scale=%.3f L=(%.3f,%.3f,%.3f) R=(%.3f,%.3f,%.3f)",
            velocity_scale,
            float(left6[0]),
            float(left6[1]),
            float(left6[2]),
            float(right6[0]),
            float(right6[1]),
            float(right6[2]),
        )
        err_l: list[BaseException] = []
        err_r: list[BaseException] = []
        pose_l: list[list[float]] = []
        pose_r: list[list[float]] = []

        def _run_l() -> None:
            try:
                pose_l.append(
                    self.left.move_line(left6, velocity_scale=velocity_scale)
                )
            except BaseException as exc:  # noqa: BLE001 — collect for both arms
                err_l.append(exc)

        def _run_r() -> None:
            try:
                pose_r.append(
                    self.right.move_line(right6, velocity_scale=velocity_scale)
                )
            except BaseException as exc:  # noqa: BLE001
                err_r.append(exc)

        if parallel and not (self.left.dry_run or self.dry_run):
            t1 = threading.Thread(target=_run_l, name="dual-L", daemon=True)
            t2 = threading.Thread(target=_run_r, name="dual-R", daemon=True)
            t1.start()
            t2.start()
            t1.join()
            t2.join()
        else:
            _run_l()
            _run_r()

        contact = False
        side = ""
        if err_l or err_r:
            msgs = []
            if err_l:
                msgs.append(f"left: {err_l[0]}")
                if "contact" in str(err_l[0]).lower():
                    contact = True
                    side = "left" if not side else "both"
            if err_r:
                msgs.append(f"right: {err_r[0]}")
                if "contact" in str(err_r[0]).lower():
                    contact = True
                    side = "right" if not side else ("both" if side == "left" else side)
            # If only one failed with contact, still return current poses
            pl = pose_l[0] if pose_l else self.left.get_flange_pose()
            pr = pose_r[0] if pose_r else self.right.get_flange_pose()
            if err_l and err_r and not contact:
                raise RuntimeError("; ".join(msgs))
            return DualMoveResult(
                left=pl,
                right=pr,
                contact=contact,
                contact_side=side,
                ok=contact,  # contact during squeeze path is success-ish
                message="; ".join(msgs),
            )
        return DualMoveResult(
            left=pose_l[0] if pose_l else self.left.get_flange_pose(),
            right=pose_r[0] if pose_r else self.right.get_flange_pose(),
            ok=True,
        )

    def squeeze(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        force_n: float = 5.0,
        velocity_scale: float = 0.01,
    ) -> DualMoveResult:
        """Move both arms toward squeeze targets with lowered TipFT threshold.

        On single-side contact, the other arm continues to its target (补走).
        """
        old_l = float(self.left.contact_threshold_n)
        old_r = float(self.right.contact_threshold_n)
        thr = max(0.5, float(force_n))
        self.left.contact_threshold_n = thr
        self.right.contact_threshold_n = thr
        LOG.info(
            "DUAL squeeze force=%.1fN scale=%.3f → L=(%.3f,%.3f,%.3f) R=(%.3f,%.3f,%.3f)",
            thr,
            velocity_scale,
            float(left_target[0]),
            float(left_target[1]),
            float(left_target[2]),
            float(right_target[0]),
            float(right_target[1]),
            float(right_target[2]),
        )
        try:
            first = self.move_both(
                left_target, right_target, velocity_scale=velocity_scale, parallel=True
            )
            if first.ok and not first.contact:
                return first
            # one side contacted — other arm finishes remaining travel
            side = first.contact_side
            if side == "left":
                LOG.info("left contact — right finishes squeeze")
                try:
                    pr = self.right.move_line(
                        right_target, velocity_scale=velocity_scale
                    )
                    return DualMoveResult(
                        left=first.left,
                        right=pr,
                        contact=True,
                        contact_side="left",
                        ok=True,
                        message=first.message,
                    )
                except Exception as exc:
                    return DualMoveResult(
                        left=first.left,
                        right=self.right.get_flange_pose(),
                        contact=True,
                        contact_side="both" if "contact" in str(exc).lower() else "left",
                        ok=True,
                        message=f"{first.message}; right follow: {exc}",
                    )
            if side == "right":
                LOG.info("right contact — left finishes squeeze")
                try:
                    pl = self.left.move_line(left_target, velocity_scale=velocity_scale)
                    return DualMoveResult(
                        left=pl,
                        right=first.right,
                        contact=True,
                        contact_side="right",
                        ok=True,
                        message=first.message,
                    )
                except Exception as exc:
                    return DualMoveResult(
                        left=self.left.get_flange_pose(),
                        right=first.right,
                        contact=True,
                        contact_side="both" if "contact" in str(exc).lower() else "right",
                        ok=True,
                        message=f"{first.message}; left follow: {exc}",
                    )
            return DualMoveResult(
                left=first.left,
                right=first.right,
                contact=True,
                contact_side=side or "both",
                ok=True,
                message=first.message or "both contacted / stopped",
            )
        finally:
            self.left.contact_threshold_n = old_l
            self.right.contact_threshold_n = old_r

    def estop(self) -> None:
        """Best-effort hold current joints on both arms."""
        for arm_m, name in ((self.left, "A"), (self.right, "B")):
            if arm_m.dry_run or arm_m.robot is None:
                continue
            try:
                q = arm_m._read_joints()  # noqa: SLF001
                arm_m.robot.set_joint_position_cmd(arm=name, joint=q)
            except Exception as exc:
                LOG.warning("estop %s failed: %s", name, exc)


class DryRunDualMotion:
    """Pose-only dual arm for offline geometry tests."""

    def __init__(self) -> None:
        self.left_pose = [0.40, 0.35, 0.05, -math.pi, 0.0, -0.5 * math.pi]
        self.right_pose = [0.40, -0.35, 0.05, math.pi, 0.0, 0.5 * math.pi]
        self.dry_run = True

    def connect(self) -> None:
        LOG.info("DryRunDualMotion connected")

    def close(self) -> None:
        pass

    def get_flange_poses(self) -> tuple[list[float], list[float]]:
        return list(self.left_pose), list(self.right_pose)

    def move_both(
        self,
        left6: Sequence[float],
        right6: Sequence[float],
        *,
        velocity_scale: float = 0.08,
        parallel: bool = True,
    ) -> DualMoveResult:
        _ = velocity_scale, parallel
        self.left_pose = [float(x) for x in left6[:6]]
        self.right_pose = [float(x) for x in right6[:6]]
        LOG.info(
            "DRY dual move_both L=%s R=%s",
            [round(x, 3) for x in self.left_pose[:3]],
            [round(x, 3) for x in self.right_pose[:3]],
        )
        return DualMoveResult(left=list(self.left_pose), right=list(self.right_pose))

    def squeeze(
        self,
        left_target: Sequence[float],
        right_target: Sequence[float],
        *,
        force_n: float = 5.0,
        velocity_scale: float = 0.01,
    ) -> DualMoveResult:
        _ = force_n, velocity_scale
        LOG.info("DRY dual squeeze")
        return self.move_both(left_target, right_target)

    def estop(self) -> None:
        LOG.info("DRY estop")
