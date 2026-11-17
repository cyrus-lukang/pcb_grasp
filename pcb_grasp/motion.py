"""Motion backends: dry-run default; plug in real arm SDK later."""

from __future__ import annotations

import logging
import math
import time
from abc import ABC, abstractmethod
from typing import Sequence

LOG = logging.getLogger("pcb_grasp.motion")


def _pose_delta(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    """Return (xyz_m, rpy_rad) L2 deltas."""
    dxyz = math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3)))
    drpy = math.sqrt(sum((float(a[i]) - float(b[i])) ** 2 for i in range(3, 6)))
    return dxyz, drpy


class MotionBackend(ABC):
    """Minimal arm interface used by GraspPipeline."""

    @abstractmethod
    def get_flange_pose(self) -> list[float]:
        """Current flange xyzrpy in base (m, rad)."""

    @abstractmethod
    def move_line(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        keep_orientation: bool = False,
        allow_partial: bool = False,
    ) -> list[float]:
        """Linear move to flange target [x,y,z,rx,ry,rz]. Returns pose after move."""

    def get_joints(self) -> list[float]:
        """Current 7 joint angles (deg). Default stub for dry-run."""
        return [0.0] * 7

    def move_joints(
        self,
        joints: Sequence[float],
        *,
        tol_deg: float = 2.0,
        timeout_s: float = 90.0,
        rate_hz: float = 50.0,
        max_deg_s: float = 12.0,
    ) -> list[float]:
        """Move to joint target (deg). Backends may override."""
        LOG.info("MOVE_JOINTS (noop base) %s", [round(float(x), 1) for x in joints[:7]])
        return [float(x) for x in joints[:7]]

    def set_gripper(self, open_: bool) -> None:
        LOG.info("gripper %s", "OPEN" if open_ else "CLOSE")

    def wait_until_still(
        self,
        *,
        settle_s: float = 0.4,
        pos_tol_m: float = 0.001,
        rot_tol_rad: float = 0.01,
        hold_s: float = 0.25,
        timeout_s: float = 5.0,
        poll_s: float = 0.05,
    ) -> list[float]:
        """运动结束后等待法兰静止，再允许识别。返回静止时位姿。"""
        if settle_s > 0:
            time.sleep(settle_s)
        t0 = time.time()
        prev = self.get_flange_pose()
        still_since: float | None = None
        while True:
            time.sleep(poll_s)
            cur = self.get_flange_pose()
            dxyz, drpy = _pose_delta(prev, cur)
            prev = cur
            if dxyz <= pos_tol_m and drpy <= rot_tol_rad:
                if still_since is None:
                    still_since = time.time()
                elif time.time() - still_since >= hold_s:
                    LOG.info(
                        "ARM STILL xyz=(%.4f,%.4f,%.4f) (ready for detect)",
                        cur[0],
                        cur[1],
                        cur[2],
                    )
                    return cur
            else:
                still_since = None
            if time.time() - t0 > timeout_s:
                LOG.warning(
                    "wait_until_still timeout (%.1fs); proceed with last pose dxyz=%.4f drpy=%.4f",
                    timeout_s,
                    dxyz,
                    drpy,
                )
                return cur


class DryRunMotion(MotionBackend):
    """Logs commanded poses; keeps an internal virtual flange pose."""

    def __init__(self, initial_pose: Sequence[float] | None = None) -> None:
        # Default: hanging above table, tool down
        self._pose = list(
            initial_pose
            if initial_pose is not None
            else [0.3, 0.0, 0.4, 3.141592653589793, 0.0, 0.0]
        )
        self.history: list[list[float]] = []

    def get_flange_pose(self) -> list[float]:
        return list(self._pose)

    def move_line(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        keep_orientation: bool = False,
        allow_partial: bool = False,
    ) -> list[float]:
        tgt = [float(x) for x in xyzrpy[:6]]
        if keep_orientation:
            tgt = [tgt[0], tgt[1], tgt[2], *self._pose[3:6]]
        LOG.info(
            "MOVE_LINE scale=%.2f keep_ori=%s partial=%s xyz=(%.4f,%.4f,%.4f) rpy=(%.3f,%.3f,%.3f)",
            velocity_scale,
            keep_orientation,
            allow_partial,
            tgt[0],
            tgt[1],
            tgt[2],
            tgt[3],
            tgt[4],
            tgt[5],
        )
        self.history.append(tgt)
        self._pose = tgt
        time.sleep(0.01)
        return self.get_flange_pose()

    def set_gripper(self, open_: bool) -> None:
        super().set_gripper(open_)


class CallbackMotion(MotionBackend):
    """Wrap user-provided callables (e.g. RealMan / Tianji SDK)."""

    def __init__(
        self,
        get_pose,
        move_line,
        set_gripper=None,
    ) -> None:
        self._get = get_pose
        self._move = move_line
        self._grip = set_gripper

    def get_flange_pose(self) -> list[float]:
        return list(self._get())

    def move_line(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        keep_orientation: bool = False,
        allow_partial: bool = False,
    ) -> list[float]:
        try:
            self._move(
                list(xyzrpy[:6]),
                velocity_scale,
                keep_orientation=keep_orientation,
                allow_partial=allow_partial,
            )
        except TypeError:
            try:
                self._move(
                    list(xyzrpy[:6]), velocity_scale, keep_orientation=keep_orientation
                )
            except TypeError:
                self._move(list(xyzrpy[:6]), velocity_scale)
        return self.get_flange_pose()

    def set_gripper(self, open_: bool) -> None:
        if self._grip is not None:
            self._grip(open_)
        else:
            super().set_gripper(open_)
