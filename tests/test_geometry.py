#!/usr/bin/env python3
"""Unit tests for rimbot-aligned rz / flange math (no GPU)."""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pcb_grasp.geometry import (  # noqa: E402
    ObjectPose,
    base_y_angle_to_gripper_rz,
    compute_flange_target,
    fold_deg_180,
    rz_from_line_xy,
)
from pcb_grasp.config import GraspConfig  # noqa: E402
from pcb_grasp.motion import DryRunMotion  # noqa: E402


class TestRz(unittest.TestCase):
    def test_edge_along_plus_x(self):
        # p0=(0,0), p1=(1,0) → edge yaw 0
        self.assertAlmostEqual(rz_from_line_xy([0, 0, 0], [1, 0, 0]), 0.0, places=6)

    def test_edge_along_plus_y(self):
        self.assertAlmostEqual(rz_from_line_xy([0, 0, 0], [0, 1, 0]), math.pi / 2, places=6)

    def test_base_y_gripper_rz_plus_y(self):
        # edge along +Y: angleWithBaseY=0 → yaw_deg = -90 → rz = -π/2
        rz = base_y_angle_to_gripper_rz([0, 0, 0], [0, 1, 0])
        self.assertAlmostEqual(rz, -math.pi / 2, places=5)

    def test_fold_180(self):
        self.assertAlmostEqual(fold_deg_180(100.0), -80.0, places=6)
        self.assertAlmostEqual(fold_deg_180(-100.0), 80.0, places=6)

    def test_flange_default_offset(self):
        # tool_rz_offset=π/2 → flange_rz ≈ input rz; tcp z=0.28, rx=π → flange_z ≈ p_z+0.28
        f = compute_flange_target([1.0, 2.0, 0.0], rz_rad=-math.pi / 2)
        self.assertAlmostEqual(f[0], 1.0, places=5)
        self.assertAlmostEqual(f[1], 2.0, places=5)
        self.assertAlmostEqual(f[2], 0.28, places=5)
        self.assertAlmostEqual(f[3], math.pi, places=5)
        self.assertAlmostEqual(f[4], 0.0, places=5)
        self.assertAlmostEqual(f[5], -math.pi / 2, places=5)


class TestFineHoldZ(unittest.TestCase):
    def test_fine_freezes_z(self):
        # Import pipeline lazily (needs cv2 only here)
        from pcb_grasp.pipeline import GraspPipeline

        cfg = GraspConfig()
        cfg.fine_xy_only = True
        cfg.fine_apply_rz = True
        cfg.tcp_offset = [0.0, 0.0, 0.28]
        pipe = GraspPipeline(cfg=cfg, motion=DryRunMotion([0.1, 0.2, 0.55, math.pi, 0.0, 0.1]))
        obj = ObjectPose(
            x=0.4,
            y=-0.1,
            z=0.05,
            rz=-0.5,
            edge_yaw=0.2,
            corners_base=[[0, 0, 0]] * 4,
        )
        cur = [0.1, 0.2, 0.55, math.pi, 0.0, 0.1]
        flange = pipe.fine_flange_xy_hold_z(obj, cur)
        self.assertAlmostEqual(flange[2], 0.55, places=6)
        self.assertNotAlmostEqual(flange[0], cur[0], places=3)
        self.assertAlmostEqual(flange[5], -0.5, places=5)
        self.assertAlmostEqual(flange[3], -0.5 * math.pi, places=5)
        self.assertAlmostEqual(flange[4], 0.0, places=5)


if __name__ == "__main__":
    unittest.main()
