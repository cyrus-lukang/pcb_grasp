"""按 grasp_config.motion_backend 创建真机运控（rim / tj）。

rim：``python -m rim_python.move_to_pose --execute``，不手动切 POSITION。
tj：旧 TJ_demo（仅兼容）。
"""

from __future__ import annotations

import logging
from typing import Any

LOG = logging.getLogger("pcb_grasp.robot_factory")


def connect_real_motion(cfg: Any, *, record_grasp_z: bool = False):
    """返回 ``(motion, dual)``。dual 在双臂场景非 None。"""
    cfg.dry_run = False
    backend = str(getattr(cfg, "motion_backend", "rim") or "rim").lower()

    if backend == "rim":
        from .dual_rim_motion import DualRimMotion
        from .rim_motion import RimMotion

        rim_kw = dict(
            frame_bundle=str(cfg.rim_frame_bundle),
            mc=str(cfg.rim_mc or ""),
            max_linear_vel=float(cfg.rim_max_linear_vel),
            max_angular_vel=float(cfg.rim_max_angular_vel),
            connect_timeout_s=float(cfg.rim_connect_timeout_s),
            plan_timeout_s=float(cfg.rim_plan_timeout_s),
            dry_run=False,
        )
        if cfg.large_board_enable and not record_grasp_z:
            dual = DualRimMotion(
                **rim_kw,
                ready_joints_A=cfg.ready_joints_A,
                ready_joints_B=cfg.ready_joints_B,
            )
            dual.connect()
            motion = dual.left
            motion.safe_z_m = float(cfg.safe_z_m)
            _apply_gripper(motion, cfg)
            _apply_gripper(dual.right, cfg)
            try:
                motion._setup_gripper()  # noqa: SLF001
                dual.right._setup_gripper()  # noqa: SLF001
            except Exception as exc:  # noqa: BLE001
                LOG.warning("rim gripper setup: %s", exc)
            LOG.info(
                "REAL RIM dual @ %s (move_to_pose; no POSITION switch)",
                cfg.rim_frame_bundle,
            )
            return motion, dual

        motion = RimMotion(arm=cfg.tj_arm, **rim_kw)
        motion.connect()
        ready = cfg.ready_joints_A if str(cfg.tj_arm).upper() == "A" else cfg.ready_joints_B
        if ready:
            motion.preferred_ik_seed = list(ready)
        motion.safe_z_m = float(cfg.safe_z_m)
        _apply_gripper(motion, cfg)
        try:
            motion._setup_gripper()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            LOG.warning("rim gripper setup: %s", exc)
        LOG.info(
            "REAL RIM arm=%s @ %s grasp_opening=%.3f (move_to_pose; no POSITION switch)",
            cfg.tj_arm,
            cfg.rim_frame_bundle,
            float(cfg.gripper_grasp_position),
        )
        return motion, None

    # ---- TJ 兼容 ----
    from .dual_tj_motion import DualTJMotion
    from .tj_motion import TJDemoMotion

    if cfg.large_board_enable and not record_grasp_z:
        dual = DualTJMotion(
            robot_ip=cfg.tj_robot_ip,
            use_impedance=bool(
                cfg.tj_use_impedance or getattr(cfg, "large_use_impedance", True)
            ),
            contact_threshold_n=cfg.tj_contact_threshold_n,
            cart_vel_mm_s=cfg.cart_vel_mm_s,
            cart_acc_mm_s2=cfg.cart_acc_mm_s2,
            vel_ratio=cfg.tj_vel_ratio,
            acc_ratio=cfg.tj_acc_ratio,
            dry_run=False,
            ready_joints_A=cfg.ready_joints_A,
            ready_joints_B=cfg.ready_joints_B,
        )
        dual.connect()
        motion = dual.left
        _apply_gripper(motion, cfg)
        try:
            motion._setup_gripper()  # noqa: SLF001
        except Exception as exc:  # noqa: BLE001
            LOG.warning("gripper: %s", exc)
        motion.safe_z_m = float(cfg.safe_z_m)
        LOG.info("REAL TJ dual A+B @ %s", cfg.tj_robot_ip)
        return motion, dual

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
    _apply_gripper(motion, cfg)
    motion.connect()
    arm = str(cfg.tj_arm).upper()
    ready = cfg.ready_joints_A if arm == "A" else cfg.ready_joints_B
    if ready:
        motion.preferred_ik_seed = list(ready)
    motion.safe_z_m = float(cfg.safe_z_m)
    LOG.info("REAL TJ arm=%s @ %s", cfg.tj_arm, cfg.tj_robot_ip)
    return motion, None


def _apply_gripper(motion: Any, cfg: Any) -> None:
    motion.gripper_com = int(cfg.gripper_com_ch)
    motion.gripper_grasp_position = float(cfg.gripper_grasp_position)
    motion.gripper_release_position = float(cfg.gripper_release_position)
    motion.gripper_speed = float(cfg.gripper_speed)
    motion.gripper_torque_grasp = float(cfg.gripper_torque_grasp)
    motion.gripper_torque_release = float(cfg.gripper_torque_release)
    # RIM 运臂时夹爪仍走 Marvin 知行 RS485
    if hasattr(motion, "gripper_robot_ip"):
        motion.gripper_robot_ip = str(
            getattr(cfg, "tj_robot_ip", "") or "192.168.1.190"
        )
