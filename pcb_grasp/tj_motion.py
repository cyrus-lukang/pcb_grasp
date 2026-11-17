"""TJ_demo real-arm backend: Link_Base pose + cart impedance + TipFT contact stop.

Units inside GraspPipeline: meters / radians (Link_Base).
TJ SDK: millimeters / degrees.

Contact stop ≈ rimbot ``tipft_force_protect`` / TJ ``filter_dpos``:
if |F_i| >= threshold and motion pushes into contact (F·dpos < 0), abort.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .motion import MotionBackend

LOG = logging.getLogger("pcb_grasp.tj_motion")

TJ_ROOT = Path("/home/ubuntu/workspace/TJ_demo")
DEFAULT_IP = "192.168.1.190"
DEFAULT_KINE = TJ_ROOT / "DEMO_PYTHON" / "ccs_m6_40.MvKDCfg"
DEFAULT_TIPFT_A = TJ_ROOT / "DEMO_PYTHON" / "tipft_calib_A.json"
DEFAULT_TIPFT_B = TJ_ROOT / "DEMO_PYTHON" / "tipft_calib_B.json"

DEFAULT_CART_K = [3000.0, 3000.0, 3000.0, 1500.0, 1500.0, 1500.0, 5.0]
DEFAULT_CART_D = [0.8, 0.8, 0.8, 0.35, 0.35, 0.35, 1.0]


def _ensure_tj_path() -> None:
    for p in (TJ_ROOT, TJ_ROOT / "DEMO_PYTHON", TJ_ROOT / "spacemouse_impedance_demo"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _deg(rad: float) -> float:
    return float(rad) * 180.0 / math.pi


def _rad(deg: float) -> float:
    return float(deg) * math.pi / 180.0


def pose_m_rad_to_mm_deg(xyzrpy: Sequence[float]) -> list[float]:
    return [
        float(xyzrpy[0]) * 1000.0,
        float(xyzrpy[1]) * 1000.0,
        float(xyzrpy[2]) * 1000.0,
        _deg(xyzrpy[3]),
        _deg(xyzrpy[4]),
        _deg(xyzrpy[5]),
    ]


def pose_mm_deg_to_m_rad(xyzabc: Sequence[float]) -> list[float]:
    return [
        float(xyzabc[0]) / 1000.0,
        float(xyzabc[1]) / 1000.0,
        float(xyzabc[2]) / 1000.0,
        _rad(xyzabc[3]),
        _rad(xyzabc[4]),
        _rad(xyzabc[5]),
    ]


class TJDemoMotion(MotionBackend):
    """Real Marvin arm via TJ_demo Concise SDK + kinematics + TipFT guard."""

    def __init__(
        self,
        *,
        robot_ip: str = DEFAULT_IP,
        arm: str = "A",
        kine_config: str | Path = DEFAULT_KINE,
        use_impedance: bool = True,
        cart_k: Sequence[float] | None = None,
        cart_d: Sequence[float] | None = None,
        vel_ratio: int = 40,
        acc_ratio: int = 40,
        cart_vel_mm_s: float = 80.0,
        cart_acc_mm_s2: float = 200.0,
        contact_threshold_n: float = 10.0,
        tipft_calib: str | Path | None = None,
        tool_kine: Sequence[float] | None = None,
        tool_dyn: Sequence[float] | None = None,
        dry_run: bool = False,
    ) -> None:
        _ensure_tj_path()
        self.robot_ip = robot_ip
        self.arm = arm.upper()
        self.arm_index = 0 if self.arm == "A" else 1
        self.kine_config = Path(kine_config)
        self.use_impedance = use_impedance
        self.cart_k = list(cart_k or DEFAULT_CART_K)
        self.cart_d = list(cart_d or DEFAULT_CART_D)
        self.vel_ratio = int(vel_ratio)
        self.acc_ratio = int(acc_ratio)
        self.cart_vel = float(cart_vel_mm_s)
        self.cart_acc = float(cart_acc_mm_s2)
        self.contact_threshold_n = float(contact_threshold_n)
        if tipft_calib is None:
            tipft_calib = DEFAULT_TIPFT_A if self.arm == "A" else DEFAULT_TIPFT_B
        self.tipft_calib = Path(tipft_calib)
        self.tool_kine = list(tool_kine or [0.0] * 6)
        self.tool_dyn = list(tool_dyn) if tool_dyn is not None else None
        self.dry_run = dry_run

        self.robot = None
        self.kine = None
        self.dcss = None
        self.xf = None
        self.tipft = None
        self.last_contact_reason = ""
        self.history: list[list[float]] = []
        self._last_ik_flags: dict = {}
        self._last_ik_joints: list[float] | None = None
        self._lmt_p: list[float] = [170.0, 120.0, 170.0, 60.0, 170.0, 60.0, 90.0]
        self._lmt_n: list[float] = [-170.0, -120.0, -170.0, -145.0, -170.0, -60.0, -90.0]
        self.preferred_ik_seed: list[float] | None = None
        self.safe_z_m: float = 0.02  # rimbot ICT safe_z=20mm
        # 左臂进治具区必经途经点（ict left_test_via_pose, mm/deg）
        self.left_via_xyzabc_mm: list[float] = [360.0, 400.0, 20.0, -180.0, 0.0, -90.0]
        self.via_x_trigger_mm: float = 250.0  # 目标 X 超过此值 → 先走 via
        self._sdk_lock = threading.Lock()
        self._gripper = None
        self._gripper_rx = None
        self.gripper_com: int = 2  # 2=COM1
        # 默认对齐 rimbot_poc end_effector；可由 GraspConfig 覆盖
        self.gripper_grasp_position: float = 0.41
        self.gripper_release_position: float = 1.0
        self.gripper_speed: float = 50.0
        self.gripper_torque_grasp: float = 30.0
        self.gripper_torque_release: float = 60.0

    def connect(self) -> None:
        from SDK_PYTHON.frame_transform import FrameTransformer
        from SDK_PYTHON.fx_kine import Marvin_Kine
        from SDK_PYTHON.fx_robot import Concise_Marvin_Robot, DCSS
        from tipft import TipFTEstimator

        if self.dry_run:
            LOG.warning("TJDemoMotion dry_run=True — no hardware connect")
            return

        self.robot = Concise_Marvin_Robot()
        if not self.robot.connect(robot_ip=self.robot_ip, log_switch=1):
            raise RuntimeError(f"TJ connect failed: {self.robot_ip}")
        self.dcss = DCSS()
        self.kine = Marvin_Kine()
        self.kine.log_switch(0)
        # SDK: load_config(arm_type, config_path) → 再 initial_kine，否则 FK 必失败
        ini = self.kine.load_config(
            arm_type=self.arm_index, config_path=str(self.kine_config)
        )
        if not ini:
            raise RuntimeError(f"kine load_config failed: {self.kine_config}")
        if not self.kine.initial_kine(
            robot_type=ini["TYPE"][self.arm_index],
            dh=ini["DH"][self.arm_index],
            pnva=ini["PNVA"][self.arm_index],
            j67=ini["BD"][self.arm_index],
        ):
            raise RuntimeError("kine initial_kine failed")
        # PNVA 前两列 = 关节正负限位（与 rimbot cartesian_planner 一致）
        try:
            pnva = ini["PNVA"][self.arm_index]
            self._lmt_p = [max(float(r[0]), float(r[1])) for r in pnva[:7]]
            self._lmt_n = [min(float(r[0]), float(r[1])) for r in pnva[:7]]
        except Exception as e:
            LOG.warning("PNVA limits fallback: %s", e)
        self.xf = FrameTransformer(kine=self.kine)

        self.robot.check_error_and_clear()
        if self.tool_dyn is not None:
            if not self.robot.set_tool(self.arm, self.tool_kine, self.tool_dyn):
                LOG.warning("set_tool failed; continuing")

        if self.use_impedance:
            ok = self.robot.set_imp_cart_state(
                arm=self.arm,
                velRatio=self.vel_ratio,
                AccRatio=self.acc_ratio,
                K=self.cart_k,
                D=self.cart_d,
                rot_type=2,
                cart_ctrl_para=[0.0] * 7,
            )
            if not ok:
                raise RuntimeError("set_imp_cart_state failed")
            LOG.info("Cartesian impedance ON K=%s thr=%.1fN", self.cart_k, self.contact_threshold_n)
        else:
            if not self.robot.set_position_state(
                arm=self.arm, velRatio=self.vel_ratio, AccRatio=self.acc_ratio
            ):
                raise RuntimeError("set_position_state failed")
            LOG.info("Position mode ON")

        if self.tipft_calib.is_file():
            self.tipft = TipFTEstimator(str(self.tipft_calib), self.arm_index)
            self.tipft.capture_startup(
                robot=self.robot,
                dcss=self.dcss,
                kine=self.kine,
                duration_s=1.0,
                hz=50.0,
            )
        else:
            LOG.warning("No tipFT calib %s — using est_cart_fn fallback", self.tipft_calib)

        self._setup_gripper()

    def connect_as_peer(self, primary: "TJDemoMotion") -> None:
        """Share robot/dcss from an already-connected arm; init this arm's kine/TipFT."""
        from SDK_PYTHON.frame_transform import FrameTransformer
        from SDK_PYTHON.fx_kine import Marvin_Kine
        from tipft import TipFTEstimator

        if primary.dry_run or primary.robot is None:
            self.dry_run = True
            LOG.warning("connect_as_peer dry_run — arm=%s no hardware", self.arm)
            return
        self.dry_run = False
        self.robot = primary.robot
        self.dcss = primary.dcss
        self.kine = Marvin_Kine()
        self.kine.log_switch(0)
        ini = self.kine.load_config(
            arm_type=self.arm_index, config_path=str(self.kine_config)
        )
        if not ini:
            raise RuntimeError(f"peer kine load_config failed: {self.kine_config}")
        if not self.kine.initial_kine(
            robot_type=ini["TYPE"][self.arm_index],
            dh=ini["DH"][self.arm_index],
            pnva=ini["PNVA"][self.arm_index],
            j67=ini["BD"][self.arm_index],
        ):
            raise RuntimeError("peer kine initial_kine failed")
        try:
            pnva = ini["PNVA"][self.arm_index]
            self._lmt_p = [max(float(r[0]), float(r[1])) for r in pnva[:7]]
            self._lmt_n = [min(float(r[0]), float(r[1])) for r in pnva[:7]]
        except Exception as e:
            LOG.warning("peer PNVA limits fallback: %s", e)
        self.xf = FrameTransformer(kine=self.kine)
        if self.tool_dyn is not None:
            if not self.robot.set_tool(self.arm, self.tool_kine, self.tool_dyn):
                LOG.warning("peer set_tool failed arm=%s", self.arm)
        if self.use_impedance:
            ok = self.robot.set_imp_cart_state(
                arm=self.arm,
                velRatio=self.vel_ratio,
                AccRatio=self.acc_ratio,
                K=self.cart_k,
                D=self.cart_d,
                rot_type=2,
                cart_ctrl_para=[0.0] * 7,
            )
            if not ok:
                raise RuntimeError(f"peer set_imp_cart_state failed arm={self.arm}")
        else:
            if not self.robot.set_position_state(
                arm=self.arm, velRatio=self.vel_ratio, AccRatio=self.acc_ratio
            ):
                raise RuntimeError(f"peer set_position_state failed arm={self.arm}")
        if self.tipft_calib.is_file():
            self.tipft = TipFTEstimator(str(self.tipft_calib), self.arm_index)
            self.tipft.capture_startup(
                robot=self.robot,
                dcss=self.dcss,
                kine=self.kine,
                duration_s=1.0,
                hz=50.0,
            )
        else:
            LOG.warning("peer no tipFT calib %s", self.tipft_calib)
        # 大板对夹不依赖夹爪开合；跳过 gripper setup
        LOG.info("peer arm=%s attached (shared robot)", self.arm)

    def _setup_gripper(self) -> None:
        """知行夹爪（与 TJ verify_zhixing_gripper / spacemouse 同协议）。"""
        if self.dry_run or self.robot is None:
            return
        try:
            from verify_zhixing_gripper import (  # type: ignore
                MarvinModbus,
                Rs485RxPump,
                ZhiXingGripper,
            )
        except Exception as exc:
            LOG.warning("gripper import failed: %s", exc)
            return

        class _ConciseRs485Adapter:
            def __init__(self, robot, lock: threading.Lock) -> None:
                self._robot = robot
                self._lock = lock

            def clear_485_cache(self, arm: str):
                with self._lock:
                    return self._robot.clear_ch_data(arm)

            def set_485_data(self, arm: str, data, size_int: int, com: int):
                if isinstance(data, str):
                    payload = bytes(int(x, 16) for x in data.split())
                else:
                    payload = bytes(data)
                with self._lock:
                    sent = self._robot.set_ch_data(arm, payload, len(payload), com)
                ok = sent is not None and int(sent) > 0
                return ok, sent

            def get_485_data(self, arm: str, com: int):
                with self._lock:
                    result = self._robot.get_ch_data(arm, com)
                tag = result[0]
                receive_hex = result[1]
                if isinstance(receive_hex, (bytes, bytearray)):
                    receive_hex = " ".join(f"{b:02X}" for b in receive_hex)
                return tag, receive_hex or ""

        rx = None
        try:
            adapter = _ConciseRs485Adapter(self.robot, self._sdk_lock)
            rx = Rs485RxPump(adapter, self.arm, int(self.gripper_com))
            rx.start()
            grip = ZhiXingGripper(MarvinModbus(adapter, rx, self.arm, int(self.gripper_com)))
            if not grip.initialize():
                rx.stop()
                LOG.warning("gripper enable failed")
                return
            self._gripper_rx = rx
            self._gripper = grip
            LOG.info("Zhixing gripper ready arm=%s com=%s", self.arm, self.gripper_com)
        except Exception as exc:
            LOG.warning("gripper setup failed: %s", exc)
            if rx is not None:
                try:
                    rx.stop()
                except Exception:
                    pass
            self._gripper = None
            self._gripper_rx = None

    def close(self) -> None:
        if self._gripper_rx is not None:
            try:
                self._gripper_rx.stop()
            except Exception:
                pass
            self._gripper_rx = None
        self._gripper = None
        if self.robot is not None:
            try:
                self.robot.release_robot()
            except Exception:
                pass
            self.robot = None

    def __enter__(self) -> "TJDemoMotion":
        self.connect()
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def _read_joints(self) -> list[float]:
        sub = self.robot.subscribe(self.dcss)
        return [float(v) for v in sub["outputs"][self.arm_index]["fb_joint_pos"]]

    def get_flange_pose(self) -> list[float]:
        if self.dry_run or self.robot is None:
            return [0.3, 0.0, 0.4, math.pi, 0.0, 0.0]
        joints = self._read_joints()
        fk = self.kine.fk(joints)
        if not fk:
            raise RuntimeError("fk failed")
        xyzabc_arm = self.kine.mat4x4_to_xyzabc(fk)
        xyzabc_base = self.xf.arm_xyzabc_to_base(self.arm, xyzabc_arm)
        return pose_mm_deg_to_m_rad(xyzabc_base)

    def _read_force_base(self) -> np.ndarray:
        sub = self.robot.subscribe(self.dcss)
        if self.tipft is not None and self.tipft.ready:
            return np.asarray(self.tipft.update(sub), dtype=np.float64)
        fn = sub["outputs"][self.arm_index].get("est_cart_fn")
        if fn is None:
            return np.zeros(3)
        return np.asarray([float(fn[0]), float(fn[1]), float(fn[2])], dtype=np.float64)

    def _contact_blocks(self, dpos_m: Sequence[float]) -> str:
        f = self._read_force_base()
        thr = self.contact_threshold_n
        blocked = []
        for i, name in enumerate("XYZ"):
            fi, di = float(f[i]), float(dpos_m[i])
            if abs(fi) >= thr and fi * di < 0.0:
                blocked.append(f"{name}(F={fi:.1f}N)")
        return ",".join(blocked)

    def move_line(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        keep_orientation: bool = False,
        allow_partial: bool = False,
    ) -> list[float]:
        """Move flange in Link_Base. Returns pose after move (m/rad).

        ``keep_orientation=True``: only XYZ from ``xyzrpy``, keep current rotation
        (TJ cart_control 同款；避免粗定位强制 rx=π 导致 J6 超限)。

        ``allow_partial=True``: keep_ori 达不到全目标时，沿直线走到最远可达点后返回
        （不抛错），供「先抬一点再平移」使用。
        """
        from SDK_PYTHON.frame_transform import xyzabc_to_mat4

        tgt_m = [float(x) for x in xyzrpy[:6]]
        self.history.append(tgt_m)
        LOG.info(
            "TJ MOVE_LINE scale=%.2f keep_ori=%s partial=%s xyz=(%.4f,%.4f,%.4f) rpy=(%.3f,%.3f,%.3f)",
            velocity_scale,
            keep_orientation,
            allow_partial,
            *tgt_m,
        )
        if self.dry_run or self.robot is None:
            if isinstance(getattr(self, "_pose", None), list):
                if keep_orientation and len(self._pose) >= 6:
                    self._pose = [tgt_m[0], tgt_m[1], tgt_m[2], *self._pose[3:6]]
                else:
                    self._pose = list(tgt_m)
            return self.get_flange_pose()

        cur = self.get_flange_pose()
        dpos = [tgt_m[i] - cur[i] for i in range(3)]
        reason = self._contact_blocks(dpos)
        if reason:
            self.last_contact_reason = reason
            raise RuntimeError(f"contact block before move: {reason}")

        joints0 = self._read_joints()
        cur_mm = pose_m_rad_to_mm_deg(cur)
        # 用 FK 矩阵保姿态，避免 xyzabc 往返丢精度
        T_arm_now = self.kine.fk(joints0)
        if not T_arm_now:
            raise RuntimeError("FK failed at current joints")
        T_base_now = np.asarray(
            self.xf.arm_mat_to_base(self.arm, T_arm_now), dtype=np.float64
        )

        if keep_orientation:
            T_base_tgt = T_base_now.copy()
            T_base_tgt[0, 3] = tgt_m[0] * 1000.0
            T_base_tgt[1, 3] = tgt_m[1] * 1000.0
            T_base_tgt[2, 3] = tgt_m[2] * 1000.0
        else:
            tgt_mm = pose_m_rad_to_mm_deg(tgt_m)
            T_base_tgt = xyzabc_to_mat4(tgt_mm, kine=self.kine)

        end_j = self._solve_ik(T_base_tgt, joints0)
        if end_j is None:
            if keep_orientation:
                reached = self._approach_keep_ori(
                    T_base_now,
                    [tgt_m[0] * 1000.0, tgt_m[1] * 1000.0, tgt_m[2] * 1000.0],
                    joints0,
                    velocity_scale=velocity_scale,
                    dpos_hint=dpos,
                )
                msg = (
                    "IK unreachable for full target (joint limit). "
                    f"cur_mm={[round(x, 1) for x in cur_mm]} "
                    f"tgt_m={[round(x, 4) for x in tgt_m[:3]]} "
                    f"furthest_reachable_mm={[round(x, 1) for x in reached]} "
                    f"ik={self._last_ik_flags}."
                )
                if allow_partial:
                    LOG.warning("partial move OK — %s", msg)
                    return self.get_flange_pose()
                raise RuntimeError(
                    msg + " 可先抬高再平移；若仍失败查头外参/板 XY 是否在左臂可达区。"
                )
            j_try = self._last_ik_joints
            raise RuntimeError(
                "IK failed / unsafe for target: "
                f"out_range={self._last_ik_flags.get('out_range')} "
                f"jnt_exd={self._last_ik_flags.get('jnt_exd')} "
                f"exd_abs={self._last_ik_flags.get('exd_abs')} "
                f"joints={[round(x, 2) for x in (j_try or [])]} "
                f"cur_mm={[round(x, 1) for x in cur_mm]}"
            )

        self._run_joint_path(joints0, end_j, velocity_scale=velocity_scale, dpos_hint=dpos)
        return self.get_flange_pose()

    def _ik_posture_cost(self, q: Sequence[float], ref: Sequence[float]) -> float:
        """选解代价：贴当前种子 + 贴 preferred（避免肘型漂成丧尸姿）。"""
        c_ref = max(abs(a - b) for a, b in zip(q, ref))
        pref = self.preferred_ik_seed
        if pref and len(pref) >= 7:
            c_pref = max(abs(a - b) for a, b in zip(q, pref[:7]))
            # 略惩罚极端腕/肩（J1、J5 贴近限位更容易「翻肘」）
            margin = min(
                self._lmt_p[0] - abs(q[0]),
                self._lmt_p[4] - abs(q[4]),
                self._lmt_p[5] - abs(q[5]),
            )
            pen = 0.0 if margin > 25.0 else (25.0 - margin) * 0.5
            return 0.55 * c_ref + 0.45 * c_pref + pen
        return c_ref

    def _fold_to_limits(
        self, q: Sequence[float], seed: Sequence[float]
    ) -> list[float] | None:
        """把 AllJoint 候选折到 seed 附近且落入 PNVA 限位（rimbot foldToLimits）。"""
        out: list[float] = []
        for i, (v, s) in enumerate(zip(q, seed)):
            d = float(v) - float(s)
            while d > 180.0:
                d -= 360.0
            while d < -180.0:
                d += 360.0
            qf = float(s) + d
            lo, hi = self._lmt_n[i], self._lmt_p[i]
            if lo - 1e-3 <= qf <= hi + 1e-3:
                out.append(qf)
                continue
            hit = False
            for k in (-360.0, 360.0):
                if lo - 1e-3 <= qf + k <= hi + 1e-3:
                    out.append(qf + k)
                    hit = True
                    break
            if not hit:
                return None
        return out

    def _solve_ik(self, T_base_tgt, ref_joints: Sequence[float]) -> list[float] | None:
        """rimbot 同款：多分支折限位 → 失败再 IK_NSP 臂角扫描脱困。"""
        from SDK_PYTHON.fx_kine import FX_InvKineSolvePara, convert_to_8x8_matrix

        ref = [float(x) for x in ref_joints[:7]]
        T_arm_tgt = self.xf.base_mat_to_arm(self.arm, T_base_tgt)
        mat16 = self.kine.mat4x4_to_mat1x16(np.asarray(T_arm_tgt).tolist())
        sp = FX_InvKineSolvePara()
        sp.set_input_ik_target_tcp(mat16)
        sp.set_input_ik_ref_joint(ref)
        sp.set_input_ik_zsp_type(0)
        sp.set_input_zsp_angle(0.0)
        ik = self.kine.ik(sp)
        self._last_ik_flags = {
            "out_range": None if not ik else bool(ik.m_Output_IsOutRange),
            "jnt_exd": None if not ik else bool(ik.m_Output_IsJntExd),
            "exd_abs": None if not ik else float(getattr(ik, "m_Output_JntExdABS", 0.0)),
            "method": None,
        }
        self._last_ik_joints = (
            None if not ik else [float(v) for v in ik.m_Output_RetJoint.to_list()]
        )
        if not ik or ik.m_Output_IsOutRange:
            return None

        best: tuple[float, list[float]] | None = None
        rows = convert_to_8x8_matrix(ik.m_OutPut_AllJoint.to_list())
        for i in range(max(int(ik.m_OutPut_Result_Num), 0)):
            q0 = [float(rows[i][j]) for j in range(7)]
            qf = self._fold_to_limits(q0, ref)
            if qf is None or abs(qf[1]) < 0.05:
                continue
            sp2 = FX_InvKineSolvePara()
            sp2.set_input_ik_target_tcp(mat16)
            sp2.set_input_ik_ref_joint(qf)
            sp2.set_input_ik_zsp_type(0)
            sp2.set_input_zsp_angle(0.0)
            ik2 = self.kine.ik(sp2)
            if not ik2 or ik2.m_Output_IsOutRange:
                continue
            if ik2.m_Output_IsJntExd or any(ik2.m_Output_IsDeg):
                continue
            qr = [float(v) for v in ik2.m_Output_RetJoint.to_list()]
            cost = self._ik_posture_cost(qr, ref)
            if best is None or cost < best[0]:
                best = (cost, qr)
        if best is not None:
            self._last_ik_joints = list(best[1])
            self._last_ik_flags.update(
                {"out_range": False, "jnt_exd": False, "exd_abs": 0.0, "method": "ik"}
            )
            return list(best[1])

        # —— IK_NSP 臂角扫描（rimbot solveIk NSP 脱困）——
        base_ext = float(getattr(ik, "m_Output_JntExdABS", 0.0))
        e_pos = e_neg = 1e9
        sp.set_input_zsp_angle(0.01)
        r = self.kine.ik_nsp(sp)
        if r and not r.m_Output_IsOutRange:
            e_pos = float(r.m_Output_JntExdABS)
        sp.set_input_zsp_angle(-0.01)
        r = self.kine.ik_nsp(sp)
        if r and not r.m_Output_IsOutRange:
            e_neg = float(r.m_Output_JntExdABS)
        if base_ext > 0.0 and min(e_pos, e_neg) >= base_ext:
            self._last_ik_flags["method"] = "nsp_no_improve"
            return None
        direction = -1 if e_neg < e_pos else 1
        found: tuple[float, list[float], float] | None = None
        for sign in (direction, -direction):
            for step in np.arange(2.0, 60.1, 2.0):
                zsp = float(sign * step)
                sp.set_input_zsp_angle(zsp)
                r = self.kine.ik_nsp(sp)
                if not r or r.m_Output_IsOutRange:
                    continue
                if r.m_Output_IsJntExd or float(r.m_Output_JntExdABS) > 1e-5:
                    continue
                if any(r.m_Output_IsDeg):
                    continue
                qr = [float(v) for v in r.m_Output_RetJoint.to_list()]
                if abs(qr[1]) < 0.05:
                    continue
                qf = self._fold_to_limits(qr, ref)
                if qf is None:
                    continue
                cost = self._ik_posture_cost(qf, ref)
                found = (cost, qf, zsp)
                break
            if found is not None:
                break
        if found is None:
            self._last_ik_flags["method"] = "nsp_fail"
            return None
        self._last_ik_joints = list(found[1])
        self._last_ik_flags.update(
            {
                "out_range": False,
                "jnt_exd": False,
                "exd_abs": 0.0,
                "method": f"nsp@{found[2]:.0f}",
            }
        )
        LOG.info("IK_NSP recovered ZSP=%.0fdeg cost=%.1f", found[2], found[0])
        return list(found[1])

    def _ik_valid_solutions(
        self, T_base_tgt, ref_joints: Sequence[float]
    ) -> list[list[float]]:
        q = self._solve_ik(T_base_tgt, ref_joints)
        return [q] if q is not None else []

    def _solve_xyzabc_mm(
        self, xyzabc_mm: Sequence[float], ref_joints: Sequence[float]
    ) -> list[float] | None:
        from SDK_PYTHON.frame_transform import xyzabc_to_mat4

        T = xyzabc_to_mat4([float(v) for v in xyzabc_mm[:6]], kine=self.kine)
        return self._solve_ik(T, ref_joints)

    def _search_tool_down(
        self,
        xyz_mm: Sequence[float],
        ref_joints: Sequence[float],
        *,
        prefer_rz_deg: float | None = None,
        rz_step_deg: float = 15.0,
    ) -> tuple[float, list[float]] | None:
        """竖直朝下(rx=±180,ry=0)搜可行 rz；优先 prefer_rz（检测/POC 常用 -90）。"""
        x, y, z = float(xyz_mm[0]), float(xyz_mm[1]), float(xyz_mm[2])
        ref = [float(v) for v in ref_joints]
        candidates: list[float] = []
        if prefer_rz_deg is not None:
            pr = float(prefer_rz_deg)
            candidates.extend([pr, pr - 180.0, pr + 180.0])
        # POC 治具/途经常用 -90；托盘等待常用 0
        for rz in (-90.0, 0.0, 90.0, 180.0, -180.0):
            candidates.append(rz)
        rz = -180.0
        while rz < 180.0 - 1e-9:
            candidates.append(float(rz))
            rz += float(rz_step_deg)
        # dedupe
        seen_rz: set[float] = set()
        ordered: list[float] = []
        for rz in candidates:
            key = round(((rz + 180.0) % 360.0) - 180.0, 1)
            if key in seen_rz:
                continue
            seen_rz.add(key)
            ordered.append(float(rz))

        best: tuple[float, float, list[float]] | None = None
        for rz in ordered:
            for rx in (180.0, -180.0):
                q = self._solve_xyzabc_mm([x, y, z, rx, 0.0, float(rz)], ref)
                if q is None:
                    continue
                cost = max(abs(a - b) for a, b in zip(q, ref))
                if best is None or cost < best[0]:
                    best = (cost, float(rz), q)
                # prefer_rz 命中即可返回（连续路径）
                if prefer_rz_deg is not None and abs(rz - prefer_rz_deg) < 1.0:
                    return float(rz), q
        if best is None:
            return None
        return best[1], best[2]

    def _stream_joint_series(
        self,
        series: list[list[float]],
        *,
        label: str,
        period: float = 0.02,
    ) -> None:
        """连续下发关节序列（多轴同时动），中间不停顿。"""
        if not series:
            return
        LOG.info("%s stream %d pts @%.0fHz", label, len(series), 1.0 / period)
        stopped = False
        dpos_hint = [0.0, 0.0, 0.0]
        try:
            for i, q in enumerate(series):
                reason = self._contact_blocks(dpos_hint)
                if reason:
                    self.last_contact_reason = reason
                    hold = self._read_joints()
                    self.robot.set_joint_position_cmd(arm=self.arm, joint=hold)
                    stopped = True
                    break
                if not self.robot.set_joint_position_cmd(arm=self.arm, joint=list(q)):
                    LOG.error("%s cmd fail at %d/%d", label, i, len(series))
                    stopped = True
                    break
                time.sleep(period)
        finally:
            pass
        if stopped:
            raise RuntimeError(
                f"{label} aborted: {self.last_contact_reason or 'cmd/contact'}"
            )
        # 终点多喂几拍，压住阻抗滞后
        hold = list(series[-1])
        for _ in range(25):
            self.robot.set_joint_position_cmd(arm=self.arm, joint=hold)
            time.sleep(period)

    def _exec_joint_waypoints(
        self,
        waypoints: list[list[float]],
        *,
        max_deg_s: float,
        label: str,
        velocity_scale: float = 0.12,
    ) -> None:
        """整段匀速关节插值连续下发。

        不再对每个 knot 单独 movL_KeepJA（那段自带加减速，串起来就是走一下停一下）。
        """
        _ = velocity_scale
        if not waypoints:
            return
        if self.dry_run or self.robot is None:
            LOG.info("%s dry-run %d wps", label, len(waypoints))
            return
        q_prev = self._read_joints()
        chain = [list(q_prev)] + [list(q) for q in waypoints]
        period = 0.02
        step = max(0.15, float(max_deg_s) * period)  # 每拍最大关节步长（度）
        series: list[list[float]] = []
        for i in range(len(chain) - 1):
            a, b = chain[i], chain[i + 1]
            dq = max(abs(float(x) - float(y)) for x, y in zip(a, b))
            if dq < 0.15:
                continue
            n = max(1, int(math.ceil(dq / step)))
            for k in range(1, n + 1):
                t = k / n
                series.append(
                    [
                        float(a[j]) + (float(b[j]) - float(a[j])) * t
                        for j in range(7)
                    ]
                )
        if not series:
            series = [list(waypoints[-1])]
        LOG.info(
            "%s continuous joint stream knots=%d pts=%d max=%.1fdeg/s (no per-seg KeepJA)",
            label,
            len(chain) - 1,
            len(series),
            max_deg_s,
        )
        self._stream_joint_series(series, label=label, period=period)

    @staticmethod
    def _wrap_deg(a: float) -> float:
        return float((a + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _ang_diff_deg(a: float, b: float) -> float:
        """Shortest signed delta a→b in degrees."""
        return float((b - a + 180.0) % 360.0 - 180.0)

    def _plan_segment_joints(
        self,
        p0_mm: Sequence[float],
        p1_mm: Sequence[float],
        rpy_deg: Sequence[float],
        seed: Sequence[float],
        *,
        n_seg: int,
        label: str,
        rz_end_deg: float | None = None,
    ) -> list[list[float]]:
        """沿直线笛卡尔段逐点 IK(+NSP)，返回关节路点（不含起点）。

        rz 默认锁定；若给 rz_end_deg，则沿路点最短角插值到终点 rz
       （远距起步时避免一上来就用目标 rz 导致中间点超限）。
        """
        p0 = np.asarray(p0_mm, dtype=np.float64)
        p1 = np.asarray(p1_mm, dtype=np.float64)
        rpy0 = [float(v) for v in rpy_deg[:3]]
        rz0 = float(rpy0[2])
        rz1 = float(rz_end_deg) if rz_end_deg is not None else rz0
        d_rz = self._ang_diff_deg(rz0, rz1)
        last_q = [float(v) for v in seed]
        out: list[list[float]] = []
        for i in range(1, n_seg + 1):
            t = i / n_seg
            p = p0 + (p1 - p0) * t
            rz = self._wrap_deg(rz0 + d_rz * t)
            rpy = [rpy0[0], rpy0[1], rz]
            q = self._solve_xyzabc_mm([p[0], p[1], p[2], *rpy], last_q)
            if q is None and abs(abs(rpy[0]) - 180.0) < 1e-6:
                q = self._solve_xyzabc_mm(
                    [p[0], p[1], p[2], -rpy[0], rpy[1], rpy[2]], last_q
                )
            if q is None:
                # 该路点允许小幅搜 rz（仍靠近插值角），避免整段报死
                hit = self._search_tool_down(
                    [p[0], p[1], p[2]],
                    last_q,
                    prefer_rz_deg=rz,
                    rz_step_deg=10.0,
                )
                if hit is not None and abs(self._ang_diff_deg(hit[0], rz)) <= 25.0:
                    q = hit[1]
                    rz = hit[0]
            if q is None:
                raise RuntimeError(
                    f"{label} IK fail at {i}/{n_seg} "
                    f"xyz_mm={np.round(p,1).tolist()} rpy={[rpy0[0], rpy0[1], rz]} "
                    f"flags={self._last_ik_flags}"
                )
            out.append(q)
            last_q = q
            LOG.info(
                "%s wp %d/%d xyz_mm=%s rz=%.1f method=%s",
                label,
                i,
                n_seg,
                np.round(p, 1).tolist(),
                rz,
                self._last_ik_flags.get("method"),
            )
        return out

    def _move_corridor_to(
        self,
        tgt_xy_mm: Sequence[float],
        tgt_z_mm: float,
        *,
        safe_z_mm: float,
        prefer_rz_deg: float,
        seed_joints: Sequence[float],
        max_deg_s: float,
        label: str,
        velocity_scale: float = 0.12,
    ) -> list[float]:
        """先到目标高度并竖直，再同高平移到目标 XY。不用固定 safe_z。

        抬升/平移用「起点可行 rz」，到点后再转到目标 rz —— 避免从等待位
        直接锁死目标 rz 导致 TRANSIT 中间点 jnt_exd。
        """
        _ = safe_z_mm  # 兼容旧调用；高度一律用 tgt_z
        last_q = [float(v) for v in seed_joints]
        cur = pose_m_rad_to_mm_deg(self.get_flange_pose())
        prefer_rz = float(prefer_rz_deg)
        tgt_z = float(tgt_z_mm)
        start_rz_hint = float(cur[5])

        goal_hit = self._search_tool_down(
            [float(tgt_xy_mm[0]), float(tgt_xy_mm[1]), tgt_z],
            last_q,
            prefer_rz_deg=prefer_rz,
            rz_step_deg=10.0,
        )
        if goal_hit is None:
            raise RuntimeError(
                f"{label} unreachable: xy={[round(float(x),1) for x in tgt_xy_mm]} "
                f"z={tgt_z:.1f} rz={prefer_rz:.0f} flags={self._last_ik_flags}"
            )
        goal_rz, goal_q = goal_hit
        LOG.info(
            "%s goal IK ok rz=%.0f method=%s q=%s",
            label,
            goal_rz,
            self._last_ik_flags.get("method"),
            [round(x, 1) for x in goal_q],
        )

        def _tool_down_at(
            xyz_mm: Sequence[float],
            seed: Sequence[float],
            prefer: float,
        ) -> tuple[float, list[float]]:
            for rx in (-180.0, 180.0):
                q = self._solve_xyzabc_mm(
                    [xyz_mm[0], xyz_mm[1], xyz_mm[2], rx, 0.0, prefer], seed
                )
                if q is not None:
                    return float(prefer), q
            hit = self._search_tool_down(
                list(xyz_mm[:3]), seed, prefer_rz_deg=prefer, rz_step_deg=10.0
            )
            if hit is None:
                raise RuntimeError(
                    f"{label} tool-down IK fail xyz={[round(float(x),1) for x in xyz_mm[:3]]}"
                )
            return float(hit[0]), hit[1]

        LOG.info(
            "%s path: cur_z=%.1f → tgt_z=%.1f then XY; lift_rz~%.0f → goal_rz=%.0f",
            label,
            float(cur[2]),
            tgt_z,
            start_rz_hint,
            goal_rz,
        )

        knots: list[list[float]] = []
        xyz_cursor = [cur[0], cur[1], cur[2]]

        # 1) 当前 XY → 目标高度；rz 优先保持起点附近（等待位多为 -90）
        lift_rz, q_ao = _tool_down_at(
            [xyz_cursor[0], xyz_cursor[1], tgt_z], last_q, start_rz_hint
        )
        knots.append(q_ao)
        last_q = q_ao
        xyz_cursor[2] = tgt_z
        LOG.info("%s lift ok rz=%.0f (start_hint=%.0f)", label, lift_rz, start_rz_hint)

        # 2) 同高平移：rz 从 lift → goal 最短角插值
        transit_tgt = [float(tgt_xy_mm[0]), float(tgt_xy_mm[1]), tgt_z]
        dxy = math.hypot(transit_tgt[0] - xyz_cursor[0], transit_tgt[1] - xyz_cursor[1])
        if dxy >= 10.0:
            n_tr = max(6, int(math.ceil(dxy / 25.0)))
            wps = self._plan_segment_joints(
                [xyz_cursor[0], xyz_cursor[1], tgt_z],
                transit_tgt,
                [-180.0, 0.0, lift_rz],
                last_q,
                n_seg=n_tr,
                label=f"{label}/TRANSIT",
                rz_end_deg=goal_rz,
            )
            knots.extend(wps)
            last_q = wps[-1]

        knots.append(goal_q)
        knots[-1] = list(goal_q)
        LOG.info("%s continuous path %d knots", label, len(knots))
        self._exec_joint_waypoints(
            knots,
            max_deg_s=max_deg_s,
            label=label,
            velocity_scale=velocity_scale,
        )
        pose = pose_m_rad_to_mm_deg(self.get_flange_pose())
        LOG.info(
            "%s done rpy_deg=[%.1f,%.1f,%.1f] (want rx=±180 ry=0 rz=%.0f)",
            label,
            pose[3],
            pose[4],
            pose[5],
            goal_rz,
        )
        return list(goal_q)

    def move_tool_down_hover(
        self,
        xyz_m: Sequence[float],
        *,
        velocity_scale: float = 0.08,
        n_seg: int = 10,
        max_deg_s: float = 10.0,
        safe_z_m: float | None = None,
        prefer_rz_rad: float | None = None,
    ) -> list[float]:
        """到目标高度后同高平移；末端竖直朝下；不用固定 safe_z。

        目标 Z/rz 无 IK 时：先放宽 rz（-90），再把悬停高度钳到当前高度附近重试，
        避免粗定位 Z 偏高（如 +280mm）导致 jnt_exd / nsp_fail。
        """
        _ = (velocity_scale, n_seg, safe_z_m)
        tgt_xy = [float(xyz_m[0]) * 1000.0, float(xyz_m[1]) * 1000.0]
        tgt_z = float(xyz_m[2]) * 1000.0
        prefer_rz = (
            None if prefer_rz_rad is None else float(prefer_rz_rad) * 180.0 / math.pi
        )
        if prefer_rz is None and len(xyz_m) >= 6:
            prefer_rz = float(xyz_m[5]) * 180.0 / math.pi
        if prefer_rz is None:
            prefer_rz = -90.0

        joints = self._read_joints()
        cur = pose_m_rad_to_mm_deg(self.get_flange_pose())
        cur_z = float(cur[2])
        # 候选高度：目标 → 当前 → 二者中点 → 略高于当前（抬 80mm）
        z_try = [tgt_z, cur_z, 0.5 * (tgt_z + cur_z), cur_z + 80.0]
        # 去重（1mm）
        z_ordered: list[float] = []
        for z in z_try:
            if all(abs(z - u) > 1.0 for u in z_ordered):
                z_ordered.append(float(z))
        rz_try = [prefer_rz, -90.0, 0.0, 90.0, prefer_rz - 180.0, prefer_rz + 180.0]

        last_err: Exception | None = None
        used_z, used_rz = tgt_z, prefer_rz
        for z_mm in z_ordered:
            for rz_deg in rz_try:
                # 先确认目标点本身有解，再走走廊
                hit = self._search_tool_down(
                    [tgt_xy[0], tgt_xy[1], z_mm],
                    list(joints),
                    prefer_rz_deg=float(rz_deg),
                    rz_step_deg=10.0,
                )
                if hit is None:
                    continue
                LOG.info(
                    "MoveDirectHeight from_mm=%s → xy=%s z=%.1f rz=%.0f "
                    "(requested z=%.1f rz=%.0f)",
                    [round(x, 1) for x in cur],
                    [round(x, 1) for x in tgt_xy],
                    z_mm,
                    hit[0],
                    tgt_z,
                    prefer_rz,
                )
                try:
                    self._move_corridor_to(
                        tgt_xy,
                        z_mm,
                        safe_z_mm=0.0,
                        prefer_rz_deg=float(hit[0]),
                        seed_joints=list(joints),
                        max_deg_s=max_deg_s,
                        label="HOVER",
                        velocity_scale=max(0.08, float(velocity_scale)),
                    )
                    used_z, used_rz = z_mm, float(hit[0])
                    last_err = None
                    break
                except Exception as exc:  # noqa: BLE001
                    last_err = exc
                    LOG.warning("HOVER attempt z=%.1f rz=%.0f failed: %s", z_mm, rz_deg, exc)
                    continue
            else:
                continue
            break
        else:
            raise RuntimeError(
                f"HOVER unreachable: xy={[round(float(x),1) for x in tgt_xy]} "
                f"z={tgt_z:.1f} rz={prefer_rz:.0f} tried_z={[round(z,1) for z in z_ordered]} "
                f"flags={self._last_ik_flags} last={last_err}"
            )
        pose = self.get_flange_pose()
        # 误差相对「实际采用」的悬停高度（可能已回退），不是原始请求 Z
        err = math.sqrt(
            (pose[0] - float(xyz_m[0])) ** 2
            + (pose[1] - float(xyz_m[1])) ** 2
            + (pose[2] - used_z / 1000.0) ** 2
        )
        rx_deg = pose[3] * 180.0 / math.pi
        ry_deg = pose[4] * 180.0 / math.pi
        tool_ok = abs(abs(rx_deg) - 180.0) <= 3.0 and abs(ry_deg) <= 3.0
        LOG.info(
            "hover done pose_mm=%s rpy_deg=[%.1f,%.1f,%.1f] |err|=%.0fmm "
            "tool_down=%s used_z=%.1f used_rz=%.0f",
            [round(x * 1000.0, 1) for x in pose[:3]],
            rx_deg,
            ry_deg,
            pose[5] * 180.0 / math.pi,
            err * 1000.0,
            tool_ok,
            used_z,
            used_rz,
        )
        if not tool_ok:
            raise RuntimeError(
                f"hover arrived but NOT tool-down: rpy_deg=[{rx_deg:.1f},{ry_deg:.1f}] "
                f"(need |rx|~180, |ry|~0)"
            )
        if err > 0.08:
            raise RuntimeError(
                f"hover incomplete |err|={err*1000:.0f}mm "
                f"toward {[round(float(xyz_m[i])*1000,1) for i in range(3)]}"
            )
        return pose

    def _approach_keep_ori(
        self,
        T_base_now: np.ndarray,
        tgt_xyz_mm: Sequence[float],
        joints0: Sequence[float],
        *,
        velocity_scale: float,
        dpos_hint: Sequence[float],
        n_seg: int = 10,
    ) -> list[float]:
        """沿直线插值 keep-ori 逼近，返回最远可达 XYZ(mm)。"""
        p0 = T_base_now[:3, 3].copy()
        p1 = np.asarray(tgt_xyz_mm, dtype=np.float64)
        joints = list(joints0)
        last = p0.copy()
        T_base = T_base_now.copy()
        for i in range(1, n_seg + 1):
            p = p0 + (p1 - p0) * (i / n_seg)
            T_try = T_base.copy()
            T_try[0, 3], T_try[1, 3], T_try[2, 3] = p
            end_j = self._solve_ik(T_try, joints)
            if end_j is None:
                LOG.warning(
                    "approach stop at %d/%d xyz_mm=%s ik=%s",
                    i - 1,
                    n_seg,
                    np.round(last, 1).tolist(),
                    self._last_ik_flags,
                )
                break
            self._run_joint_path(
                joints, end_j, velocity_scale=velocity_scale, dpos_hint=dpos_hint
            )
            joints = end_j
            last = p
            T_base = T_try
            LOG.info("approach waypoint %d/%d xyz_mm=%s", i, n_seg, np.round(p, 1).tolist())
        return [float(last[0]), float(last[1]), float(last[2])]

    def _run_joint_path(
        self,
        joints0: Sequence[float],
        end_j: Sequence[float],
        *,
        velocity_scale: float,
        dpos_hint: Sequence[float],
    ) -> None:
        vel = max(10.0, self.cart_vel * float(velocity_scale) / 0.3)
        acc = max(20.0, self.cart_acc * float(velocity_scale) / 0.3)
        points, pset = self.kine.movL_KeepJA(
            start_joints=list(joints0),
            end_joints=list(end_j),
            vel=vel,
            acc=acc,
            freq_hz=50,
        )
        if not points or len(points) < 2:
            try:
                if pset:
                    self.kine.destroy_point_set(pset)
            except Exception:
                pass
            raise RuntimeError("movL_KeepJA failed")

        period = 0.02
        stopped = False
        try:
            for i, q in enumerate(points):
                reason = self._contact_blocks(dpos_hint)
                if reason:
                    self.last_contact_reason = reason
                    LOG.warning("CONTACT STOP at %d/%d: %s", i, len(points), reason)
                    hold = self._read_joints()
                    self.robot.set_joint_position_cmd(arm=self.arm, joint=hold)
                    stopped = True
                    break
                if not self.robot.set_joint_position_cmd(arm=self.arm, joint=list(q)):
                    LOG.error("set_joint_position_cmd failed at %d", i)
                    stopped = True
                    break
                time.sleep(period)
        finally:
            try:
                if pset:
                    self.kine.destroy_point_set(pset)
            except Exception:
                pass
        if stopped:
            raise RuntimeError(
                f"TJ move aborted: {self.last_contact_reason or 'cmd/contact'}"
            )

    def get_joints(self) -> list[float]:
        if self.dry_run or self.robot is None:
            return [0.0] * 7
        return self._read_joints()

    def move_joints(
        self,
        joints: Sequence[float],
        *,
        tol_deg: float = 2.0,
        timeout_s: float = 180.0,
        rate_hz: float = 50.0,
        max_deg_s: float = 12.0,
    ) -> list[float]:
        """慢速关节插值到位（不一次甩到终点；默认 ≤12°/s）。"""
        tgt = [float(x) for x in joints[:7]]
        if len(tgt) != 7:
            raise ValueError(f"need 7 joints, got {len(tgt)}")
        if self.dry_run or self.robot is None:
            LOG.info("TJ MOVE_JOINTS dry-run tgt=%s", [round(x, 1) for x in tgt])
            return list(tgt)

        period = 1.0 / max(1.0, float(rate_hz))
        step = max(0.1, float(max_deg_s) * period)  # 每周期最大关节步长
        cmd = self._read_joints()
        err0 = max(abs(a - b) for a, b in zip(cmd, tgt))
        eta = err0 / max(1e-6, float(max_deg_s))
        LOG.info(
            "TJ MOVE_JOINTS slow max=%.1fdeg/s step=%.2fdeg eta~%.0fs "
            "from=%s → %s",
            max_deg_s,
            step,
            eta,
            [round(x, 1) for x in cmd],
            [round(x, 1) for x in tgt],
        )
        t0 = time.time()
        last_log = 0.0
        cmd_done_since: float | None = None
        while True:
            # 向目标有限步长逼近（指令侧限速，避免阻抗下一次跳终点）
            nxt = []
            for a, b in zip(cmd, tgt):
                d = b - a
                if abs(d) <= step:
                    nxt.append(b)
                else:
                    nxt.append(a + step * (1.0 if d > 0 else -1.0))
            cmd = nxt
            if not self.robot.set_joint_position_cmd(arm=self.arm, joint=cmd):
                raise RuntimeError("set_joint_position_cmd failed in move_joints")
            time.sleep(period)
            fb = self._read_joints()
            err_cmd = max(abs(a - b) for a, b in zip(cmd, tgt))
            err_fb = max(abs(a - b) for a, b in zip(fb, tgt))
            now = time.time() - t0
            if now - last_log >= 1.0:
                LOG.info(
                    "MOVE_JOINTS t=%.0fs err_fb=%.1fdeg err_cmd=%.1fdeg fb=%s",
                    now,
                    err_fb,
                    err_cmd,
                    [round(x, 1) for x in fb],
                )
                last_log = now
            if err_fb <= tol_deg and err_cmd <= tol_deg:
                LOG.info(
                    "MOVE_JOINTS done t=%.1fs err=%.2fdeg fb=%s",
                    now,
                    err_fb,
                    [round(x, 1) for x in fb],
                )
                return fb
            # 阻抗下常差几度跟不满：指令已到终点且反馈稳定一段时间 → 软接受继续
            if err_cmd <= 0.05:
                if cmd_done_since is None:
                    cmd_done_since = now
                elif now - cmd_done_since >= 2.0 and err_fb <= max(tol_deg, 6.0):
                    LOG.warning(
                        "MOVE_JOINTS soft-arrive t=%.1fs err_fb=%.1fdeg (impedance lag) fb=%s",
                        now,
                        err_fb,
                        [round(x, 1) for x in fb],
                    )
                    return fb
            else:
                cmd_done_since = None
            if now > timeout_s:
                raise RuntimeError(
                    f"move_joints timeout err_fb={err_fb:.1f}deg "
                    f"fb={[round(x,1) for x in fb]} tgt={[round(x,1) for x in tgt]}"
                )

    def set_gripper(self, open_: bool) -> None:
        LOG.info(
            "gripper %s pos=%.2f speed=%.0f torque=%.0f",
            "OPEN" if open_ else "CLOSE",
            self.gripper_release_position if open_ else self.gripper_grasp_position,
            self.gripper_speed,
            self.gripper_torque_release if open_ else self.gripper_torque_grasp,
        )
        if self._gripper is not None:
            try:
                if open_:
                    ok = self._gripper.move(
                        self.gripper_release_position,
                        self.gripper_speed,
                        self.gripper_torque_release,
                    )
                else:
                    ok = self._gripper.move(
                        self.gripper_grasp_position,
                        self.gripper_speed,
                        self.gripper_torque_grasp,
                    )
                if not ok:
                    LOG.warning("gripper command returned False")
            except Exception as exc:
                LOG.warning("gripper command failed: %s", exc)
        else:
            super().set_gripper(open_)
