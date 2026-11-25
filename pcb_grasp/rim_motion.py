"""RIM 运控后端：定位算完目标位姿后，只交给 ``rim_python.move_to_pose`` 去动。

分工：
- ``pcb_grasp``：只负责视觉定位（粗/精）并算出目标 xyzrpy
- 运动：``perception/.venv-sdk`` + ``pipelines`` 下的 CLI，本类不做路径拆段 / 死区绕行

等价手动命令::

  source ~/workspace/perception/.venv-sdk/bin/activate
  cd /home/ubuntu/workspace/perception/projects/yuantu/pipelines
  python -m rim_python.move_to_pose --arm left|right --pose x,y,z,r,p,y --execute

- 位姿相对 ``base_link``，目标帧 ``arm_l_linkTCP`` / ``arm_r_linkTCP``
- CLI：位置毫米、姿态角度；本类内部仍用米/弧度
- **绝不**切 HAL POSITION；位置模式须在 arm-mode 页事先确认
"""

from __future__ import annotations

import logging
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from .motion import MotionBackend

LOG = logging.getLogger("pcb_grasp.rim_motion")

_PIPELINES = Path("/home/ubuntu/workspace/perception/projects/yuantu/pipelines")
_SDK_PY = Path("/home/ubuntu/workspace/perception/.venv-sdk/bin/python")
DEFAULT_FRAME_BUNDLE = "10.3.8.7:50066"

# 到位软判据：子进程失败但 TCP+姿态都够近才当成功（纯转姿态时位置Δ≈0，不能只看位移）
_ARRIVE_OK_MM = 20.0
_ARRIVE_OK_Z_MM = 12.0  # tipZ（如 grasp_z=681）单独收紧，防 XY 近但高度差 3~4cm
_ARRIVE_OK_DEG = 8.0
# MC PLAN_TO_POSE 对约 ≤5cm 的位移常直接 SUCCEEDED 且 TCP 不动（现场 40~45mm 必现）
_MC_DEADZONE_MM = 70.0
_MC_BREAKOUT_MM = 90.0
# 中途失败但已明显移动：从新位姿续跑（MC 瞬时拒规划 / arrive timeout 常见）
_RETRY_MOVED_MM = 15.0
_RETRY_ORI_DEG = 3.0
_EXECUTE_RETRIES = 3
_RETRY_SLEEP_S = 0.8


def _ori_err_deg(a: Sequence[float], b: Sequence[float]) -> float:
    """两姿态旋转角（度），R = Raᵀ Rb。"""
    from .geometry import pose_xyz_rpy_to_mat4

    Ra = pose_xyz_rpy_to_mat4(*[float(x) for x in a[:6]])[:3, :3]
    Rb = pose_xyz_rpy_to_mat4(*[float(x) for x in b[:6]])[:3, :3]
    R = Ra.T @ Rb
    c = float(np.clip((float(np.trace(R)) - 1.0) * 0.5, -1.0, 1.0))
    return float(math.degrees(math.acos(c)))


def _pose_close_enough(cur: Sequence[float], tgt: Sequence[float]) -> bool:
    remain_mm = float(
        np.linalg.norm(np.asarray(cur[:3], dtype=float) - np.asarray(tgt[:3], dtype=float))
        * 1000.0
    )
    dz_mm = abs(float(cur[2]) - float(tgt[2])) * 1000.0
    return (
        remain_mm <= _ARRIVE_OK_MM
        and dz_mm <= _ARRIVE_OK_Z_MM
        and _ori_err_deg(cur, tgt) <= _ARRIVE_OK_DEG
    )


def _arm_side(arm: str) -> str:
    a = str(arm).strip().upper()
    if a in ("A", "L", "LEFT"):
        return "left"
    if a in ("B", "R", "RIGHT"):
        return "right"
    raise ValueError(f"arm 只能是 A/B 或 left/right，收到 {arm!r}")


def _sdk_python() -> str:
    if _SDK_PY.is_file():
        return str(_SDK_PY)
    return sys.executable


def _env_with_pipelines() -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_PIPELINES) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    # 严禁任何切 POSITION 的旁路环境变量
    env.pop("RIM_FORCE_POSITION", None)
    return env


def _pose_mm_deg_str(xyzrpy_m_rad: Sequence[float]) -> str:
    p = [float(x) for x in xyzrpy_m_rad[:6]]
    return (
        f"{p[0]*1000:.4f},{p[1]*1000:.4f},{p[2]*1000:.4f},"
        f"{math.degrees(p[3]):.4f},{math.degrees(p[4]):.4f},{math.degrees(p[5]):.4f}"
    )


def _parse_pose_line(out: str, label: str = "current") -> list[float] | None:
    """从 move_to_pose 打印解析 ``current|final x,y,z,r,p,y`` → 米/弧度。"""
    m = re.search(
        rf"{label}\s+(-?\d+(?:\.\d+)?(?:,-?\d+(?:\.\d+)?){{5}})",
        out,
    )
    if not m:
        return None
    mm_deg = [float(x) for x in m.group(1).split(",")]
    return [
        mm_deg[0] / 1000.0,
        mm_deg[1] / 1000.0,
        mm_deg[2] / 1000.0,
        math.radians(mm_deg[3]),
        math.radians(mm_deg[4]),
        math.radians(mm_deg[5]),
    ]


class RimMotion(MotionBackend):
    def __init__(
        self,
        *,
        frame_bundle: str = DEFAULT_FRAME_BUNDLE,
        mc: str = "",
        arm: str = "A",
        connect_timeout_s: float = 10.0,
        plan_timeout_s: float = 120.0,
        max_linear_vel: float = 0.06,
        max_angular_vel: float = 0.25,
        dry_run: bool = False,
    ) -> None:
        self.frame_bundle = frame_bundle
        self.mc = mc
        self.arm = arm
        self.side = _arm_side(arm)
        self.connect_timeout_s = float(connect_timeout_s)
        self.plan_timeout_s = float(plan_timeout_s)
        self.max_linear_vel = float(max_linear_vel)
        self.max_angular_vel = float(max_angular_vel)
        self.dry_run = bool(dry_run)
        self.safe_z_m = 0.02
        # 路径规划：尽量少动底座 j1（优先肘腕/升降/径向），θ 变化放到高处+外侧再转
        self.j1_theta_min_deg = 3.0
        self.j1_swing_r_min_m = 0.55
        self.j1_min_clear_z_m = 0.25
        self.preferred_ik_seed: list[float] | None = None
        self.history: list[list[float]] = []
        self._pose = [0.4, 0.4, 1.0, 0.0, 0.0, -math.pi / 2]
        self._connected = False
        self.gripper_com = 2
        self.gripper_grasp_position = 0.4
        self.gripper_release_position = 1.0
        self.gripper_speed = 50.0
        self.gripper_torque_grasp = 30.0
        self.gripper_torque_release = 60.0
        # 知行夹爪走 Marvin RS485（臂仍用 RIM）；空=不连
        self.gripper_robot_ip: str = "192.168.1.190"
        self._zhixing = None

    def connect(self) -> None:
        if self.dry_run:
            self._connected = True
            LOG.info("RimMotion dry_run side=%s", self.side)
            return
        pose = self._read_tcp_m_rad()
        self._pose = list(pose)
        self._connected = True
        LOG.info(
            "RimMotion connected side=%s @ %s pose=(%.3f,%.3f,%.3f) "
            "(move via rim_python.move_to_pose; no HAL POSITION switch)",
            self.side,
            self.frame_bundle,
            pose[0],
            pose[1],
            pose[2],
        )

    def connect_as_peer(self, primary: "RimMotion") -> None:
        self.frame_bundle = primary.frame_bundle
        self.mc = primary.mc
        self.dry_run = primary.dry_run
        self._connected = primary._connected
        if not self.dry_run:
            self._pose = self._read_tcp_m_rad()
        LOG.info("RimMotion peer side=%s", self.side)

    def close(self) -> None:
        self._connected = False
        if self._zhixing is not None:
            try:
                self._zhixing.close()
            except Exception:  # noqa: BLE001
                pass
            self._zhixing = None

    def _run_move_to_pose(
        self,
        extra: list[str],
        *,
        timeout_s: float,
    ) -> subprocess.CompletedProcess:
        cmd = [
            _sdk_python(),
            "-m",
            "rim_python.move_to_pose",
            self.frame_bundle,
            "--arm",
            self.side,
            "--connect-timeout",
            str(self.connect_timeout_s),
            *extra,
        ]
        if self.mc:
            cmd.extend(["--mc", self.mc])
        LOG.debug("RIM cmd: %s", " ".join(cmd))
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_env_with_pipelines(),
            cwd=str(_PIPELINES),
            timeout=timeout_s,
        )

    def _read_tcp_m_rad(self) -> list[float]:
        proc = self._run_move_to_pose(
            [],
            timeout_s=self.connect_timeout_s + 20.0,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        pose = _parse_pose_line(out, "current") or _parse_pose_line(out, "final")
        if pose is None:
            raise RuntimeError(
                f"读不到 {self.side} TCP @ {self.frame_bundle}: "
                f"{(out.strip().splitlines() or ['无输出'])[-1]}"
            )
        return pose

    def get_flange_pose(self) -> list[float]:
        """当前末端 TCP（arm_*_linkTCP）相对 base_link，米/弧度。"""
        if self.dry_run or not self._connected:
            return list(self._pose)
        try:
            self._pose = self._read_tcp_m_rad()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("read TCP failed: %s; use cache", exc)
        return list(self._pose)

    def _estimate_duration_s(self, start: Sequence[float], goal: Sequence[float]) -> float:
        a = np.asarray(start[:6], dtype=np.float64)
        b = np.asarray(goal[:6], dtype=np.float64)
        dist = float(np.linalg.norm(b[:3] - a[:3]))
        # 粗估转角（欧拉差，仅作超时预算）
        dang = float(np.linalg.norm(b[3:] - a[3:]))
        t = dist / max(self.max_linear_vel, 1e-3) + dang / max(self.max_angular_vel, 1e-3)
        return float(min(max(t + 8.0, 12.0), self.plan_timeout_s))

    @staticmethod
    def _yaw_diff_rad(a: float, b: float) -> float:
        return abs((float(a) - float(b) + math.pi) % (2.0 * math.pi) - math.pi)

    @staticmethod
    def _pose_xyz_far(a: Sequence[float], b: Sequence[float], *, mm: float = 30.0) -> bool:
        return float(np.linalg.norm(np.asarray(a[:3]) - np.asarray(b[:3]))) >= (mm / 1000.0)

    def _waypoints_via_height(
        self, start: Sequence[float], goal: Sequence[float]
    ) -> list[list[float]]:
        """大位移时先水平到目标 XY（保持当前高度），再落到目标 Z。"""
        s = [float(x) for x in start[:6]]
        g = [float(x) for x in goal[:6]]
        d_xy = float(math.hypot(g[0] - s[0], g[1] - s[1]))
        d_z = float(abs(g[2] - s[2]))
        # 短距或几乎纯竖直：一段到位
        if d_xy < 0.05 or d_z < 0.08:
            return [g]
        via_z = max(float(s[2]), float(g[2]), float(self.safe_z_m))
        via = [g[0], g[1], via_z, g[3], g[4], g[5]]
        # via 与起点/终点过近则跳过
        if math.hypot(via[0] - s[0], via[1] - s[1]) < 0.03 and abs(via[2] - s[2]) < 0.03:
            return [g]
        if abs(via[2] - g[2]) < 0.03:
            return [g]
        return [via, g]

    @staticmethod
    def _pose_in_workspace(p: Sequence[float]) -> bool:
        """粗/复位允许工作区（米）。超出则禁止自动规划。"""
        r = float(math.hypot(float(p[0]), float(p[1])))
        return (
            r <= 0.90
            and abs(float(p[1])) <= 0.70
            and 0.40 <= float(p[2]) <= 0.98
        )

    def _waypoints_lift_orient_slide(
        self,
        start: Sequence[float],
        goal: Sequence[float],
        *,
        lift_m: float = 0.05,
    ) -> list[list[float]]:
        """两段式接近（不再绕圈、不顶高）：

        1) 保持当前姿态，平移到目标 XY、高度取 min(当前Z, 目标Z+lift, 0.95)
        2) 再转到目标完整位姿（含姿态与最终 Z）

        起点或终点超工作区直接报错（例如上次限位卡在 y≈1.1），避免再甩 1m。
        """
        s = [float(x) for x in start[:6]]
        g = [float(x) for x in goal[:6]]
        if not self._pose_in_workspace(s):
            raise RuntimeError(
                f"当前末端已在工作区外 xyz=({s[0]:.3f},{s[1]:.3f},{s[2]:.3f}) "
                f"r={math.hypot(s[0], s[1]):.2f}m。"
                f"请先在示教器/手动把臂挪回托盘上方（建议 |y|<0.6、z≈0.8），"
                f"再点接近或复位——程序不会从怪姿势自动硬拖。"
            )
        if not self._pose_in_workspace(g):
            raise RuntimeError(
                f"目标超工作区 xyz=({g[0]:.3f},{g[1]:.3f},{g[2]:.3f}) "
                f"r={math.hypot(g[0], g[1]):.2f}m；拒绝下发"
            )

        d_xy = float(math.hypot(g[0] - s[0], g[1] - s[1]))
        ori = _ori_err_deg(s, g)
        if d_xy < 0.04 and ori <= _ARRIVE_OK_DEG and abs(g[2] - s[2]) < 0.04:
            return [g]

        # 平移高度：不超过 0.95，且不要比当前位置更高太多（防顶）
        # 段1 高度：取当前/目标较高者，最多再抬 lift，且 ≤0.95（少做无谓抬升）
        via_z = float(
            min(0.95, max(float(s[2]), float(g[2])) + max(0.02, float(lift_m)))
        )
        wps: list[list[float]] = []
        # 段1：保持当前姿态，平移到目标 XY（同高或略抬）
        # 若 via 与起点过近（MC 常对 <~5cm 直接 SUCCEEDED 却 TCP 差 3~5cm），跳过抬升段
        via = [g[0], g[1], via_z, s[3], s[4], s[5]]
        d_via = float(np.linalg.norm(np.asarray(via[:3]) - np.asarray(s[:3])))
        if d_via >= 0.06:
            self._append_wp(wps, s, via)
        else:
            LOG.info(
                "RIM simple path: skip via (|via−start|=%.0fmm <60mm)，直达目标",
                d_via * 1000.0,
            )
        # 段2：目标位姿（含 camera-down / 复位姿态）
        self._append_wp(wps, s, g)
        if not wps:
            wps = [g]
        LOG.info(
            "RIM simple path: slide(keep ori)→goal |Δxy|=%.0fmm ori=%.1f° "
            "via_z=%.0fmm segments=%d",
            d_xy * 1000.0,
            ori,
            via_z * 1000.0,
            len(wps),
        )
        return wps

    def _append_wp(
        self, wps: list[list[float]], ref: Sequence[float], wp: Sequence[float]
    ) -> list[float]:
        """与上一点够远才追加；返回当前末点。"""
        cur = list(wps[-1]) if wps else [float(x) for x in ref[:6]]
        nxt = [float(x) for x in wp[:6]]
        # 位置或姿态任一变化明显才落点（姿态用欧拉差粗判）
        d_xyz = float(np.linalg.norm(np.asarray(nxt[:3]) - np.asarray(cur[:3])))
        d_rpy = float(np.linalg.norm(np.asarray(nxt[3:6]) - np.asarray(cur[3:6])))
        if d_xyz >= 0.03 or d_rpy >= math.radians(4.0):
            wps.append(nxt)
            return nxt
        return cur

    def _waypoints_j1_last(
        self, start: Sequence[float], goal: Sequence[float]
    ) -> list[list[float]]:
        """尽量少动底座 j1：优先升降 / 径向 / 肘腕姿态，θ 最后且在外侧高处转。

        PLAN_TO_POSE 无法指定关节序或权重；用柱坐标拆段逼近：
        1) 原地抬升（保 θ、r、姿态）——几乎不动 j1
        2) 固定 θ_start：外侧半径 + 目标姿态（肘腕为主）
        3) 仅当 Δθ 超阈：在高处+外侧再转方位角（j1）
        4) 收到目标半径 → 落到目标
        短距且几乎不转 θ 时仍走高度走廊，避免过度拆段。
        """
        s = [float(x) for x in start[:6]]
        g = [float(x) for x in goal[:6]]
        th_s = math.atan2(s[1], s[0])
        th_g = math.atan2(g[1], g[0])
        r_s = float(math.hypot(s[0], s[1]))
        r_g = float(math.hypot(g[0], g[1]))
        d_th = self._yaw_diff_rad(th_g, th_s)
        d_xy = float(math.hypot(g[0] - s[0], g[1] - s[1]))
        d_xyz = float(np.linalg.norm(np.asarray(g[:3]) - np.asarray(s[:3])))
        theta_min = math.radians(float(self.j1_theta_min_deg))

        # 精定位/抓取近距：禁止甩到 r_swing 外侧（否则像「乱走」）
        if d_xy < 0.25:
            if d_xyz < 0.05:
                return [g]
            return self._waypoints_via_height(s, g)

        # 短距且几乎不转底座：旧高度分段即可
        if d_th < theta_min and d_xyz < 0.12:
            return self._waypoints_via_height(s, g)

        high_z = max(
            float(s[2]),
            float(g[2]),
            float(self.safe_z_m),
            float(self.j1_min_clear_z_m),
        )
        r_swing = max(r_s, r_g, float(self.j1_swing_r_min_m))

        wps: list[list[float]] = []
        # 1) 纯抬升：保当前 xy + 姿态
        self._append_wp(wps, s, [s[0], s[1], high_z, s[3], s[4], s[5]])
        # 2) 固定 θ_s：外侧 + 目标姿态（先远端关节）
        self._append_wp(
            wps,
            s,
            [
                r_swing * math.cos(th_s),
                r_swing * math.sin(th_s),
                high_z,
                g[3],
                g[4],
                g[5],
            ],
        )
        # 3) j1 摆角：仅必要时，且在 high_z + r_swing（远离头部）
        if d_th >= theta_min:
            self._append_wp(
                wps,
                s,
                [
                    r_swing * math.cos(th_g),
                    r_swing * math.sin(th_g),
                    high_z,
                    g[3],
                    g[4],
                    g[5],
                ],
            )
        # 4) 收到目标半径（仍高处）
        self._append_wp(
            wps,
            s,
            [r_g * math.cos(th_g), r_g * math.sin(th_g), high_z, g[3], g[4], g[5]],
        )
        # 5) 落到目标
        self._append_wp(wps, s, g)
        if not wps:
            return [g]
        LOG.info(
            "RIM min-j1 staging: Δθ=%.1fdeg r_s=%.0fmm r_g=%.0fmm r_swing=%.0fmm "
            "high_z=%.0fmm segments=%d (lift→distal@θ_s→swing_j1→radial→down)",
            math.degrees(d_th),
            r_s * 1000.0,
            r_g * 1000.0,
            r_swing * 1000.0,
            high_z * 1000.0,
            len(wps),
        )
        return wps

    def _mc_deadzone_via(
        self, start: Sequence[float], goal: Sequence[float]
    ) -> list[float] | None:
        """短距落在 MC 到位死区时，造一个 ≥_MC_BREAKOUT_MM 的中间点强迫真正动臂。"""
        s = [float(x) for x in start[:6]]
        g = [float(x) for x in goal[:6]]
        need = _MC_BREAKOUT_MM / 1000.0

        def _ok(via: list[float]) -> bool:
            d = float(
                np.linalg.norm(np.asarray(via[:3], dtype=float) - np.asarray(s[:3]))
            )
            return d >= need * 0.95 and self._pose_in_workspace(via)

        # 1) 越过目标再 breakout：保证 start→via 与 via→goal 都 ≥死区
        #    via = goal + dir*(need)；返回段 |via−goal|=need，不会再假成功
        delta = np.asarray(g[:3], dtype=float) - np.asarray(s[:3], dtype=float)
        dist = float(np.linalg.norm(delta))
        if dist >= 1e-6:
            direction = delta / dist
            via_xyz = np.asarray(g[:3], dtype=float) + direction * need
            via_push = [
                float(via_xyz[0]),
                float(via_xyz[1]),
                float(via_xyz[2]),
                g[3],
                g[4],
                g[5],
            ]
            if _ok(via_push):
                return via_push
        # 2) 原地抬高（顶到 0.95）；够 breakout 才用，否则 MC 仍会假成功
        lift_z = float(min(0.95, s[2] + need))
        via_lift = [s[0], s[1], lift_z, g[3], g[4], g[5]]
        if _ok(via_lift):
            return via_lift
        # 3) 水平外推：沿 XY 投影方向
        dxy = math.hypot(g[0] - s[0], g[1] - s[1])
        if dxy >= 1e-6:
            ux, uy = (g[0] - s[0]) / dxy, (g[1] - s[1]) / dxy
            via_xy = [s[0] + ux * need, s[1] + uy * need, s[2], g[3], g[4], g[5]]
            if _ok(via_xy):
                return via_xy
        return None

    def _execute_pose(
        self,
        start: Sequence[float],
        tgt: Sequence[float],
        *,
        velocity_scale: float,
        tag: str = "",
    ) -> list[float]:
        """定位给出目标后，直接调 CLI；不做死区 via / 续跑。"""
        del start, velocity_scale
        return self._cli_move_to_pose(tgt, tag=tag)

    def _cli_move_to_pose_once(
        self,
        tgt: Sequence[float],
        *,
        tag: str = "",
        max_linear_vel: float | None = None,
        timeout_s: float | None = None,
    ) -> tuple[list[float] | None, int, str]:
        """单次 CLI 下发；返回 (final_pose|None, returncode, tail_msg)。"""
        goal = [float(x) for x in tgt[:6]]
        pose_str = _pose_mm_deg_str(goal)
        wait_s = float(
            self.plan_timeout_s if timeout_s is None else timeout_s
        )
        extra = [
            "--pose",
            pose_str,
            "--execute",
            "--allow-large",
            "--timeout",
            f"{wait_s:.1f}",
        ]
        if max_linear_vel is not None and float(max_linear_vel) > 0.0:
            extra.extend(["--max-linear-vel", f"{float(max_linear_vel):.4f}"])
        LOG.info(
            "localize→CLI%s side=%s pose_mm_deg=%s  "
            "(cd %s && python -m rim_python.move_to_pose --arm %s --pose %s --execute)",
            f"[{tag}]" if tag else "",
            self.side,
            pose_str,
            _PIPELINES,
            self.side,
            pose_str,
        )
        proc = self._run_move_to_pose(
            extra,
            timeout_s=max(wait_s + self.connect_timeout_s + 15.0, 30.0),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        for line in out.strip().splitlines()[-20:]:
            LOG.info("[rim:%s] %s", self.side, line)

        after = _parse_pose_line(out, "final")
        if after is None:
            try:
                after = self._read_tcp_m_rad()
            except Exception:
                after = None
        tail = (out.strip().splitlines() or [f"exit {proc.returncode}"])[-1]
        return after, int(proc.returncode), str(tail)

    def _cli_move_to_pose(
        self,
        tgt: Sequence[float],
        *,
        tag: str = "",
        max_linear_vel: float | None = None,
        timeout_s: float | None = None,
    ) -> list[float]:
        """下发目标并校验 TCP 真到位；MC 假成功 / 死区停半路会自动 via+重试。"""
        goal = [float(x) for x in tgt[:6]]
        last: list[float] | None = list(self._pose) if self._pose else None
        last_tail = ""

        for attempt in range(1, int(_EXECUTE_RETRIES) + 1):
            # 剩余落在 MC 死区：先经 via 拉出，否则再 PLAN 目标会假 SUCCEEDED
            if last is not None:
                remain0 = float(
                    np.linalg.norm(
                        np.asarray(last[:3], dtype=float)
                        - np.asarray(goal[:3], dtype=float)
                    )
                    * 1000.0
                )
                if _pose_close_enough(last, goal):
                    self._pose = list(last)
                    return list(self._pose)
                if 1.0 < remain0 <= float(_MC_DEADZONE_MM):
                    via = self._mc_deadzone_via(last, goal)
                    if via is not None:
                        LOG.warning(
                            "RIM %s attempt %d/%d remain=%.0fmm ∈死区→ via "
                            "(%.0f,%.0f,%.0f)mm 再投 goal",
                            self.side,
                            attempt,
                            _EXECUTE_RETRIES,
                            remain0,
                            via[0] * 1000.0,
                            via[1] * 1000.0,
                            via[2] * 1000.0,
                        )
                        via_pose, _, _ = self._cli_move_to_pose_once(
                            via,
                            tag=f"{tag}|via{attempt}" if tag else f"via{attempt}",
                            max_linear_vel=max_linear_vel,
                            timeout_s=timeout_s,
                        )
                        if via_pose is not None:
                            last = via_pose
                            self._pose = list(via_pose)

            after, rc, tail = self._cli_move_to_pose_once(
                goal,
                tag=f"{tag}|try{attempt}" if tag else f"try{attempt}",
                max_linear_vel=max_linear_vel,
                timeout_s=timeout_s,
            )
            last_tail = tail
            if after is not None:
                last = after
                self._pose = list(after)
                remain = float(
                    np.linalg.norm(
                        np.asarray(after[:3], dtype=float)
                        - np.asarray(goal[:3], dtype=float)
                    )
                    * 1000.0
                )
                ori_err = _ori_err_deg(after, goal)
                dz_mm = abs(float(after[2]) - float(goal[2])) * 1000.0
                if _pose_close_enough(after, goal):
                    if rc != 0:
                        LOG.warning(
                            "move_to_pose exit≠0 但 TCP 已近 remain=%.1fmm "
                            "ori=%.1f° ΔZ=%.1fmm，按到位",
                            remain,
                            ori_err,
                            dz_mm,
                        )
                    elif attempt > 1:
                        LOG.info(
                            "RIM %s arrived after retry %d remain=%.1fmm ΔZ=%.1fmm",
                            self.side,
                            attempt,
                            remain,
                            dz_mm,
                        )
                    return list(self._pose)
                LOG.warning(
                    "RIM %s attempt %d/%d MC=%s 但未到位 "
                    "remain=%.0fmm ΔZ=%.0fmm ori=%.1f° → 续跑",
                    self.side,
                    attempt,
                    _EXECUTE_RETRIES,
                    "ok" if rc == 0 else tail,
                    remain,
                    dz_mm,
                    ori_err,
                )
            else:
                LOG.warning(
                    "RIM %s attempt %d/%d 无 final TCP (%s) → 续跑",
                    self.side,
                    attempt,
                    _EXECUTE_RETRIES,
                    tail,
                )
            if attempt < int(_EXECUTE_RETRIES):
                time.sleep(float(_RETRY_SLEEP_S))

        detail = ""
        if last is not None:
            remain = float(
                np.linalg.norm(
                    np.asarray(last[:3], dtype=float)
                    - np.asarray(goal[:3], dtype=float)
                )
                * 1000.0
            )
            dz_mm = abs(float(last[2]) - float(goal[2])) * 1000.0
            detail = (
                f" (TCP remain={remain:.0f}mm ΔZ={dz_mm:.0f}mm "
                f"ori={_ori_err_deg(last, goal):.1f}°)"
            )
            self._pose = list(last)
        raise RuntimeError(
            f"rim_python.move_to_pose 未到位 after {_EXECUTE_RETRIES} tries: "
            f"{last_tail}{detail}"
        )

    def move_jog(
        self,
        xyzrpy: Sequence[float],
        *,
        max_linear_vel: float = 0.12,
        timeout_s: float = 8.0,
    ) -> list[float]:
        """短距点动：不再先读 TCP，带线速度上限，短超时。供大板内收。"""
        tgt = [float(x) for x in xyzrpy[:6]]
        self.history.append(list(tgt))
        if self.dry_run or not self._connected:
            self._pose = list(tgt)
            return list(self._pose)
        LOG.info(
            "RIM JOG side=%s vmax=%.3f xyz=(%.4f,%.4f,%.4f)",
            self.side,
            max_linear_vel,
            tgt[0],
            tgt[1],
            tgt[2],
        )
        return self._cli_move_to_pose(
            tgt,
            tag="jog",
            max_linear_vel=float(max_linear_vel),
            timeout_s=float(timeout_s),
        )

    def move_line(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        keep_orientation: bool = False,
        allow_partial: bool = False,
        direct: bool = False,
        refresh_start: bool = False,
    ) -> list[float]:
        """``move_to_pose --execute`` 单段直达（与 CLI 相同）。

        默认用缓存 TCP 打 from= 日志，不再每次先起 CLI 读位姿——
        否则双臂串行 L→R 时，左臂刚结束还会空等 ~1s 才发右臂规划。
        CLI 自己会读当前位姿；仅 ``keep_orientation`` / ``refresh_start`` 才强制刷新。
        """
        del allow_partial
        tgt = [float(x) for x in xyzrpy[:6]]
        if keep_orientation or refresh_start or not self._pose:
            cur = self.get_flange_pose()
        else:
            cur = list(self._pose)
        if keep_orientation:
            tgt = [tgt[0], tgt[1], tgt[2], cur[3], cur[4], cur[5]]
        self.history.append(list(tgt))
        d_xy = math.hypot(tgt[0] - cur[0], tgt[1] - cur[1])
        d_xyz = float(
            np.linalg.norm(np.asarray(tgt[:3]) - np.asarray(cur[:3]))
        )
        LOG.info(
            "RIM MOVE_LINE side=%s scale=%.2f xyz=(%.4f,%.4f,%.4f) rpy=(%.3f,%.3f,%.3f) "
            "from=(%.3f,%.3f,%.3f) |Δxy|=%.0fmm |Δxyz|=%.0fmm "
            "(localize only → rim_python.move_to_pose CLI)",
            self.side,
            velocity_scale,
            *tgt,
            cur[0],
            cur[1],
            cur[2],
            d_xy * 1000.0,
            d_xyz * 1000.0,
        )
        if self.dry_run or not self._connected:
            self._pose = list(tgt)
            return list(self._pose)

        # 不做工作区门禁：位姿算完直接交给 rim_python.move_to_pose
        del direct, velocity_scale
        return self._cli_move_to_pose(tgt, tag="line")

    def _run_waypoints(
        self,
        start: Sequence[float],
        waypoints: Sequence[Sequence[float]],
        *,
        velocity_scale: float,
        tag: str,
    ) -> list[float]:
        wps = [list(map(float, wp[:6])) for wp in waypoints]
        if not wps:
            return list(self.get_flange_pose())
        if len(wps) > 1:
            LOG.info(
                "RIM %s staged %d segment(s): "
                "first xyz=(%.3f,%.3f,%.3f) rpy_deg=(%.1f,%.1f,%.1f) → "
                "last xyz=(%.3f,%.3f,%.3f) rpy_deg=(%.1f,%.1f,%.1f)",
                tag,
                len(wps),
                wps[0][0],
                wps[0][1],
                wps[0][2],
                math.degrees(wps[0][3]),
                math.degrees(wps[0][4]),
                math.degrees(wps[0][5]),
                wps[-1][0],
                wps[-1][1],
                wps[-1][2],
                math.degrees(wps[-1][3]),
                math.degrees(wps[-1][4]),
                math.degrees(wps[-1][5]),
            )
        for i, wp in enumerate(wps):
            LOG.info(
                "RIM %s wp%d/%d xyz=(%.4f,%.4f,%.4f) rpy_deg=(%.1f,%.1f,%.1f)",
                tag,
                i + 1,
                len(wps),
                wp[0],
                wp[1],
                wp[2],
                math.degrees(wp[3]),
                math.degrees(wp[4]),
                math.degrees(wp[5]),
            )
        pose = [float(x) for x in start[:6]]
        ori_scale = max(float(velocity_scale), 0.45)
        for i, wp in enumerate(wps):
            d_xyz = float(
                np.linalg.norm(np.asarray(wp[:3]) - np.asarray(pose[:3]))
            )
            d_ori = _ori_err_deg(pose, wp)
            scale = (
                ori_scale
                if (d_xyz < 0.025 and d_ori > _ARRIVE_OK_DEG)
                else float(velocity_scale)
            )
            pose = self._execute_pose(
                pose,
                wp,
                velocity_scale=scale,
                tag=f"{tag} {i+1}/{len(wps)}",
            )
        return list(self._pose)

    def move_reset(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.35,
        lift_m: float = 0.06,
    ) -> list[float]:
        """复位 / 粗接近：单段 ``move_to_pose``（与 CLI 相同）。"""
        del lift_m  # 不再分段抬升；保留参数兼容调用方
        tgt = [float(x) for x in xyzrpy[:6]]
        cur = self.get_flange_pose()
        ori = _ori_err_deg(cur, tgt)
        d_xy = float(math.hypot(tgt[0] - cur[0], tgt[1] - cur[1]))
        LOG.info(
            "RIM MOVE_RESET side=%s scale=%.2f |Δxy|=%.0fmm ori_err=%.1f° "
            "(single move_to_pose) → xyz=(%.4f,%.4f,%.4f) rpy_deg=(%.1f,%.1f,%.1f)",
            self.side,
            velocity_scale,
            d_xy * 1000.0,
            ori,
            tgt[0],
            tgt[1],
            tgt[2],
            tgt[3] * 180.0 / math.pi,
            tgt[4] * 180.0 / math.pi,
            tgt[5] * 180.0 / math.pi,
        )
        if self.dry_run or not self._connected:
            self._pose = list(tgt)
            return list(self._pose)
        # 不做工作区门禁：直接交给 rim_python.move_to_pose
        del velocity_scale
        return self._cli_move_to_pose(tgt, tag="reset")

    def move_hover(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.35,
        lift_m: float = 0.05,
    ) -> list[float]:
        """粗定位悬停：单段 move_to_pose 直达。"""
        return self.move_reset(
            xyzrpy, velocity_scale=velocity_scale, lift_m=lift_m
        )

    def set_gripper(self, open_: bool) -> None:
        """优先知行 RS485（位置 0=合 1=开，收夹默认 0.4）；否则回退 rim_python。"""
        pos = (
            float(self.gripper_release_position)
            if open_
            else float(self.gripper_grasp_position)
        )
        pos = float(min(1.0, max(0.0, pos)))
        cmd_name = "open" if open_ else "close"
        LOG.info(
            "gripper %s side=%s pos=%.3f (知行:0=合1=开) backend=%s",
            cmd_name,
            self.side,
            pos,
            "zhixing" if self._zhixing is not None else "rim_python",
        )
        if self.dry_run or not self._connected:
            return

        if self._zhixing is not None:
            from .zhixing_grip import move_zhixing

            ok = move_zhixing(
                self._zhixing,
                position=pos,
                speed=float(self.gripper_speed),
                torque=float(
                    self.gripper_torque_release
                    if open_
                    else self.gripper_torque_grasp
                ),
            )
            if not ok:
                raise RuntimeError(f"知行夹爪 {cmd_name} 失败 pos={pos:.3f}")
            return

        # 回退 rim_python：命令开度 0=闭 1=开（与知行位置同向）；勿再 1-pos 反转
        opening = pos
        cmd = [
            _sdk_python(),
            "-m",
            "rim_python.gripper",
            cmd_name,
            self.frame_bundle,
            "--arm",
            self.side,
            "--opening",
            f"{opening:.4f}",
            "--execute",
            "--connect-timeout",
            str(self.connect_timeout_s),
            "--duration",
            "2.0",
        ]
        if self.mc:
            cmd.extend(["--mc", self.mc])
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=_env_with_pipelines(),
            cwd=str(_PIPELINES),
            timeout=max(40.0, self.connect_timeout_s + 30.0),
        )
        out = ((proc.stdout or "") + (proc.stderr or "")).strip()
        for line in out.splitlines()[-12:]:
            LOG.info("[rim-grip:%s] %s", self.side, line)
        if proc.returncode != 0:
            tail = (out.splitlines() or [f"exit {proc.returncode}"])[-1]
            raise RuntimeError(f"RIM gripper {cmd_name} 失败: {tail}")

    def _setup_gripper(self) -> None:
        """优先连 Marvin 上的知行夹爪；失败则仅标记 rim_python 回退。"""
        ip = str(getattr(self, "gripper_robot_ip", "") or "").strip()
        if ip and not self.dry_run:
            from .zhixing_grip import connect_zhixing

            self._zhixing = connect_zhixing(
                robot_ip=ip,
                arm=str(self.arm),
                com_ch=int(self.gripper_com),
            )
        if self._zhixing is not None:
            LOG.info(
                "gripper=zhixing side=%s grasp=%.3f release=%.3f @ %s",
                self.side,
                float(self.gripper_grasp_position),
                float(self.gripper_release_position),
                ip,
            )
        else:
            LOG.warning(
                "gripper=rim_python fallback side=%s grasp=%.3f "
                "(知行未连上时用；现场夹爪可能仍不动)",
                self.side,
                float(self.gripper_grasp_position),
            )

    def move_tool_down_hover(
        self,
        xyzrpy: Sequence[float],
        *,
        velocity_scale: float = 0.3,
        safe_z_m: float | None = None,
        max_deg_s: float | None = None,
        prefer_rz_rad: float | None = None,
        direct: bool = True,
    ) -> list[float]:
        """抓取段默认 direct：tip 朝下近距移动，不走粗接近外侧绕行。"""
        del max_deg_s, prefer_rz_rad
        if safe_z_m is not None:
            self.safe_z_m = float(safe_z_m)
        return self.move_line(
            xyzrpy, velocity_scale=velocity_scale, direct=bool(direct)
        )
