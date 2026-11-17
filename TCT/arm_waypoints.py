#!/usr/bin/env python3
"""示教 2 个途经点，再按固定扫描路径驱动 RealMan 机械臂。

流程：
  1) 拖动/示教器挪到目标位姿 → 回车记录（默认 2 点：途经点1、途经点2）
  2) 录完后自动保存 JSON，输入 YES 开始运动

也可分开跑：
  python arm_waypoints.py --record                 # 只录点
  python arm_waypoints.py --record --execute       # 录完确认后运动
  python arm_waypoints.py --play --execute         # 只回放
  python arm_waypoints.py --play --execute --yes   # 跳过确认

位姿单位：Base 系 xyz(m) + rpy(rad)。基座 +Z 竖直向上。

默认姿态：末端垂直向下（工具 Z ∥ 基座 -Z）
  - 回放只改 XYZ，RPY 全程锁定为垂直向下
  - 偏航默认继承开始运动时当前末端绕竖直轴朝向（可用 --ee-yaw-deg 覆盖）
  - --cam-level：改回旧行为「相机与桌面平行」（需手眼）
  - --lock-current-ori / --no-lock-ori：其它姿态模式

默认路径（只用录到的 2 点自动展开，抬高默认 8cm）：
  1 → 2 → 抬高 → 回 2 → 1 → 抬高 → 回 1 → 中点 → 抬高

运动默认 rm_movel；Ctrl+C → 急停。
"""
from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

Pose = List[float]

DEFAULT_ARM_IP = "192.168.1.18"
DEFAULT_ARM_PORT = 8080
ROOT = Path(__file__).resolve().parent
DEFAULT_WP = ROOT / "waypoints" / "waypoints_latest.json"
DEFAULT_HANDEYE = ROOT / "shape_match" / "calib" / "wrist_d405_T_tool_cam.yaml"


def _add_calibration_path() -> Optional[Path]:
    import os

    roots = []
    env = os.environ.get("CALIBRATION_ROOT")
    if env:
        roots.append(Path(env))
    roots += [
        ROOT.parent / "Calibration",
        Path("/home/ubuntu/workspace/lk_ceshi/Calibration"),
    ]
    for root in roots:
        ctrl = (
            root
            / "calibration_runtime"
            / "modules"
            / "robot"
            / "realman"
            / "legacy_arm_controller.py"
        )
        if ctrl.is_file():
            sys.path.insert(0, str(root))
            sys.path.insert(0, str(root / "calibration_runtime"))
            return root
    return None


def _fmt(pose: Pose) -> str:
    return (
        "xyz=["
        + ", ".join(f"{v: .4f}" for v in pose[:3])
        + "]  rpy=["
        + ", ".join(f"{v: .4f}" for v in pose[3:])
        + "]"
    )


def rpy_to_R(rx: float, ry: float, rz: float) -> np.ndarray:
    """RPY = Rz @ Ry @ Rx（RealMan / yuantu 约定）。"""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )


def R_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """逆解 Rz@Ry@Rx。"""
    ry = math.asin(float(np.clip(-R[2, 0], -1.0, 1.0)))
    cy = math.cos(ry)
    if abs(cy) > 1e-8:
        rx = math.atan2(float(R[2, 1]), float(R[2, 2]))
        rz = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        rx = math.atan2(float(-R[1, 2]), float(R[1, 1]))
        rz = 0.0
    return rx, ry, rz


def load_T_tool_cam(path: Path) -> np.ndarray:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(text)
        if isinstance(data, dict):
            if "T_tool_cam" in data:
                M = np.asarray(data["T_tool_cam"], dtype=np.float64)
                if M.shape == (4, 4):
                    return M
            block = data.get("camera2_eye_in_hand") or data.get("camera2")
            if isinstance(block, dict) and "matrix" in block:
                M = np.asarray(block["matrix"], dtype=np.float64)
                if M.shape == (4, 4):
                    return M
    except Exception:
        pass

    # 简易回退解析
    token = "T_tool_cam:"
    idx = text.find(token)
    if idx < 0:
        raise SystemExit(f"无法解析手眼矩阵: {path}")
    rows: list[list[float]] = []
    for line in text[idx + len(token) :].splitlines():
        line = line.strip()
        if not line.startswith("-"):
            if rows:
                break
            continue
        if "[" in line:
            inner = line[line.index("[") : line.rindex("]") + 1]
            rows.append([float(x) for x in inner.strip("[]").split(",")])
            if len(rows) == 4:
                return np.asarray(rows, dtype=np.float64)
    raise SystemExit(f"无法解析手眼矩阵: {path}")


def _orthonormalize(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(R)
    R2 = U @ Vt
    if np.linalg.det(R2) < 0:
        U = U.copy()
        U[:, -1] *= -1
        R2 = U @ Vt
    return R2


def camera_yaw_about_base_z(R_base_cam: np.ndarray) -> float:
    """相机 X 轴在水平面投影相对基座 +X 的偏航角（rad）。"""
    x = R_base_cam[:3, 0].astype(np.float64)
    xy = np.array([x[0], x[1], 0.0], dtype=np.float64)
    n = float(np.linalg.norm(xy))
    if n < 1e-9:
        # 退化：用相机 Y 投影
        y = R_base_cam[:3, 1].astype(np.float64)
        xy = np.array([y[0], y[1], 0.0], dtype=np.float64)
        n = float(np.linalg.norm(xy))
        if n < 1e-9:
            return 0.0
        xy /= n
        return math.atan2(float(xy[1]), float(xy[0])) - math.pi / 2
    xy /= n
    return math.atan2(float(xy[1]), float(xy[0]))


def make_R_base_cam_level(yaw_about_z: float, look_sign: float = -1.0) -> np.ndarray:
    """构造相机系：Z_cam = look_sign * Z_base（默认 -Z 俯视），绕竖直轴偏航 yaw。"""
    z = np.array([0.0, 0.0, look_sign], dtype=np.float64)
    x0 = np.array([math.cos(yaw_about_z), math.sin(yaw_about_z), 0.0], dtype=np.float64)
    y = np.cross(z, x0)
    yn = float(np.linalg.norm(y))
    if yn < 1e-12:
        raise SystemExit("无法构造水平相机姿态（叉积退化）")
    y /= yn
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def compute_cam_level_rpy(
    current_xyzrpy: Sequence[float],
    T_tool_cam: np.ndarray,
    *,
    cam_yaw_rad: Optional[float] = None,
    look_sign: float = -1.0,
) -> List[float]:
    """由当前位姿/手眼，求使相机与桌面平行的 EE RPY。

    look_sign=-1：相机光轴朝基座 -Z（俯视桌面，与桌面平行）。
    偏航默认继承当前姿态下相机绕竖直轴的朝向。
    """
    R_tc = T_tool_cam[:3, :3]
    rx, ry, rz = (float(current_xyzrpy[3]), float(current_xyzrpy[4]), float(current_xyzrpy[5]))
    R_be = rpy_to_R(rx, ry, rz)
    R_bc = R_be @ R_tc
    if cam_yaw_rad is None:
        cam_yaw_rad = camera_yaw_about_base_z(R_bc)
    R_bc_lvl = make_R_base_cam_level(float(cam_yaw_rad), look_sign=look_sign)
    R_be_lvl = _orthonormalize(R_bc_lvl @ R_tc.T)
    # 校验
    cam_z = (R_be_lvl @ R_tc)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(abs(cam_z[2]), -1.0, 1.0))))
    if tilt > 0.5:
        print(f"警告：平行姿态残差倾角 {tilt:.3f}°（期望≈0）", flush=True)
    return list(R_to_rpy(R_be_lvl))


def make_R_base_ee_down(yaw_about_z: float) -> np.ndarray:
    """构造末端系：工具 Z = 基座 -Z（垂直向下），绕竖直轴偏航 yaw。"""
    return make_R_base_cam_level(yaw_about_z, look_sign=-1.0)


def compute_ee_down_rpy(
    current_xyzrpy: Sequence[float],
    *,
    ee_yaw_rad: Optional[float] = None,
) -> List[float]:
    """求使机械臂末端垂直向下的 RPY（工具 Z ∥ 基座 -Z）。"""
    rx, ry, rz = (float(current_xyzrpy[3]), float(current_xyzrpy[4]), float(current_xyzrpy[5]))
    R_be = rpy_to_R(rx, ry, rz)
    if ee_yaw_rad is None:
        ee_yaw_rad = camera_yaw_about_base_z(R_be)
    R_down = _orthonormalize(make_R_base_ee_down(float(ee_yaw_rad)))
    tool_z = R_down[:, 2]
    align = float(np.dot(tool_z, np.array([0.0, 0.0, -1.0])))
    tilt = math.degrees(math.acos(float(np.clip(abs(align), -1.0, 1.0))))
    if tilt > 0.5:
        print(f"警告：末端垂直向下残差倾角 {tilt:.3f}°（期望≈0）", flush=True)
    return list(R_to_rpy(R_down))


def build_two_point_scan_path(poses: List[Pose], lift_m: float) -> List[Pose]:
    """由 2 个途经点展开扫描路径。

    顺序：1 → 2 → 抬高 → 回2 → 1 → 抬高 → 回1 → 中点 → 抬高
    """
    if len(poses) < 2:
        raise SystemExit("扫描路径需要至少 2 个途经点")
    p1 = [float(v) for v in poses[0]]
    p2 = [float(v) for v in poses[1]]
    if len(poses) > 2:
        print(f"提示：录了 {len(poses)} 点，扫描路径只用前 2 点", flush=True)

    def with_xyz(src: Pose, x: float, y: float, z: float) -> Pose:
        return [x, y, z, float(src[3]), float(src[4]), float(src[5])]

    def lift(p: Pose) -> Pose:
        return with_xyz(p, p[0], p[1], p[2] + lift_m)

    mid = with_xyz(
        p1,
        0.5 * (p1[0] + p2[0]),
        0.5 * (p1[1] + p2[1]),
        0.5 * (p1[2] + p2[2]),
    )
    path = [
        p1,  # 先到途经点1
        p2,  # 1 → 2
        lift(p2),  # 抬高
        p2,  # 回 2
        p1,  # → 1
        lift(p1),  # 抬高
        p1,  # 回 1
        mid,  # 中点
        lift(mid),  # 抬高
    ]
    labels = [
        "途经点1",
        "途经点2",
        f"途经点2+抬高{lift_m*100:.0f}cm",
        "途经点2",
        "途经点1",
        f"途经点1+抬高{lift_m*100:.0f}cm",
        "途经点1",
        "中点",
        f"中点+抬高{lift_m*100:.0f}cm",
    ]
    print("扫描路径：", flush=True)
    for i, (lab, pose) in enumerate(zip(labels, path), start=1):
        print(f"  [{i}/{len(path)}] {lab}: {_fmt(pose)}", flush=True)
    return path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="录制/回放机械臂途经点（默认末端垂直向下 + 两点扫描路径）"
    )
    p.add_argument("--arm-ip", default=DEFAULT_ARM_IP)
    p.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT)
    p.add_argument("--count", type=int, default=2, help="途经点个数（默认 2）")
    p.add_argument(
        "--waypoints",
        default=str(DEFAULT_WP),
        help=f"路点 JSON（默认 {DEFAULT_WP}）",
    )
    p.add_argument("--speed", type=float, default=8.0, help="速度百分比 1-100（默认 8，较慢）")
    p.add_argument("--dwell", type=float, default=0.5, help="每点到位后停留秒")
    p.add_argument("--loop", type=int, default=1, help="整条路径重复次数")
    p.add_argument(
        "--lift-cm",
        type=float,
        default=8.0,
        help="扫描路径抬高高度（cm，默认 8）",
    )
    p.add_argument(
        "--raw-path",
        action="store_true",
        help="不展开两点扫描路径，按录制顺序原样走",
    )
    p.add_argument("--record", action="store_true", help="只示教录点")
    p.add_argument("--play", action="store_true", help="只回放已有路点")
    p.add_argument("--execute", action="store_true", help="真正运动（否则 dry-run）")
    p.add_argument("--yes", action="store_true", help="跳过 YES 确认")
    p.add_argument(
        "--handeye",
        default=str(DEFAULT_HANDEYE),
        help="腕部 T_tool_cam YAML（仅 --cam-level 时需要）",
    )
    ori = p.add_mutually_exclusive_group()
    ori.add_argument(
        "--cam-level",
        action="store_true",
        help="改用旧行为：强制相机与桌面平行（需手眼）",
    )
    ori.add_argument(
        "--no-lock-ori",
        action="store_true",
        help="不锁定姿态：各途经点用各自录制的 RPY",
    )
    ori.add_argument(
        "--lock-current-ori",
        action="store_true",
        help="锁定为开始运动时的当前 RPY",
    )
    p.add_argument(
        "--ee-yaw-deg",
        type=float,
        default=None,
        help="末端绕竖直轴偏航（度）。默认继承开始运动时当前朝向",
    )
    p.add_argument(
        "--cam-yaw-deg",
        type=float,
        default=None,
        help="[仅 --cam-level] 相机绕竖直轴偏航（度）",
    )
    p.add_argument(
        "--no-linear",
        action="store_true",
        help="禁用 rm_movel，改用 movej_p（关节插值途中姿态会漂）",
    )
    p.add_argument(
        "--interp-mm",
        type=float,
        default=0.0,
        help="途经点之间按 XYZ 直线插密集点（mm）；>0 时即使回退 movej_p 也更易保姿态",
    )
    return p.parse_args()


def load_poses(path: Path) -> List[Pose]:
    data = json.loads(path.read_text(encoding="utf-8"))
    poses: List[Pose] = []
    for item in data.get("poses", []):
        xyzrpy = item.get("xyzrpy")
        if xyzrpy is None or len(xyzrpy) != 6:
            raise SystemExit(f"坏点: {item}")
        poses.append([float(v) for v in xyzrpy])
    if not poses:
        raise SystemExit(f"{path} 没有 poses")
    return poses


def save_poses(
    path: Path,
    poses: List[Pose],
    *,
    arm_ip: str,
    arm_port: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "unit": "m_rad",
        "frame": "base_ee",
        "arm_ip": arm_ip,
        "arm_port": arm_port,
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(poses),
        "poses": [
            {"name": f"p{i + 1}", "xyzrpy": [round(v, 6) for v in pose]}
            for i, pose in enumerate(poses)
        ],
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.write_text(text, encoding="utf-8")
    latest = DEFAULT_WP
    if path.resolve() != latest.resolve():
        latest.parent.mkdir(parents=True, exist_ok=True)
        latest.write_text(text, encoding="utf-8")


def record_waypoints(arm, count: int) -> List[Pose]:
    poses: List[Pose] = []
    print(
        f"将录 {count} 个途经点。示教器/拖动到位后回车记录；输入 q 提前结束。\n"
        "提示：默认录 2 点后按「1→2→抬高→回2→1→抬高→回1→中点→抬高」走；"
        "回放默认强制「末端垂直向下」，只走 XYZ；录点时姿态可随意。"
    )
    while len(poses) < count:
        ok, cur = arm.get_current_pose()
        if not ok:
            raise SystemExit(f"读位姿失败: {cur}")
        cur = [float(v) for v in cur]
        n = len(poses) + 1
        print(f"\n[{n}/{count}] 当前 {_fmt(cur)}")
        cmd = input("回车=记录此点 / r=刷新 / q=结束保存 > ").strip().lower()
        if cmd == "q":
            break
        if cmd == "r":
            continue
        if cmd not in ("", "y", "yes"):
            print("未识别，已忽略。")
            continue
        poses.append(cur)
        print(f"  ✓ 已记录点{len(poses)}: {_fmt(cur)}")
        time.sleep(0.05)
    return poses


def clear_err(arm) -> None:
    robot = getattr(arm, "_robot", None)
    if robot is None:
        return
    try:
        ret = robot.rm_clear_system_err()
        print(f"  rm_clear_system_err → {ret}", flush=True)
    except Exception as exc:  # pylint: disable=broad-except
        print(f"  clear_err 异常: {exc}", flush=True)


def lock_orientation(poses: List[Pose], rpy: Sequence[float]) -> List[Pose]:
    """途经点只保留 XYZ，姿态统一成给定 RPY。"""
    rx, ry, rz = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return [[float(p[0]), float(p[1]), float(p[2]), rx, ry, rz] for p in poses]


def densify_xyz(poses: List[Pose], step_mm: float) -> List[Pose]:
    """在相邻途经点 XYZ 之间按步长插点，RPY 原样保留。"""
    if step_mm <= 0 or len(poses) < 2:
        return poses
    step = step_mm / 1000.0
    out: List[Pose] = [poses[0]]
    for nxt in poses[1:]:
        prev = out[-1]
        dx = nxt[0] - prev[0]
        dy = nxt[1] - prev[1]
        dz = nxt[2] - prev[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-9:
            out.append(nxt)
            continue
        n = max(1, int(math.ceil(dist / step)))
        for k in range(1, n + 1):
            t = k / n
            out.append(
                [
                    prev[0] + dx * t,
                    prev[1] + dy * t,
                    prev[2] + dz * t,
                    nxt[3],
                    nxt[4],
                    nxt[5],
                ]
            )
    return out


def move_to_pose_linear(arm, pose: Pose, speed: float, *, linear: bool) -> bool:
    """优先 rm_movel（途中姿态恒定），失败清错后回退 movej_p。"""
    robot = getattr(arm, "_robot", None)
    if linear and robot is not None:
        try:
            ret = robot.rm_movel(
                pose=pose, v=int(round(speed)), r=0, connect=0, block=1
            )
            if ret == 0:
                return True
            print(f"  rm_movel 失败 ret={ret}，清错后改 movej_p ...", flush=True)
            clear_err(arm)
            time.sleep(0.2)
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  rm_movel 异常: {exc}，清错后改 movej_p ...", flush=True)
            clear_err(arm)
            time.sleep(0.2)

    if arm.move_to_pose(pose, speed=speed):
        return True
    clear_err(arm)
    time.sleep(0.3)
    return arm.move_to_pose(pose, speed=speed)


def _ori_mode(args: argparse.Namespace) -> str:
    if args.no_lock_ori:
        return "free"
    if args.lock_current_ori:
        return "current"
    if args.cam_level:
        return "cam_level"
    return "ee_down"


def play_waypoints(
    arm,
    poses: List[Pose],
    *,
    speed: float,
    dwell: float,
    loop: int,
    ori_mode: str = "ee_down",
    handeye: Optional[Path] = None,
    cam_yaw_deg: Optional[float] = None,
    ee_yaw_deg: Optional[float] = None,
    linear: bool = True,
    interp_mm: float = 0.0,
    lift_cm: float = 8.0,
    raw_path: bool = False,
) -> int:
    stopped = {"v": False}

    def _on_sigint(sig, frame):  # noqa: ANN001
        print("\n>>> SIGINT：急停 ...", flush=True)
        try:
            arm.emergency_stop()
        except Exception as exc:  # pylint: disable=broad-except
            print(f"急停失败: {exc}", flush=True)
        stopped["v"] = True

    signal.signal(signal.SIGINT, _on_sigint)
    clear_err(arm)

    ok, cur = arm.get_current_pose()
    if not ok:
        raise SystemExit(f"读当前姿态失败: {cur}")
    cur = [float(v) for v in cur]

    fixed_rpy: Optional[List[float]] = None
    if ori_mode == "ee_down":
        yaw = None if ee_yaw_deg is None else math.radians(float(ee_yaw_deg))
        fixed_rpy = compute_ee_down_rpy(cur, ee_yaw_rad=yaw)
        poses = lock_orientation(poses, fixed_rpy)
        print(
            "末端姿态锁定：垂直向下（工具 Z ∥ 基座 -Z）  "
            f"rpy=[{fixed_rpy[0]:.4f},{fixed_rpy[1]:.4f},{fixed_rpy[2]:.4f}]  "
            "只移动 XYZ",
            flush=True,
        )
        if ee_yaw_deg is not None:
            print(f"  ee_yaw={ee_yaw_deg:.2f}°（手动）", flush=True)
        else:
            print("  ee_yaw=继承当前姿态绕竖直轴朝向", flush=True)
    elif ori_mode == "cam_level":
        he = Path(handeye) if handeye else DEFAULT_HANDEYE
        if not he.is_file():
            raise SystemExit(f"找不到手眼文件: {he}")
        T_tc = load_T_tool_cam(he)
        yaw = None if cam_yaw_deg is None else math.radians(float(cam_yaw_deg))
        fixed_rpy = compute_cam_level_rpy(cur, T_tc, cam_yaw_rad=yaw, look_sign=-1.0)
        poses = lock_orientation(poses, fixed_rpy)
        print(
            "末端姿态锁定：相机与桌面平行（光轴∥基座 -Z）  "
            f"rpy=[{fixed_rpy[0]:.4f},{fixed_rpy[1]:.4f},{fixed_rpy[2]:.4f}]  "
            "只移动 XYZ",
            flush=True,
        )
        if cam_yaw_deg is not None:
            print(f"  cam_yaw={cam_yaw_deg:.2f}°（手动）", flush=True)
        else:
            print("  cam_yaw=继承当前姿态绕竖直轴朝向", flush=True)
    elif ori_mode == "current":
        fixed_rpy = [float(v) for v in cur[3:6]]
        poses = lock_orientation(poses, fixed_rpy)
        print(
            "末端姿态锁定（当前 RPY）  "
            f"rpy=[{fixed_rpy[0]:.4f},{fixed_rpy[1]:.4f},{fixed_rpy[2]:.4f}]  "
            "只移动 XYZ",
            flush=True,
        )
    else:
        print("末端姿态未锁定：各途经点用各自录制的 RPY", flush=True)

    if not raw_path:
        poses = build_two_point_scan_path(poses, lift_m=float(lift_cm) / 100.0)
    else:
        print("按录制顺序原样走（未展开扫描路径）", flush=True)

    if interp_mm > 0 and fixed_rpy is not None:
        before = len(poses)
        poses = densify_xyz(poses, interp_mm)
        print(f"XYZ 插密 {interp_mm:.1f} mm：{before} → {len(poses)} 点", flush=True)

    mode = "rm_movel(笛卡尔直线, 途中保姿态)" if linear else "rm_movej_p(关节插值, 途中可能漂)"
    print(f"运动模式：{mode}", flush=True)

    # 出发前：当前位置先摆正姿态，避免第一段就带着倾斜走
    if fixed_rpy is not None:
        level_here = [cur[0], cur[1], cur[2], *fixed_rpy]
        dr = math.sqrt(sum((cur[i + 3] - fixed_rpy[i]) ** 2 for i in range(3)))
        if dr > 1e-3:
            print(f"先摆正当前姿态 → {_fmt(level_here)}", flush=True)
            if not move_to_pose_linear(arm, level_here, speed, linear=linear):
                print("摆正姿态失败，中止。", flush=True)
                arm.emergency_stop()
                return 3
            print("  ✓ 已摆正到垂直向下", flush=True)

    t0 = time.time()
    for lap in range(loop):
        print(f"--- 第 {lap + 1}/{loop} 圈 ---")
        for i, pose in enumerate(poses):
            if stopped["v"]:
                return 2
            print(f"→ p{i + 1}/{len(poses)} {_fmt(pose)}", flush=True)
            if not move_to_pose_linear(arm, pose, speed, linear=linear):
                ok2, cur2 = arm.get_current_pose()
                if ok2:
                    print(f"  当前位姿 {_fmt([float(v) for v in cur2])}")
                print("运动失败。请示教器确认该点可达，或减小跨度后重录。")
                arm.emergency_stop()
                return 3
            print(f"  ✓ p{i + 1} 到位", flush=True)
            time.sleep(max(0.0, dwell))
    print(f"完成，用时 {time.time() - t0:.1f}s")
    return 0


def _confirm_label(ori_mode: str, linear: bool) -> str:
    bits = []
    if ori_mode == "ee_down":
        bits.append("末端垂直向下")
    elif ori_mode == "cam_level":
        bits.append("相机∥桌面")
    elif ori_mode == "current":
        bits.append("姿态锁定=当前RPY")
    bits.append("movel" if linear else "movej_p")
    return f"（{' / '.join(bits)}）" if bits else ""


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if not (0 < args.speed <= 100):
        raise SystemExit("--speed must be in (0, 100]")

    # 无 flag：录+走；--record：只录；--record --execute：录完就走；--play：只回放
    if args.record and args.play:
        do_record, do_play = True, True
    elif args.record:
        do_record = True
        do_play = bool(args.execute)
    elif args.play:
        do_record, do_play = False, True
    else:
        do_record, do_play = True, True

    ori_mode = _ori_mode(args)
    wp_path = Path(args.waypoints)
    poses: List[Pose] = []

    play_kw = dict(
        speed=args.speed,
        dwell=args.dwell,
        loop=args.loop,
        ori_mode=ori_mode,
        handeye=Path(args.handeye),
        cam_yaw_deg=args.cam_yaw_deg,
        ee_yaw_deg=args.ee_yaw_deg,
        linear=not args.no_linear,
        interp_mm=float(args.interp_mm),
        lift_cm=float(args.lift_cm),
        raw_path=bool(args.raw_path),
    )

    if do_record:
        if _add_calibration_path() is None:
            raise SystemExit("找不到 Calibration，请 export CALIBRATION_ROOT=...")
        from modules.robot.realman.legacy_arm_controller import (  # type: ignore
            ArmController,
        )

        arm = ArmController(args.arm_ip, args.arm_port)
        if not arm.connect():
            raise SystemExit(f"连接失败 {args.arm_ip}:{args.arm_port}")
        try:
            poses = record_waypoints(arm, args.count)
            if not poses:
                print("未记录任何点，不写文件。")
                return 1
            if wp_path.resolve() == DEFAULT_WP.resolve():
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                wp_path = ROOT / "waypoints" / f"waypoints_{stamp}.json"
            save_poses(wp_path, poses, arm_ip=args.arm_ip, arm_port=args.arm_port)
            print(f"\n已保存 {len(poses)} 点 → {wp_path}")
            print(f"同时更新 → {DEFAULT_WP}")
            for i, pose in enumerate(poses):
                print(f"  p{i + 1}: {_fmt(pose)}")
            if len(poses) >= 2:
                d = sum((poses[1][j] - poses[0][j]) ** 2 for j in range(3)) ** 0.5
                if d < 1e-4:
                    print(
                        "警告：点1 与 点2 几乎重合，扫描路径会退化。"
                        "请拖到不同位置再录。",
                        flush=True,
                    )

            if not do_play:
                print("已只录点。要运动请：python arm_waypoints.py --play --execute")
                return 0

            if not args.execute:
                print("\n[DRY-RUN] 未运动。要真正走途经点请加 --execute")
                ans = input("现在就要运动？输入 YES 继续，其它键跳过: ").strip()
                if ans != "YES":
                    print("已跳过运动。之后可：python arm_waypoints.py --play --execute")
                    return 0
            elif not args.yes:
                ans = input(
                    f"将用 @ {args.speed}% 走扫描路径×{args.loop} 圈"
                    f"{_confirm_label(ori_mode, not args.no_linear)}。"
                    "输入 YES 继续: "
                ).strip()
                if ans != "YES":
                    print("已取消。")
                    return 1

            return play_waypoints(arm, poses, **play_kw)
        finally:
            arm.disconnect()

    if not wp_path.is_file():
        raise SystemExit(f"找不到路点文件: {wp_path}\n请先: python arm_waypoints.py --record")
    poses = load_poses(wp_path)
    print(f"路点文件: {wp_path}  ({len(poses)} 点)")
    for i, pose in enumerate(poses):
        print(f"  p{i + 1}: {_fmt(pose)}")
        if i > 0:
            a, b = poses[i - 1], pose
            dist = sum((b[j] - a[j]) ** 2 for j in range(3)) ** 0.5
            print(f"       ← 距上点 {dist * 1000:.0f} mm")
    if ori_mode == "ee_down":
        print(
            "回放将强制末端垂直向下：只走 XYZ；"
            f"默认 {'rm_movel 保姿态' if not args.no_linear else 'movej_p'}"
        )
    elif ori_mode == "cam_level":
        print(
            "回放将强制相机与桌面平行：只走 XYZ；"
            f"默认 {'rm_movel 保姿态' if not args.no_linear else 'movej_p'}"
        )
    elif ori_mode == "current":
        print("回放将锁定末端姿态：只走 XYZ，RPY=开始运动时的当前姿态")
    if not args.raw_path:
        print(
            f"将展开两点扫描路径（抬高 {args.lift_cm:.1f} cm）："
            "1→2→抬高→回2→1→抬高→回1→中点→抬高"
        )

    if not args.execute:
        print("\n[DRY-RUN] 未运动。确认后加 --execute。")
        return 0

    if _add_calibration_path() is None:
        raise SystemExit("找不到 Calibration，请 export CALIBRATION_ROOT=...")
    from modules.robot.realman.legacy_arm_controller import ArmController  # type: ignore

    if not args.yes:
        ans = input(
            f"将用 @ {args.speed}% 走 {len(poses)} 点×{args.loop} 圈"
            f"{_confirm_label(ori_mode, not args.no_linear)}。"
            "输入 YES 继续: "
        ).strip()
        if ans != "YES":
            print("已取消。")
            return 1

    arm = ArmController(args.arm_ip, args.arm_port)
    if not arm.connect():
        raise SystemExit(f"连接失败 {args.arm_ip}:{args.arm_port}")
    try:
        return play_waypoints(arm, poses, **play_kw)
    finally:
        arm.disconnect()


if __name__ == "__main__":
    raise SystemExit(main())
