#!/usr/bin/env python3
"""腕部 D405 采集：逐帧视频 + RGB/深度/IR + EE 位姿。

默认相机 serial=260322274901（wrist_camera_01）。
默认同时读机械臂 TCP（RealMan 8080）记下 T_base_ee / T_base_cam。

默认：相机每一帧都采集（不抽样），并写 color 视频 color.mp4。
仍按帧落盘 jpg/深度/IR/位姿，便于后续标注与匹配。

目录结构：

  <out>/
    session.json
    color.mp4                     # 彩色视频（逐帧）
    color/color_0000.jpg
    depth_aligned/  depth_raw/  ir_left/  ir_right/  preview/
    meta/color_0000.json          # 含 ee_xyz_rpy / T_base_ee / T_base_cam
    poses.jsonl                   # 每行一帧位姿
    ee_poses.json                 # shape_match --ee-poses 用
    K.txt

按键：q/Esc 退出  空格=暂停/继续  s=立刻多采一帧
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SERIAL = "260322274901"
DEFAULT_ARM_IP = "192.168.1.18"
DEFAULT_ARM_PORT = 8080


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="腕部 D405 采集 RGB/深度/IR")
    p.add_argument("--serial", default=DEFAULT_SERIAL)
    p.add_argument(
        "--out",
        default="",
        help="输出目录。默认 TCT/captures/<serial>_<stamp>",
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument(
        "--emitter",
        type=int,
        default=1,
        choices=(0, 1),
        help="红外点阵：1=开(深度更稳) 0=关(IR 更干净，方便立体)",
    )
    p.add_argument(
        "--auto-exposure",
        dest="auto_exposure",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="彩色自动曝光（默认开）。PCB 反光建议：--no-auto-exposure --exposure 30 --gain 16",
    )
    p.add_argument(
        "--exposure",
        type=float,
        default=None,
        help="彩色手动曝光(µs)。设了会自动关掉 AE。D405 常见可用约 1~10000+",
    )
    p.add_argument(
        "--gain",
        type=float,
        default=None,
        help="彩色增益。反光场景优先降 exposure，gain 尽量低",
    )
    p.add_argument(
        "--auto-white-balance",
        dest="auto_white_balance",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="彩色自动白平衡（默认开）",
    )
    p.add_argument("--list", action="store_true", help="列出已连接 RealSense 后退出")
    p.add_argument(
        "--auto",
        action="store_true",
        help="无窗口，按间隔连拍 --count 张后退出",
    )
    p.add_argument("--count", type=int, default=1)
    p.add_argument(
        "--interval",
        type=float,
        default=0.0,
        help="采样间隔秒；0=每一帧都采（默认）。>0 则按间隔抽样",
    )
    p.add_argument(
        "--no-continuous",
        action="store_true",
        help="预览模式不自动连拍（仅空格/s 手动）",
    )
    p.add_argument(
        "--no-video",
        action="store_true",
        help="不写 color.mp4（默认写）",
    )
    p.add_argument(
        "--video-name",
        default="color.mp4",
        help="彩色视频文件名（默认 color.mp4）",
    )
    p.add_argument("--start-index", type=int, default=0)
    p.add_argument(
        "--no-ee-pose",
        action="store_true",
        help="不读机械臂（默认会采集 EE / T_base_cam）",
    )
    p.add_argument(
        "--allow-no-arm",
        action="store_true",
        help="连不上机械臂时继续采图，该帧位姿记 null",
    )
    p.add_argument("--arm-ip", default=DEFAULT_ARM_IP)
    p.add_argument("--arm-port", type=int, default=DEFAULT_ARM_PORT)
    p.add_argument("--work-frame", default="Base")
    p.add_argument(
        "--handeye",
        default=str(
            Path(__file__).resolve().parent
            / "shape_match"
            / "calib"
            / "wrist_d405_T_tool_cam.yaml"
        ),
        help="腕部 T_tool_cam YAML，用来算 T_base_cam = T_base_ee @ T_tool_cam",
    )
    return p.parse_args()


def list_devices() -> dict[str, str]:
    import pyrealsense2 as rs

    ctx = rs.context()
    found: dict[str, str] = {}
    for d in ctx.devices:
        serial = d.get_info(rs.camera_info.serial_number)
        name = d.get_info(rs.camera_info.name)
        fw = d.get_info(rs.camera_info.firmware_version)
        found[serial] = f"{name} fw={fw}"
        print(f"  {serial}  {name}  fw={fw}")
    if not found:
        print("未发现 RealSense。关掉 realsense-viewer / 其它占用后再试。")
    return found


def _try_set_option(sensor, option, value: float, label: str) -> bool:
    if not sensor.supports(option):
        print(f"  {label}: 不支持，跳过")
        return False
    try:
        rng = sensor.get_option_range(option)
        v = float(np.clip(value, rng.min, rng.max))
        if abs(v - value) > 1e-6:
            print(f"  {label}: {value} → 钳到 [{rng.min}, {rng.max}] = {v}")
        sensor.set_option(option, v)
        cur = float(sensor.get_option(option))
        print(f"  {label}: {cur}  (range {rng.min}..{rng.max} step {rng.step})")
        return True
    except RuntimeError as e:
        print(f"  {label}: 设置失败（忽略）：{e}")
        return False


def apply_color_exposure(device, args: argparse.Namespace) -> dict:
    """Configure RGB exposure/gain for specular PCB scenes."""
    import pyrealsense2 as rs

    sensors = list(device.query_sensors())
    color_sensor = None
    for s in sensors:
        try:
            name = s.get_info(rs.camera_info.name)
        except Exception:
            name = ""
        if "RGB" in name.upper() or "COLOR" in name.upper():
            color_sensor = s
            break
    if color_sensor is None:
        try:
            color_sensor = device.first_color_sensor()
        except Exception:
            color_sensor = None
    if color_sensor is None:
        print("未找到彩色传感器，跳过曝光设置")
        return {"ok": False}

    print(f"彩色传感器曝光设置（{color_sensor.get_info(rs.camera_info.name)}）：")
    want_manual = (
        (args.exposure is not None)
        or (args.gain is not None)
        or (not args.auto_exposure)
    )
    meta: dict = {
        "ok": True,
        "auto_exposure": bool(args.auto_exposure) and not want_manual,
        "exposure_us": args.exposure,
        "gain": args.gain,
        "auto_white_balance": bool(args.auto_white_balance),
    }

    if want_manual:
        _try_set_option(
            color_sensor, rs.option.enable_auto_exposure, 0.0, "enable_auto_exposure"
        )
        meta["auto_exposure"] = False
        if args.exposure is not None:
            _try_set_option(
                color_sensor, rs.option.exposure, float(args.exposure), "exposure_us"
            )
        if args.gain is not None:
            _try_set_option(color_sensor, rs.option.gain, float(args.gain), "gain")
    else:
        _try_set_option(
            color_sensor, rs.option.enable_auto_exposure, 1.0, "enable_auto_exposure"
        )

    if color_sensor.supports(rs.option.enable_auto_white_balance):
        _try_set_option(
            color_sensor,
            rs.option.enable_auto_white_balance,
            1.0 if args.auto_white_balance else 0.0,
            "enable_auto_white_balance",
        )

    try:
        if color_sensor.supports(rs.option.exposure):
            meta["exposure_us_readback"] = float(
                color_sensor.get_option(rs.option.exposure)
            )
        if color_sensor.supports(rs.option.gain):
            meta["gain_readback"] = float(color_sensor.get_option(rs.option.gain))
        if color_sensor.supports(rs.option.enable_auto_exposure):
            meta["auto_exposure_readback"] = bool(
                color_sensor.get_option(rs.option.enable_auto_exposure)
            )
    except RuntimeError:
        pass
    return meta


def intrinsics_dict(intr) -> dict:
    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.ppx),
        "cy": float(intr.ppy),
        "model": str(intr.model),
        "coeffs": [float(x) for x in intr.coeffs],
    }


def extrinsics_dict(ext) -> dict:
    return {
        "rotation": [float(x) for x in ext.rotation],
        "translation": [float(x) for x in ext.translation],
    }


def write_k_txt(path: Path, K: np.ndarray, baseline_m: float) -> None:
    flat = " ".join(f"{x:.8f}" for x in K.reshape(-1).tolist())
    path.write_text(f"{flat}\n{baseline_m:.8f}\n", encoding="utf-8")


def depth_colormap(depth_u16: np.ndarray, depth_scale: float) -> np.ndarray:
    depth_m = depth_u16.astype(np.float32) * depth_scale
    valid = depth_m > 0
    if valid.any():
        lo, hi = np.percentile(depth_m[valid], [2, 98])
        n = np.clip((depth_m - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        gray = ((1.0 - n) * 255).astype(np.uint8)
    else:
        gray = np.zeros(depth_u16.shape, dtype=np.uint8)
    color = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def pose_xyz_rpy_to_mat4(
    x: float, y: float, z: float, rx: float, ry: float, rz: float
) -> np.ndarray:
    """RPY = Rz @ Ry @ Rx（与 Calibration / yuantu 一致）。"""
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    R = np.array(
        [
            [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
            [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
            [-sy, cy * sx, cy * cx],
        ],
        dtype=np.float64,
    )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = (x, y, z)
    return T


def load_T_tool_cam(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        import yaml  # type: ignore

        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            if "T_tool_cam" in data:
                return np.asarray(data["T_tool_cam"], dtype=np.float64)
            block = data.get("camera2_eye_in_hand") or data.get("camera2")
            if isinstance(block, dict) and "matrix" in block:
                return np.asarray(block["matrix"], dtype=np.float64)
    except Exception:
        pass
    return None


class RealmanArmClient:
    """RealMan TCP-JSON：pose 单位 um / 0.001 rad → m / rad。"""

    def __init__(self, ip: str, port: int, work_frame: str = "Base") -> None:
        self.ip = ip
        self.port = int(port)
        self.work_frame = work_frame
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(3.0)
        self._sock.connect((ip, int(port)))
        self._send(
            json.dumps({"command": "set_change_work_frame", "frame_name": work_frame}),
            get_pose=False,
        )
        xyzrpy = self.get_ee_xyzrpy()
        print(
            f"RealMan EE  {ip}:{port} frame={work_frame}\n"
            f"  xyz(m)=[{xyzrpy[0]:.4f},{xyzrpy[1]:.4f},{xyzrpy[2]:.4f}]  "
            f"rpy(rad)=[{xyzrpy[3]:.4f},{xyzrpy[4]:.4f},{xyzrpy[5]:.4f}]"
        )

    def _send(self, cmd: str, get_pose: bool = True) -> dict | None:
        self._sock.send(cmd.encode("utf-8"))
        if not get_pose:
            try:
                self._sock.recv(1024)
            except socket.timeout:
                pass
            return None
        time.sleep(0.05)
        response = self._sock.recv(4096).decode("utf-8", errors="ignore")
        decoder = json.JSONDecoder()
        data_list = []
        index = 0
        while index < len(response):
            while index < len(response) and response[index].isspace():
                index += 1
            if index >= len(response):
                break
            try:
                obj, idx = decoder.raw_decode(response[index:])
            except json.JSONDecodeError:
                break
            data_list.append(obj)
            index += idx
        target = None
        for data in reversed(data_list):
            if data.get("state") == "current_arm_state":
                target = data
                break
        if not target:
            raise RuntimeError(f"无 current_arm_state: {response[:200]}")
        err = target.get("arm_state", {}).get("err", [0])
        if err not in ([0], 0, [0.0]):
            raise RuntimeError(f"Arm error: {err}")
        return target

    def get_ee_xyzrpy(self) -> list[float]:
        target = self._send('{"command":"get_current_arm_state"}')
        assert target is not None
        pose_raw = target["arm_state"]["pose"]
        return [
            float(pose_raw[0]) / 1_000_000.0,
            float(pose_raw[1]) / 1_000_000.0,
            float(pose_raw[2]) / 1_000_000.0,
            float(pose_raw[3]) / 1000.0,
            float(pose_raw[4]) / 1000.0,
            float(pose_raw[5]) / 1000.0,
        ]

    def close(self) -> None:
        try:
            self._sock.close()
        except Exception:
            pass


def next_index(color_dir: Path, start: int) -> int:
    n = start
    while (color_dir / f"color_{n:04d}.jpg").exists():
        n += 1
    return n


def save_frame(
    out: Path,
    idx: int,
    *,
    color_bgr: np.ndarray,
    depth_aligned_u16: np.ndarray,
    depth_raw_u16: np.ndarray,
    ir_l: np.ndarray,
    ir_r: np.ndarray,
    depth_scale: float,
    jpeg_quality: int,
    extra_meta: dict,
) -> Path:
    stem = f"color_{idx:04d}"
    jpeg = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    png16 = [cv2.IMWRITE_PNG_COMPRESSION, 3]

    color_path = out / "color" / f"{stem}.jpg"
    cv2.imwrite(str(color_path), color_bgr, jpeg)
    cv2.imwrite(str(out / "depth_aligned" / f"{stem}.png"), depth_aligned_u16, png16)
    cv2.imwrite(str(out / "depth_raw" / f"{stem}.png"), depth_raw_u16, png16)
    cv2.imwrite(str(out / "ir_left" / f"{stem}.png"), ir_l)
    cv2.imwrite(str(out / "ir_right" / f"{stem}.png"), ir_r)

    depth_m = depth_aligned_u16.astype(np.float32) * float(depth_scale)
    np.save(out / "depth_aligned" / f"{stem}.npy", depth_m)
    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    cv2.imwrite(str(out / "depth_aligned" / f"{stem}_mm.png"), depth_mm)

    cmap = depth_colormap(depth_aligned_u16, depth_scale)
    overlay = cv2.addWeighted(color_bgr, 0.55, cmap, 0.45, 0)
    cv2.imwrite(str(out / "preview" / f"{stem}_depth.jpg"), cmap, jpeg)
    cv2.imwrite(str(out / "preview" / f"{stem}_overlay.jpg"), overlay, jpeg)

    valid = depth_m > 0
    meta = {
        "image": f"color/{stem}.jpg",
        "index": idx,
        "saved_at": datetime.now().astimezone().isoformat(),
        "color_shape": list(color_bgr.shape),
        "depth_aligned_png": f"depth_aligned/{stem}.png",
        "depth_aligned_npy": f"depth_aligned/{stem}.npy",
        "depth_raw_png": f"depth_raw/{stem}.png",
        "ir_left": f"ir_left/{stem}.png",
        "ir_right": f"ir_right/{stem}.png",
        "depth_valid_ratio": float(valid.mean()) if valid.size else 0.0,
        "depth_min_m": float(depth_m[valid].min()) if valid.any() else None,
        "depth_max_m": float(depth_m[valid].max()) if valid.any() else None,
        **extra_meta,
    }
    (out / "meta" / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    ee = extra_meta.get("ee_xyz_rpy")
    ee_s = ""
    if isinstance(ee, (list, tuple)) and len(ee) >= 6:
        ee_s = (
            f"  ee=[{ee[0]:.3f},{ee[1]:.3f},{ee[2]:.3f} "
            f"{ee[3]:.3f},{ee[4]:.3f},{ee[5]:.3f}]"
        )
    print(
        f"saved {stem}  depth_valid={meta['depth_valid_ratio']*100:.1f}%  "
        f"z=[{meta['depth_min_m']},{meta['depth_max_m']}]{ee_s}"
    )
    return color_path


def main() -> int:
    args = parse_args()
    import pyrealsense2 as rs

    print("RealSense 设备：")
    found = list_devices()
    if args.list:
        return 0 if found else 1

    serial = args.serial.strip()
    if serial not in found:
        raise SystemExit(f"找不到腕部相机 {serial}。当前：{list(found) or '无'}")

    root = Path(__file__).resolve().parent
    if args.out:
        out = Path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = root / "captures" / f"{serial}_{stamp}"
    for sub in (
        "color",
        "depth_aligned",
        "depth_raw",
        "ir_left",
        "ir_right",
        "preview",
        "meta",
    ):
        (out / sub).mkdir(parents=True, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.infrared, 1, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.infrared, 2, args.width, args.height, rs.format.y8, args.fps)
    config.enable_stream(rs.stream.depth, args.width, args.height, rs.format.z16, args.fps)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)

    profile = pipeline.start(config)
    arm: RealmanArmClient | None = None
    poses_jsonl = None
    video_writer: cv2.VideoWriter | None = None
    video_path = out / str(args.video_name)
    try:
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()
        if depth_sensor.supports(rs.option.emitter_enabled):
            try:
                depth_sensor.set_option(rs.option.emitter_enabled, float(args.emitter))
            except RuntimeError as e:
                print(f"emitter 设置失败（忽略）：{e}")
        depth_scale = float(depth_sensor.get_depth_scale())
        color_exposure_meta = apply_color_exposure(device, args)

        left_prof = profile.get_stream(rs.stream.infrared, 1).as_video_stream_profile()
        right_prof = profile.get_stream(rs.stream.infrared, 2).as_video_stream_profile()
        color_prof = profile.get_stream(rs.stream.color).as_video_stream_profile()
        depth_prof = profile.get_stream(rs.stream.depth).as_video_stream_profile()
        left_intr = left_prof.get_intrinsics()
        color_intr = color_prof.get_intrinsics()
        depth_intr = depth_prof.get_intrinsics()
        ext_rl = right_prof.get_extrinsics_to(left_prof)
        ext_dc = depth_prof.get_extrinsics_to(color_prof)
        baseline_m = abs(float(ext_rl.translation[0]))
        if baseline_m < 1e-4:
            baseline_m = float(np.linalg.norm(ext_rl.translation))

        K_ir = np.array(
            [
                [left_intr.fx, 0.0, left_intr.ppx],
                [0.0, left_intr.fy, left_intr.ppy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        write_k_txt(out / "K.txt", K_ir, baseline_m)

        session = {
            "camera": "wrist",
            "serial": serial,
            "name": found[serial],
            "out": str(out),
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
            "emitter": args.emitter,
            "color_exposure": color_exposure_meta,
            "depth_scale_m_per_unit": depth_scale,
            "baseline_m": baseline_m,
            "K_left_ir": K_ir.tolist(),
            "intrinsics_color": intrinsics_dict(color_intr),
            "intrinsics_left_ir": intrinsics_dict(left_intr),
            "intrinsics_depth": intrinsics_dict(depth_intr),
            "extrinsics_right_to_left": extrinsics_dict(ext_rl),
            "extrinsics_depth_to_color": extrinsics_dict(ext_dc),
            "started_at": datetime.now().astimezone().isoformat(),
        }
        T_tool_cam = None
        if not args.no_ee_pose:
            try:
                arm = RealmanArmClient(args.arm_ip, args.arm_port, args.work_frame)
            except Exception as e:
                msg = f"机械臂连不上 {args.arm_ip}:{args.arm_port}：{e}"
                if args.allow_no_arm:
                    print(f"WARNING {msg}  （继续采图，位姿为 null）")
                else:
                    pipeline.stop()
                    raise SystemExit(
                        msg + "\n加 --allow-no-arm 可跳过位姿，或 --no-ee-pose 完全不采。"
                    )
            T_tool_cam = load_T_tool_cam(Path(args.handeye))
            if T_tool_cam is None:
                print(f"WARNING 未读到手眼 {args.handeye}，只存 T_base_ee")
        session["arm_ip"] = args.arm_ip
        session["arm_port"] = args.arm_port
        session["work_frame"] = args.work_frame
        session["ee_pose_enabled"] = not args.no_ee_pose
        session["handeye"] = args.handeye if T_tool_cam is not None else None
        if T_tool_cam is not None:
            session["T_tool_cam"] = T_tool_cam.tolist()
        (out / "session.json").write_text(
            json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        poses_jsonl = (out / "poses.jsonl").open("a", encoding="utf-8")
        ee_poses_map: dict[str, list[float]] = {}
        if (out / "ee_poses.json").is_file():
            try:
                ee_poses_map = json.loads(
                    (out / "ee_poses.json").read_text(encoding="utf-8")
                )
            except Exception:
                ee_poses_map = {}

        if not args.no_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                float(args.fps),
                (int(args.width), int(args.height)),
            )
            if not video_writer.isOpened():
                print(f"WARNING 打不开视频写入器，改用 avi：{video_path}")
                video_path = out / (Path(args.video_name).stem + ".avi")
                fourcc = cv2.VideoWriter_fourcc(*"XVID")
                video_writer = cv2.VideoWriter(
                    str(video_path),
                    fourcc,
                    float(args.fps),
                    (int(args.width), int(args.height)),
                )
            if not video_writer.isOpened():
                print("WARNING 视频写入失败，继续只存逐帧文件")
                video_writer = None
            else:
                session["video"] = str(video_path.name)
                session["video_fps"] = float(args.fps)
                (out / "session.json").write_text(
                    json.dumps(session, indent=2, ensure_ascii=False), encoding="utf-8"
                )

        align = rs.align(rs.stream.color)
        every = "每一帧" if float(args.interval) <= 0 else f"每 {args.interval:.2f}s"
        print(
            f"Using {found[serial]} serial={serial}\n"
            f"  {args.width}x{args.height}@{args.fps}  "
            f"depth_scale={depth_scale:.6g}  baseline={baseline_m*1000:.2f}mm  "
            f"emitter={args.emitter}\n"
            f"  pose={'ON' if arm else 'off'}  采样={every}  "
            f"video={'ON→'+video_path.name if video_writer else 'off'}  → {out}"
        )

        def grab():
            frames = pipeline.wait_for_frames(5000)
            ir_l = frames.get_infrared_frame(1)
            ir_r = frames.get_infrared_frame(2)
            depth_raw_f = frames.get_depth_frame()
            aligned = align.process(frames)
            depth_a = aligned.get_depth_frame()
            color_f = aligned.get_color_frame()
            if not ir_l or not ir_r or not depth_raw_f or not depth_a or not color_f:
                return None
            return {
                "ir_l": np.asanyarray(ir_l.get_data()).copy(),
                "ir_r": np.asanyarray(ir_r.get_data()).copy(),
                "depth_raw": np.asanyarray(depth_raw_f.get_data()).copy(),
                "depth_aligned": np.asanyarray(depth_a.get_data()).copy(),
                "color": np.asanyarray(color_f.get_data()).copy(),
                "color_ts_ms": float(color_f.get_timestamp()),
                "frame_number": int(color_f.get_frame_number()),
            }

        for i in range(args.warmup):
            grab()
            if (i + 1) % 10 == 0:
                print(f"warmup {i+1}/{args.warmup}")

        idx = next_index(out / "color", args.start_index)

        def persist(got: dict) -> None:
            nonlocal idx
            extra = {
                "serial": serial,
                "depth_scale_m_per_unit": depth_scale,
                "depth_aligned_to": "color",
                "frame_number": got["frame_number"],
                "timestamp_ms": got["color_ts_ms"],
                "ee_xyz_rpy": None,
                "ee_units": "xyz_m + rpy_rad",
                "T_base_ee": None,
                "T_base_cam": None,
            }
            if arm is not None:
                xyzrpy = arm.get_ee_xyzrpy()
                T_ee = pose_xyz_rpy_to_mat4(*xyzrpy)
                extra["ee_xyz_rpy"] = xyzrpy
                extra["T_base_ee"] = T_ee.tolist()
                if T_tool_cam is not None:
                    extra["T_base_cam"] = (T_ee @ T_tool_cam).tolist()
            stem = f"color_{idx:04d}"
            save_frame(
                out,
                idx,
                color_bgr=got["color"],
                depth_aligned_u16=got["depth_aligned"],
                depth_raw_u16=got["depth_raw"],
                ir_l=got["ir_l"],
                ir_r=got["ir_r"],
                depth_scale=depth_scale,
                jpeg_quality=args.jpeg_quality,
                extra_meta=extra,
            )
            if video_writer is not None:
                video_writer.write(got["color"])
            row = {
                "image": f"{stem}.jpg",
                "ee_xyz_rpy": extra["ee_xyz_rpy"],
                "T_base_ee": extra["T_base_ee"],
                "T_base_cam": extra["T_base_cam"],
                "timestamp_ms": extra["timestamp_ms"],
            }
            poses_jsonl.write(json.dumps(row, ensure_ascii=False) + "\n")
            poses_jsonl.flush()
            if extra["ee_xyz_rpy"] is not None:
                ee_poses_map[f"{stem}.jpg"] = extra["ee_xyz_rpy"]
                (out / "ee_poses.json").write_text(
                    json.dumps(ee_poses_map, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            idx += 1

        if args.auto:
            n = max(1, args.count)
            for k in range(n):
                got = grab()
                if got is None:
                    raise RuntimeError("抓帧失败")
                persist(got)
                if k + 1 < n and float(args.interval) > 0:
                    time.sleep(float(args.interval))
            print(f"done  {n} frames → {out}")
            print(f"RGB 目录（标注/匹配）：{out / 'color'}")
            if video_writer is not None:
                print(f"视频：{video_path}")
            print(f"位姿：{out / 'ee_poses.json'}  {out / 'poses.jsonl'}")
            return 0

        win = f"wrist {serial}"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        # 默认持续连拍；--no-continuous 则先暂停，只预览
        capturing = not args.no_continuous
        last_save = 0.0
        last_ee: list[float] | None = None
        hud_i = 0
        rate_hud = "every-frame" if float(args.interval) <= 0 else f"{args.interval:.2f}s"
        print(
            f"持续拍摄：{rate_hud}  "
            f"{'ON' if capturing else 'PAUSED'}  "
            f"video={'ON' if video_writer else 'off'}  "
            f"空格=暂停/继续  s=立刻采  q=退出"
        )
        while True:
            got = grab()
            if got is None:
                continue
            cmap = depth_colormap(got["depth_aligned"], depth_scale)
            ir_vis = cv2.cvtColor(
                np.hstack([got["ir_l"], got["ir_r"]]), cv2.COLOR_GRAY2BGR
            )
            ir_small = cv2.resize(ir_vis, (got["color"].shape[1], got["color"].shape[0] // 2))
            top = np.hstack([got["color"], cmap])
            # pad IR row to same width
            pad_w = top.shape[1] - ir_small.shape[1]
            if pad_w > 0:
                ir_small = cv2.copyMakeBorder(
                    ir_small, 0, 0, 0, pad_w, cv2.BORDER_CONSTANT, value=(0, 0, 0)
                )
            vis = np.vstack([top, ir_small])
            hud_i += 1
            if arm is not None and hud_i % 15 == 1:
                try:
                    last_ee = arm.get_ee_xyzrpy()
                except Exception:
                    last_ee = None
            pose_hud = "pose=off"
            if arm is not None:
                if last_ee is not None:
                    p = last_ee
                    pose_hud = (
                        f"ee [{p[0]:.3f} {p[1]:.3f} {p[2]:.3f}] "
                        f"rpy [{p[3]:.2f} {p[4]:.2f} {p[5]:.2f}]"
                    )
                else:
                    pose_hud = "pose=ERR"
            status = "CAPTURING" if capturing else "PAUSED"
            hud = (
                f"idx={idx}  {pose_hud}  {status}@{rate_hud}  "
                f"SPC=pause  s=now  q=quit"
            )
            cv2.putText(
                vis, hud, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2
            )
            cv2.imshow(win, vis)
            key = cv2.waitKey(1) & 0xFF
            now = time.time()
            if key == ord(" "):
                capturing = not capturing
                print(f"capture={'ON' if capturing else 'PAUSED'}")
            do_save = key in (ord("s"), ord("S"))
            if capturing:
                if float(args.interval) <= 0:
                    do_save = True
                elif (now - last_save) >= float(args.interval):
                    do_save = True
            if do_save:
                persist(got)
                last_save = now
            if key in (ord("q"), ord("Q"), 27):
                break
        cv2.destroyAllWindows()
        print(f"done → {out}")
        print(f"RGB 目录（标注/匹配）：{out / 'color'}")
        if video_writer is not None:
            print(f"视频：{video_path}")
        print(f"位姿：{out / 'ee_poses.json'}  {out / 'poses.jsonl'}")
        return 0
    finally:
        try:
            if video_writer is not None:
                video_writer.release()
        except Exception:
            pass
        try:
            poses_jsonl.close()
        except Exception:
            pass
        try:
            if arm is not None:
                arm.close()
        except Exception:
            pass
        pipeline.stop()


if __name__ == "__main__":
    raise SystemExit(main())
