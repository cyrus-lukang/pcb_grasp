"""Load grasp YAML / JSON config."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "grasp_config.yaml"


@dataclass
class GraspConfig:
    # models / calib paths
    head_pose_weights: str = ""
    wrist_pose_weights: str = ""
    # 精定位：外轮廓模板（TCT shape match）
    shape_model_path: str = ""
    fine_use_shape_match: bool = True
    shape_angle_offset_deg: float = 0.0
    shape_angle_range_deg: float = 30.0
    shape_edge_mode: str = "outer-ring"
    shape_score_thresh: float = 5.2
    stereo_calib_json: str = ""
    stereo_maps_yml: str = ""
    # 头定位用内参 yaml（camera_matrix）；有则覆盖 stereo K_rect，与手眼一致
    head_intrinsics_yaml: str = ""
    head_fx: float | None = None
    head_fy: float | None = None
    head_cx: float | None = None
    head_cy: float | None = None
    T_base_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_tool_cam: np.ndarray = field(default_factory=lambda: np.eye(4))
    T_tool_cam_right: np.ndarray | None = None  # 右腕手眼；大板精定位看 P2
    wrist_fx: float = 615.0
    wrist_fy: float = 615.0
    wrist_cx: float = 630.0
    wrist_cy: float = 401.0
    wrist_dist_coeffs: list[float] | None = None  # OpenCV k1,k2,p1,p2,k3,k4,k5,k6
    wrist_depth_scale: float = 0.001  # Gemini305 uint16 mm → m；D405 常用 0.0001
    # True：精定位 camZ 固定用 hover_height（粗定位悬停高），不采腕深图
    wrist_use_fixed_hover_depth: bool = True

    # detection
    head_conf: float = 0.4
    wrist_conf: float = 0.5
    kpt_conf: float = 0.35
    imgsz: int = 640
    head_imgsz: int = 480
    wrist_imgsz: int = 960
    device: int | str = 0
    # 头模类别过滤；2cls 时 [0]=只 PCB。None/[]=不过滤
    head_class_ids: list[int] | None = None
    kp_i: int = 0
    kp_j: int = 1
    detect_n_frames: int = 10
    detect_outlier_mad_k: float = 3.0
    detect_frame_gap_s: float = 0.03

    # depth fallback for head (camera Z meters) when no depth map / no stereo
    head_fixed_depth_cam_m: float | None = 0.85
    # 托盘 PCB 上表面在 base 的 Z（m）；有则头目粗定位用射线-平面求交（抗倾斜+飞点）
    head_tray_z_m: float | None = -0.48
    ffs_weights: str = ""
    ffs_cfg: str = ""

    # motion / grasp
    hover_height_m: float = 0.45  # 粗定位悬停：相机光心在 PCB 正上方 (m)
    hover_camera_down: bool = True  # True=悬停相机朝下；精定位后再切 (-90°,0) 抓取
    # rimbot MoveViaSafeHeight 走廊高度（m）；ICT json safe_z=20mm
    safe_z_m: float = 0.02
    tcp_offset: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.28])
    tool_rz_offset_rad: float = 1.57079632679
    move_velocity_scale: float = 0.35  # 略快；实际 ≈ rim.max_linear_vel * scale
    cart_vel_mm_s: float = 40.0
    cart_acc_mm_s2: float = 80.0
    tj_vel_ratio: int = 18
    tj_acc_ratio: int = 18
    fine_xy_only: bool = True  # fine: update XY (+ rz); keep current Z
    fine_apply_rz: bool = True
    # 已并入 hover_height；默认不再横移/再抬
    fine_clear_dy_m: float = 0.0
    fine_clear_dz_m: float = 0.0
    # 精定位到位：先 XY，再降 Z
    fine_descend_dz_m: float = 0.10  # 无 grasp_z 时的相对下降（旧逻辑）
    # 示教夹取高度（法兰 Z，m）；有则精定位 XY 后降到此高度再收夹
    grasp_z_m: float | None = None
    grasp_z_file: str = ""
    # 相对示教高度上抬（m）；正值=更高，避免压板
    grasp_z_lift_m: float = 0.005
    target_board_id: int = 1  # 1=最近板
    board_match_max_dist_m: float = 0.10
    dry_run: bool = True

    # 运控后端：rim = 10.3.8.7 PLAN_TO_POSE；tj = 旧 TJ_demo / 192.168.1.190
    motion_backend: str = "rim"
    # RIM
    rim_frame_bundle: str = "10.3.8.7:50066"
    rim_mc: str = ""
    rim_connect_timeout_s: float = 10.0
    rim_plan_timeout_s: float = 120.0
    rim_max_linear_vel: float = 0.06
    rim_max_angular_vel: float = 0.25
    # TJ_demo real robot（motion_backend=tj 时用）
    tj_robot_ip: str = "192.168.1.190"
    tj_arm: str = "A"
    tj_use_impedance: bool = True
    tj_contact_threshold_n: float = 10.0

    # 夹爪开度：rim_python 约定 0=闭 1=开；收夹用 grasp（默认 0.4）
    gripper_grasp_position: float = 0.4
    gripper_release_position: float = 1.0
    gripper_speed: float = 50.0
    gripper_torque_grasp: float = 30.0
    gripper_torque_release: float = 60.0
    gripper_com_ch: int = 2

    # live cameras
    # head_source: portal = rim watch portal（与 capture_images 同源）；v4l2 = 本机 /dev/video*
    cam_head_source: str = "portal"
    cam_portal: str = "10.3.8.7:8787"
    cam_portal_head_left_id: int = 7
    cam_portal_head_right_id: int = 6
    cam_portal_max_hz: float = 8.0
    cam_portal_timeout_s: float = 60.0
    cam_head_left: str = "/dev/video7"
    cam_head_right: str = "/dev/video6"
    cam_head_width: int = 1920
    cam_head_height: int = 1536
    cam_s36_sdk: str = "/home/ubuntu/workspace/ccg3_8m_sdk/ccg3_camera_python"
    # wrist_source: portal=Gemini305 门户；d405=本机 RealSense
    cam_wrist_source: str = "portal"
    cam_portal_wrist_id: int = 1  # 左臂彩色
    cam_portal_wrist_depth_id: int = 0  # 左臂 Z16
    cam_portal_wrist_right_id: int = 3  # 右臂彩色
    cam_portal_wrist_right_depth_id: int = 2  # 右臂 Z16
    cam_wrist_need_depth: bool = True
    cam_wrist_undistort: bool = True
    cam_wrist_serial: str = "262422274334"  # 仅 d405
    cam_wrist_width: int = 1280
    cam_wrist_height: int = 800
    cam_wrist_fps: int = 30
    live_settle_s: float = 1.0
    still_pos_tol_m: float = 0.0015
    still_rot_tol_rad: float = 0.015
    still_hold_s: float = 0.15
    still_timeout_s: float = 3.0

    # optional home / ready joints (deg)
    home_pose: list[float] | None = None
    ready_joints_A: list[float] | None = None
    ready_joints_B: list[float] | None = None

    # class ids（同模训大板后：0 小板 / 1 nopcb / 2 大板）
    cls_pcb_small: int = 0
    cls_nopcb: int = 1
    cls_pcb_large: int = 2

    # 大板双臂对夹
    large_board_enable: bool = True
    large_tool_y_m: float = 0.01
    large_tool_z_m: float = 0.27
    large_open_margin_m: float = 0.05
    large_squeeze_close_m: float = 0.0
    large_squeeze_velocity_scale: float = 0.01
    large_squeeze_force_n: float = 5.0
    large_use_impedance: bool = True
    # 标定后填：固定 tip/板面 Z、两 tip 夹取间距（基座米）；null=暂用托盘/视觉
    large_grasp_z_m: float | None = None
    large_tip_span_m: float | None = None
    large_hover_z_m: float | None = None
    large_approach_velocity_scale: float = 0.08
    large_half_width_m: float | None = None
    # 大板头定位：False=头 FFS 角点 Z（默认）；True=射线∩固定板面
    large_use_fixed_grasp_z_plane: bool = False
    # True：夹取 tipZ = 左臂当前法兰 Z − below（大板已弃用，改用 FFS）
    large_use_left_current_z: bool = False
    large_grasp_z_below_left_m: float = 0.0
    # 大板末端 rz：parallel_p2p3=夹爪中间线∥P2→P3（默认；23⊥03）
    large_orient_edge: str = "parallel_p2p3"
    large_orient_rz_offset_deg: float = 0.0
    # 接近：mid(P2,P3) 后 x+=dx；左 y+=dy、右 y-=dy（基座系右臂 Y 为负侧）
    large_approach_dx_m: float = 0.15
    large_approach_dy_m: float = 0.08
    # 大板腕部精定位：左看 P3、右看 P2（上方 hover）→ 合成精 2/3
    large_fine_enable: bool = True
    large_wrist_pose_weights: str = ""
    # V2·220e76 单角模型：训练 imgsz=480、每框 1 关键点（勿用小板 wrist_imgsz=960 / n_corners=4）
    large_wrist_imgsz: int = 480
    large_wrist_n_corners: int = 1
    large_coarse_hover_height_m: float = 0.35
    large_fine_hover_height_m: float = 0.35
    # 接近停靠相对板面抬高；None → 复用 large_fine_hover_height_m
    large_approach_hover_height_m: float | None = None
    large_fine_match_max_dist_m: float = 0.15
    # 大板单臂支路 B：无头相机；腕部粗定位 → 左臂精看 P2/P3 → 停 mid23
    large_single_enable: bool = True
    # 粗定位固定法兰 [x,y,z,rx,ry,rz]（米/弧度）；null=后面写死，未配则用当前 TCP
    large_single_coarse_flange: list[float] | None = None
    # 精定位后停在 mid(P2,P3) 的 tip/板面上方相机高度（米）；null=后面写死
    large_single_stop_z_m: float | None = None
    # 单臂精定位悬停高度（相对板面）；null 复用 large_fine_hover_height_m
    large_single_fine_hover_height_m: float | None = None


def _load_raw(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ImportError as e:
            raise RuntimeError("PyYAML required for .yaml config") from e
        data = yaml.safe_load(text) or {}
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config root must be mapping: {path}")
    return data


def _mat4(val: Any) -> np.ndarray:
    M = np.asarray(val, dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"Expected 4x4 matrix, got {M.shape}")
    return M


def _resolve_path(base: Path, value: str) -> str:
    """相对路径相对 pcb_grasp 根目录；绝对路径原样。"""
    if not value:
        return ""
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = (base / p).resolve()
    return str(p)


def load_config(path: str | Path | None = None) -> GraspConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    raw = _load_raw(path) if path.is_file() else {}
    root = ROOT

    def g(*keys, default=None):
        cur: Any = raw
        for k in keys:
            if not isinstance(cur, dict) or k not in cur:
                return default
            cur = cur[k]
        return cur

    cfg = GraspConfig()
    cfg.head_pose_weights = _resolve_path(
        root, str(g("models", "head_pose_weights", default=cfg.head_pose_weights) or "")
    )
    cfg.wrist_pose_weights = _resolve_path(
        root,
        str(g("models", "wrist_pose_weights", default=cfg.wrist_pose_weights) or cfg.head_pose_weights),
    )
    cfg.shape_model_path = _resolve_path(
        root,
        str(
            g("models", "shape_model", default="")
            or g("models", "shape_model_path", default="")
            or "TCT/shape_match/models/d435_pcb.npz"
        ),
    )
    sm = g("shape_match", default={}) or {}
    cfg.fine_use_shape_match = bool(
        sm.get("enable", g("detect", "fine_use_shape_match", default=cfg.fine_use_shape_match))
    )
    cfg.shape_angle_offset_deg = float(sm.get("angle_offset_deg", cfg.shape_angle_offset_deg))
    cfg.shape_angle_range_deg = float(sm.get("angle_range_deg", cfg.shape_angle_range_deg))
    cfg.shape_edge_mode = str(sm.get("edge_mode", cfg.shape_edge_mode))
    cfg.shape_score_thresh = float(sm.get("score_thresh", cfg.shape_score_thresh))
    cfg.stereo_calib_json = _resolve_path(
        root, str(g("calib", "stereo_calib_json", default="") or "")
    )
    cfg.stereo_maps_yml = _resolve_path(
        root, str(g("calib", "stereo_maps_yml", default="") or "")
    )
    hi_path = str(g("calib", "head_intrinsics_yaml", default="") or "").strip()
    cfg.head_intrinsics_yaml = _resolve_path(root, hi_path) if hi_path else ""
    if cfg.head_intrinsics_yaml and Path(cfg.head_intrinsics_yaml).is_file():
        try:
            import yaml

            hi = yaml.safe_load(Path(cfg.head_intrinsics_yaml).read_text(encoding="utf-8")) or {}
            Km = hi.get("camera_matrix") or hi.get("K")
            if Km is not None:
                K = np.asarray(Km, dtype=np.float64).reshape(3, 3)
                cfg.head_fx = float(K[0, 0])
                cfg.head_fy = float(K[1, 1])
                cfg.head_cx = float(K[0, 2])
                cfg.head_cy = float(K[1, 2])
        except Exception:
            pass

    Tbc = g("calib", "T_base_cam")
    if Tbc is not None:
        cfg.T_base_cam = _mat4(Tbc)
    Ttc = g("calib", "T_tool_cam")
    if Ttc is not None:
        cfg.T_tool_cam = _mat4(Ttc)
    Ttc_r = g("calib", "T_tool_cam_right")
    if Ttc_r is not None and Ttc_r not in ("null", ""):
        cfg.T_tool_cam_right = _mat4(Ttc_r)

    wi = g("calib", "wrist_intrinsics", default={}) or {}
    cfg.wrist_fx = float(wi.get("fx", cfg.wrist_fx))
    cfg.wrist_fy = float(wi.get("fy", cfg.wrist_fy))
    cfg.wrist_cx = float(wi.get("cx", cfg.wrist_cx))
    cfg.wrist_cy = float(wi.get("cy", cfg.wrist_cy))
    dist = wi.get("distortion_coefficients") or wi.get("dist_coeffs")
    if dist is not None:
        cfg.wrist_dist_coeffs = [float(x) for x in dist]
    else:
        # 兼容拆开写的 k1..k6 / p1 p2
        keys = ("k1", "k2", "p1", "p2", "k3", "k4", "k5", "k6")
        if any(k in wi for k in keys):
            cfg.wrist_dist_coeffs = [float(wi.get(k, 0.0)) for k in keys]
    cfg.wrist_depth_scale = float(
        g("calib", "wrist_depth_scale", default=cfg.wrist_depth_scale)
    )
    cfg.wrist_use_fixed_hover_depth = bool(
        g(
            "calib",
            "wrist_use_fixed_hover_depth",
            default=g(
                "motion",
                "wrist_use_fixed_hover_depth",
                default=cfg.wrist_use_fixed_hover_depth,
            ),
        )
    )

    det = g("detect", default={}) or {}
    cfg.head_conf = float(det.get("head_conf", cfg.head_conf))
    cfg.wrist_conf = float(det.get("wrist_conf", cfg.wrist_conf))
    cfg.kpt_conf = float(det.get("kpt_conf", cfg.kpt_conf))
    cfg.imgsz = int(det.get("imgsz", cfg.imgsz))
    cfg.head_imgsz = int(det.get("head_imgsz", det.get("imgsz", cfg.head_imgsz)))
    cfg.wrist_imgsz = int(det.get("wrist_imgsz", det.get("imgsz", cfg.wrist_imgsz)))
    cfg.device = det.get("device", cfg.device)
    raw_cls = det.get("head_class_ids", cfg.head_class_ids)
    if raw_cls is None or raw_cls == [] or raw_cls == "null":
        cfg.head_class_ids = None
    else:
        cfg.head_class_ids = [int(x) for x in list(raw_cls)]
    cfg.kp_i = int(det.get("kp_i", cfg.kp_i))
    cfg.kp_j = int(det.get("kp_j", cfg.kp_j))
    cfg.detect_n_frames = int(det.get("n_frames", cfg.detect_n_frames))
    cfg.detect_outlier_mad_k = float(det.get("outlier_mad_k", cfg.detect_outlier_mad_k))
    cfg.detect_frame_gap_s = float(det.get("frame_gap_s", cfg.detect_frame_gap_s))

    head_fd = g("calib", "head_fixed_depth_cam_m", default=cfg.head_fixed_depth_cam_m)
    cfg.head_fixed_depth_cam_m = None if head_fd in (None, "null") else float(head_fd)
    head_tz = g("calib", "head_tray_z_m", default=cfg.head_tray_z_m)
    cfg.head_tray_z_m = None if head_tz in (None, "null") else float(head_tz)
    cfg.ffs_weights = _resolve_path(root, str(g("depth", "ffs_weights", default="") or ""))
    cfg.ffs_cfg = _resolve_path(root, str(g("depth", "ffs_cfg", default="") or ""))

    motion = g("motion", default={}) or {}
    cfg.hover_height_m = float(motion.get("hover_height_m", cfg.hover_height_m))
    if "wrist_use_fixed_hover_depth" in motion:
        cfg.wrist_use_fixed_hover_depth = bool(motion["wrist_use_fixed_hover_depth"])
    cfg.hover_camera_down = bool(
        motion.get("hover_camera_down", cfg.hover_camera_down)
    )
    cfg.safe_z_m = float(motion.get("safe_z_m", cfg.safe_z_m))
    cfg.tcp_offset = list(motion.get("tcp_offset", cfg.tcp_offset))
    cfg.tool_rz_offset_rad = float(motion.get("tool_rz_offset_rad", cfg.tool_rz_offset_rad))
    cfg.move_velocity_scale = float(motion.get("velocity_scale", cfg.move_velocity_scale))
    cfg.cart_vel_mm_s = float(motion.get("cart_vel_mm_s", cfg.cart_vel_mm_s))
    cfg.cart_acc_mm_s2 = float(motion.get("cart_acc_mm_s2", cfg.cart_acc_mm_s2))
    cfg.tj_vel_ratio = int(motion.get("vel_ratio", cfg.tj_vel_ratio))
    cfg.tj_acc_ratio = int(motion.get("acc_ratio", cfg.tj_acc_ratio))
    cfg.fine_xy_only = bool(motion.get("fine_xy_only", cfg.fine_xy_only))
    cfg.fine_apply_rz = bool(motion.get("fine_apply_rz", cfg.fine_apply_rz))
    cfg.fine_clear_dy_m = float(motion.get("fine_clear_dy_m", cfg.fine_clear_dy_m))
    cfg.fine_clear_dz_m = float(motion.get("fine_clear_dz_m", cfg.fine_clear_dz_m))
    cfg.fine_descend_dz_m = float(
        motion.get("fine_descend_dz_m", cfg.fine_descend_dz_m)
    )
    cfg.target_board_id = int(motion.get("target_board_id", cfg.target_board_id))
    cfg.board_match_max_dist_m = float(
        motion.get("board_match_max_dist_m", cfg.board_match_max_dist_m)
    )
    cfg.grasp_z_file = _resolve_path(
        root, str(motion.get("grasp_z_file", "config/grasp_z.json") or "config/grasp_z.json")
    )
    gz = motion.get("grasp_z_m", None)
    if gz is not None and gz != "null":
        cfg.grasp_z_m = float(gz)
    else:
        try:
            from .grasp_z import load_grasp_z_m

            cfg.grasp_z_m = load_grasp_z_m(cfg.grasp_z_file)
        except Exception:
            cfg.grasp_z_m = None
    cfg.grasp_z_lift_m = float(motion.get("grasp_z_lift_m", cfg.grasp_z_lift_m))
    cfg.dry_run = bool(motion.get("dry_run", cfg.dry_run))
    hp = motion.get("home_pose")
    cfg.home_pose = list(hp) if hp is not None else None
    rj_a = motion.get("ready_joints_A")
    cfg.ready_joints_A = [float(x) for x in rj_a] if rj_a else None
    rj_b = motion.get("ready_joints_B")
    cfg.ready_joints_B = [float(x) for x in rj_b] if rj_b else None

    cfg.motion_backend = str(
        g("motion_backend", default=None)
        or (g("robot", default={}) or {}).get("backend")
        or cfg.motion_backend
    ).lower()
    rim = g("rim", default={}) or {}
    cfg.rim_frame_bundle = str(rim.get("frame_bundle", cfg.rim_frame_bundle))
    cfg.rim_mc = str(rim.get("mc", cfg.rim_mc) or "")
    cfg.rim_connect_timeout_s = float(
        rim.get("connect_timeout_s", cfg.rim_connect_timeout_s)
    )
    cfg.rim_plan_timeout_s = float(rim.get("plan_timeout_s", cfg.rim_plan_timeout_s))
    cfg.rim_max_linear_vel = float(rim.get("max_linear_vel", cfg.rim_max_linear_vel))
    cfg.rim_max_angular_vel = float(rim.get("max_angular_vel", cfg.rim_max_angular_vel))

    tj = g("tj_demo", default={}) or {}
    cfg.tj_robot_ip = str(tj.get("robot_ip", cfg.tj_robot_ip))
    cfg.tj_arm = str(tj.get("arm", cfg.tj_arm))
    cfg.tj_use_impedance = bool(tj.get("use_impedance", cfg.tj_use_impedance))
    cfg.tj_contact_threshold_n = float(
        tj.get("contact_threshold_n", cfg.tj_contact_threshold_n)
    )

    grip = tj.get("gripper", {}) or {}
    cfg.gripper_grasp_position = float(
        grip.get("grasp_position", cfg.gripper_grasp_position)
    )
    cfg.gripper_release_position = float(
        grip.get("release_position", cfg.gripper_release_position)
    )
    cfg.gripper_speed = float(grip.get("speed", cfg.gripper_speed))
    cfg.gripper_torque_grasp = float(
        grip.get("torque_grasp", cfg.gripper_torque_grasp)
    )
    cfg.gripper_torque_release = float(
        grip.get("torque_release", cfg.gripper_torque_release)
    )
    cfg.gripper_com_ch = int(grip.get("com_ch", cfg.gripper_com_ch))

    cam = g("cameras", default={}) or {}
    cfg.cam_head_source = str(cam.get("head_source", cfg.cam_head_source))
    cfg.cam_portal = str(cam.get("portal", cfg.cam_portal))
    cfg.cam_portal_head_left_id = int(
        cam.get("portal_head_left_id", cfg.cam_portal_head_left_id)
    )
    cfg.cam_portal_head_right_id = int(
        cam.get("portal_head_right_id", cfg.cam_portal_head_right_id)
    )
    cfg.cam_portal_max_hz = float(cam.get("portal_max_hz", cfg.cam_portal_max_hz))
    cfg.cam_portal_timeout_s = float(
        cam.get("portal_timeout_s", cfg.cam_portal_timeout_s)
    )
    cfg.cam_head_left = str(cam.get("head_left", cfg.cam_head_left))
    cfg.cam_head_right = str(cam.get("head_right", cfg.cam_head_right))
    cfg.cam_head_width = int(cam.get("head_width", cfg.cam_head_width))
    cfg.cam_head_height = int(cam.get("head_height", cfg.cam_head_height))
    cfg.cam_s36_sdk = str(cam.get("s36_sdk", cfg.cam_s36_sdk))
    cfg.cam_wrist_source = str(cam.get("wrist_source", cfg.cam_wrist_source) or "portal")
    cfg.cam_portal_wrist_id = int(cam.get("portal_wrist_id", cfg.cam_portal_wrist_id))
    cfg.cam_portal_wrist_depth_id = int(
        cam.get("portal_wrist_depth_id", cfg.cam_portal_wrist_depth_id)
    )
    cfg.cam_portal_wrist_right_id = int(
        cam.get("portal_wrist_right_id", cfg.cam_portal_wrist_right_id)
    )
    cfg.cam_portal_wrist_right_depth_id = int(
        cam.get("portal_wrist_right_depth_id", cfg.cam_portal_wrist_right_depth_id)
    )
    cfg.cam_wrist_need_depth = bool(
        cam.get("wrist_need_depth", cfg.cam_wrist_need_depth)
    )
    cfg.cam_wrist_undistort = bool(cam.get("wrist_undistort", cfg.cam_wrist_undistort))
    cfg.cam_wrist_serial = str(cam.get("wrist_serial", cfg.cam_wrist_serial) or "")
    cfg.cam_wrist_width = int(cam.get("wrist_width", cfg.cam_wrist_width))
    cfg.cam_wrist_height = int(cam.get("wrist_height", cfg.cam_wrist_height))
    cfg.cam_wrist_fps = int(cam.get("wrist_fps", cfg.cam_wrist_fps))
    cfg.live_settle_s = float(cam.get("settle_s", motion.get("settle_s", cfg.live_settle_s)))
    cfg.still_pos_tol_m = float(motion.get("still_pos_tol_m", cfg.still_pos_tol_m))
    cfg.still_rot_tol_rad = float(motion.get("still_rot_tol_rad", cfg.still_rot_tol_rad))
    cfg.still_hold_s = float(motion.get("still_hold_s", cfg.still_hold_s))
    cfg.still_timeout_s = float(motion.get("still_timeout_s", cfg.still_timeout_s))

    classes = g("classes", default={}) or {}
    cfg.cls_pcb_small = int(classes.get("pcb_small", cfg.cls_pcb_small))
    cfg.cls_nopcb = int(classes.get("nopcb", cfg.cls_nopcb))
    cfg.cls_pcb_large = int(classes.get("pcb_large", cfg.cls_pcb_large))

    lb = g("large_board", default={}) or {}
    cfg.large_board_enable = bool(lb.get("enable", cfg.large_board_enable))
    cfg.large_tool_y_m = float(lb.get("tool_y_m", cfg.large_tool_y_m))
    cfg.large_tool_z_m = float(lb.get("tool_z_m", cfg.large_tool_z_m))
    cfg.large_open_margin_m = float(lb.get("open_margin_m", cfg.large_open_margin_m))
    cfg.large_squeeze_close_m = float(lb.get("squeeze_close_m", cfg.large_squeeze_close_m))
    cfg.large_squeeze_velocity_scale = float(
        lb.get("squeeze_velocity_scale", cfg.large_squeeze_velocity_scale)
    )
    cfg.large_squeeze_force_n = float(lb.get("squeeze_force_n", cfg.large_squeeze_force_n))
    cfg.large_use_impedance = bool(lb.get("use_impedance", cfg.large_use_impedance))
    gz = lb.get("grasp_z_m", cfg.large_grasp_z_m)
    cfg.large_grasp_z_m = None if gz in (None, "null") else float(gz)
    ts = lb.get("tip_span_m", cfg.large_tip_span_m)
    cfg.large_tip_span_m = None if ts in (None, "null") else float(ts)
    hz = lb.get("hover_z_m", cfg.large_hover_z_m)
    cfg.large_hover_z_m = None if hz in (None, "null") else float(hz)
    cfg.large_approach_velocity_scale = float(
        lb.get("approach_velocity_scale", cfg.large_approach_velocity_scale)
    )
    hw = lb.get("half_width_m", cfg.large_half_width_m)
    cfg.large_half_width_m = None if hw in (None, "null") else float(hw)
    cfg.large_use_fixed_grasp_z_plane = bool(
        lb.get("use_fixed_grasp_z_plane", cfg.large_use_fixed_grasp_z_plane)
    )
    cfg.large_use_left_current_z = bool(
        lb.get("use_left_current_z", cfg.large_use_left_current_z)
    )
    cfg.large_grasp_z_below_left_m = float(
        lb.get("grasp_z_below_left_m", cfg.large_grasp_z_below_left_m)
    )
    cfg.large_orient_edge = str(
        lb.get("orient_edge", cfg.large_orient_edge) or "parallel_p2p3"
    ).strip().lower()
    cfg.large_orient_rz_offset_deg = float(
        lb.get("orient_rz_offset_deg", cfg.large_orient_rz_offset_deg)
    )
    cfg.large_approach_dx_m = float(
        lb.get("approach_dx_m", cfg.large_approach_dx_m)
    )
    cfg.large_approach_dy_m = float(
        lb.get("approach_dy_m", cfg.large_approach_dy_m)
    )
    cfg.large_fine_enable = bool(lb.get("fine_enable", cfg.large_fine_enable))
    wpath = lb.get("wrist_pose_weights", cfg.large_wrist_pose_weights)
    if wpath in (None, "null", ""):
        cfg.large_wrist_pose_weights = ""
    else:
        cfg.large_wrist_pose_weights = _resolve_path(root, str(wpath))
    cfg.large_wrist_imgsz = int(
        lb.get("wrist_imgsz", cfg.large_wrist_imgsz) or cfg.large_wrist_imgsz
    )
    cfg.large_wrist_n_corners = int(
        lb.get("wrist_n_corners", cfg.large_wrist_n_corners)
        or cfg.large_wrist_n_corners
    )
    cfg.large_coarse_hover_height_m = float(
        lb.get("coarse_hover_height_m", cfg.large_coarse_hover_height_m)
    )
    cfg.large_fine_hover_height_m = float(
        lb.get("fine_hover_height_m", cfg.large_fine_hover_height_m)
    )
    ah = lb.get("approach_hover_height_m", cfg.large_approach_hover_height_m)
    cfg.large_approach_hover_height_m = (
        None if ah in (None, "null", "") else float(ah)
    )
    cfg.large_fine_match_max_dist_m = float(
        lb.get("fine_match_max_dist_m", cfg.large_fine_match_max_dist_m)
    )
    # 单臂支路 B
    sa = lb.get("single_arm", {}) or {}
    cfg.large_single_enable = bool(sa.get("enable", cfg.large_single_enable))
    cfl = sa.get("coarse_flange_pose", cfg.large_single_coarse_flange)
    if cfl in (None, "null", ""):
        cfg.large_single_coarse_flange = None
    else:
        cfg.large_single_coarse_flange = [float(x) for x in list(cfl)[:6]]
    stop_z = sa.get("stop_z_m", cfg.large_single_stop_z_m)
    cfg.large_single_stop_z_m = (
        None if stop_z in (None, "null", "") else float(stop_z)
    )
    sh = sa.get("fine_hover_height_m", cfg.large_single_fine_hover_height_m)
    cfg.large_single_fine_hover_height_m = (
        None if sh in (None, "null", "") else float(sh)
    )
    return cfg


def load_T_from_yaml_matrix(path: str | Path) -> np.ndarray:
    """Load 4x4 from handeye yaml with top-level ``matrix``."""
    try:
        import yaml
    except ImportError as e:
        raise RuntimeError("PyYAML required") from e
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, dict) and "matrix" in data:
        return _mat4(data["matrix"])
    raise RuntimeError(f"No matrix in {path}")
