"""Coarse (head) → hover 30cm → fine (wrist XY+rz, Z frozen) grasp pipeline."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .config import GraspConfig, load_config
from .board_id import (
    assign_board_ids_by_distance,
    cls_name,
    select_grasp_board,
    match_fine_to_coarse,
    select_board,
)
from .detect import (
    PcbDetection,
    YoloPoseDetector,
    draw_multi_board_vis,
    draw_pose_vis,
)
from .geometry import (
    END_DOWN_RX,
    END_DOWN_RY,
    ObjectPose,
    compute_flange_target,
    nearest_rz_rad,
)
from .localize import Intrinsics, format_pose, head_localize, wrist_localize
from .motion import DryRunMotion, MotionBackend
from .robust import median_detection, median_object_pose
from .undistort import HeadRectifyMaps, build_maps_from_stereo_json, load_maps_from_yml

LOG = logging.getLogger("pcb_grasp.pipeline")


def _pose_dist(a: ObjectPose, b: ObjectPose) -> float:
    """Distance for picking the frame closest to the robust median."""
    dxyz = (a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2
    drz = math.atan2(math.sin(a.rz - b.rz), math.cos(a.rz - b.rz))
    return float(dxyz + (0.05 * drz) ** 2)


def _pick_frame_closest_to_median(
    poses: list[ObjectPose],
    med: ObjectPose,
    keep_mask: list[bool] | None = None,
) -> int:
    """Index of the sample closest to median (only among kept if mask given)."""
    best_i, best_d = 0, float("inf")
    for i, p in enumerate(poses):
        if keep_mask is not None and not keep_mask[i]:
            continue
        d = _pose_dist(p, med)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def _det_center_keep_mask(dets: list[PcbDetection], mad_k: float) -> list[bool]:
    """Same center-MAD keep rule as median_detection, as a per-frame mask."""
    import numpy as np
    from .robust import _mad_keep

    if len(dets) <= 1:
        return [True] * len(dets)
    cx = [d.center_uv[0] for d in dets]
    cy = [d.center_uv[1] for d in dets]
    keep = _mad_keep(cx, mad_k) & _mad_keep(cy, mad_k)
    if not bool(np.any(keep)):
        return [True] * len(dets)
    return [bool(x) for x in keep]


def _pose_keep_mask(poses: list[ObjectPose], mad_k: float) -> list[bool]:
    import numpy as np
    from .robust import _mad_keep

    if len(poses) <= 1:
        return [True] * len(poses)
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    zs = [p.z for p in poses]
    keep = _mad_keep(xs, mad_k) & _mad_keep(ys, mad_k) & _mad_keep(zs, mad_k)
    rzs = [p.rz for p in poses]
    ref = math.atan2(
        float(np.mean([math.sin(r) for r in rzs])),
        float(np.mean([math.cos(r) for r in rzs])),
    )
    rz_off = [math.atan2(math.sin(r - ref), math.cos(r - ref)) for r in rzs]
    keep = keep & _mad_keep(rz_off, mad_k)
    if not bool(np.any(keep)):
        return [True] * len(poses)
    return [bool(x) for x in keep]


@dataclass
class GraspResult:
    coarse: ObjectPose | None = None
    fine: ObjectPose | None = None
    coarse_boards: list[ObjectPose] = field(default_factory=list)
    fine_boards: list[ObjectPose] = field(default_factory=list)
    target_board_id: int | None = None
    coarse_flange: list[float] | None = None
    fine_flange: list[float] | None = None
    hover_flange: list[float] | None = None
    current_flange: list[float] | None = None  # 识别时左臂 TCP，米/弧度
    current_flange_right: list[float] | None = None  # 识别时右臂 TCP，米/弧度
    ok: bool = False
    message: str = ""
    debug: dict = field(default_factory=dict)


class GraspPipeline:
    """
    一套 live 流程（按头模 cls 选抓取方式）
    ------------------------------------
    1. 头相机粗定：多板 + 距离 ID → 选目标
    2. 接近：小板悬停在板上方；大板双臂到左右夹点外开位
    3. 抓取：小板腕精(相机朝下) → tip竖直朝下接近(板上方同高) → 示教 Z 下降 → 收夹；
       大板双臂同高水平对挤（不收夹爪）
    """

    def __init__(
        self,
        cfg: GraspConfig | None = None,
        config_path: str | Path | None = None,
        motion: MotionBackend | None = None,
        dual_motion=None,
        force_class: str | None = None,
    ) -> None:
        self.cfg = cfg or load_config(config_path)
        self.motion = motion or DryRunMotion(
            initial_pose=self.cfg.home_pose
            if self.cfg.home_pose
            else None
        )
        self.dual_motion = dual_motion
        # "large" | "small" | None — override detected class for dry tests
        self.force_class = (
            None if force_class is None else str(force_class).strip().lower() or None
        )
        self._head_maps: HeadRectifyMaps | None = None
        self._head_K: Intrinsics | None = None
        self._head_det: YoloPoseDetector | None = None
        self._wrist_det: YoloPoseDetector | None = None
        self._wrist_K = Intrinsics(
            self.cfg.wrist_fx, self.cfg.wrist_fy, self.cfg.wrist_cx, self.cfg.wrist_cy
        )
        self._stereo_depth = None
        self._shape_model = None
        self._last_head_pose_vis: np.ndarray | None = None
        self._last_wrist_pose_vis: np.ndarray | None = None
        self._last_wrist_match_vis: np.ndarray | None = None
        self._last_head_rect: np.ndarray | None = None
        self._last_head_raw_l: np.ndarray | None = None
        self._last_head_raw_r: np.ndarray | None = None
        self._last_head_depth_u16: np.ndarray | None = None
        self._last_head_depth_stats: dict | None = None
        self._last_wrist_bgr: np.ndarray | None = None
        self._last_wrist_depth: np.ndarray | None = None

    def _ensure_shape_model(self):
        if self._shape_model is not None:
            return self._shape_model
        from .shape_refine import load_shape_model

        path = getattr(self.cfg, "shape_model_path", "") or ""
        if not path or not Path(path).is_file():
            raise RuntimeError(
                f"shape model missing: {path!r} "
                "(set models.shape_model in grasp_config.yaml)"
            )
        self._shape_model = load_shape_model(path)
        LOG.info("loaded shape model %s n=%d", path, len(self._shape_model.points))
        return self._shape_model

    def _ensure_stereo_depth(self):
        if self._stereo_depth is not None:
            return self._stereo_depth
        from .stereo_depth import HeadStereoDepth

        maps = self._ensure_head_maps()
        kw = {}
        if getattr(self.cfg, "ffs_weights", None):
            kw["weights"] = self.cfg.ffs_weights
        if getattr(self.cfg, "ffs_cfg", None):
            kw["cfg"] = self.cfg.ffs_cfg
        self._stereo_depth = HeadStereoDepth(
            maps=maps,
            stereo_calib_json=self.cfg.stereo_calib_json or None,
            **kw,
        )
        return self._stereo_depth

    # ------------------------------------------------------------------ setup
    def _ensure_head_maps(self) -> HeadRectifyMaps:
        if self._head_maps is not None:
            return self._head_maps
        if self.cfg.stereo_calib_json:
            maps = build_maps_from_stereo_json(self.cfg.stereo_calib_json)
            if self.cfg.stereo_maps_yml and Path(self.cfg.stereo_maps_yml).is_file():
                # Prefer precomputed maps but keep K from stereoRectify
                pre = load_maps_from_yml(self.cfg.stereo_maps_yml, K_rect=maps.K_rect)
                self._head_maps = pre
            else:
                self._head_maps = maps
        elif self.cfg.stereo_maps_yml:
            self._head_maps = load_maps_from_yml(self.cfg.stereo_maps_yml)
        else:
            raise RuntimeError("Need calib.stereo_calib_json or stereo_maps_yml for head undistort")
        self._head_K = Intrinsics.from_K(self._head_maps.K_rect)
        return self._head_maps

    def _ensure_detectors(self) -> None:
        if self._head_det is None:
            if not self.cfg.head_pose_weights:
                raise RuntimeError("models.head_pose_weights not set")
            self._head_det = YoloPoseDetector(
                self.cfg.head_pose_weights,
                conf=self.cfg.head_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=int(getattr(self.cfg, "head_imgsz", self.cfg.imgsz)),
                class_ids=getattr(self.cfg, "head_class_ids", None),
            )
        if self._wrist_det is None:
            w = self.cfg.wrist_pose_weights or self.cfg.head_pose_weights
            self._wrist_det = YoloPoseDetector(
                w,
                conf=self.cfg.wrist_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=int(getattr(self.cfg, "wrist_imgsz", self.cfg.imgsz)),
            )

    # --------------------------------------------------------------- perception
    def undistort_head(self, fisheye_bgr: np.ndarray) -> np.ndarray:
        maps = self._ensure_head_maps()
        return maps.undistort_left(fisheye_bgr)

    def _plane_z_for_cls(self, cls_id: int | None) -> float:
        """头定位板面 Z：大板用标定 tip/板面高；其它用托盘高。"""
        large_cls = int(self.cfg.cls_pcb_large)
        use_fixed = bool(getattr(self.cfg, "large_use_fixed_grasp_z_plane", True))
        if use_fixed and cls_id is not None and int(cls_id) == large_cls:
            gz = getattr(self.cfg, "large_grasp_z_m", None)
            if gz is not None:
                return float(gz)
            LOG.warning(
                "large grasp_z_m 未标定 → 头定位暂用 head_tray_z_m=%.4f（下午标定后写入 large_board.grasp_z_m）",
                float(self.cfg.head_tray_z_m),
            )
        return float(self.cfg.head_tray_z_m)

    def _head_localize_det(
        self,
        det: PcbDetection,
        *,
        depth: np.ndarray | None = None,
        tray_z_m: float | None = None,
    ) -> ObjectPose:
        """单板头定位。大板默认只走固定板面求交，不采角点深度。"""
        assert self._head_K is not None
        cls_id = int(getattr(det, "cls_id", 0) or 0)
        large_cls = int(self.cfg.cls_pcb_large)
        use_fixed_large = bool(
            getattr(self.cfg, "large_use_fixed_grasp_z_plane", True)
        ) and cls_id == large_cls
        z_plane = (
            float(tray_z_m)
            if tray_z_m is not None
            else self._plane_z_for_cls(cls_id)
        )
        if use_fixed_large:
            # 大板：固定板面高度（标定 tip Z），不用 FFS/深度图角点
            pose = head_localize(
                det,
                T_base_cam=self.cfg.T_base_cam,
                K=self._head_K,
                depth=None,
                depth_scale=0.001,
                fixed_depth_cam_m=None,
                tray_z_m=z_plane,
                kp_i=self.cfg.kp_i,
                kp_j=self.cfg.kp_j,
            )
        else:
            pose = head_localize(
                det,
                T_base_cam=self.cfg.T_base_cam,
                K=self._head_K,
                depth=depth,
                depth_scale=0.001,
                fixed_depth_cam_m=self.cfg.head_fixed_depth_cam_m,
                tray_z_m=z_plane,
                kp_i=self.cfg.kp_i,
                kp_j=self.cfg.kp_j,
            )
        pose.cls_id = cls_id
        return pose

    def localize_head(
        self,
        fisheye_bgr: np.ndarray,
        *,
        depth: np.ndarray | None = None,
        right_fisheye_bgr: np.ndarray | None = None,
        already_undistorted: bool = False,
        require_depth_ok: bool = True,
    ) -> tuple[ObjectPose, PcbDetection, np.ndarray]:
        """Head localize. Prefer stereo FFS depth from left+right; else depth map / fixed Z."""
        self._ensure_detectors()
        assert self._head_det is not None
        maps = self._ensure_head_maps()
        assert self._head_K is not None

        depth_stats = None
        if depth is None and right_fisheye_bgr is not None:
            est = self._ensure_stereo_depth()
            if already_undistorted:
                # both left/right assumed already rectified
                rect = fisheye_bgr
                depth, _valid, depth_stats = est.estimate_rectified(
                    rect, right_fisheye_bgr
                )
            else:
                rect, _rr, depth, _valid, depth_stats = est.estimate_raw(
                    fisheye_bgr, right_fisheye_bgr
                )
            if require_depth_ok and depth_stats is not None and not depth_stats.ok():
                raise RuntimeError(
                    f"Head FFS depth not trustworthy: {depth_stats.summary()} "
                    f"— run tools/check_head_depth.py first"
                )
        else:
            rect = fisheye_bgr if already_undistorted else maps.undistort_left(fisheye_bgr)

        det = self._head_det.detect_best(rect)
        if det is None:
            raise RuntimeError("Head: no PCB detected")
        pose = self._head_localize_det(det, depth=depth)
        return pose, det, rect

    def localize_wrist(
        self,
        wrist_bgr: np.ndarray,
        wrist_depth: np.ndarray,
        ee_xyz_rpy: Sequence[float] | None = None,
        *,
        make_vis: bool = True,
    ) -> tuple[ObjectPose, PcbDetection]:
        self._ensure_detectors()
        assert self._wrist_det is not None
        det = self._wrist_det.detect_best(wrist_bgr)
        if det is None:
            raise RuntimeError("Wrist: no PCB detected")
        ee = ee_xyz_rpy if ee_xyz_rpy is not None else self.motion.get_flange_pose()
        if getattr(self.cfg, "fine_use_shape_match", True):
            from .shape_refine import wrist_localize_shape_match

            pose, info = wrist_localize_shape_match(
                wrist_bgr,
                wrist_depth,
                det,
                shape_model=self._ensure_shape_model(),
                T_tool_cam=self.cfg.T_tool_cam,
                ee_xyz_rpy=ee,
                K=self._wrist_K,
                depth_scale=self.cfg.wrist_depth_scale,
                fallback_depth_m=float(self.cfg.hover_height_m),
                force_fixed_depth=bool(
                    getattr(self.cfg, "wrist_use_fixed_hover_depth", True)
                ),
                angle_offset_deg=float(self.cfg.shape_angle_offset_deg),
                angle_half_range=float(self.cfg.shape_angle_range_deg),
                score_thresh=float(self.cfg.shape_score_thresh),
                edge_mode=str(self.cfg.shape_edge_mode),
                make_vis=make_vis,
            )
            if make_vis and info.vis is not None:
                self._last_wrist_match_vis = info.vis
            return pose, det

        pose = wrist_localize(
            det,
            T_tool_cam=self.cfg.T_tool_cam,
            ee_xyz_rpy=ee,
            K=self._wrist_K,
            depth=wrist_depth,
            depth_scale=self.cfg.wrist_depth_scale,
            fallback_depth_m=float(self.cfg.hover_height_m),
            force_fixed_depth=bool(
                getattr(self.cfg, "wrist_use_fixed_hover_depth", True)
            ),
            kp_i=self.cfg.kp_i,
            kp_j=self.cfg.kp_j,
        )
        return pose, det

    # ------------------------------------------------------------------- motion
    def flange_above(self, obj: ObjectPose, *, z_override: float | None = None) -> list[float]:
        """旧悬停：板上方 + compute_flange_target（默认 tip rx=π）；一般不用。"""
        z = float(z_override) if z_override is not None else float(obj.z + self.cfg.hover_height_m)
        p = [obj.x, obj.y, z]
        return compute_flange_target(
            p,
            obj.rz,
            tcp_offset=[0.0, 0.0, 0.0],
            tool_rz_offset_rad=self.cfg.tool_rz_offset_rad,
        )

    def flange_above_end_down(
        self, obj: ObjectPose, *, z_override: float | None = None
    ) -> list[float]:
        """板上方 + 末端竖直向下 (rx=-90°, ry=0, rz)。抓取姿态用，不作粗定位悬停。"""
        flange = self.flange_above(obj, z_override=z_override)
        flange[3], flange[4] = END_DOWN_RX, END_DOWN_RY
        return flange

    def flange_above_camera_down(
        self,
        obj: ObjectPose,
        *,
        z_override: float | None = None,
        keep_rz_rad: float | None = None,
        cam_yaw_rad: float | None = None,
        current_flange: Sequence[float] | None = None,
    ) -> list[float]:
        """粗定位悬停：腕相机光心对准板心，光轴竖直向下。

        默认按当前法兰选最近 ``cam_yaw``（避免 yaw=0 时 rz 跳 ~90° 拧限位）。
        ``keep_rz_rad`` 已废弃，忽略。
        """
        from .geometry import (
            flange_pose_camera_down_nearest,
            flange_pose_camera_optical_down,
        )

        z_cam = (
            float(z_override)
            if z_override is not None
            else float(obj.z + self.cfg.hover_height_m)
        )
        cam_xyz = [float(obj.x), float(obj.y), z_cam]
        cur = (
            [float(x) for x in current_flange[:6]]
            if current_flange is not None
            else None
        )
        if cur is None:
            try:
                cur = [float(x) for x in self.motion.get_flange_pose()[:6]]
            except Exception:
                cur = None

        if cam_yaw_rad is not None:
            yaw = float(cam_yaw_rad)
            flange = flange_pose_camera_optical_down(
                cam_xyz,
                T_tool_cam=self.cfg.T_tool_cam,
                cam_yaw_rad=yaw,
            )
        else:
            flange, yaw = flange_pose_camera_down_nearest(
                cam_xyz,
                T_tool_cam=self.cfg.T_tool_cam,
                current_xyzrpy=cur,
            )
        if keep_rz_rad is not None:
            LOG.warning(
                "flange_above_camera_down: ignore keep_rz "
                "(would offset camera center off board)"
            )
        reach = math.hypot(flange[0], flange[1])
        LOG.info(
            "camera-down hover: cam_center→board_xy=(%.4f,%.4f) camZ=%.3f "
            "cam_yaw=%.1f° reach=%.0fmm → flange xyz=(%.4f,%.4f,%.4f) "
            "rpy_deg=(%.1f,%.1f,%.1f)",
            obj.x,
            obj.y,
            z_cam,
            yaw * 180.0 / math.pi,
            reach * 1000.0,
            flange[0],
            flange[1],
            flange[2],
            flange[3] * 180.0 / math.pi,
            flange[4] * 180.0 / math.pi,
            flange[5] * 180.0 / math.pi,
        )
        # 不做工作区门禁：算出法兰位姿后交给 rim_python.move_to_pose
        return flange

    def fine_flange_xy_hold_z(
        self,
        obj: ObjectPose,
        current_flange: Sequence[float],
    ) -> list[float]:
        """抓取样式法兰：物体 XY (+ rz)，**保持当前法兰 Z**；末端竖直向下 (-90,0)。"""
        cur = [float(x) for x in current_flange[:6]]
        z_hold = cur[2]
        flange = compute_flange_target(
            [obj.x, obj.y, z_hold],
            obj.rz if self.cfg.fine_apply_rz else cur[5],
            tcp_offset=[0.0, 0.0, 0.0],
            tool_rz_offset_rad=self.cfg.tool_rz_offset_rad,
        )
        flange[2] = z_hold
        flange[3], flange[4] = END_DOWN_RX, END_DOWN_RY
        if not self.cfg.fine_apply_rz:
            flange[5] = cur[5]
        return flange

    def grasp_orient_hold_xyz(
        self,
        obj: ObjectPose,
        current_flange: Sequence[float],
    ) -> list[float]:
        """只改抓取姿态：XYZ 冻结在当前悬停位，rx=-90°,ry=0，rz 取精定位。"""
        cur = [float(x) for x in current_flange[:6]]
        rz = float(obj.rz) if self.cfg.fine_apply_rz else cur[5]
        flange = compute_flange_target(
            [cur[0], cur[1], cur[2]],
            rz,
            tcp_offset=[0.0, 0.0, 0.0],
            tool_rz_offset_rad=self.cfg.tool_rz_offset_rad,
        )
        flange[0], flange[1], flange[2] = cur[0], cur[1], cur[2]
        flange[3], flange[4] = END_DOWN_RX, END_DOWN_RY
        if not self.cfg.fine_apply_rz:
            flange[5] = cur[5]
        return flange

    def _move_tool_down_pose(
        self,
        pose6: Sequence[float],
        *,
        tag: str,
        direct: bool = True,
        force_tip_down: bool = True,
    ) -> list[float]:
        """移到目标位姿。``force_tip_down=True`` 时强制抓取姿态 (rx=-90°,ry=0)；
        粗定位相机朝下悬停不要开它。"""
        tgt = [float(x) for x in pose6[:6]]
        if force_tip_down:
            tgt[3], tgt[4] = END_DOWN_RX, END_DOWN_RY
        LOG.info(
            "PHASE: %s → xyz=(%.3f,%.3f,%.3f) rpy_deg=(%.1f,%.1f,%.1f) direct=%s end_down=%s",
            tag,
            tgt[0],
            tgt[1],
            tgt[2],
            tgt[3] * 180.0 / math.pi,
            tgt[4] * 180.0 / math.pi,
            tgt[5] * 180.0 / math.pi,
            bool(direct),
            bool(force_tip_down),
        )
        if hasattr(self.motion, "move_tool_down_hover") and force_tip_down:
            arm = str(getattr(self.cfg, "tj_arm", "A")).upper()
            ready = (
                self.cfg.ready_joints_A if arm == "A" else self.cfg.ready_joints_B
            )
            if ready and hasattr(self.motion, "preferred_ik_seed"):
                self.motion.preferred_ik_seed = list(ready)
            safe_z = float(getattr(self.cfg, "safe_z_m", 0.02))
            ee = self.motion.get_flange_pose()
            corridor = max(float(ee[2]), float(tgt[2]), safe_z)
            self.motion.move_tool_down_hover(
                tgt,
                velocity_scale=self.cfg.move_velocity_scale,
                max_deg_s=10.0,
                safe_z_m=corridor,
                prefer_rz_rad=float(tgt[5]),
                direct=bool(direct),
            )
        else:
            # 保持当前 rx/ry：走 move_line，勿经 tip-down 重写姿态
            kw = {"velocity_scale": self.cfg.move_velocity_scale, "keep_orientation": False}
            try:
                self.motion.move_line(tgt, direct=bool(direct), **kw)
            except TypeError:
                self.motion.move_line(tgt, **kw)
        return self.motion.get_flange_pose()

    # ------------------------------------------------------------------- full run
    def run(
        self,
        head_fisheye_bgr: np.ndarray,
        wrist_bgr: np.ndarray,
        wrist_depth: np.ndarray,
        *,
        head_right_bgr: np.ndarray | None = None,
        head_depth: np.ndarray | None = None,
        already_undistorted: bool = False,
        do_grasp: bool = False,
        save_vis_dir: str | Path | None = None,
    ) -> GraspResult:
        result = GraspResult()
        try:
            coarse, head_det, rect = self.localize_head(
                head_fisheye_bgr,
                depth=head_depth,
                right_fisheye_bgr=head_right_bgr,
                already_undistorted=already_undistorted,
            )
            result.coarse = coarse
            LOG.info("COARSE %s", format_pose(coarse))

            # 粗定位悬停：相机光心对准板心，光轴朝下（选最近 cam_yaw）
            hover = self.flange_above_camera_down(
                coarse, current_flange=self.motion.get_flange_pose()
            )
            result.hover_flange = hover
            result.coarse_flange = hover
            LOG.info(
                "HOVER camera-down cam_center@board camZ=+%.2fm "
                "flange xyz=(%.4f,%.4f,%.4f) rpy_deg=(%.1f,%.1f,%.1f)",
                self.cfg.hover_height_m,
                hover[0],
                hover[1],
                hover[2],
                hover[3] * 180.0 / math.pi,
                hover[4] * 180.0 / math.pi,
                hover[5] * 180.0 / math.pi,
            )
            # 粗定位悬停不碰夹爪（避免一上来误收夹）
            self.motion.move_line(hover, velocity_scale=self.cfg.move_velocity_scale)

            # After arriving, re-read EE for wrist eye-in-hand
            ee = self.motion.get_flange_pose()
            fine, wrist_det = self.localize_wrist(wrist_bgr, wrist_depth, ee_xyz_rpy=ee)
            result.fine = fine
            LOG.info("FINE   %s", format_pose(fine))

            fine_flange = self.fine_flange_xy_hold_z(fine, ee)
            result.fine_flange = fine_flange
            LOG.info(
                "FINE flange (Z held=%.4f) xyz=(%.4f,%.4f,%.4f) rz=%.3f",
                ee[2],
                fine_flange[0],
                fine_flange[1],
                fine_flange[2],
                fine_flange[5],
            )
            self.motion.move_line(fine_flange, velocity_scale=self.cfg.move_velocity_scale)

            if do_grasp:
                # Descend / close left for a later stage — not enabled by default
                LOG.warning("do_grasp=True but descend Z is intentionally not implemented yet")
                self.motion.set_gripper(False)

            if save_vis_dir:
                self._save_stage_vis(Path(save_vis_dir), head_rect=rect, head_det=head_det)

            result.ok = True
            result.message = "coarse+fine OK (Z frozen on fine)"
            result.debug = {
                "hover_height_m": self.cfg.hover_height_m,
                "fine_xy_only": self.cfg.fine_xy_only,
                "fine_use_shape_match": bool(self.cfg.fine_use_shape_match),
                "kp": [self.cfg.kp_i, self.cfg.kp_j],
            }
            return result
        except Exception as e:
            result.ok = False
            result.message = str(e)
            LOG.exception("Grasp pipeline failed: %s", e)
            return result

    def localize_head_multiframe(
        self,
        grab_head,
        *,
        n_frames: int | None = None,
        mad_k: float | None = None,
        frame_gap_s: float | None = None,
        require_depth_ok: bool = True,
        target_board_id: int | None = None,
    ) -> tuple[ObjectPose, PcbDetection, np.ndarray, np.ndarray, np.ndarray]:
        """采 N 帧选代表帧 → 该帧多板检测 → 基座距离赋 ID → 返回目标板。

        Returns ``(target_pose, target_det, rect, raw_left, raw_right)``.
        全部粗定位板存在 ``self._last_coarse_boards``。
        """
        import time

        self._ensure_detectors()
        assert self._head_det is not None
        maps = self._ensure_head_maps()
        assert self._head_K is not None

        n = int(n_frames if n_frames is not None else self.cfg.detect_n_frames)
        gap = float(
            frame_gap_s if frame_gap_s is not None else self.cfg.detect_frame_gap_s
        )
        n = max(1, n)
        tid = int(
            self.cfg.target_board_id if target_board_id is None else target_board_id
        )

        # 选检出板数最多的一帧做代表（深度+多板 ID）
        best_pack: tuple[int, np.ndarray, np.ndarray, np.ndarray, list[PcbDetection]] | None = None
        LOG.info(
            "HEAD multi-frame n=%d → multi-board + distance ID (target=%d)",
            n,
            tid,
        )
        for i in range(n):
            left, right = grab_head()
            rect = maps.undistort_left(left)
            dets_i = self._head_det.detect(rect)
            if not dets_i:
                LOG.warning("HEAD frame %d/%d: no detect", i + 1, n)
            else:
                LOG.info("HEAD frame %d/%d: %d board(s)", i + 1, n, len(dets_i))
                if best_pack is None or len(dets_i) > len(best_pack[4]):
                    best_pack = (i, left, right, rect, dets_i)
            if gap > 0 and i + 1 < n:
                time.sleep(gap)

        if best_pack is None:
            raise RuntimeError(f"Head: no PCB in {n} frames")

        best_i, left, right, _rect0, _dets0 = best_pack
        est = self._ensure_stereo_depth()
        rect_vis, _rr, depth, _valid, depth_stats = est.estimate_raw(left, right)
        if require_depth_ok and depth_stats is not None and not depth_stats.ok():
            raise RuntimeError(
                f"Head FFS depth not trustworthy: {depth_stats.summary()} "
                f"— run tools/check_head_depth.py first"
            )

        # 在有深度的校正图上再检一遍，保证 det 与 depth 对齐
        dets = self._head_det.detect(rect_vis)
        if not dets:
            raise RuntimeError("Head: no PCB on depth frame")

        poses: list[ObjectPose] = []
        ok_dets: list[PcbDetection] = []
        for det in dets:
            try:
                pose = self._head_localize_det(det, depth=depth)
            except Exception as exc:
                LOG.warning("HEAD board localize fail: %s", exc)
                continue
            poses.append(pose)
            ok_dets.append(det)

        if not poses:
            raise RuntimeError("Head: boards detected but none localized")

        boards = assign_board_ids_by_distance(poses, metric="xy")
        # 同步 det 顺序到 boards（按距离重排）
        order = sorted(
            range(len(poses)),
            key=lambda i: (poses[i].x ** 2 + poses[i].y ** 2),
        )
        dets_ranked = [ok_dets[i] for i in order]
        # 全类都分配 ID；有大板优先，否则第 tid 近的小板（跳过 nopcb）
        target = select_grasp_board(
            boards,
            tid,
            pcb_cls=int(self.cfg.cls_pcb_small),
            large_cls=int(self.cfg.cls_pcb_large),
            prefer_large=bool(self.cfg.large_board_enable),
        )
        tgt_i = next(
            i for i, b in enumerate(boards) if b.board_id == target.board_id
        )
        tgt_det = dets_ranked[tgt_i]
        tid_grasp = int(target.board_id) if target.board_id is not None else int(tid)

        self._last_coarse_boards = boards
        self._last_coarse_dets = dets_ranked
        self._last_target_board_id = tid_grasp
        n_pcb = sum(
            1
            for b in boards
            if int(getattr(b, "cls_id", 0) or 0) == int(self.cfg.cls_pcb_small)
        )
        n_nopcb = sum(
            1
            for b in boards
            if int(getattr(b, "cls_id", 0) or 0) == int(self.cfg.cls_nopcb)
        )
        n_large = sum(
            1
            for b in boards
            if int(getattr(b, "cls_id", 0) or 0) == int(self.cfg.cls_pcb_large)
        )
        force_edge = (0, 3) if self.force_class == "large" else None
        self._last_head_pose_vis = draw_multi_board_vis(
            rect_vis,
            dets_ranked,
            [b.board_id for b in boards],
            title=(
                f"HEAD ×{len(boards)} (pcb={n_pcb} nopcb={n_nopcb} large={n_large}) "
                f"frame#{best_i + 1} grasp=ID{tid_grasp}/{cls_name(target.cls_id)} "
                f"rz={target.rz * 180.0 / math.pi:.1f}°"
            ),
            highlight_id=tid_grasp,
            kp_i=int(self.cfg.kp_i),
            kp_j=int(self.cfg.kp_j),
            large_cls=int(self.cfg.cls_pcb_large),
            force_edge_for_id=force_edge,
        )
        self._last_head_rect = rect_vis
        self._last_head_raw_l = left
        self._last_head_raw_r = right
        self._last_head_depth_u16 = depth
        self._last_head_depth_stats = (
            None
            if depth_stats is None
            else {
                "valid_ratio": float(depth_stats.valid_ratio),
                "median_m": float(depth_stats.median_m),
                "p10_m": float(depth_stats.p10_m),
                "p90_m": float(depth_stats.p90_m),
                "shape": list(depth_stats.shape),
                "summary": depth_stats.summary(),
                "rep_frame": best_i + 1,
                "n_boards": len(boards),
                "n_pcb": n_pcb,
                "n_nopcb": n_nopcb,
                "n_large": n_large,
                "target_board_id": tid_grasp,
                "board_classes": [
                    {"id": b.board_id, "cls": cls_name(b.cls_id), "cls_id": int(b.cls_id)}
                    for b in boards
                ],
            }
        )
        LOG.info(
            "HEAD %d board(s) pcb=%d nopcb=%d large=%d → grasp ID=%d %s %s",
            len(boards),
            n_pcb,
            n_nopcb,
            n_large,
            tid_grasp,
            cls_name(target.cls_id),
            format_pose(target),
        )
        return target, tgt_det, rect_vis, left, right

    def _grasp_mode(self, pose: ObjectPose) -> str:
        """Return ``large`` or ``small`` from force_class / detected cls."""
        if self.force_class == "large":
            return "large"
        if self.force_class == "small":
            return "small"
        if bool(self.cfg.large_board_enable) and int(
            getattr(pose, "cls_id", 0) or 0
        ) == int(self.cfg.cls_pcb_large):
            return "large"
        return "small"

    def _ensure_dual(self):
        dual = self.dual_motion
        if dual is not None:
            return dual
        from .dual_tj_motion import DryRunDualMotion

        dual = DryRunDualMotion()
        dual.connect()
        self.dual_motion = dual
        LOG.warning("dual_motion missing — DryRunDualMotion for this run")
        return dual

    def _read_both_tcp(self) -> tuple[list[float] | None, list[float] | None]:
        """识别时读左右臂 TCP（米/弧度）。优先 dual；否则主臂 + 对侧 Rim 只读。"""
        left: list[float] | None = None
        right: list[float] | None = None
        dual = self.dual_motion
        if dual is not None and hasattr(dual, "get_flange_poses"):
            try:
                pl, pr = dual.get_flange_poses()
                return list(pl[:6]), list(pr[:6])
            except Exception as exc:  # noqa: BLE001
                LOG.warning("dual get_flange_poses failed: %s", exc)
        try:
            primary = list(self.motion.get_flange_pose()[:6])
        except Exception as exc:  # noqa: BLE001
            LOG.warning("read primary TCP failed: %s", exc)
            primary = None
        side = str(getattr(self.motion, "side", "left") or "left").lower()
        if side.startswith("r"):
            right = primary
        else:
            left = primary
        # 对侧：同 frame_bundle 再读一次
        try:
            from .rim_motion import RimMotion

            fb = str(
                getattr(self.motion, "frame_bundle", None)
                or getattr(self.cfg, "rim_frame_bundle", None)
                or "10.3.8.7:50066"
            )
            other_arm = "A" if side.startswith("r") else "B"
            peer = RimMotion(
                frame_bundle=fb,
                arm=other_arm,
                mc=str(getattr(self.motion, "mc", "") or ""),
                connect_timeout_s=float(
                    getattr(self.motion, "connect_timeout_s", 10.0) or 10.0
                ),
                dry_run=bool(getattr(self.motion, "dry_run", False)),
            )
            if (
                isinstance(self.motion, RimMotion)
                and bool(getattr(self.motion, "_connected", False))
                and not bool(getattr(self.motion, "dry_run", False))
            ):
                peer.connect_as_peer(self.motion)
            else:
                peer.connect()
            other = list(peer.get_flange_pose()[:6])
            if not isinstance(self.motion, RimMotion) or peer is not self.motion:
                try:
                    peer.close()
                except Exception:  # noqa: BLE001
                    pass
            if side.startswith("r"):
                left = other
            else:
                right = other
        except Exception as exc:  # noqa: BLE001
            LOG.warning("read other-arm TCP failed: %s", exc)
        return left, right

    def localize_wrist_multiframe(
        self,
        grab_wrist,
        ee_xyz_rpy: Sequence[float] | None = None,
        *,
        n_frames: int | None = None,
        mad_k: float | None = None,
        frame_gap_s: float | None = None,
        coarse_boards: Sequence[ObjectPose] | None = None,
        target_board_id: int | None = None,
    ) -> tuple[ObjectPose, PcbDetection, np.ndarray]:
        """多帧腕部：每帧多板 → 匹配粗 ID → 只保留目标 ID → 中值。"""
        import time

        self._ensure_detectors()
        assert self._wrist_det is not None
        n = int(n_frames if n_frames is not None else self.cfg.detect_n_frames)
        k = float(mad_k if mad_k is not None else self.cfg.detect_outlier_mad_k)
        gap = float(
            frame_gap_s if frame_gap_s is not None else self.cfg.detect_frame_gap_s
        )
        n = max(1, n)
        ee = ee_xyz_rpy if ee_xyz_rpy is not None else self.motion.get_flange_pose()
        use_shape = bool(getattr(self.cfg, "fine_use_shape_match", True))
        tid = int(
            self.cfg.target_board_id if target_board_id is None else target_board_id
        )
        coarse = list(
            coarse_boards
            if coarse_boards is not None
            else getattr(self, "_last_coarse_boards", []) or []
        )
        if not coarse:
            raise RuntimeError("Wrist fine needs coarse_boards for ID matching")
        max_dist = float(self.cfg.board_match_max_dist_m)

        poses: list[ObjectPose] = []
        dets: list[PcbDetection] = []
        bgrs: list[np.ndarray] = []
        depths: list[np.ndarray] = []
        # 回退池：任意匹配上粗 ID 的检出（目标 ID 全空时用）
        fb_poses: list[ObjectPose] = []
        fb_dets: list[PcbDetection] = []
        fb_bgrs: list[np.ndarray] = []
        fb_depths: list[np.ndarray] = []
        # 代表帧：检出板数最多的那帧（用于多板 ID 可视化）
        rep_all: tuple[np.ndarray, np.ndarray, list[PcbDetection], list[ObjectPose]] | None = None

        coarse_tgt = next((b for b in coarse if b.board_id == tid), None)

        LOG.info(
            "WRIST multi-frame n=%d shape_match=%s match→coarse ID=%d (max_dist=%.0fmm)",
            n,
            use_shape,
            tid,
            max_dist * 1000.0,
        )
        for i in range(n):
            bgr, depth = grab_wrist()
            frame_dets = self._wrist_det.detect(bgr)
            if not frame_dets:
                LOG.warning("WRIST frame %d/%d: no detect", i + 1, n)
            else:
                frame_poses: list[ObjectPose] = []
                frame_ok_dets: list[PcbDetection] = []
                for det in frame_dets:
                    try:
                        if use_shape:
                            from .shape_refine import wrist_localize_shape_match

                            pose, _info = wrist_localize_shape_match(
                                bgr,
                                depth,
                                det,
                                shape_model=self._ensure_shape_model(),
                                T_tool_cam=self.cfg.T_tool_cam,
                                ee_xyz_rpy=ee,
                                K=self._wrist_K,
                                depth_scale=self.cfg.wrist_depth_scale,
                                fallback_depth_m=float(self.cfg.hover_height_m),
                                force_fixed_depth=bool(
                                    getattr(
                                        self.cfg, "wrist_use_fixed_hover_depth", True
                                    )
                                ),
                                angle_offset_deg=float(self.cfg.shape_angle_offset_deg),
                                angle_half_range=float(self.cfg.shape_angle_range_deg),
                                score_thresh=float(self.cfg.shape_score_thresh),
                                edge_mode=str(self.cfg.shape_edge_mode),
                                make_vis=False,
                                kp_i=int(self.cfg.kp_i),
                                kp_j=int(self.cfg.kp_j),
                            )
                        else:
                            pose = wrist_localize(
                                det,
                                T_tool_cam=self.cfg.T_tool_cam,
                                ee_xyz_rpy=ee,
                                K=self._wrist_K,
                                depth=depth,
                                depth_scale=self.cfg.wrist_depth_scale,
                                fallback_depth_m=float(self.cfg.hover_height_m),
                                force_fixed_depth=bool(
                                    getattr(
                                        self.cfg, "wrist_use_fixed_hover_depth", True
                                    )
                                ),
                                kp_i=self.cfg.kp_i,
                                kp_j=self.cfg.kp_j,
                            )
                    except Exception as exc:
                        LOG.warning(
                            "WRIST frame %d/%d board localize fail: %s",
                            i + 1,
                            n,
                            exc,
                        )
                        continue
                    frame_poses.append(pose)
                    frame_ok_dets.append(det)

                if not frame_poses:
                    LOG.warning("WRIST frame %d/%d: localize all failed", i + 1, n)
                else:
                    matched = match_fine_to_coarse(
                        frame_poses, coarse, max_dist_m=max_dist
                    )
                    if rep_all is None or len(matched) > len(rep_all[3]):
                        rep_all = (bgr, depth, frame_ok_dets, matched)

                    hit_tid = False
                    for det, pose in zip(frame_ok_dets, matched):
                        if pose.board_id is None:
                            continue
                        # 任意认出的板进回退池（每帧每种 ID 留最近粗目标的一块）
                        if not any(
                            p.board_id == pose.board_id
                            and abs(p.x - pose.x) < 1e-4
                            and abs(p.y - pose.y) < 1e-4
                            for p in fb_poses[-8:]
                        ):
                            fb_poses.append(pose)
                            fb_dets.append(det)
                            fb_bgrs.append(bgr)
                            fb_depths.append(depth)
                        if pose.board_id == tid:
                            if coarse_tgt is not None:
                                pose.rz = nearest_rz_rad(
                                    float(coarse_tgt.rz), float(pose.rz)
                                )
                            poses.append(pose)
                            dets.append(det)
                            bgrs.append(bgr)
                            depths.append(depth)
                            hit_tid = True
                            break  # 每帧至多一块目标板
                    if not hit_tid:
                        LOG.warning(
                            "WRIST frame %d/%d: no board matched to ID=%d "
                            "(recognized ids=%s)",
                            i + 1,
                            n,
                            tid,
                            [p.board_id for p in matched if p.board_id is not None],
                        )
            if gap > 0 and i + 1 < n:
                time.sleep(gap)

        used_fallback = False
        if not poses:
            # 只抓腕部实际认出并匹配上的板：选离粗定位目标最近的那块 ID，再中值
            cand = [
                (p, d, b, dep)
                for p, d, b, dep in zip(fb_poses, fb_dets, fb_bgrs, fb_depths)
                if p.board_id is not None
            ]
            if not cand:
                raise RuntimeError(
                    f"Wrist: no valid PCB with board_id={tid} in {n} frames "
                    "(多板未匹配到目标 ID，且腕部也未认出其它板；"
                    "若日志大量 No wrist depth，则是深度为空而非 YOLO 未检出)"
                )
            # 按 ID 聚成代表点，挑离 coarse 目标最近的 ID
            by_id: dict[int, list[tuple]] = {}
            for item in cand:
                bid = int(item[0].board_id)
                by_id.setdefault(bid, []).append(item)
            def _id_score(bid: int) -> float:
                items = by_id[bid]
                mx = sum(t[0].x for t in items) / len(items)
                my = sum(t[0].y for t in items) / len(items)
                if coarse_tgt is None:
                    return float(bid)
                return float((mx - coarse_tgt.x) ** 2 + (my - coarse_tgt.y) ** 2)

            pick_id = min(by_id.keys(), key=_id_score)
            chosen = by_id[pick_id]
            poses = [c[0] for c in chosen]
            dets = [c[1] for c in chosen]
            bgrs = [c[2] for c in chosen]
            depths = [c[3] for c in chosen]
            used_fallback = True
            LOG.warning(
                "WRIST ID=%d 未匹配，回退到已识别板 id=%d（候选=%s，只抓能认出的）",
                tid,
                pick_id,
                sorted(by_id.keys()),
            )

        keep = _pose_keep_mask(poses, k)
        med_pose = median_object_pose(poses, mad_k=k)
        if used_fallback:
            # 回退时用中值板自身 ID，勿强行写成原 tid
            if med_pose.board_id is None:
                med_pose.board_id = poses[0].board_id
        else:
            med_pose.board_id = tid
        med_det = median_detection(dets, mad_k=k)
        rep_i = _pick_frame_closest_to_median(poses, med_pose, keep)
        rep_bgr = bgrs[rep_i]
        rep_det = dets[rep_i]

        # 多板 ID 可视化（代表帧全部匹配结果）
        if rep_all is not None:
            rbgr, rdepth, rdets, rmatched = rep_all
            self._last_wrist_pose_vis = draw_multi_board_vis(
                rbgr,
                rdets,
                [p.board_id for p in rmatched],
                title=f"WRIST boards match→ID{tid} (n={len(rmatched)})",
                highlight_id=tid,
                kp_i=int(self.cfg.kp_i),
                kp_j=int(self.cfg.kp_j),
            )
            self._last_fine_boards = list(rmatched)
        else:
            self._last_wrist_pose_vis = draw_pose_vis(
                rep_bgr,
                rep_det,
                title=f"WRIST ID={tid} median→frame#{rep_i + 1}",
                kpt_dir=(1, 0),
                board_id=tid,
            )
            self._last_fine_boards = [med_pose]

        if use_shape:
            from .shape_refine import wrist_localize_shape_match

            _pose_vis, info = wrist_localize_shape_match(
                rep_bgr,
                depths[rep_i],
                rep_det,
                shape_model=self._ensure_shape_model(),
                T_tool_cam=self.cfg.T_tool_cam,
                ee_xyz_rpy=ee,
                K=self._wrist_K,
                depth_scale=self.cfg.wrist_depth_scale,
                fallback_depth_m=float(self.cfg.hover_height_m),
                force_fixed_depth=bool(
                    getattr(self.cfg, "wrist_use_fixed_hover_depth", True)
                ),
                angle_offset_deg=float(self.cfg.shape_angle_offset_deg),
                angle_half_range=float(self.cfg.shape_angle_range_deg),
                score_thresh=float(self.cfg.shape_score_thresh),
                edge_mode=str(self.cfg.shape_edge_mode),
                make_vis=True,
                kp_i=int(self.cfg.kp_i),
                kp_j=int(self.cfg.kp_j),
            )
            self._last_wrist_match_vis = info.vis
            # 代表帧 match：XY + rz 均来自轮廓方向；±180° 折到靠近头粗
            med_pose.x, med_pose.y, med_pose.z = _pose_vis.x, _pose_vis.y, _pose_vis.z
            med_pose.rz = float(_pose_vis.rz)
            med_pose.edge_yaw = float(_pose_vis.edge_yaw)
            med_pose.corners_base = list(_pose_vis.corners_base)
            if coarse_tgt is not None:
                before = med_pose.rz
                med_pose.rz = nearest_rz_rad(float(coarse_tgt.rz), float(med_pose.rz))
                if abs(med_pose.rz - before) > 1e-6:
                    LOG.info(
                        "WRIST contour-rz unwrap to coarse: %.1f° → %.1f° (coarse=%.1f°)",
                        before * 180.0 / math.pi,
                        med_pose.rz * 180.0 / math.pi,
                        float(coarse_tgt.rz) * 180.0 / math.pi,
                    )
        else:
            self._last_wrist_match_vis = None

        self._last_wrist_bgr = rep_bgr
        self._last_wrist_depth = depths[rep_i]
        LOG.info(
            "WRIST target ID=%d median from %d frames (vis=frame#%d) → %s",
            tid,
            len(poses),
            rep_i + 1,
            format_pose(med_pose),
        )
        return med_pose, med_det, rep_bgr

    def _wait_still(self, *, settle_s: float | None = None) -> list[float]:
        """动臂结束后等静止；静止前不做识别。"""
        return self.motion.wait_until_still(
            settle_s=float(self.cfg.live_settle_s if settle_s is None else settle_s),
            pos_tol_m=float(self.cfg.still_pos_tol_m),
            rot_tol_rad=float(self.cfg.still_rot_tol_rad),
            hold_s=float(self.cfg.still_hold_s),
            timeout_s=float(self.cfg.still_timeout_s),
        )

    def _save_stage_vis(
        self,
        out: Path,
        *,
        head_rect: np.ndarray | None = None,
        head_det: PcbDetection | None = None,
        extra_meta: dict | None = None,
    ) -> None:
        """本轮实验批次：识别/匹配图 + 头 FFS 深度 + 腕深度。"""
        from .experiment_log import save_experiment_artifacts

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        head_pose = self._last_head_pose_vis
        if head_pose is None and head_rect is not None and head_det is not None:
            head_pose = draw_pose_vis(
                head_rect, head_det, title="HEAD", kpt_dir=(0, 1)
            )
            self._last_head_pose_vis = head_pose
        meta = {
            "grasp_z_m": getattr(self.cfg, "grasp_z_m", None),
            "wrist_serial": getattr(self.cfg, "cam_wrist_serial", ""),
            "shape_model": str(getattr(self.cfg, "shape_model_path", "")),
        }
        if extra_meta:
            meta.update(extra_meta)
        save_experiment_artifacts(
            out,
            head_pose_vis=head_pose,
            head_rect=self._last_head_rect if self._last_head_rect is not None else head_rect,
            head_raw_left=self._last_head_raw_l,
            head_raw_right=self._last_head_raw_r,
            head_depth_u16_mm=self._last_head_depth_u16,
            head_depth_stats=self._last_head_depth_stats,
            wrist_pose_vis=self._last_wrist_pose_vis,
            wrist_match_vis=self._last_wrist_match_vis,
            wrist_color=self._last_wrist_bgr,
            wrist_depth=self._last_wrist_depth,
            wrist_depth_scale=float(self.cfg.wrist_depth_scale),
            meta=meta,
        )

    def run_live(
        self,
        grab_head,
        grab_wrist,
        *,
        settle_s: float | None = None,
        do_grasp: bool = False,
        save_vis_dir: str | Path | None = None,
        stage: str = "all",
        large_confirm=None,
        plan_only: bool = False,
        on_progress=None,
    ) -> GraspResult:
        """一套流程：头粗定 → 按 cls 选抓取方式 → 接近 → 抓取。

        ``stage``:
        - ``all``：头粗 → 接近 →（小板腕精+示教高+收夹 / 大板确认对挤）
        - ``coarse``：头粗 → 接近（小板悬停 / 大板外开）；若 ``do_grasp`` 且大板则继续确认对挤
        - ``fine``：仅小板——头定 ID → 悬停 → 腕精按 ID → 示教高 → 收夹；大板直接拒绝

        ``plan_only``: 只识别并画出抓取点（不动臂、不收夹）。用于相机预览。

        ``large_confirm``: 可选回调 ``(info: dict) -> bool``。大板在双臂到达
        接近位（tip 左右各外开 open_margin、同高）之后、合拢之前调用；
        返回 False 则停在接近位。小板不调用。
        ``info`` 含 ``plan`` / ``board_id`` / ``vis``(BGR) / ``message``。

        ``on_progress``: 可选 ``(result, phase: str) -> None``，粗/精定位一出就回调
        （运动开始前），供网页即时打印坐标。
        """
        from .large_board import draw_large_board_plan, plan_large_board_grasp

        def _progress(phase: str) -> None:
            if on_progress is None:
                return
            try:
                on_progress(result, phase)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("on_progress(%s) failed: %s", phase, exc)

        result = GraspResult()
        rect = None
        head_det = None
        large_plan = None
        grasp_mode = "small"
        try:
            # —— 1. 头粗定（共用）——
            LOG.info(
                "PHASE: wait STILL → HEAD detect (%d frames)%s",
                self.cfg.detect_n_frames,
                " [plan_only]" if plan_only else "",
            )
            if not plan_only:
                self._wait_still(
                    settle_s=0.0 if settle_s is None else min(settle_s, 0.2)
                )
            coarse, head_det, rect, _hl, _hr = self.localize_head_multiframe(grab_head)
            result.coarse = coarse
            result.coarse_boards = list(getattr(self, "_last_coarse_boards", []) or [])
            result.target_board_id = int(
                getattr(self, "_last_target_board_id", self.cfg.target_board_id)
            )
            grasp_mode = self._grasp_mode(coarse)
            if grasp_mode == "large" and self.force_class == "large":
                coarse.cls_id = int(self.cfg.cls_pcb_large)
            LOG.info(
                "COARSE ID=%d cls=%s → grasp_mode=%s | boards=%s",
                result.target_board_id,
                cls_name(coarse.cls_id),
                grasp_mode,
                [
                    f"ID{b.board_id}/{cls_name(b.cls_id)}:({b.x:.3f},{b.y:.3f})"
                    for b in result.coarse_boards
                ],
            )
            cur_l, cur_r = self._read_both_tcp()
            result.current_flange = cur_l
            result.current_flange_right = cur_r
            result.debug = {
                **(result.debug or {}),
                "grasp_mode": grasp_mode,
                "stage": "after_coarse_detect",
                "live": True,
                "current_flange": result.current_flange,
                "current_flange_right": result.current_flange_right,
            }
            _progress("coarse")

            if str(stage) == "fine" and grasp_mode == "large":
                if save_vis_dir:
                    self._save_stage_vis(
                        Path(save_vis_dir), head_rect=rect, head_det=head_det
                    )
                result.debug = {
                    "grasp_mode": grasp_mode,
                    "stage": "fine",
                    "live": True,
                }
                result.ok = False
                result.message = (
                    "精定位仅支持小板（按 ID 腕精→示教高→收夹）；"
                    "大板请点「粗定位」走接近/对挤"
                )
                return result

            # —— 2. 接近（按 mode）——
            d_xy = 0.0
            d_xyz = 0.0
            if grasp_mode == "large":
                tool_z = float(self.cfg.large_tool_z_m)
                # 夹取高度：默认两臂法兰 Z = 左臂当前高度 → tipZ = 左臂Z − tool_z
                grasp_z_tip = self.cfg.large_grasp_z_m
                z_src = "calib_grasp_z_m"
                if bool(getattr(self.cfg, "large_use_left_current_z", True)):
                    cur_l = result.current_flange
                    if cur_l is None or len(cur_l) < 3:
                        try:
                            cur_l = list(self.motion.get_flange_pose()[:6])
                            result.current_flange = cur_l
                        except Exception:  # noqa: BLE001
                            cur_l = None
                    if cur_l is not None and len(cur_l) >= 3:
                        flange_z = float(cur_l[2])
                        grasp_z_tip = flange_z - tool_z
                        z_src = "left_current_flange_z"
                        LOG.info(
                            "LARGE grasp height from left arm: flangeZ=%.4f → tipZ=%.4f "
                            "(both arms)",
                            flange_z,
                            float(grasp_z_tip),
                        )
                    else:
                        LOG.warning(
                            "LARGE use_left_current_z 但读不到左臂位姿 → 回退 grasp_z_m/tray"
                        )
                # 大板：角点按 tip/板面高度固定平面求交（不用头深）
                if head_det is not None and bool(
                    getattr(self.cfg, "large_use_fixed_grasp_z_plane", True)
                ):
                    z_plane = (
                        float(grasp_z_tip)
                        if grasp_z_tip is not None
                        else self._plane_z_for_cls(int(self.cfg.cls_pcb_large))
                    )
                    try:
                        coarse_lg = self._head_localize_det(
                            head_det, depth=None, tray_z_m=z_plane
                        )
                        coarse_lg.board_id = coarse.board_id
                        coarse_lg.cls_id = int(self.cfg.cls_pcb_large)
                        coarse = coarse_lg
                        result.coarse = coarse
                        LOG.info(
                            "LARGE fixed-plane z=%.4f (%s) → %s",
                            z_plane,
                            z_src,
                            format_pose(coarse),
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("LARGE re-localize on grasp_z plane failed: %s", exc)
                large_plan = plan_large_board_grasp(
                    coarse,
                    tool_y_m=float(self.cfg.large_tool_y_m),
                    tool_z_m=tool_z,
                    open_margin_m=float(self.cfg.large_open_margin_m),
                    squeeze_close_m=float(self.cfg.large_squeeze_close_m),
                    hover_z_m=self.cfg.large_hover_z_m,
                    half_width_m=self.cfg.large_half_width_m,
                    grasp_z_m=grasp_z_tip,
                    tip_span_m=self.cfg.large_tip_span_m,
                    orient_edge=str(
                        getattr(self.cfg, "large_orient_edge", "p0p1") or "p0p1"
                    ),
                    orient_rz_offset_deg=float(
                        getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
                    ),
                )
                self._last_large_plan = large_plan
                if head_det is not None:
                    base_vis = (
                        self._last_head_pose_vis
                        if self._last_head_pose_vis is not None
                        else rect
                    )
                    self._last_head_pose_vis = draw_large_board_plan(
                        base_vis,
                        head_det.corners,
                        title=(
                            f"grasp_mode=large ID={coarse.board_id} "
                            f"w={large_plan.width_m*1000:.0f}mm"
                        ),
                    )
                # 规划一出即落盘（含头图）；避免全流程中途停导致空 batch
                if save_vis_dir and not plan_only:
                    self._save_stage_vis(
                        Path(save_vis_dir),
                        head_rect=rect,
                        head_det=head_det,
                        extra_meta={
                            "grasp_mode": grasp_mode,
                            "stage": "large_plan_ready",
                            "plan": large_plan.as_dict(),
                        },
                    )
                result.coarse_flange = list(large_plan.left_approach)
                result.hover_flange = list(large_plan.left_approach)
                if plan_only:
                    plan_d = large_plan.as_dict()

                    def _pose_csv(p: list[float] | None) -> str | None:
                        if not p or len(p) < 6:
                            return None
                        return (
                            f"{p[0]*1000:.4f},{p[1]*1000:.4f},{p[2]*1000:.4f},"
                            f"{math.degrees(p[3]):.4f},{math.degrees(p[4]):.4f},"
                            f"{math.degrees(p[5]):.4f}"
                        )

                    def _cli(arm: str, pose_csv: str | None) -> str:
                        if not pose_csv:
                            return "—"
                        return (
                            f"python -m rim_python.move_to_pose "
                            f"--arm {arm} --pose {pose_csv} "
                            f"--execute --allow-large"
                        )

                    left_app = list(large_plan.left_approach)
                    right_app = list(large_plan.right_approach)
                    left_grasp = list(large_plan.left_grasp)
                    right_grasp = list(large_plan.right_grasp)
                    left_app_csv = _pose_csv(left_app)
                    right_app_csv = _pose_csv(right_app)
                    left_grasp_csv = _pose_csv(left_grasp)
                    right_grasp_csv = _pose_csv(right_grasp)
                    cur_l = result.current_flange
                    cur_r = result.current_flange_right
                    cur_l_csv = _pose_csv(cur_l)
                    cur_r_csv = _pose_csv(cur_r)
                    cli = {
                        "arm": "dual",
                        "current": _cli("left", cur_l_csv),
                        "current_left": _cli("left", cur_l_csv),
                        "current_right": _cli("right", cur_r_csv),
                        "left_approach": _cli("left", left_app_csv),
                        "right_approach": _cli("right", right_app_csv),
                        "left_grasp": _cli("left", left_grasp_csv),
                        "right_grasp": _cli("right", right_grasp_csv),
                        # 兼容旧字段：coarse=左接近，fine=右接近
                        "coarse": _cli("left", left_app_csv),
                        "fine": _cli("right", right_app_csv),
                    }
                    if save_vis_dir:
                        self._save_stage_vis(
                            Path(save_vis_dir),
                            head_rect=rect,
                            head_det=head_det,
                            extra_meta={
                                "grasp_mode": grasp_mode,
                                "stage": "plan_only",
                                "plan": plan_d,
                                "left_approach": left_app,
                                "right_approach": right_app,
                                "left_grasp": left_grasp,
                                "right_grasp": right_grasp,
                                "cli": cli,
                            },
                        )
                    result.debug = {
                        "grasp_mode": grasp_mode,
                        "stage": "plan_only",
                        "live": True,
                        "move": False,
                        "arm": "dual",
                        "plan": plan_d,
                        "current_flange": result.current_flange,
                        "current_flange_right": result.current_flange_right,
                        "left_approach": left_app,
                        "right_approach": right_app,
                        "left_grasp": left_grasp,
                        "right_grasp": right_grasp,
                        "cli": cli,
                        "note": {
                            "edge": "P0→P3（夹持边）",
                            "approach": "左右 tip 外开 open_margin，同高",
                            "grasp": "合拢到 tipL/tipR；两臂法兰 Z=左臂当前高度",
                            "z_source": z_src,
                        },
                        "print_xy": {
                            "mid_mode": plan_d.get("mid_mode"),
                            "p0_xy": plan_d.get("p0_xy"),
                            "p3_xy": plan_d.get("p3_xy"),
                            "tip_left_xy": plan_d.get("tip_left_xy"),
                            "tip_right_xy": plan_d.get("tip_right_xy"),
                            "grasp_z_m": plan_d.get("grasp_z_m"),
                            "tip_span_m": plan_d.get("tip_span_m"),
                            "width_m": plan_d.get("width_m"),
                            "calib_pending": plan_d.get("calib_pending"),
                            "left_approach_xyzrpy": left_app,
                            "right_approach_xyzrpy": right_app,
                            "left_grasp_xyzrpy": left_grasp,
                            "right_grasp_xyzrpy": right_grasp,
                        },
                    }
                    result.ok = True
                    tl = plan_d.get("tip_left_xy") or []
                    tr = plan_d.get("tip_right_xy") or []
                    tip_z = float(large_plan.tip_left[2])
                    pending = " 待标定高度/间距" if plan_d.get("calib_pending") else ""
                    result.message = (
                        "未动臂 · 大板双臂末端 xyzrpy(mm,deg) 可复制命令：\n"
                        f"# 当前左臂\n{cli['current_left']}\n"
                        f"# 当前右臂\n{cli['current_right']}\n"
                        f"# 左臂接近\n{cli['left_approach']}\n"
                        f"# 右臂接近\n{cli['right_approach']}\n"
                        f"# tipL=({tl[0]:.4f},{tl[1]:.4f}) tipR=({tr[0]:.4f},{tr[1]:.4f}) "
                        f"span={large_plan.width_m*1000:.1f}mm Z={tip_z:.4f}{pending}"
                    )
                    LOG.info("当前左臂: %s", cli["current_left"])
                    LOG.info("当前右臂: %s", cli["current_right"])
                    _progress("coarse")
                    return result
                dual = self._ensure_dual()
                # 先只到接近位：同 tip 高度，左/右相对 tip 各外开 open_margin（默认 5cm）
                # 不对 tip 位直接落点；确认抓取后再合拢（对齐托盘：运动→确认→抓取）
                LOG.info(
                    "PHASE: APPROACH outer±%.0fmm sameZ tipL=%s tipR=%s width=%.0fmm",
                    float(self.cfg.large_open_margin_m) * 1000.0,
                    [round(x, 3) for x in large_plan.tip_left],
                    [round(x, 3) for x in large_plan.tip_right],
                    large_plan.width_m * 1000.0,
                )
                approach = dual.move_both(
                    large_plan.left_approach,
                    large_plan.right_approach,
                    velocity_scale=float(self.cfg.large_approach_velocity_scale),
                )
                # MC 常假成功：缺半边再补跑一次，避免观感「一边近一边远」
                try:
                    pl, pr = dual.get_flange_poses()
                    def _remain_mm(cur, goal) -> float:
                        return float(
                            math.hypot(
                                float(cur[0]) - float(goal[0]),
                                float(cur[1]) - float(goal[1]),
                            )
                            * 1000.0
                        )

                    def _dist_center_mm(pose, center_xy) -> float:
                        return float(
                            math.hypot(
                                float(pose[0]) - float(center_xy[0]),
                                float(pose[1]) - float(center_xy[1]),
                            )
                            * 1000.0
                        )

                    rem_l = _remain_mm(pl, large_plan.left_approach)
                    rem_r = _remain_mm(pr, large_plan.right_approach)
                    ctr = [
                        0.5
                        * (
                            float(large_plan.tip_left[0])
                            + float(large_plan.tip_right[0])
                        ),
                        0.5
                        * (
                            float(large_plan.tip_left[1])
                            + float(large_plan.tip_right[1])
                        ),
                    ]
                    goal_dl = _dist_center_mm(large_plan.left_approach, ctr)
                    goal_dr = _dist_center_mm(large_plan.right_approach, ctr)
                    cur_dl = _dist_center_mm(pl, ctr)
                    cur_dr = _dist_center_mm(pr, ctr)
                    LOG.info(
                        "APPROACH remain L=%.0fmm R=%.0fmm | "
                        "dist→center goal L/R=%.0f/%.0f arrived L/R=%.0f/%.0f "
                        "(diff goal=%.0f arrived=%.0f mm)",
                        rem_l,
                        rem_r,
                        goal_dl,
                        goal_dr,
                        cur_dl,
                        cur_dr,
                        goal_dl - goal_dr,
                        cur_dl - cur_dr,
                    )
                    if rem_l > 40.0 or rem_r > 40.0:
                        LOG.warning(
                            "APPROACH incomplete → retry lagging arm(s) "
                            "L=%.0fmm R=%.0fmm",
                            rem_l,
                            rem_r,
                        )
                        if rem_l > 40.0:
                            dual.left.move_line(
                                large_plan.left_approach,
                                velocity_scale=float(
                                    self.cfg.large_approach_velocity_scale
                                ),
                            )
                        if rem_r > 40.0:
                            dual.right.move_line(
                                large_plan.right_approach,
                                velocity_scale=float(
                                    self.cfg.large_approach_velocity_scale
                                ),
                            )
                        pl, pr = dual.get_flange_poses()
                        approach = type(approach)(
                            left=list(pl[:6]),
                            right=list(pr[:6]),
                            ok=True,
                            message=(
                                f"retried remain L={_remain_mm(pl, large_plan.left_approach):.0f} "
                                f"R={_remain_mm(pr, large_plan.right_approach):.0f} "
                                f"distCtr L/R={_dist_center_mm(pl, ctr):.0f}/"
                                f"{_dist_center_mm(pr, ctr):.0f}"
                            ),
                        )
                        LOG.info(
                            "APPROACH after retry dist→center L/R=%.0f/%.0f "
                            "diff=%.0fmm remain L/R=%.0f/%.0f",
                            _dist_center_mm(pl, ctr),
                            _dist_center_mm(pr, ctr),
                            _dist_center_mm(pl, ctr) - _dist_center_mm(pr, ctr),
                            _remain_mm(pl, large_plan.left_approach),
                            _remain_mm(pr, large_plan.right_approach),
                        )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("APPROACH remain check failed: %s", exc)
                result.hover_flange = list(approach.left)
                try:
                    self._wait_still(settle_s=settle_s)
                except Exception:
                    pass
                # 一到接近位先落盘可视化（勿等内收结束；否则中途停/杀 UI 会空 batch）
                if save_vis_dir:
                    self._save_stage_vis(
                        Path(save_vis_dir),
                        head_rect=rect,
                        head_det=head_det,
                        extra_meta={
                            "grasp_mode": grasp_mode,
                            "stage": "approach_reached",
                            "plan": large_plan.as_dict(),
                        },
                    )
                # 已到接近位：网页「内收 5mm」步进；无一次合拢
                if large_confirm is not None:
                    margin_mm = float(self.cfg.large_open_margin_m) * 1000.0
                    large_confirm(
                        {
                            "plan": large_plan.as_dict(),
                            "board_id": int(result.target_board_id),
                            "vis": self._last_head_pose_vis,
                            "message": (
                                f"大板 ID={result.target_board_id} "
                                f"已到接近位（tip 外开各 {margin_mm:.0f}mm，同高）；"
                                f"宽 {large_plan.width_m*1000:.0f}mm。"
                                "点「内收 5mm」逐步夹近；点「结束」退出（无自动合拢）"
                            ),
                        }
                    )
                    if save_vis_dir:
                        self._save_stage_vis(
                            Path(save_vis_dir),
                            head_rect=rect,
                            head_det=head_det,
                            extra_meta={
                                "grasp_mode": grasp_mode,
                                "stage": "approach_inward_done",
                                "plan": large_plan.as_dict(),
                            },
                        )
                    result.debug = {
                        "grasp_mode": grasp_mode,
                        "stage": "approach_inward_done",
                        "live": True,
                        "plan": large_plan.as_dict(),
                    }
                    result.ok = True
                    result.message = "大板已到接近位；内收会话结束（未自动合拢）"
                    return result
            else:
                # 小板：粗定位 → 相机光心对准板心 + 光轴朝下；精定位后再切 (-90,0,rz)
                ee0 = self.motion.get_flange_pose()
                hover = self.flange_above_camera_down(coarse, current_flange=ee0)
                result.coarse_flange = list(hover)
                result.hover_flange = list(hover)
                if plan_only:
                    # 只打印 xyzrpy，不动臂。coarse=悬停目标；fine=示教高抓取目标（需腕精）
                    want_wrist = str(stage) in ("fine", "all")
                    fine_msg = ""
                    if want_wrist:
                        try:
                            ee_plan = self.motion.get_flange_pose()
                            fine, _wdet, _wbgr = self.localize_wrist_multiframe(
                                grab_wrist,
                                ee_xyz_rpy=ee_plan,
                                coarse_boards=result.coarse_boards,
                                target_board_id=result.target_board_id,
                            )
                            result.fine = fine
                            result.fine_boards = list(
                                getattr(self, "_last_fine_boards", []) or []
                            )
                            # 精定位末端：xy + (-90,0,rz) 用腕精；z=标定抓取高+10cm
                            grasp_z_taught = getattr(self.cfg, "grasp_z_m", None)
                            fine_above_m = 0.10
                            gz = (
                                float(ee_plan[2])
                                if grasp_z_taught is None
                                else float(grasp_z_taught) + fine_above_m
                            )
                            fine_rz = float(fine.rz)
                            if result.coarse is not None:
                                fine_rz = nearest_rz_rad(
                                    float(result.coarse.rz), fine_rz
                                )
                            result.fine_flange = [
                                float(fine.x),
                                float(fine.y),
                                float(gz),
                                END_DOWN_RX,
                                END_DOWN_RY,
                                fine_rz,
                            ]
                            fine.rz = fine_rz
                            dx = (float(fine.x) - float(coarse.x)) * 1000.0
                            dy = (float(fine.y) - float(coarse.y)) * 1000.0
                            fine_msg = (
                                f"；腕精 OK Δxy=({dx:.1f},{dy:.1f})mm"
                            )
                            LOG.info(
                                "plan_only WRIST fine %s Δxy=(%.1f,%.1f)mm",
                                format_pose(fine),
                                dx,
                                dy,
                            )
                            _progress("fine")
                        except Exception as exc:  # noqa: BLE001
                            fine_msg = f"；腕部未检出（{exc}）"
                            LOG.info("plan_only wrist fine skip: %s", exc)
                    if save_vis_dir:
                        self._save_stage_vis(
                            Path(save_vis_dir),
                            head_rect=rect,
                            head_det=head_det,
                            extra_meta={
                                "grasp_mode": grasp_mode,
                                "stage": "plan_only",
                                "hover_flange": list(hover),
                                "fine_flange": result.fine_flange,
                                "current_flange": result.current_flange,
                            },
                        )

                    def _pose_csv(p: list[float] | None) -> str | None:
                        """末端 xyzrpy → CLI：mm,deg 逗号串。"""
                        if not p or len(p) < 6:
                            return None
                        return (
                            f"{p[0]*1000:.4f},{p[1]*1000:.4f},{p[2]*1000:.4f},"
                            f"{math.degrees(p[3]):.4f},{math.degrees(p[4]):.4f},"
                            f"{math.degrees(p[5]):.4f}"
                        )

                    arm_a = str(getattr(self.cfg, "tj_arm", "A") or "A").upper()
                    arm_side = "left" if arm_a in ("A", "L", "LEFT") else "right"

                    def _cli(pose_csv: str | None) -> str:
                        if not pose_csv:
                            return "—"
                        return (
                            f"python -m rim_python.move_to_pose "
                            f"--arm {arm_side} --pose {pose_csv} "
                            f"--execute --allow-large"
                        )

                    # current_flange=左，current_flange_right=右（_read_both_tcp）
                    pose_l = result.current_flange
                    pose_r = result.current_flange_right
                    cur_left_csv = _pose_csv(pose_l)
                    cur_right_csv = _pose_csv(pose_r)
                    cur_left_cmd = (
                        f"python -m rim_python.move_to_pose --arm left --pose "
                        f"{cur_left_csv} --execute --allow-large"
                        if cur_left_csv
                        else "—"
                    )
                    cur_right_cmd = (
                        f"python -m rim_python.move_to_pose --arm right --pose "
                        f"{cur_right_csv} --execute --allow-large"
                        if cur_right_csv
                        else "—"
                    )
                    # 粗：相机竖直向下悬停；精：(-90°,0,rz)+示教 Z
                    coarse_csv = _pose_csv(result.hover_flange)
                    fine_csv = _pose_csv(result.fine_flange)
                    cur_cmd = cur_left_cmd if arm_side == "left" else cur_right_cmd
                    coarse_cmd = _cli(coarse_csv)
                    fine_cmd = _cli(fine_csv)
                    result.debug = {
                        "grasp_mode": grasp_mode,
                        "stage": "plan_only",
                        "live": True,
                        "move": False,
                        "arm": arm_side,
                        "current_flange": result.current_flange,
                        "current_flange_right": result.current_flange_right,
                        "hover_flange": list(hover),
                        "fine_flange": result.fine_flange,
                        "wrist_fine": result.fine is not None,
                        "note": {
                            "coarse": "末端=相机竖直向下悬停",
                            "fine": "末端=(-90°,0,rz) · z=标定抓取高+10cm",
                        },
                        "cli": {
                            "current": cur_cmd,
                            "current_left": cur_left_cmd,
                            "current_right": cur_right_cmd,
                            "coarse": coarse_cmd,
                            "fine": fine_cmd,
                        },
                    }
                    result.ok = True
                    result.message = (
                        "未动臂 · 末端 xyzrpy(mm,deg) 可复制命令：\n"
                        f"# 当前左臂\n{cur_left_cmd}\n"
                        f"# 当前右臂\n{cur_right_cmd}\n"
                        f"# 粗定位（相机竖直向下）\n{coarse_cmd}\n"
                        f"# 精定位（-90,0,rz · z=标定抓取高+10cm）\n{fine_cmd}"
                        f"{fine_msg}"
                    )
                    LOG.info("当前左臂: %s", cur_left_cmd)
                    LOG.info("当前右臂: %s", cur_right_cmd)
                    LOG.info("粗定位(相机竖直向下): %s", coarse_cmd)
                    LOG.info("精定位(-90,0,rz z=grasp+10cm): %s", fine_cmd)
                    return result
                # 粗定位接近：不碰夹爪；完整 camera-down 姿态保证光心在板心上方
                ee_now = self.motion.get_flange_pose()
                LOG.info(
                    "PHASE: APPROACH camera-center@board board_id=%d "
                    "board_xy=(%.3f,%.3f) camZ=+%.0fcm "
                    "flange=(%.3f,%.3f,%.3f) rpy_deg=(%.1f,%.1f,%.1f) "
                    "from=(%.3f,%.3f,%.3f)",
                    result.target_board_id,
                    float(coarse.x),
                    float(coarse.y),
                    self.cfg.hover_height_m * 100.0,
                    hover[0],
                    hover[1],
                    hover[2],
                    hover[3] * 180.0 / math.pi,
                    hover[4] * 180.0 / math.pi,
                    hover[5] * 180.0 / math.pi,
                    ee_now[0],
                    ee_now[1],
                    ee_now[2],
                )
                # 短走廊接近：抬高→转 camera-down→直线滑到板心上方（禁止外侧绕圈）
                if hasattr(self.motion, "move_hover"):
                    self.motion.move_hover(
                        hover,
                        velocity_scale=self.cfg.move_velocity_scale,
                        lift_m=0.05,
                    )
                elif hasattr(self.motion, "move_reset"):
                    self.motion.move_reset(
                        hover,
                        velocity_scale=self.cfg.move_velocity_scale,
                        lift_m=0.05,
                    )
                else:
                    self.motion.move_line(
                        hover,
                        velocity_scale=self.cfg.move_velocity_scale,
                        keep_orientation=False,
                        direct=False,
                    )
                ee_arr = self.motion.get_flange_pose()
                d_xy = float(
                    ((ee_arr[0] - hover[0]) ** 2 + (ee_arr[1] - hover[1]) ** 2) ** 0.5
                )
                d_xyz = float(
                    sum((ee_arr[i] - hover[i]) ** 2 for i in range(3)) ** 0.5
                )
                result.hover_flange = list(ee_arr[:6])
                from .geometry import camera_xyz_from_flange

                cam_xyz = camera_xyz_from_flange(
                    ee_arr, T_tool_cam=self.cfg.T_tool_cam
                )
                cam_err_xy = float(
                    math.hypot(cam_xyz[0] - float(coarse.x), cam_xyz[1] - float(coarse.y))
                )
                LOG.info(
                    "approach done |flange_err_xy|=%.0fmm |flange_err_xyz|=%.0fmm "
                    "|cam_center_vs_board_xy|=%.0fmm "
                    "cam_xy=(%.3f,%.3f) board_xy=(%.3f,%.3f)",
                    d_xy * 1000.0,
                    d_xyz * 1000.0,
                    cam_err_xy * 1000.0,
                    cam_xyz[0],
                    cam_xyz[1],
                    float(coarse.x),
                    float(coarse.y),
                )
                if d_xy > 0.08 or cam_err_xy > 0.08:
                    LOG.warning(
                        "悬停未对准板心 flange_err=%.0fmm cam_err=%.0fmm "
                        "ee=(%.3f,%.3f,%.3f) want=(%.3f,%.3f,%.3f)",
                        d_xy * 1000.0,
                        cam_err_xy * 1000.0,
                        ee_arr[0],
                        ee_arr[1],
                        ee_arr[2],
                        hover[0],
                        hover[1],
                        hover[2],
                    )
                    msg = (
                        f"粗悬停未到位：法兰仍偏 {d_xy*1000:.0f}mm "
                        f"(相机偏 {cam_err_xy*1000:.0f}mm)。"
                        f"常见原因：MC 报成功但臂没动（请确认 arm-mode=位置）。"
                        f"当前=({ee_arr[0]:.3f},{ee_arr[1]:.3f}) "
                        f"目标=({hover[0]:.3f},{hover[1]:.3f})"
                    )
                    if stage == "coarse":
                        self._wait_still(settle_s=settle_s)
                        if save_vis_dir:
                            self._save_stage_vis(
                                Path(save_vis_dir), head_rect=rect, head_det=head_det
                            )
                        result.debug = {
                            "grasp_mode": grasp_mode,
                            "stage": "coarse",
                            "hover_err_m": d_xy,
                            "cam_center_err_m": cam_err_xy,
                            "live": True,
                        }
                        result.ok = False
                        result.message = msg
                        return result
                    raise RuntimeError(msg)

            # —— 3. coarse-only 出口：小板停悬停；大板若未要夹也停接近位 ——
            if str(stage) == "coarse" and not (
                grasp_mode == "large" and do_grasp
            ):
                self._wait_still(settle_s=settle_s)
                if save_vis_dir:
                    self._save_stage_vis(
                        Path(save_vis_dir),
                        head_rect=rect,
                        head_det=head_det,
                        extra_meta={
                            "grasp_mode": grasp_mode,
                            "stage": "coarse",
                            "plan": None if large_plan is None else large_plan.as_dict(),
                        },
                    )
                result.debug = {
                    "grasp_mode": grasp_mode,
                    "stage": "coarse",
                    "camera": "head",
                    "n_frames": self.cfg.detect_n_frames,
                    "live": True,
                    "hover_err_m": d_xy,
                    "hover_err_xyz_m": d_xyz,
                    "plan": None if large_plan is None else large_plan.as_dict(),
                }
                result.ok = True
                if grasp_mode == "large":
                    result.message = (
                        f"粗定位 OK：大板 ID={result.target_board_id} 已到接近位"
                        "（未勾选收夹，停在外开）"
                    )
                else:
                    result.message = (
                        f"粗定位 OK：小板 ID={result.target_board_id} "
                        "头定位→camera-down 悬停（已调 move_to_pose）；"
                        "下一步点「精定位」腕精+示教高+收夹"
                    )
                return result

            # —— 4. 抓取（按 mode）——
            squeeze_info = None
            grasp_z_taught = getattr(self.cfg, "grasp_z_m", None)
            lift = float(getattr(self.cfg, "grasp_z_lift_m", 0.0) or 0.0)
            grasp_z = (
                None
                if grasp_z_taught is None
                else float(grasp_z_taught) + lift
            )
            if grasp_z is not None and abs(lift) > 1e-9:
                LOG.info(
                    "grasp_z taught=%.1fmm + lift=%.1fmm → %.1fmm",
                    float(grasp_z_taught) * 1000.0,
                    lift * 1000.0,
                    float(grasp_z) * 1000.0,
                )
            if grasp_mode == "large":
                assert large_plan is not None
                # 大板不再自动合拢；接近后由网页「内收 5mm」步进
                result.fine_flange = list(result.hover_flange or [])
                result.message = "live OK: 大板已接近（合拢改为网页内收步进）"
                if save_vis_dir:
                    self._save_stage_vis(
                        Path(save_vis_dir),
                        head_rect=rect,
                        head_det=head_det,
                        extra_meta={
                            "grasp_mode": grasp_mode,
                            "stage": "after_grasp",
                            "plan": large_plan.as_dict(),
                            "squeeze": squeeze_info,
                        },
                    )
            else:
                LOG.info(
                    "PHASE: wait STILL → WRIST fine (%d frames) target_id=%d",
                    self.cfg.detect_n_frames,
                    result.target_board_id,
                )
                ee = self._wait_still(settle_s=settle_s)
                try:
                    fine, _wdet, _wbgr = self.localize_wrist_multiframe(
                        grab_wrist,
                        ee_xyz_rpy=ee,
                        coarse_boards=result.coarse_boards,
                        target_board_id=result.target_board_id,
                    )
                except Exception:
                    if save_vis_dir is not None:
                        try:
                            out = Path(save_vis_dir)
                            out.mkdir(parents=True, exist_ok=True)
                            bgr, _dep = grab_wrist()
                            cv2.imwrite(str(out / "wrist_fail_color.jpg"), bgr)
                        except Exception:
                            pass
                        self._save_stage_vis(
                            Path(save_vis_dir), head_rect=rect, head_det=head_det
                        )
                    raise
                result.fine = fine
                result.fine_boards = list(getattr(self, "_last_fine_boards", []) or [])
                LOG.info(
                    "FINE wrist board_id=%s %s",
                    fine.board_id,
                    format_pose(fine),
                )
                if save_vis_dir:
                    self._save_stage_vis(
                        Path(save_vis_dir),
                        head_rect=rect,
                        head_det=head_det,
                        extra_meta={
                            "grasp_mode": grasp_mode,
                            "stage": "after_fine_detect",
                            "target_board_id": result.target_board_id,
                            "fine": {
                                "id": fine.board_id,
                                "cls": cls_name(fine.cls_id),
                                "xyz": list(fine.xyz),
                                "rz": float(fine.rz),
                            },
                        },
                    )
                ee = self.motion.get_flange_pose()
                if grasp_z is None:
                    raise RuntimeError(
                        "grasp_z_m 未设置：请先 ./run_live.sh --real --record-grasp-z"
                    )
                # 腕精目标 = 板心 (fine.x/y)。抓取：法兰 XY 对准板心 + (-90°,0,rz)，再降 Z。
                # 不要用「相机朝下法兰 + (fine−coarse)」——手眼偏置会让夹爪偏心。
                rx, ry = END_DOWN_RX, END_DOWN_RY
                rz = float(ee[5])
                if self.cfg.fine_apply_rz:
                    fine_rz = float(fine.rz)
                    if result.coarse is not None:
                        fine_rz = nearest_rz_rad(float(result.coarse.rz), fine_rz)
                    tmp = compute_flange_target(
                        [float(fine.x), float(fine.y), float(ee[2])],
                        fine_rz,
                        tcp_offset=[0.0, 0.0, 0.0],
                        tool_rz_offset_rad=self.cfg.tool_rz_offset_rad,
                    )
                    rz = float(tmp[5])
                    fine.rz = fine_rz
                # tcp_offset=0：末端 XY = 板心
                orient = [
                    float(fine.x),
                    float(fine.y),
                    float(ee[2]),
                    rx,
                    ry,
                    rz,
                ]
                descend = [
                    orient[0],
                    orient[1],
                    float(grasp_z),
                    rx,
                    ry,
                    rz,
                ]
                dxy_x = float(fine.x) - float(ee[0])
                dxy_y = float(fine.y) - float(ee[1])
                result.fine_flange = list(descend)
                result.debug = {
                    **(result.debug or {}),
                    "grasp_mode": grasp_mode,
                    "stage": "before_grasp_move",
                    "board_center_xy_m": [round(float(fine.x), 4), round(float(fine.y), 4)],
                    "flange_to_center_dxy_mm": [
                        round(dxy_x * 1000.0, 1),
                        round(dxy_y * 1000.0, 1),
                    ],
                    "grasp_rpy_deg": [-90.0, 0.0, round(rz * 180.0 / math.pi, 1)],
                    "live": True,
                }
                _progress("fine")
                LOG.info(
                    "PHASE: FINE→GRASP board-center=(%.4f,%.4f) "
                    "end-down=(-90,0,rz=%.1f°) Δflange_xy=(%.1f,%.1f)mm "
                    "→ descend Z=%.1fmm → %s",
                    float(fine.x),
                    float(fine.y),
                    rz * 180.0 / math.pi,
                    dxy_x * 1000.0,
                    dxy_y * 1000.0,
                    float(grasp_z) * 1000.0,
                    "CLOSE gripper" if do_grasp else "hold",
                )
                # 降之前先张开，到位后再收夹（粗定位阶段不动夹爪）
                if do_grasp:
                    LOG.info("PHASE: GRIPPER OPEN before descend")
                    self.motion.set_gripper(True)
                self._move_tool_down_pose(
                    orient,
                    tag="FINE_ORIENT(center XY + end-down -90,0,rz)",
                    direct=True,
                    force_tip_down=True,
                )
                self._move_tool_down_pose(
                    descend,
                    tag=f"GRASP_Z(center Z={float(grasp_z)*1000:.1f}mm keep -90,0,rz)",
                    direct=True,
                    force_tip_down=True,
                )
                self._wait_still(settle_s=settle_s)
                if do_grasp:
                    LOG.info("PHASE: GRIPPER CLOSE pos=%.3f", float(self.cfg.gripper_grasp_position))
                    self.motion.set_gripper(False)
                    LOG.info("GRASP gripper closed")
                result.message = (
                    f"live OK: 抓板心 ({fine.x:.3f},{fine.y:.3f}) "
                    f"(-90,0,rz) → 降 Z={float(grasp_z)*1000:.1f}mm"
                    + (" → 收夹" if do_grasp else " → 未收夹")
                )

            result.ok = True
            result.debug = {
                "grasp_mode": grasp_mode,
                "target_board_id": result.target_board_id,
                "n_coarse_boards": len(result.coarse_boards),
                "n_fine_boards": len(result.fine_boards),
                "do_grasp": bool(do_grasp),
                "stage": "all",
                "live": True,
                "hover_height_m": self.cfg.hover_height_m,
                "grasp_z_m": None if grasp_z is None else float(grasp_z),
                "fine_use_shape_match": bool(self.cfg.fine_use_shape_match),
                "plan": None if large_plan is None else large_plan.as_dict(),
                "squeeze": squeeze_info,
            }
            return result
        except Exception as e:
            result.ok = False
            result.message = str(e)
            result.debug = {
                **(result.debug or {}),
                "grasp_mode": grasp_mode,
                "live": True,
            }
            LOG.exception("Live grasp pipeline failed: %s", e)
            if save_vis_dir is not None:
                try:
                    self._save_stage_vis(
                        Path(save_vis_dir), head_rect=rect, head_det=head_det
                    )
                except Exception:
                    pass
            return result
