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
    detect_with_flip_fallback,
    draw_grasp_points_on_head,
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
from .localize import Intrinsics, format_pose, head_localize, sample_depth_m, wrist_localize
from .motion import DryRunMotion, MotionBackend
from .robust import (
    median_detection,
    median_object_pose,
    pick_closest_to_mean_pose,
    pick_closest_to_mean_xyz,
    pose_keep_mask,
)
from .undistort import (
    HeadRectifyMaps,
    T_base_from_rectified,
    build_maps_from_stereo_json,
    load_maps_from_yml,
)

LOG = logging.getLogger("pcb_grasp.pipeline")


def _announce_targets(title: str, named: Sequence[tuple[str, Sequence[float]]]) -> None:
    """把阶段目标打到日志和终端。位姿为米/弧度，打印成毫米/度。"""
    lines = [title]
    for name, pose in named:
        p = [float(x) for x in pose]
        if len(p) >= 6:
            lines.append(
                f"  {name}  xyz_mm=({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, {p[2] * 1000:.1f})"
                f"  rpy_deg=({math.degrees(p[3]):.1f}, {math.degrees(p[4]):.1f}, {math.degrees(p[5]):.1f})"
            )
        else:
            lines.append(
                f"  {name}  xyz_mm=({p[0] * 1000:.1f}, {p[1] * 1000:.1f}, {p[2] * 1000:.1f})"
            )
    text = "\n".join(lines)
    LOG.info("\n%s", text)
    print(text, flush=True)


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
        # 校正左目 → 基座。手眼是 6 号原相机，这里已乘上 R1.T。
        self._T_base_head: np.ndarray | None = None
        self._head_det: YoloPoseDetector | None = None
        self._wrist_det: YoloPoseDetector | None = None
        self._large_wrist_det: YoloPoseDetector | None = None
        self._large_single_coarse_det: YoloPoseDetector | None = None
        self._large_single_fine_det: YoloPoseDetector | None = None
        self._wrist_K = Intrinsics(
            self.cfg.wrist_fx, self.cfg.wrist_fy, self.cfg.wrist_cx, self.cfg.wrist_cy
        )
        # 右腕内参（大板单臂 cam3）；缺省与左腕相同
        _rfx = getattr(self.cfg, "wrist_fx_right", None)
        if _rfx is not None:
            self._wrist_K_right = Intrinsics(
                float(_rfx),
                float(getattr(self.cfg, "wrist_fy_right", _rfx)),
                float(getattr(self.cfg, "wrist_cx_right", self.cfg.wrist_cx)),
                float(getattr(self.cfg, "wrist_cy_right", self.cfg.wrist_cy)),
            )
        else:
            self._wrist_K_right = self._wrist_K
        self._stereo_depth = None
        self._shape_model = None
        self._last_head_pose_vis: np.ndarray | None = None
        self._last_wrist_pose_vis: np.ndarray | None = None
        self._last_wrist_fine_left_vis: np.ndarray | None = None
        self._last_wrist_fine_right_vis: np.ndarray | None = None
        self._last_wrist_match_vis: np.ndarray | None = None
        self._last_head_rect: np.ndarray | None = None
        self._last_head_raw_l: np.ndarray | None = None
        self._last_head_raw_r: np.ndarray | None = None
        self._last_head_depth_u16: np.ndarray | None = None
        self._last_head_depth_stats: dict | None = None
        self._last_wrist_bgr: np.ndarray | None = None
        self._last_wrist_depth: np.ndarray | None = None
        # 大板单臂：粗定位腕图 + 当时 T_base_cam（可视化叠在这张图上）
        self._last_wrist_coarse_bgr: np.ndarray | None = None
        self._last_wrist_coarse_T_base_cam: np.ndarray | None = None
        # 粗/精预测标签缓存（落盘供人工标注对照）
        self._last_coarse_dets: list[PcbDetection] = []
        self._last_coarse_boards: list[ObjectPose] = []
        self._last_fine_dets: list[PcbDetection] = []
        self._last_fine_boards: list[ObjectPose] = []
        self._last_target_board_id: int | None = None

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
        # 检测图画在 7 号的 CAM0 校正图上，内参是 K_rect。
        # 手眼是 7 号物理相机：p_cam7 = R1.T @ p_rect。
        self._head_K = Intrinsics.from_K(self._head_maps.K_rect)
        self._T_base_head = T_base_from_rectified(
            self.cfg.T_base_cam, self._head_maps.R1
        )
        LOG.info(
            "head stereo left=cam7 CAM0  K_rect fx=%.2f fy=%.2f cx=%.2f cy=%.2f",
            self._head_K.fx,
            self._head_K.fy,
            self._head_K.cx,
            self._head_K.cy,
        )
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
            # 小板腕精：models.wrist_pose_weights（与大板角点模分开）
            w = self.cfg.wrist_pose_weights or self.cfg.head_pose_weights
            if not w or not Path(w).is_file():
                LOG.warning(
                    "SMALL wrist weights missing (%s) → 跳过加载小板腕模",
                    w,
                )
                return
            self._wrist_det = YoloPoseDetector(
                w,
                conf=self.cfg.wrist_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=int(getattr(self.cfg, "wrist_imgsz", self.cfg.imgsz)),
            )
            LOG.info("SMALL wrist fine detector ← %s", w)

    def _ensure_large_wrist_det(self) -> YoloPoseDetector | None:
        """大板腕部角点精定位专用 YOLO（corner · 单角）；未配则返回 None。"""
        path = str(getattr(self.cfg, "large_wrist_pose_weights", "") or "").strip()
        if not path:
            return None
        if self._large_wrist_det is None:
            imgsz = int(
                getattr(self.cfg, "large_wrist_imgsz", 0)
                or getattr(self.cfg, "wrist_imgsz", self.cfg.imgsz)
            )
            n_corners = int(getattr(self.cfg, "large_wrist_n_corners", 1) or 1)
            self._large_wrist_det = YoloPoseDetector(
                path,
                conf=self.cfg.wrist_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=imgsz,
                n_corners=n_corners,
            )
            LOG.info(
                "LARGE wrist corner detector ← %s (imgsz=%d n_corners=%d)",
                path,
                imgsz,
                n_corners,
            )
        return self._large_wrist_det

    def _ensure_large_single_coarse_det(self) -> YoloPoseDetector | None:
        """单臂粗：pcb-large-1 · 4 角。"""
        path = str(
            getattr(self.cfg, "large_single_coarse_pose_weights", "") or ""
        ).strip()
        if not path:
            return None
        if self._large_single_coarse_det is None:
            imgsz = int(
                getattr(self.cfg, "large_single_coarse_imgsz", 0) or 480
            )
            n_corners = int(
                getattr(self.cfg, "large_single_coarse_n_corners", 0) or 4
            )
            self._large_single_coarse_det = YoloPoseDetector(
                path,
                conf=self.cfg.wrist_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=imgsz,
                n_corners=n_corners,
            )
            LOG.info(
                "LARGE single coarse det ← %s (imgsz=%d n_corners=%d)",
                path,
                imgsz,
                n_corners,
            )
        return self._large_single_coarse_det

    def _ensure_large_single_fine_det(self) -> YoloPoseDetector | None:
        """单臂精：corner-1 · 单角。"""
        path = str(
            getattr(self.cfg, "large_single_fine_pose_weights", "") or ""
        ).strip()
        if not path:
            # 回落大板腕精模
            return self._ensure_large_wrist_det()
        if self._large_single_fine_det is None:
            imgsz = int(getattr(self.cfg, "large_single_fine_imgsz", 0) or 480)
            n_corners = int(
                getattr(self.cfg, "large_single_fine_n_corners", 0) or 1
            )
            self._large_single_fine_det = YoloPoseDetector(
                path,
                conf=self.cfg.wrist_conf,
                kpt_conf=self.cfg.kpt_conf,
                device=self.cfg.device,
                imgsz=imgsz,
                n_corners=n_corners,
            )
            LOG.info(
                "LARGE single fine det ← %s (imgsz=%d n_corners=%d)",
                path,
                imgsz,
                n_corners,
            )
        return self._large_single_fine_det

    # --------------------------------------------------------------- perception
    def undistort_head(self, fisheye_bgr: np.ndarray) -> np.ndarray:
        maps = self._ensure_head_maps()
        return maps.undistort_left(fisheye_bgr)

    def _plane_z_for_cls(self, cls_id: int | None) -> float:
        """头定位板面 Z：大板默认走 FFS（不锁平面）；仅显式固定平面时才用 grasp_z/托盘。"""
        large_cls = int(self.cfg.cls_pcb_large)
        use_fixed = bool(getattr(self.cfg, "large_use_fixed_grasp_z_plane", False))
        if use_fixed and cls_id is not None and int(cls_id) == large_cls:
            gz = getattr(self.cfg, "large_grasp_z_m", None)
            if gz is not None:
                return float(gz)
            LOG.warning(
                "large grasp_z_m 未设 → 固定平面暂用 head_tray_z_m=%.4f",
                float(self.cfg.head_tray_z_m),
            )
        return float(self.cfg.head_tray_z_m)

    def _head_localize_det(
        self,
        det: PcbDetection,
        *,
        depth: np.ndarray | None = None,
        tray_z_m: float | None = None,
        depth_m_per_corner: Sequence[float] | None = None,
    ) -> ObjectPose:
        """单板头定位。角点深度用校正左目（cam7）上的 FFS。"""
        assert self._head_K is not None
        T_head = self._T_base_head if self._T_base_head is not None else self.cfg.T_base_cam
        cls_id = int(getattr(det, "cls_id", 0) or 0)
        large_cls = int(self.cfg.cls_pcb_large)
        is_large = cls_id == large_cls
        use_fixed_large = bool(
            getattr(self.cfg, "large_use_fixed_grasp_z_plane", False)
        ) and is_large
        if tray_z_m is not None or use_fixed_large:
            z_plane = (
                float(tray_z_m)
                if tray_z_m is not None
                else self._plane_z_for_cls(cls_id)
            )
            pose = head_localize(
                det,
                T_base_cam=T_head,
                K=self._head_K,
                depth=None,
                depth_scale=0.001,
                fixed_depth_cam_m=None,
                tray_z_m=z_plane,
                kp_i=self.cfg.kp_i,
                kp_j=self.cfg.kp_j,
            )
        else:
            # 大板：只用 FFS 角点深度反投影 XY/Z，禁止固定深度/托盘平面
            if is_large and depth is None and depth_m_per_corner is None:
                raise RuntimeError(
                    "大板头定位需要 FFS 角点深度，禁止固定深度/托盘平面"
                )
            pose = head_localize(
                det,
                T_base_cam=T_head,
                K=self._head_K,
                depth=depth,
                depth_scale=0.001,
                fixed_depth_cam_m=(
                    None
                    if (is_large or depth is not None or depth_m_per_corner is not None)
                    else self.cfg.head_fixed_depth_cam_m
                ),
                tray_z_m=None if is_large else self._plane_z_for_cls(cls_id),
                kp_i=self.cfg.kp_i,
                kp_j=self.cfg.kp_j,
                depth_m_per_corner=depth_m_per_corner,
                corner_z_max=1.0,
                required_corner_indices=(2, 3) if is_large else None,
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
                rect = fisheye_bgr
                right_rect = right_fisheye_bgr
                depth, _valid, depth_stats = est.estimate_rectified(
                    rect, right_fisheye_bgr
                )
            else:
                rect, _right_rect, depth, _valid, depth_stats = est.estimate_raw(
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

    def refine_large_board_p0p3(
        self,
        coarse: ObjectPose,
        grab_wrist,
        *,
        grasp_z_m: float,
        plan_only: bool = False,
        grab_wrist_right=None,
        save_vis_dir: str | Path | None = None,
    ) -> ObjectPose:
        """大板精定位（P2/P3）：粗到位拍腕 → 精 XY → 接近加 dx/dy。

        - 粗定位到达：末端在角点 XY、Z=grasp_z+fine_hover（左右统一），姿态垂直向下
        - 腕精：深度=当前相机光心到写死 grasp_z 的光轴距离，反投影得精 XY
        - 夹取：tip = 精角点 XY + (dx, ±dy)，tipZ = grasp_z（写死）
        - ``plan_only``：不动臂；当前腕图能认出角点则精修并产出 tip，认不出才跳过
        """
        from .geometry import ObjectPose as _OP, pose_xyz_rpy_to_mat4
        from .large_board import gripper_rz_large_parallel_p2p3, tip_to_flange
        from .localize import cam_depth_to_plane_m, corner_uv_to_base

        self._large_wrist_fine_ok = False
        if not bool(getattr(self.cfg, "large_fine_enable", True)):
            LOG.info("LARGE fine disabled → keep coarse P2/P3（不产出 tip）")
            return coarse
        det_model = self._ensure_large_wrist_det()
        if det_model is None:
            LOG.warning(
                "LARGE fine: large_board.wrist_pose_weights 未配置 → 跳过精定位，"
                "不产出 tip"
            )
            return coarse
        if coarse.corners_base is None or len(coarse.corners_base) < 4:
            LOG.warning("LARGE fine: coarse 无四角 → skip")
            return coarse

        hover_h = float(
            getattr(self.cfg, "large_fine_hover_height_m", 0.35) or 0.35
        )
        max_d = float(getattr(self.cfg, "large_fine_match_max_dist_m", 0.15))
        dx = float(getattr(self.cfg, "large_approach_dx_m", 0.15) or 0.15)
        dy = float(getattr(self.cfg, "large_approach_dy_m", 0.08) or 0.08)
        tool_y = float(getattr(self.cfg, "large_tool_y_m", 0.0) or 0.0)
        tool_z = float(getattr(self.cfg, "large_tool_z_m", 0.0) or 0.0)
        coarse_boards = list(getattr(self, "_last_coarse_boards", []) or [coarse])
        tid = coarse.board_id
        p2_c = coarse.corners_base[2]
        p3_c = coarse.corners_base[3]
        grasp_z_fixed = getattr(self.cfg, "large_grasp_z_m", None)
        if grasp_z_fixed is None:
            raise RuntimeError("large_board.grasp_z_m 未写死")
        z_board = float(grasp_z_fixed)
        # 拍腕高度统一：grasp_z + hover（不再跟 FFS 角点 Z，避免左右差十几毫米）
        z_arrive = z_board + hover_h
        z_p3_arrive = z_arrive
        z_p2_arrive = z_arrive
        hover_p3 = [float(p3_c[0]), float(p3_c[1])]
        hover_p2 = [float(p2_c[0]), float(p2_c[1])]
        grip_rz = gripper_rz_large_parallel_p2p3(
            p2_c,
            p3_c,
            orient_rz_offset_deg=float(
                getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
            ),
        )
        # tip_to_flange 外开方向（tool_y=0 时不影响 xyz）
        _ux = float(hover_p2[0] - hover_p3[0])
        _uy = float(hover_p2[1] - hover_p3[1])
        _n = math.hypot(_ux, _uy) or 1.0
        grasp_ux, grasp_uy = _ux / _n, _uy / _n
        self._last_wrist_fine_left_vis = None
        self._last_wrist_fine_right_vis = None
        LOG.info(
            "LARGE: 粗到位 XY=FFS角点 Z=grasp_z+%.3f=%.4f（统一）→ "
            "P3=(%.4f,%.4f,%.4f) P2=(%.4f,%.4f,%.4f)；"
            "腕精深度=相机到写死Z %.3f；夹取 tipZ=%.4f "
            "后加 dx=%.3f dy=%.3f board_id=%s",
            hover_h,
            z_arrive,
            hover_p3[0],
            hover_p3[1],
            z_p3_arrive,
            hover_p2[0],
            hover_p2[1],
            z_p2_arrive,
            z_board,
            z_board,
            dx,
            dy,
            tid,
        )

        T_left = np.asarray(self.cfg.T_tool_cam, dtype=np.float64)
        T_right = getattr(self.cfg, "T_tool_cam_right", None)
        if T_right is None:
            T_right = T_left
            LOG.warning(
                "LARGE fine: T_tool_cam_right 未配置 → 右腕暂用左腕手眼（请尽快标定写入）"
            )
        else:
            T_right = np.asarray(T_right, dtype=np.float64)

        dual = self._ensure_dual()
        right_motion = getattr(dual, "right", None)
        left_motion = getattr(dual, "left", None) or self.motion
        # 拍完不回退；后续 run_live 去 tip（角点 x+dx / y±dy）
        coarse_fl_l = None
        coarse_fl_r = None

        # 右腕取流：优先回调，否则临时开 cam3
        right_cam = None
        grab_r = grab_wrist_right

        def _ensure_grab_right():
            nonlocal right_cam, grab_r
            if grab_r is not None:
                return grab_r
            from .cameras import WristPortalCam

            portal = (
                getattr(self.cfg, "rim_frame_bundle", None)
                or getattr(self.cfg, "cam_portal", "10.3.8.7:50066")
            )
            dist = getattr(self.cfg, "wrist_dist_coeffs", None)
            right_id = int(getattr(self.cfg, "cam_portal_wrist_right_id", 3))
            right_cam = WristPortalCam(
                portal,
                cam_id=right_id,
                max_hz=float(getattr(self.cfg, "cam_portal_max_hz", 15.0)),
                timeout_s=float(getattr(self.cfg, "cam_portal_timeout_s", 60.0)),
                need_depth=False,
                fx=float(getattr(self.cfg, "wrist_fx", 615.0)),
                fy=float(getattr(self.cfg, "wrist_fy", 615.0)),
                cx=float(getattr(self.cfg, "wrist_cx", 630.0)),
                cy=float(getattr(self.cfg, "wrist_cy", 401.0)),
                dist_coeffs=list(dist) if dist else None,
                undistort=bool(getattr(self.cfg, "cam_wrist_undistort", False)),
                depth_cam_id=int(getattr(self.cfg, "cam_portal_wrist_right_depth_id", 2)),
            )
            grab_r = right_cam.grab
            LOG.info("LARGE fine: opened right wrist cam%d @ %s", right_id, portal)
            return grab_r

        # —— 只定位：不动臂，但若当前腕图能认出角点则精修并产出 tip ——
        if plan_only:
            from .detect import draw_pose_vis
            from .localize import cam_depth_to_plane_m, corner_uv_to_base

            def _detect_stationary(
                *,
                side: str,
                corner_idx: int,
                corner_xy: Sequence[float],
                grab_fn,
                motion_backend,
                T_tc: np.ndarray,
            ) -> tuple[list[float] | None, PcbDetection | None, np.ndarray | None]:
                if grab_fn is None or motion_backend is None:
                    return None, None, None
                if self._wrist_K is None:
                    return None, None, None
                try:
                    ee = list(motion_backend.get_flange_pose()[:6])
                    wrist_bgr, _wd = grab_fn()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("LARGE plan_only %s grab/TCP fail: %s", side, exc)
                    return None, None, None
                if side.startswith("l"):
                    self._last_wrist_bgr = wrist_bgr
                T_base_cam = pose_xyz_rpy_to_mat4(*[float(x) for x in ee[:6]]) @ T_tc
                try:
                    cam_depth = cam_depth_to_plane_m(T_base_cam, z_board)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("LARGE plan_only %s cam_depth fail: %s", side, exc)
                    return None, None, wrist_bgr
                dets = det_model.detect(wrist_bgr)
                if not dets:
                    LOG.info("LARGE plan_only %s: 腕图未检出角点", side)
                    if save_vis_dir is not None:
                        try:
                            import cv2

                            out_w = Path(save_vis_dir) / "wrist"
                            out_w.mkdir(parents=True, exist_ok=True)
                            vis = wrist_bgr.copy()
                            cv2.putText(
                                vis,
                                f"LARGE plan_only {side} P{corner_idx} NO DET",
                                (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                            name = (
                                "pose_left.jpg"
                                if side.startswith("l")
                                else "pose_right.jpg"
                            )
                            cv2.imwrite(str(out_w / name), vis)
                            if side.startswith("l"):
                                self._last_wrist_fine_left_vis = vis
                            else:
                                self._last_wrist_fine_right_vis = vis
                            self._last_wrist_pose_vis = vis
                        except Exception:
                            pass
                    return None, None, wrist_bgr
                best_d = 1e9
                best_xyz: list[float] | None = None
                best_det: PcbDetection | None = None
                for d in dets:
                    if not d.corners:
                        continue
                    u, v = d.corners[0]
                    try:
                        xyz = corner_uv_to_base(
                            float(u),
                            float(v),
                            T_base_cam=T_base_cam,
                            K=self._wrist_K,
                            fixed_depth_cam_m=float(cam_depth),
                        )
                    except Exception:
                        continue
                    dxy = math.hypot(
                        float(xyz[0]) - float(corner_xy[0]),
                        float(xyz[1]) - float(corner_xy[1]),
                    )
                    if dxy < best_d:
                        best_d = dxy
                        best_xyz = [
                            float(xyz[0]),
                            float(xyz[1]),
                            float(z_board),
                        ]
                        best_det = d
                if best_xyz is None or best_d > max_d:
                    LOG.info(
                        "LARGE plan_only %s: 检出但 Δcoarse=%.1fmm > max_d → 不用",
                        side,
                        best_d * 1000.0 if best_xyz is not None else -1.0,
                    )
                    return None, best_det, wrist_bgr
                LOG.info(
                    "LARGE plan_only %s OK P%d=(%.4f,%.4f) Δcoarse=%.1fmm conf=%.2f",
                    side,
                    corner_idx,
                    best_xyz[0],
                    best_xyz[1],
                    best_d * 1000.0,
                    float(getattr(best_det, "conf", 0.0) or 0.0),
                )
                try:
                    vis = draw_pose_vis(
                        wrist_bgr,
                        best_det,
                        title=(
                            f"LARGE plan_only {side} P{corner_idx} "
                            f"Δ={best_d*1000:.0f}mm"
                        ),
                    )
                    self._last_wrist_pose_vis = vis
                    if side.startswith("l"):
                        self._last_wrist_fine_left_vis = vis
                    else:
                        self._last_wrist_fine_right_vis = vis
                    if save_vis_dir is not None:
                        import cv2

                        out_w = Path(save_vis_dir) / "wrist"
                        out_w.mkdir(parents=True, exist_ok=True)
                        name = (
                            "pose_left.jpg"
                            if side.startswith("l")
                            else "pose_right.jpg"
                        )
                        cv2.imwrite(str(out_w / name), vis)
                        cv2.imwrite(str(out_w / "color.jpg"), wrist_bgr)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("LARGE plan_only %s vis fail: %s", side, exc)
                return best_xyz, best_det, wrist_bgr

            LOG.info(
                "LARGE fine plan_only: 不动臂，用当前腕图试认 P2/P3 "
                "(能认出则产出 tip)"
            )
            fine_p3, _d3, _b3 = _detect_stationary(
                side="left",
                corner_idx=3,
                corner_xy=p3_c,
                grab_fn=grab_wrist,
                motion_backend=left_motion or self.motion,
                T_tc=T_left,
            )
            fine_p2 = None
            r_motion = right_motion
            use_right = (
                r_motion is not None
                and not bool(getattr(r_motion, "dry_run", False))
            )
            if use_right:
                try:
                    grab_fn_r = _ensure_grab_right()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("LARGE plan_only: 开右腕失败: %s", exc)
                    grab_fn_r = None
                if grab_fn_r is not None:
                    fine_p2, _d2, _b2 = _detect_stationary(
                        side="right",
                        corner_idx=2,
                        corner_xy=p2_c,
                        grab_fn=grab_fn_r,
                        motion_backend=r_motion,
                        T_tc=T_right,
                    )
            if fine_p2 is None:
                # 未连右臂 / 右腕未认出：同一左腕图再匹配 P2（视野里有则用）
                fine_p2, _d2, _b2 = _detect_stationary(
                    side="left",
                    corner_idx=2,
                    corner_xy=p2_c,
                    grab_fn=grab_wrist,
                    motion_backend=left_motion or self.motion,
                    T_tc=T_left,
                )
            if right_cam is not None:
                try:
                    right_cam.close()
                except Exception:
                    pass
            if fine_p3 is None and fine_p2 is None:
                LOG.info("LARGE plan_only: 腕部未认出角点 → 不产出 tip")
                return coarse
            new_corners = [list(c) for c in coarse.corners_base]
            if fine_p3 is not None:
                new_corners[3] = list(fine_p3)
            if fine_p2 is not None:
                new_corners[2] = list(fine_p2)
            new_corners[2][2] = z_board
            new_corners[3][2] = z_board
            z_board_out = 0.5 * (
                float(new_corners[2][2]) + float(new_corners[3][2])
            )
            rz = gripper_rz_large_parallel_p2p3(
                new_corners[2],
                new_corners[3],
                orient_rz_offset_deg=float(
                    getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
                ),
            )
            mid_x = 0.5 * (float(new_corners[2][0]) + float(new_corners[3][0]))
            mid_y = 0.5 * (float(new_corners[2][1]) + float(new_corners[3][1]))
            refined = _OP(
                x=mid_x,
                y=mid_y,
                z=float(z_board_out),
                rz=float(rz),
                edge_yaw=math.atan2(
                    float(new_corners[3][1]) - float(new_corners[2][1]),
                    float(new_corners[3][0]) - float(new_corners[2][0]),
                ),
                corners_base=new_corners,
                conf=float(coarse.conf),
                board_id=coarse.board_id,
                cls_id=int(self.cfg.cls_pcb_large),
            )
            LOG.info(
                "LARGE plan_only fine OK id=%s mid23=(%.4f,%.4f) "
                "P2%s P3%s",
                refined.board_id,
                refined.x,
                refined.y,
                "=腕" if fine_p2 is not None else "=粗",
                "=腕" if fine_p3 is not None else "=粗",
            )
            self._large_wrist_fine_ok = True
            return refined

        def _pick_corner(
            *,
            side: str,
            corner_idx: int,
            hover_xy: Sequence[float],
            corner_xy: Sequence[float],
            z_arrive: float,
            grab_fn,
            motion_backend,
            T_tc: np.ndarray,
            home_flange: Sequence[float] | None,
        ) -> tuple[list[float] | None, PcbDetection | None, np.ndarray | None]:
            """粗定位到达：末端在角点 XY、Z=grasp_z+hover（左右统一），夹爪垂直向下。"""
            hover = tip_to_flange(
                [float(hover_xy[0]), float(hover_xy[1]), float(z_arrive)],
                gripper_rz=float(grip_rz),
                grasp_ux=grasp_ux,
                grasp_uy=grasp_uy,
                side=side,
                tool_y_m=0.0,
                tool_z_m=0.0,
            )
            LOG.info(
                "PHASE: LARGE 粗定位到达 %s P%d tipXYZ=(%.3f,%.3f,%.3f) "
                "rpy_deg=(%.1f,%.1f,%.1f) (末端垂直向下 · grasp_z+%.2f)",
                side,
                corner_idx,
                float(hover[0]),
                float(hover[1]),
                float(hover[2]),
                float(hover[3]) * 180.0 / math.pi,
                float(hover[4]) * 180.0 / math.pi,
                float(hover[5]) * 180.0 / math.pi,
                hover_h,
            )
            motion_backend.move_line(
                hover, velocity_scale=float(self.cfg.move_velocity_scale)
            )
            try:
                self._wait_still(settle_s=0.3)
            except Exception:
                pass

            corner_xyz: list[float] | None = None
            used_det: PcbDetection | None = None
            bgr_out: np.ndarray | None = None
            n_f = max(1, int(self.cfg.detect_n_frames))
            gap_f = float(self.cfg.detect_frame_gap_s)
            mad_k = float(self.cfg.detect_outlier_mad_k)
            try:
                import time as _time

                corner_samples: list[list[float]] = []
                det_samples: list[PcbDetection] = []
                bgr_samples: list[np.ndarray] = []
                for fi in range(n_f):
                    wrist_bgr, wrist_depth = grab_fn()
                    if side.startswith("l"):
                        self._last_wrist_bgr = wrist_bgr
                        self._last_wrist_depth = wrist_depth
                    ee = list(motion_backend.get_flange_pose()[:6])
                    dets = det_model.detect(wrist_bgr)
                    if not dets:
                        LOG.warning(
                            "LARGE fine %s frame %d/%d: 未检出",
                            side,
                            fi + 1,
                            n_f,
                        )
                    else:
                        T_base_tool = pose_xyz_rpy_to_mat4(
                            *[float(x) for x in ee[:6]]
                        )
                        T_base_cam = T_base_tool @ T_tc
                        cam_depth = cam_depth_to_plane_m(T_base_cam, z_board)
                        LOG.info(
                            "LARGE fine %s frame %d camZ=%.3f depth_to_z%.3f=%.3f "
                            "n_det=%d",
                            side,
                            fi + 1,
                            float(T_base_cam[2, 3]),
                            z_board,
                            cam_depth,
                            len(dets),
                        )
                        # V2·220e76：每检出=一个角点；取距粗 P2/P3 最近且 ≤max_d
                        best_d = 1e9
                        best_xyz: list[float] | None = None
                        best_det: PcbDetection | None = None
                        for d in dets:
                            if not d.corners:
                                continue
                            u, v = d.corners[0]
                            try:
                                xyz = corner_uv_to_base(
                                    float(u),
                                    float(v),
                                    T_base_cam=T_base_cam,
                                    K=self._wrist_K,
                                    fixed_depth_cam_m=float(cam_depth),
                                )
                            except Exception as exc:  # noqa: BLE001
                                LOG.warning(
                                    "LARGE fine %s lift fail: %s", side, exc
                                )
                                continue
                            dxy = math.hypot(
                                float(xyz[0]) - float(corner_xy[0]),
                                float(xyz[1]) - float(corner_xy[1]),
                            )
                            if dxy < best_d:
                                best_d = dxy
                                best_xyz = [
                                    float(xyz[0]),
                                    float(xyz[1]),
                                    float(z_board),
                                ]
                                best_det = d
                        if best_xyz is not None and best_d <= max_d:
                            corner_samples.append(best_xyz)
                            det_samples.append(best_det)  # type: ignore[arg-type]
                            bgr_samples.append(wrist_bgr)
                            if side.startswith("l"):
                                self._last_fine_dets = list(dets)
                                self._last_fine_boards = []
                            LOG.info(
                                "LARGE fine %s frame %d/%d: P%d cand "
                                "Δcoarse=%.1fmm conf=%.2f",
                                side,
                                fi + 1,
                                n_f,
                                corner_idx,
                                best_d * 1000.0,
                                float(getattr(best_det, "conf", 0.0) or 0.0),
                            )
                        elif best_xyz is not None:
                            LOG.warning(
                                "LARGE fine %s frame %d/%d: 最近角点 "
                                "Δ=%.1fmm > max_d=%.0fmm → 丢弃",
                                side,
                                fi + 1,
                                n_f,
                                best_d * 1000.0,
                                max_d * 1000.0,
                            )
                    if gap_f > 0 and fi + 1 < n_f:
                        _time.sleep(gap_f)

                if not corner_samples:
                    LOG.warning(
                        "LARGE fine %s: %d 帧均无有效 P%d", side, n_f, corner_idx
                    )
                    # 无检出也落盘最后一帧，方便看高度/视野
                    if save_vis_dir is not None and wrist_bgr is not None:
                        try:
                            import cv2

                            out_w = Path(save_vis_dir) / "wrist"
                            out_w.mkdir(parents=True, exist_ok=True)
                            vis = wrist_bgr.copy()
                            cv2.putText(
                                vis,
                                f"LARGE {side} fine P{corner_idx} NO DET",
                                (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                            )
                            name = (
                                "pose_left.jpg"
                                if side.startswith("l")
                                else "pose_right.jpg"
                            )
                            cv2.imwrite(str(out_w / name), vis)
                            cv2.imwrite(str(out_w / "color.jpg"), wrist_bgr)
                            if side.startswith("l"):
                                self._last_wrist_fine_left_vis = vis
                            else:
                                self._last_wrist_fine_right_vis = vis
                            self._last_wrist_pose_vis = vis
                            LOG.info("LARGE fine saved %s/%s (no det)", out_w, name)
                        except Exception as exc:  # noqa: BLE001
                            LOG.warning(
                                "LARGE fine %s save no-det vis failed: %s", side, exc
                            )
                else:
                    corner_xyz, pi, _km = pick_closest_to_mean_xyz(
                        corner_samples, mad_k=mad_k
                    )
                    used_det = det_samples[pi]
                    bgr_out = bgr_samples[pi]
                    d_c = math.hypot(
                        corner_xyz[0] - float(corner_xy[0]),
                        corner_xyz[1] - float(corner_xy[1]),
                    )
                    LOG.info(
                        "LARGE fine %s OK P%d=(%.4f,%.4f) pick#%d/%d "
                        "Δcoarse=%.1fmm",
                        side,
                        corner_idx,
                        corner_xyz[0],
                        corner_xyz[1],
                        pi + 1,
                        len(corner_samples),
                        d_c * 1000.0,
                    )
                    try:
                        vis = draw_pose_vis(
                            bgr_out,
                            used_det,
                            title=(
                                f"LARGE {side} fine P{corner_idx} "
                                f"pick#{pi + 1}/{len(corner_samples)} id={tid}"
                            ),
                        )
                        self._last_wrist_pose_vis = vis
                        if side.startswith("l"):
                            self._last_wrist_fine_left_vis = vis
                        else:
                            self._last_wrist_fine_right_vis = vis
                        if save_vis_dir is not None:
                            import cv2

                            out_w = Path(save_vis_dir) / "wrist"
                            out_w.mkdir(parents=True, exist_ok=True)
                            name = (
                                "pose_left.jpg"
                                if side.startswith("l")
                                else "pose_right.jpg"
                            )
                            cv2.imwrite(str(out_w / name), vis)
                            cv2.imwrite(str(out_w / "pose.jpg"), vis)
                            if bgr_out is not None:
                                cv2.imwrite(str(out_w / "color.jpg"), bgr_out)
                            LOG.info("LARGE fine saved %s/%s", out_w, name)
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("LARGE fine %s save vis failed: %s", side, exc)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("LARGE fine %s detect failed: %s", side, exc)
            finally:
                # 回粗定位位置
                if home_flange is not None and len(home_flange) >= 6:
                    try:
                        LOG.info("PHASE: LARGE fine %s → return coarse flange", side)
                        motion_backend.move_line(
                            home_flange[:6],
                            velocity_scale=float(self.cfg.move_velocity_scale),
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("LARGE fine %s return home failed: %s", side, exc)
            return corner_xyz, used_det, bgr_out

        fine_p3 = fine_p2 = None
        # 右腕相机提前开好，避免左臂拍完再连相机空等
        grab_fn_r = None
        if right_motion is not None:
            try:
                grab_fn_r = _ensure_grab_right()
            except Exception as exc:  # noqa: BLE001
                LOG.warning("LARGE fine: 开右腕相机失败: %s → 跳过 P2", exc)
                grab_fn_r = None
        try:
            fine_p3, _d3, _b3 = _pick_corner(
                side="left",
                corner_idx=3,
                hover_xy=hover_p3,
                corner_xy=p3_c,
                z_arrive=z_p3_arrive,
                grab_fn=grab_wrist,
                motion_backend=left_motion,
                T_tc=T_left,
                home_flange=coarse_fl_l,
            )
            if right_motion is None:
                LOG.warning("LARGE fine: 无右臂 motion → 跳过 P2 精定位")
            elif grab_fn_r is not None:
                fine_p2, _d2, _b2 = _pick_corner(
                    side="right",
                    corner_idx=2,
                    hover_xy=hover_p2,
                    corner_xy=p2_c,
                    z_arrive=z_p2_arrive,
                    grab_fn=grab_fn_r,
                    motion_backend=right_motion,
                    T_tc=T_right,
                    home_flange=coarse_fl_r,
                )
        finally:
            if right_cam is not None:
                try:
                    right_cam.close()
                except Exception:
                    pass

        # 未检出则直接报错，禁止继续接近/夹取移动
        miss: list[str] = []
        if fine_p3 is None:
            miss.append("左腕P3")
        if right_motion is not None and fine_p2 is None:
            miss.append("右腕P2")
        if miss:
            raise RuntimeError(
                "大板腕精定位未识别到角点："
                + "、".join(miss)
                + "（模型 V2·220e76）。已中止，不再做精定位接近/夹取移动。"
            )

        new_corners = [list(c) for c in coarse.corners_base]
        new_corners[3] = list(fine_p3)
        if fine_p2 is not None:
            new_corners[2] = list(fine_p2)
        else:
            LOG.warning("LARGE fine: 无右臂 → P2 保留粗 XY，Z=FFS")
        # 夹取高度写死；精定位只提供 XY
        new_corners[2][2] = z_board
        new_corners[3][2] = z_board
        z_board_out = 0.5 * (
            float(new_corners[2][2]) + float(new_corners[3][2])
        )

        rz = gripper_rz_large_parallel_p2p3(
            new_corners[2],
            new_corners[3],
            orient_rz_offset_deg=float(
                getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
            ),
        )
        mid_x = 0.5 * (float(new_corners[2][0]) + float(new_corners[3][0]))
        mid_y = 0.5 * (float(new_corners[2][1]) + float(new_corners[3][1]))
        refined = _OP(
            x=mid_x,
            y=mid_y,
            z=float(z_board_out),
            rz=float(rz),
            edge_yaw=math.atan2(
                float(new_corners[3][1]) - float(new_corners[2][1]),
                float(new_corners[3][0]) - float(new_corners[2][0]),
            ),
            corners_base=new_corners,
            conf=float(coarse.conf),
            board_id=coarse.board_id,
            cls_id=int(self.cfg.cls_pcb_large),
        )
        LOG.info(
            "LARGE fine merge OK id=%s mid23=(%.4f,%.4f) rz=%.1f° "
            "P2=(%.4f,%.4f) P3=(%.4f,%.4f)",
            refined.board_id,
            refined.x,
            refined.y,
            refined.rz * 180.0 / math.pi,
            new_corners[2][0],
            new_corners[2][1],
            new_corners[3][0],
            new_corners[3][1],
        )
        self._large_wrist_fine_ok = True
        return refined

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
        T_tool_cam: np.ndarray | None = None,
    ) -> list[float]:
        """粗定位悬停：腕相机光心对准板心，光轴竖直向下。

        默认按当前法兰选最近 ``cam_yaw``（避免 yaw=0 时 rz 跳 ~90° 拧限位）。
        ``keep_rz_rad`` 已废弃，忽略。
        ``T_tool_cam``：左右腕手眼不同时传入（默认左腕 ``cfg.T_tool_cam``）。
        """
        from .geometry import (
            flange_pose_camera_down_nearest,
            flange_pose_camera_optical_down,
        )

        T_tc = (
            np.asarray(T_tool_cam, dtype=np.float64)
            if T_tool_cam is not None
            else np.asarray(self.cfg.T_tool_cam, dtype=np.float64)
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
                T_tool_cam=T_tc,
                cam_yaw_rad=yaw,
            )
        else:
            flange, yaw = flange_pose_camera_down_nearest(
                cam_xyz,
                T_tool_cam=T_tc,
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
        """采 N 帧 → 每帧多板 ID → 滤异常 → 取最接近均值的那帧目标板。

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
        mad_k = float(
            mad_k if mad_k is not None else self.cfg.detect_outlier_mad_k
        )
        n = max(1, n)
        tid = int(
            self.cfg.target_board_id if target_board_id is None else target_board_id
        )
        est = self._ensure_stereo_depth()

        samples: list[dict] = []
        LOG.info(
            "HEAD multi-frame n=%d mad_k=%.1f → pick closest-to-mean (target_rank=%d)",
            n,
            mad_k,
            tid,
        )
        for i in range(n):
            left, right = grab_head()
            rect_vis, _rr, depth, _valid, depth_stats = est.estimate_raw(left, right)
            if require_depth_ok and depth_stats is not None and not depth_stats.ok():
                LOG.warning(
                    "HEAD frame %d/%d depth bad: %s",
                    i + 1,
                    n,
                    depth_stats.summary(),
                )
                if gap > 0 and i + 1 < n:
                    time.sleep(gap)
                continue
            dets = self._head_det.detect(rect_vis)
            if not dets:
                LOG.warning("HEAD frame %d/%d: no detect", i + 1, n)
                if gap > 0 and i + 1 < n:
                    time.sleep(gap)
                continue
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
                LOG.warning("HEAD frame %d/%d: localize all failed", i + 1, n)
                if gap > 0 and i + 1 < n:
                    time.sleep(gap)
                continue
            boards = assign_board_ids_by_distance(poses, metric="xy")
            order = sorted(
                range(len(poses)),
                key=lambda j: (poses[j].x ** 2 + poses[j].y ** 2),
            )
            dets_ranked = [ok_dets[j] for j in order]
            try:
                target = select_grasp_board(
                    boards,
                    tid,
                    pcb_cls=int(self.cfg.cls_pcb_small),
                    large_cls=int(self.cfg.cls_pcb_large),
                    prefer_large=bool(self.cfg.large_board_enable),
                )
            except Exception as exc:
                LOG.warning("HEAD frame %d/%d select fail: %s", i + 1, n, exc)
                if gap > 0 and i + 1 < n:
                    time.sleep(gap)
                continue
            tgt_i = next(
                j for j, b in enumerate(boards) if b.board_id == target.board_id
            )
            samples.append(
                {
                    "i": i,
                    "left": left,
                    "right": right,
                    "rect": rect_vis,
                    "depth": depth,
                    "depth_stats": depth_stats,
                    "boards": boards,
                    "dets": dets_ranked,
                    "target": target,
                    "tgt_det": dets_ranked[tgt_i],
                }
            )
            LOG.info(
                "HEAD frame %d/%d: %d board(s) target=ID%s cls=%s xy=(%.3f,%.3f)",
                i + 1,
                n,
                len(boards),
                target.board_id,
                cls_name(target.cls_id),
                target.x,
                target.y,
            )
            if gap > 0 and i + 1 < n:
                time.sleep(gap)

        if not samples:
            raise RuntimeError(f"Head: no valid PCB in {n} frames")

        tgt_poses = [s["target"] for s in samples]
        _picked, pick_i, keep, mean_xyz = pick_closest_to_mean_pose(
            tgt_poses, mad_k=mad_k
        )
        samp = samples[pick_i]
        left = samp["left"]
        right = samp["right"]
        rect_vis = samp["rect"]
        depth = samp["depth"]
        depth_stats = samp["depth_stats"]
        boards = samp["boards"]
        dets_ranked = samp["dets"]
        tgt_det = samp["tgt_det"]
        # 每个角点自己在 N 帧里的深度中位数（只计 <1m），不用四角互相替换。
        n_c = len(tgt_det.corners)
        buckets: list[list[float]] = [[] for _ in range(n_c)]
        for s in samples:
            det_s = s["tgt_det"]
            if len(det_s.corners) < n_c:
                continue
            for j, (u, v) in enumerate(det_s.corners[:n_c]):
                z = sample_depth_m(s["depth"], float(u), float(v), z_max=1.0)
                if z is not None:
                    buckets[j].append(float(z))
        tgt_cls = int(getattr(samp["target"], "cls_id", 0) or 0)
        required = (
            {2, 3}
            if tgt_cls == int(self.cfg.cls_pcb_large)
            else set(range(n_c))
        )
        meds: list[float] = []
        for j, zs in enumerate(buckets):
            if not zs and j in required:
                raise RuntimeError(
                    f"Head corner P{j}: no depth <1m across {len(samples)} frames"
                )
            if zs:
                meds.append(float(np.median(np.asarray(zs, dtype=np.float64))))
            else:
                meds.append(float("nan"))
        have = [z for j, z in enumerate(meds) if j in required]
        fill = float(sorted(have)[len(have) // 2])
        meds = [fill if z != z else z for z in meds]
        LOG.info(
            "HEAD per-corner depth median over %d frames: %s counts=%s",
            len(samples),
            [round(z, 3) for z in meds],
            [len(zs) for zs in buckets],
        )
        target = self._head_localize_det(tgt_det, depth_m_per_corner=meds)
        target.board_id = samp["target"].board_id
        target.cls_id = int(getattr(samp["target"], "cls_id", 0) or 0)
        boards = [
            target if getattr(b, "board_id", None) == target.board_id else b
            for b in boards
        ]
        tid_grasp = int(target.board_id) if target.board_id is not None else int(tid)
        best_i = int(samp["i"])

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
        n_keep = sum(1 for ok in keep if ok)
        self._last_head_pose_vis = draw_multi_board_vis(
            rect_vis,
            dets_ranked,
            [b.board_id for b in boards],
            title=(
                f"HEAD ×{len(boards)} (pcb={n_pcb} nopcb={n_nopcb} large={n_large}) "
                f"pick#{best_i + 1}/{n} keep={n_keep} "
                f"grasp=ID{tid_grasp}/{cls_name(target.cls_id)} "
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
                "n_frames": n,
                "n_samples": len(samples),
                "n_keep": n_keep,
                "mean_xyz": [float(x) for x in mean_xyz],
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
        """Return ``large`` / ``large_single`` / ``small`` from force_class / cls."""
        if self.force_class in ("large_single", "large-single", "single"):
            return "large_single"
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

        # MAD 滤异常 → 取最接近平均位置的真实帧（不用合成中值当抓取位）
        picked_pose, rep_i, keep, mean_xyz = pick_closest_to_mean_pose(poses, mad_k=k)
        if used_fallback:
            if picked_pose.board_id is None:
                picked_pose.board_id = poses[0].board_id
        else:
            picked_pose.board_id = tid
        med_pose = picked_pose
        rep_bgr = bgrs[rep_i]
        rep_det = dets[rep_i]
        n_keep = sum(1 for ok in keep if ok)

        # 多板 ID 可视化（代表帧）+ 缓存预测标签
        if rep_all is not None and rep_all[0] is rep_bgr:
            rbgr, rdepth, rdets, rmatched = rep_all
            self._last_wrist_pose_vis = draw_multi_board_vis(
                rbgr,
                rdets,
                [p.board_id for p in rmatched],
                title=(
                    f"WRIST ID{tid} pick#{rep_i + 1}/{len(poses)} "
                    f"keep={n_keep} meanΔ="
                    f"{math.hypot(med_pose.x - mean_xyz[0], med_pose.y - mean_xyz[1]) * 1000:.0f}mm"
                ),
                highlight_id=tid,
                kp_i=int(self.cfg.kp_i),
                kp_j=int(self.cfg.kp_j),
            )
            self._last_fine_dets = list(rdets)
            self._last_fine_boards = list(rmatched)
        else:
            self._last_wrist_pose_vis = draw_pose_vis(
                rep_bgr,
                rep_det,
                title=(
                    f"WRIST ID={tid} closest-to-mean frame#{rep_i + 1} "
                    f"keep={n_keep}/{len(poses)}"
                ),
                kpt_dir=(1, 0),
                board_id=tid,
            )
            self._last_fine_dets = [rep_det]
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

    def _overlay_grasp_on_head(
        self,
        points: list[tuple[str, Sequence[float], tuple[int, int, int]]],
        *,
        title: str | None = None,
    ) -> None:
        """定位后的抓取点：``inv(T_base_cam)`` 投到头图并画在 ``_last_head_pose_vis``。"""
        if self._last_head_pose_vis is None and self._last_head_rect is None:
            return
        if self._head_K is None:
            try:
                self._ensure_head_maps()
            except Exception:
                return
        if self._head_K is None:
            return
        base = (
            self._last_head_pose_vis
            if self._last_head_pose_vis is not None
            else self._last_head_rect
        )
        try:
            self._last_head_pose_vis = draw_grasp_points_on_head(
                base,
                points,
                K=self._head_K,
                T_base_cam=self._T_base_head
                if self._T_base_head is not None
                else self.cfg.T_base_cam,
                title=title or "grasp←base (handeye)",
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("overlay grasp on head failed: %s", exc)

    def _save_wrist_depth_debug(
        self,
        out: Path,
        *,
        tag: str = "wrist",
        extra: dict | None = None,
    ) -> list[str]:
        """落盘腕部彩色 + 原始深度 + 伪彩 + 统计（粗失败排查用）。"""
        import json

        from .experiment_log import colorize_depth_any

        out = Path(out)
        wrist_dir = out / "wrist"
        wrist_dir.mkdir(parents=True, exist_ok=True)
        written: list[str] = []
        bgr = getattr(self, "_last_wrist_bgr", None)
        depth = getattr(self, "_last_wrist_depth", None)
        scale = float(self.cfg.wrist_depth_scale)
        if bgr is not None:
            p = wrist_dir / "color.jpg"
            cv2.imwrite(str(p), bgr)
            written.append(str(p.relative_to(out)))
        pose = getattr(self, "_last_wrist_pose_vis", None)
        if pose is not None:
            p = wrist_dir / "pose.jpg"
            cv2.imwrite(str(p), pose)
            written.append(str(p.relative_to(out)))
        stats: dict = {
            "tag": tag,
            "depth_scale": scale,
            "depth_is_none": depth is None,
        }
        if depth is not None:
            raw = np.asarray(depth)
            stats["dtype"] = str(raw.dtype)
            stats["shape"] = list(raw.shape)
            np.save(str(wrist_dir / "depth.npy"), raw)
            written.append("wrist/depth.npy")
            # 16-bit PNG（若是 uint16）方便系统看图软件打开
            if raw.dtype == np.uint16:
                cv2.imwrite(str(wrist_dir / "depth_u16.png"), raw)
                written.append("wrist/depth_u16.png")
                valid = raw[raw > 0]
            else:
                valid = raw[raw > 0.05]
            stats["nnz"] = int(valid.size)
            stats["numel"] = int(raw.size)
            stats["nnz_ratio"] = (
                float(valid.size) / float(raw.size) if raw.size else 0.0
            )
            if valid.size:
                if raw.dtype == np.uint16:
                    stats["z_min_m"] = float(valid.min()) * scale
                    stats["z_max_m"] = float(valid.max()) * scale
                    stats["z_median_m"] = float(np.median(valid)) * scale
                else:
                    stats["z_min_m"] = float(valid.min())
                    stats["z_max_m"] = float(valid.max())
                    stats["z_median_m"] = float(np.median(valid))
            try:
                color = colorize_depth_any(raw, depth_scale=scale)
                cv2.imwrite(str(wrist_dir / "depth_color.jpg"), color)
                written.append("wrist/depth_color.jpg")
            except Exception as exc:  # noqa: BLE001
                stats["colorize_error"] = str(exc)
        if extra:
            stats.update(extra)
        (wrist_dir / "depth_stats.json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append("wrist/depth_stats.json")
        LOG.info(
            "saved wrist depth debug → %s (nnz=%s ratio=%.3f)",
            wrist_dir,
            stats.get("nnz"),
            float(stats.get("nnz_ratio") or 0.0),
        )
        return written

    def _save_stage_vis(
        self,
        out: Path,
        *,
        head_rect: np.ndarray | None = None,
        head_det: PcbDetection | None = None,
        extra_meta: dict | None = None,
        save_head_pred: bool = True,
        save_wrist_pred: bool = True,
    ) -> None:
        """本轮实验批次：原图 + 预测标签 + 识别/匹配图 + 深度。

        粗/精定位后都会落盘：
        - ``head/rect.jpg`` + ``head/pred.json``（及 ``annotate/coarse_*``）
        - ``wrist/color.jpg`` + ``wrist/pred.json``（及 ``annotate/fine_*``）
        原图无绘制；标签含 corners_uv / corners_base / rz，供人工标注对照误差。
        """
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
            "target_board_id": getattr(self, "_last_target_board_id", None),
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
            wrist_pose_left_vis=getattr(self, "_last_wrist_fine_left_vis", None),
            wrist_pose_right_vis=getattr(self, "_last_wrist_fine_right_vis", None),
            wrist_match_vis=self._last_wrist_match_vis,
            wrist_color=self._last_wrist_bgr,
            wrist_depth=self._last_wrist_depth,
            wrist_depth_scale=float(self.cfg.wrist_depth_scale),
            meta=meta,
            head_pred_dets=list(getattr(self, "_last_coarse_dets", []) or []),
            head_pred_poses=list(getattr(self, "_last_coarse_boards", []) or []),
            wrist_pred_dets=list(getattr(self, "_last_fine_dets", []) or []),
            wrist_pred_poses=list(getattr(self, "_last_fine_boards", []) or []),
            target_board_id=getattr(self, "_last_target_board_id", None),
            save_head_pred=save_head_pred,
            save_wrist_pred=save_wrist_pred,
        )


    def _overlay_on_wrist_coarse(
        self,
        points: list[tuple[str, Sequence[float], tuple[int, int, int]]],
        *,
        title: str | None = None,
        base_bgr: np.ndarray | None = None,
    ) -> None:
        """单臂支路：抓取/角点叠在「粗定位前腕部照片」上。"""
        base = (
            base_bgr
            if base_bgr is not None
            else self._last_wrist_coarse_bgr
        )
        T = self._last_wrist_coarse_T_base_cam
        if base is None or T is None:
            return
        try:
            vis = draw_grasp_points_on_head(
                base,
                points,
                K=self._wrist_K,
                T_base_cam=T,
                title=title or "LARGE single ← wrist coarse",
            )
            self._last_wrist_pose_vis = vis
            # 无头图时，网页头槽也显示这张腕粗图
            if self._last_head_pose_vis is None:
                self._last_head_pose_vis = vis
        except Exception as exc:  # noqa: BLE001
            LOG.warning("overlay on wrist coarse failed: %s", exc)

    def run_large_single_arm(
        self,
        grab_wrist,
        *,
        settle_s: float | None = None,
        do_grasp: bool = False,
        save_vis_dir: str | Path | None = None,
        stage: str = "all",
        plan_only: bool = False,
        on_progress=None,
    ) -> GraspResult:
        """大板单臂支路 B：无头；右腕 cam3。

        1) 粗（pcb-large-1·0ca8f6）：真实深度抬 XY/Z，得 P2/P3；本阶段不动臂
           ※ 定位高度只用深度角点 Z；grasp_z_m 是抓取示教，不参与定位
        2) 精拍到位：TCP=(角点XY, 角点Z+15cm)，RPY=当前腕姿（非夹取 90°/0/rz）
        3) 精（corner-1·90459c）：固定 cam 深 15cm；依次 P2→P3 pose 匹配更新
        4) 停 tip：精更新后 mid XY、TCP_z=角点Z+10cm、末端朝下、rz∥精P2P3
        """
        from .geometry import pose_xyz_rpy_to_mat4
        from .large_board import gripper_rz_large_parallel_p2p3, tip_to_flange
        from .localize import corner_uv_to_base, corners_to_object_pose

        def _progress(phase: str) -> None:
            if on_progress is None:
                return
            try:
                on_progress(result, phase)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("on_progress(%s) failed: %s", phase, exc)

        result = GraspResult()
        grasp_mode = "large_single"
        if not bool(getattr(self.cfg, "large_single_enable", True)):
            result.message = "large_single 未启用（large_board.single_arm.enable）"
            return result

        arm_side = str(
            getattr(self.cfg, "large_single_arm", "right") or "right"
        ).strip().lower()
        use_right = arm_side.startswith("r")
        arm_cli = "right" if use_right else "left"

        coarse_det = self._ensure_large_single_coarse_det()
        if coarse_det is None:
            result.message = (
                "未配置 large_board.single_arm.coarse_pose_weights（pcb-large-1）"
            )
            return result
        fine_det = self._ensure_large_single_fine_det()
        if fine_det is None:
            result.message = (
                "未配置 large_board.single_arm.fine_pose_weights（corner-1）"
            )
            return result

        grasp_z_cfg = getattr(self.cfg, "large_grasp_z_m", None)
        if grasp_z_cfg is None:
            grasp_z_cfg = float(self.cfg.head_tray_z_m)
            LOG.warning(
                "LARGE single: grasp_z_m 未标定 → 暂用 head_tray_z_m=%.4f",
                float(grasp_z_cfg),
            )
        else:
            grasp_z_cfg = float(grasp_z_cfg)

        hover_h = getattr(self.cfg, "large_single_fine_hover_height_m", None)
        if hover_h is None:
            hover_h = 0.30
        else:
            hover_h = float(hover_h)
        # 精拍光心在板上方 hover；固定 cam 深默认跟 hover（可单独改）
        fixed_cam_d = float(
            getattr(self.cfg, "large_single_fine_fixed_depth_m", None)
            or hover_h
            or 0.30
        )
        board_w = float(
            getattr(self.cfg, "large_board_width_m", 0.27) or 0.27
        )
        board_len = float(
            getattr(self.cfg, "large_board_length_m", 0.36) or 0.36
        )
        # 停 tip = 板心：mid(P2,P3) + (width/2)*板内法向；stop_dy 仅调试覆盖
        _sdy_ov = getattr(self.cfg, "large_single_stop_dy_m", None)
        stop_half_w = (
            abs(float(_sdy_ov))
            if _sdy_ov is not None
            else 0.5 * board_w
        )
        stop_dy = stop_half_w  # 兼容下游 debug 字段名；语义=半宽到板心
        stop_z_above = float(
            getattr(self.cfg, "large_single_stop_z_above_board_m", 0.10) or 0.10
        )
        stop_z_override = getattr(self.cfg, "large_single_stop_z_m", None)
        max_d = float(getattr(self.cfg, "large_fine_match_max_dist_m", 0.15))

        if use_right:
            T_tc = getattr(self.cfg, "T_tool_cam_right", None)
            if T_tc is None:
                T_tc = np.asarray(self.cfg.T_tool_cam, dtype=np.float64)
                LOG.warning(
                    "LARGE single: T_tool_cam_right 未配置 → 暂用左腕手眼"
                )
            else:
                T_tc = np.asarray(T_tc, dtype=np.float64)
        else:
            T_tc = np.asarray(self.cfg.T_tool_cam, dtype=np.float64)
        cam_from_tf = bool(
            getattr(self.cfg, "large_single_cam_pose_from_tf", True)
        )

        # 运控：右臂走 dual.right
        dual = self.dual_motion
        if use_right:
            arm_motion = None
            if dual is not None:
                arm_motion = getattr(dual, "right", None)
            if arm_motion is None:
                arm_motion = self.motion
                LOG.warning(
                    "LARGE single: dual.right 不可用 → 用 primary motion "
                    "（请确认已连双臂且 primary 为右）"
                )
        else:
            arm_motion = self.motion
            if dual is not None and getattr(dual, "left", None) is not None:
                arm_motion = dual.left

        do_move = not bool(plan_only)
        depth_scale = float(self.cfg.wrist_depth_scale)
        # 右臂用右腕内参抬点
        wrist_K = self._wrist_K_right if use_right else self._wrist_K
        own_cam = None
        grab_fn = grab_wrist

        # 机上 TF：只 subprocess 读一次 → 缓存 T_tool_cam；之后 T_base_cam=TCP×缓存
        # （原先每帧 camera_pose ≈1~2s，粗 10 帧就卡 20s+）
        _tf_cache: dict = {"T_cam": None, "T_tc": None, "logged": False}

        def _invalidate_tf_tool() -> None:
            _tf_cache["T_cam"] = None
            _tf_cache["T_tc"] = None

        def _T_base_cam_now(ee: list[float]) -> np.ndarray:
            """机上 305g_frame 推活手眼；已缓存则只乘当前 TCP。"""
            T_tcp = pose_xyz_rpy_to_mat4(*[float(x) for x in ee[:6]])
            if cam_from_tf:
                if _tf_cache["T_tc"] is not None:
                    return T_tcp @ np.asarray(_tf_cache["T_tc"], dtype=np.float64)
                try:
                    from .rim_motion import read_wrist_camera_pose_mat4

                    portal = (
                        getattr(arm_motion, "frame_bundle", None)
                        or getattr(self.cfg, "rim_frame_bundle", None)
                        or "10.3.8.7:50066"
                    )
                    T_cam = np.asarray(
                        read_wrist_camera_pose_mat4(
                            "right" if use_right else "left",
                            frame_bundle=str(portal),
                            timeout_s=float(
                                getattr(self.cfg, "rim_connect_timeout_s", 10.0)
                                or 10.0
                            ),
                        ),
                        dtype=np.float64,
                    )
                    T_tc_live = np.linalg.inv(T_tcp) @ T_cam
                    _tf_cache["T_cam"] = T_cam
                    _tf_cache["T_tc"] = T_tc_live
                    if not _tf_cache["logged"]:
                        LOG.info(
                            "LARGE single: 缓存活手眼 ← %s（其后只乘 TCP）",
                            "arm_r_305g_frame"
                            if use_right
                            else "arm_l_305g_frame",
                        )
                        _tf_cache["logged"] = True
                    return T_cam
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "LARGE single: 读 arm_*_305g_frame 失败 (%s) → TCP×yaml手眼",
                        exc,
                    )
            return T_tcp @ T_tc

        def _ensure_cam_grab():
            """右腕默认开 cam3；左腕用传入 grab。"""
            nonlocal own_cam, grab_fn
            if not use_right:
                return grab_fn
            if grab_fn is not None and bool(
                getattr(self, "_large_single_grab_is_right", False)
            ):
                return grab_fn
            # 自开 cam3（与双臂精定位右腕一致）
            from .cameras import WristPortalCam

            portal = (
                getattr(self.cfg, "rim_frame_bundle", None)
                or getattr(self.cfg, "cam_portal", "10.3.8.7:50066")
            )
            dist = (
                getattr(self.cfg, "wrist_dist_coeffs_right", None)
                or getattr(self.cfg, "wrist_dist_coeffs", None)
            )
            cam_id = int(
                getattr(self.cfg, "large_single_cam_id", None)
                or getattr(self.cfg, "cam_portal_wrist_right_id", 3)
                or 3
            )
            depth_id = int(
                getattr(self.cfg, "large_single_depth_cam_id", None)
                or getattr(self.cfg, "cam_portal_wrist_right_depth_id", 2)
                or 2
            )
            Kr = self._wrist_K_right
            own_cam = WristPortalCam(
                portal,
                cam_id=cam_id,
                max_hz=float(getattr(self.cfg, "cam_portal_max_hz", 15.0)),
                timeout_s=float(getattr(self.cfg, "cam_portal_timeout_s", 60.0)),
                need_depth=True,
                fx=float(Kr.fx),
                fy=float(Kr.fy),
                cx=float(Kr.cx),
                cy=float(Kr.cy),
                dist_coeffs=list(dist) if dist else None,
                undistort=bool(getattr(self.cfg, "cam_wrist_undistort", False)),
                depth_cam_id=depth_id,
            )
            grab_fn = own_cam.grab
            LOG.info(
                "LARGE single: opened wrist cam%d depth=%d @ %s (subscribe hz)",
                cam_id,
                depth_id,
                portal,
            )
            return grab_fn

        try:
            grab = _ensure_cam_grab()

            # —— 0. 粗定位位姿（可选固定法兰）——
            coarse_pose_cfg = getattr(self.cfg, "large_single_coarse_flange", None)
            if do_move and coarse_pose_cfg is not None and len(coarse_pose_cfg) >= 6:
                LOG.info(
                    "PHASE: LARGE single → coarse flange (configured) "
                    "xyz=(%.3f,%.3f,%.3f)",
                    float(coarse_pose_cfg[0]),
                    float(coarse_pose_cfg[1]),
                    float(coarse_pose_cfg[2]),
                )
                arm_motion.move_line(
                    [float(x) for x in coarse_pose_cfg[:6]],
                    velocity_scale=float(self.cfg.move_velocity_scale),
                )
                try:
                    self._wait_still(
                        settle_s=0.3 if settle_s is None else settle_s
                    )
                except Exception:
                    pass
            elif coarse_pose_cfg is None:
                LOG.info(
                    "LARGE single: coarse_flange_pose 未写死 → 当前 TCP 拍粗图"
                )

            try:
                cur = list(arm_motion.get_flange_pose()[:6])
            except Exception:
                cur = None
            if use_right:
                result.current_flange_right = cur
                # 也记左侧便于 UI
                try:
                    if dual is not None:
                        result.current_flange = list(
                            dual.left.get_flange_pose()[:6]
                        )
                except Exception:
                    result.current_flange = None
            else:
                result.current_flange = cur
            result.hover_flange = list(cur) if cur else None
            result.coarse_flange = list(cur) if cur else None

            # —— 1. 腕粗：真实深度抬 4 角 ——
            LOG.info(
                "PHASE: LARGE single WRIST coarse (%d frames) arm=%s "
                "[plan_only=%s] depth=real",
                self.cfg.detect_n_frames,
                arm_cli,
                plan_only,
            )
            n_f = max(1, int(self.cfg.detect_n_frames))
            gap_f = float(self.cfg.detect_frame_gap_s)
            mad_k = float(self.cfg.detect_outlier_mad_k)
            pose_samples: list[ObjectPose] = []
            det_samples: list[PcbDetection] = []
            bgr_samples: list[np.ndarray] = []
            T_samples: list[np.ndarray] = []
            import time as _time

            n_rgb_hit = 0
            n_depth_miss = 0
            last_depth_diag = ""
            # 粗拍臂不动：TCP 只读一次（get_flange_pose 也是起子进程）
            ee_coarse = list(arm_motion.get_flange_pose()[:6])
            for fi in range(n_f):
                wrist_bgr, wrist_depth = grab()
                self._last_wrist_bgr = wrist_bgr
                self._last_wrist_depth = wrist_depth
                ee = ee_coarse
                T_base_cam = _T_base_cam_now(ee)
                dets = coarse_det.detect(wrist_bgr)
                if not dets:
                    LOG.warning(
                        "LARGE single coarse frame %d/%d: RGB 未检出",
                        fi + 1,
                        n_f,
                    )
                else:
                    dets_sorted = sorted(
                        dets,
                        key=lambda d: float(getattr(d, "conf", 0) or 0),
                        reverse=True,
                    )
                    d0 = dets_sorted[0]
                    if not d0.corners or len(d0.corners) < 4:
                        LOG.warning(
                            "LARGE single coarse frame %d: 角点数=%d <4",
                            fi + 1,
                            len(d0.corners) if d0.corners else 0,
                        )
                    else:
                        n_rgb_hit += 1
                        raw = (
                            np.asarray(wrist_depth)
                            if wrist_depth is not None
                            else None
                        )
                        # 整图有效深度中值（门户有深度但角点落在孔洞时用）
                        scene_med = None
                        nnz = 0
                        dmin = dmax = 0.0
                        if raw is not None and raw.size:
                            if raw.dtype == np.uint16:
                                valid = raw[(raw > 80) & (raw < 5000)]  # 0.08~5m
                                nnz = int((raw > 0).sum())
                                if valid.size:
                                    dmin = float(valid.min()) * depth_scale
                                    dmax = float(valid.max()) * depth_scale
                                    scene_med = float(np.median(valid)) * depth_scale
                            else:
                                valid = raw[(raw > 0.08) & (raw < 5.0)]
                                nnz = int((raw > 0.05).sum())
                                if valid.size:
                                    dmin = float(valid.min())
                                    dmax = float(valid.max())
                                    scene_med = float(np.median(valid))

                        depths: list[float] = []
                        miss_uv: list[tuple[float, float]] = []
                        raw_at: list[int | float | None] = []
                        for u, v in d0.corners:
                            z = None
                            raw_px = None
                            if raw is not None and raw.size:
                                ui, vi = int(round(u)), int(round(v))
                                if 0 <= vi < raw.shape[0] and 0 <= ui < raw.shape[1]:
                                    raw_px = raw[vi, ui]
                                    raw_at.append(
                                        int(raw_px)
                                        if raw.dtype == np.uint16
                                        else float(raw_px)
                                    )
                                else:
                                    raw_at.append(None)
                                z = sample_depth_m(
                                    raw,
                                    float(u),
                                    float(v),
                                    depth_scale=depth_scale,
                                    radius=20,
                                    z_min=0.08,
                                    z_max=5.0,
                                )
                            else:
                                raw_at.append(None)
                            if z is None or float(z) < 0.05:
                                miss_uv.append((float(u), float(v)))
                                depths.append(float("nan"))
                            else:
                                depths.append(float(z))

                        n_ok = sum(1 for z in depths if z == z)  # not nan
                        # 场景中值常被背景（地板/远处）拉到数米 → XY 漂到 3~4m；
                        # 腕看大板合理 camZ 一般 <1.2m，超出则不用 scene_med，走板面求交
                        scene_med_ok = (
                            scene_med is not None
                            and 0.08 < float(scene_med) <= 1.20
                        )
                        if n_ok < 4 and scene_med_ok:
                            # 角点落在孔洞：用邻域失败处填场景中值
                            depths = [
                                float(scene_med) if (z != z) else float(z)
                                for z in depths
                            ]
                            n_depth_miss += 1
                            last_depth_diag = (
                                f"角点深度孔洞 {4 - n_ok}/4 "
                                f"raw_at={raw_at} miss_uv="
                                f"{[(round(a), round(b)) for a, b in miss_uv]} "
                                f"→ 用场景中值 camZ={scene_med:.3f}m "
                                f"(nnz={nnz} z={dmin:.3f}~{dmax:.3f})"
                            )
                            LOG.warning(
                                "LARGE single coarse frame %d: %s",
                                fi + 1,
                                last_depth_diag,
                            )
                        elif n_ok < 4:
                            n_depth_miss += 1
                            last_depth_diag = (
                                f"RGB已检出但深度不可用 {4 - n_ok}/4 "
                                f"raw_at={raw_at} miss_uv="
                                f"{[(round(a), round(b)) for a, b in miss_uv]} "
                                f"nnz={nnz} z={dmin:.3f}~{dmax:.3f} "
                                f"scene_med={scene_med}"
                            )
                            LOG.warning(
                                "LARGE single coarse frame %d: %s",
                                fi + 1,
                                last_depth_diag,
                            )

                        def _append_pose(pose, *, note: str = "") -> None:
                            pose.cls_id = int(self.cfg.cls_pcb_large)
                            pose_samples.append(pose)
                            det_samples.append(d0)
                            bgr_samples.append(wrist_bgr)
                            T_samples.append(T_base_cam)
                            LOG.info(
                                "LARGE single coarse frame %d: "
                                "mid=(%.4f,%.4f,%.4f)%s",
                                fi + 1,
                                pose.x,
                                pose.y,
                                pose.z,
                                f" {note}" if note else "",
                            )

                        used_depth_lift = False
                        if n_ok == 4 or (
                            scene_med_ok and all(z == z for z in depths)
                        ):
                            try:
                                pose = corners_to_object_pose(
                                    d0,
                                    T_base_cam=T_base_cam,
                                    K=wrist_K,
                                    depth_m_per_corner=[float(z) for z in depths],
                                    kp_i=2,
                                    kp_j=3,
                                )
                                _append_pose(
                                    pose,
                                    note=(
                                        f"camZ={[round(z, 3) for z in depths]}"
                                        + (
                                            " [scene_med fill]"
                                            if n_ok < 4
                                            else ""
                                        )
                                    ),
                                )
                                used_depth_lift = True
                            except Exception as exc:  # noqa: BLE001
                                LOG.warning(
                                    "LARGE single coarse lift fail: %s", exc
                                )

                        if not used_depth_lift:
                            # 定位只用深度：不走 grasp_z 平面。角点孔洞 → 固定 cam 深兜底
                            if (
                                scene_med is not None
                                and not scene_med_ok
                            ):
                                LOG.warning(
                                    "LARGE single coarse frame %d: "
                                    "场景中值 camZ=%.3f 像背景 "
                                    "(nnz=%d z=%.3f~%.3f) → 不用；改固定cam深",
                                    fi + 1,
                                    float(scene_med),
                                    nnz,
                                    dmin,
                                    dmax,
                                )
                                n_depth_miss += 1
                                last_depth_diag = (
                                    f"场景中值过大 camZ={scene_med:.3f}m "
                                    f"→ fixed_camZ（不用 grasp_z）"
                                )
                            fb = float(
                                getattr(
                                    self.cfg,
                                    "large_single_fine_fixed_depth_m",
                                    0.15,
                                )
                                or 0.15
                            )
                            try:
                                pose = corners_to_object_pose(
                                    d0,
                                    T_base_cam=T_base_cam,
                                    K=wrist_K,
                                    fixed_depth_cam_m=fb,
                                    kp_i=2,
                                    kp_j=3,
                                )
                                _append_pose(
                                    pose,
                                    note=f"[fixed_camZ={fb:.3f}]",
                                )
                            except Exception as exc:  # noqa: BLE001
                                LOG.warning(
                                    "LARGE single coarse fixed-depth "
                                    "lift fail: %s",
                                    exc,
                                )
                                last_depth_diag = (
                                    f"深度不可用且 fixed_camZ={fb:.3f} 抬点失败: "
                                    f"{exc}"
                                )
                if gap_f > 0 and fi + 1 < n_f:
                    _time.sleep(gap_f)

            if not pose_samples:
                if n_rgb_hit > 0 and n_depth_miss > 0:
                    result.message = (
                        f"单臂腕粗：RGB检出×{n_rgb_hit} 但深度不可用，"
                        f"且板面求交也失败。{last_depth_diag}"
                    )
                elif n_rgb_hit > 0:
                    result.message = (
                        f"单臂腕粗：RGB检出×{n_rgb_hit} 但抬点失败"
                    )
                else:
                    result.message = "单臂腕粗定位 RGB 未检出四角"
                result.debug = {
                    "grasp_mode": grasp_mode,
                    "stage": "coarse_fail",
                    "n_rgb_hit": n_rgb_hit,
                    "n_depth_miss": n_depth_miss,
                    "depth_diag": last_depth_diag,
                }
                # 失败也落盘彩色+深度，便于核对 cam2
                if save_vis_dir:
                    try:
                        self._save_wrist_depth_debug(
                            Path(save_vis_dir),
                            tag="coarse_fail",
                            extra={
                                "n_rgb_hit": n_rgb_hit,
                                "n_depth_miss": n_depth_miss,
                                "depth_diag": last_depth_diag,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("save wrist depth debug failed: %s", exc)
                return result

            boards = assign_board_ids_by_distance(pose_samples, metric="xy")
            coarse, pi, _keep, _mean = pick_closest_to_mean_pose(
                pose_samples, mad_k=mad_k
            )
            if coarse.board_id is None:
                coarse.board_id = 0
            for b in boards:
                b.board_id = int(coarse.board_id)
            coarse.cls_id = int(self.cfg.cls_pcb_large)
            used_det = det_samples[pi]
            coarse_bgr = bgr_samples[pi]
            T_coarse = T_samples[pi]
            self._last_wrist_coarse_bgr = coarse_bgr.copy()
            self._last_wrist_coarse_T_base_cam = np.asarray(
                T_coarse, dtype=np.float64
            )
            self._last_coarse_dets = [used_det]
            self._last_coarse_boards = [coarse]
            self._last_target_board_id = coarse.board_id
            result.coarse = coarse
            result.coarse_boards = [coarse]
            result.target_board_id = coarse.board_id
            # 定位板面 Z = 深度抬出的角点 Z 均值（与 grasp_z_m 无关）
            zs = [
                float(c[2])
                for c in (coarse.corners_base or [])
                if c is not None and len(c) >= 3
            ]
            if not zs:
                result.message = "单臂腕粗：无角点 Z（深度抬点失败）"
                return result
            z_board = float(np.mean(zs))
            result.debug = {
                "grasp_mode": grasp_mode,
                "stage": "after_coarse_detect",
                "live": True,
                "branch": "large_single",
                "arm": arm_cli,
                "z_board_m": z_board,
                "z_board_source": "coarse_depth_corner_mean",
                "grasp_z_m_unused_for_localize": (
                    None
                    if getattr(self.cfg, "large_grasp_z_m", None) is None
                    else float(grasp_z_cfg)
                ),
            }
            try:
                self._last_wrist_pose_vis = draw_pose_vis(
                    coarse_bgr,
                    used_det,
                    title=(
                        f"LARGE single WRIST coarse id={coarse.board_id} "
                        f"arm={arm_cli}"
                    ),
                )
                self._last_head_pose_vis = self._last_wrist_pose_vis
            except Exception:
                self._last_wrist_pose_vis = coarse_bgr.copy()
                self._last_head_pose_vis = self._last_wrist_pose_vis

            # 粗四角一出就落盘+推 UI（不等算到达位 / 精定位）
            if save_vis_dir:
                try:
                    out_w = Path(save_vis_dir) / "wrist"
                    out_w.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out_w / "color.jpg"), coarse_bgr)
                    if self._last_wrist_pose_vis is not None:
                        cv2.imwrite(
                            str(out_w / "pose.jpg"),
                            self._last_wrist_pose_vis,
                        )
                        cv2.imwrite(
                            str(out_w / "coarse_pose.jpg"),
                            self._last_wrist_pose_vis,
                        )
                    LOG.info(
                        "LARGE single coarse saved wrist/coarse_pose.jpg "
                        "(四角已识别，立即存图)"
                    )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("LARGE single coarse immediate save failed: %s", exc)
                self._save_stage_vis(
                    Path(save_vis_dir),
                    head_rect=None,
                    head_det=None,
                    extra_meta={
                        "grasp_mode": grasp_mode,
                        "stage": "after_wrist_coarse",
                        "arm": arm_cli,
                    },
                    save_head_pred=False,
                    save_wrist_pred=True,
                )
            _progress("coarse")

            # —— 2. 精定位：P2 → P3 ——
            # 精拍到位：相机光轴朝下（光心在角点上方 hover）；
            # 停位：TCP 末端垂直向下，rz ∥ 精更新后的 P2→P3。
            need_fine = stage in ("all", "fine") and bool(
                getattr(self.cfg, "large_fine_enable", True)
            )
            p2_c = list(coarse.corners_base[2])
            p3_c = list(coarse.corners_base[3])
            grip_rz = gripper_rz_large_parallel_p2p3(
                p2_c,
                p3_c,
                orient_rz_offset_deg=float(
                    getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
                ),
            )
            _ux = float(p2_c[0]) - float(p3_c[0])
            _uy = float(p2_c[1]) - float(p3_c[1])
            _n = math.hypot(_ux, _uy) or 1.0
            grasp_ux, grasp_uy = _ux / _n, _uy / _n

            cur_fl = (
                list(result.current_flange_right[:6])
                if use_right and result.current_flange_right
                else list(result.current_flange[:6])
                if result.current_flange
                else None
            )
            if cur_fl is None:
                cur_fl = list(arm_motion.get_flange_pose()[:6])

            # 精拍光轴朝下用外参：优先活手眼缓存
            T_tc_view = np.asarray(T_tc, dtype=np.float64)
            if cam_from_tf:
                try:
                    _ = _T_base_cam_now(cur_fl)
                    if _tf_cache.get("T_tc") is not None:
                        T_tc_view = np.asarray(
                            _tf_cache["T_tc"], dtype=np.float64
                        )
                except Exception as exc:  # noqa: BLE001
                    LOG.warning(
                        "LARGE single: 预读活手眼失败 (%s) → yaml",
                        exc,
                    )

            def _arrive_tip_and_flange(corner_xyz: Sequence[float]):
                """光心在角点上方 hover、光轴朝下；法兰由 T_tool_cam 反推。"""
                from .geometry import flange_pose_camera_down_nearest

                cam_xyz = [
                    float(corner_xyz[0]),
                    float(corner_xyz[1]),
                    float(corner_xyz[2]) + float(hover_h),
                ]
                fl, yaw = flange_pose_camera_down_nearest(
                    cam_xyz,
                    T_tool_cam=T_tc_view,
                    current_xyzrpy=cur_fl,
                )
                tip = list(cam_xyz)  # 记录的是光心目标，非 TCP
                return tip, fl, f"cam_optical_down hover yaw={math.degrees(yaw):.1f}°"

            # 粗定位到达位（精定位开始拍腕位）：右臂依次 P2 → P3
            tip_arrive_p2, fl_arrive_p2, zsrc2 = _arrive_tip_and_flange(p2_c)
            tip_arrive_p3, fl_arrive_p3, zsrc3 = _arrive_tip_and_flange(p3_c)
            # 深度漂到数米时拒绝移动（batch_048：Y≈3.6~4.2m）
            cur_xy = None
            if use_right and result.current_flange_right:
                cur_xy = result.current_flange_right[:2]
            elif result.current_flange:
                cur_xy = result.current_flange[:2]
            if cur_xy is not None:
                for label, tip in (
                    ("P2", tip_arrive_p2),
                    ("P3", tip_arrive_p3),
                ):
                    d_xy = math.hypot(
                        float(tip[0]) - float(cur_xy[0]),
                        float(tip[1]) - float(cur_xy[1]),
                    )
                    if d_xy > 1.5:
                        result.ok = False
                        result.message = (
                            f"粗定位 {label} 到达位距当前 TCP "
                            f"{d_xy*1000:.0f}mm（>1.5m），像深度漂到背景；"
                            f"tip=({tip[0]:.3f},{tip[1]:.3f},{tip[2]:.3f}) "
                            f"请检查腕深/手眼，勿执行"
                        )
                        result.debug = {
                            **(result.debug or {}),
                            "grasp_mode": grasp_mode,
                            "stage": "coarse_xy_outlier",
                            "arrive_p2_tip": list(tip_arrive_p2),
                            "arrive_p3_tip": list(tip_arrive_p3),
                            "outlier": label,
                            "d_xy_m": d_xy,
                        }
                        return result
            result.debug = {
                **(result.debug or {}),
                "arrive_p2_tip": list(tip_arrive_p2),
                "arrive_p3_tip": list(tip_arrive_p3),
                "arrive_p2_flange": list(fl_arrive_p2),
                "arrive_p3_flange": list(fl_arrive_p3),
                "arrive_z_source": {"p2": zsrc2, "p3": zsrc3},
            }
            # 粗完成后就把到达位推到网页 CLI
            def _pose_csv(p):
                if not p or len(p) < 6:
                    return None
                return (
                    f"{p[0]*1000:.4f},{p[1]*1000:.4f},{p[2]*1000:.4f},"
                    f"{math.degrees(p[3]):.4f},{math.degrees(p[4]):.4f},"
                    f"{math.degrees(p[5]):.4f}"
                )

            def _cli(arm, pose_csv):
                if not pose_csv:
                    return "—"
                # --pose= 避免负数被 argparse 当成选项
                return (
                    f"python -m rim_python.move_to_pose "
                    f"--arm {arm} --pose={pose_csv} --execute --allow-large"
                )

            cur_for_cli = (
                result.current_flange_right
                if use_right
                else result.current_flange
            )
            result.debug["cli"] = {
                "arm": arm_cli,
                "current": _cli(arm_cli, _pose_csv(cur_for_cli)),
                "current_right": _cli(arm_cli, _pose_csv(cur_for_cli))
                if use_right
                else None,
                "current_left": _cli(arm_cli, _pose_csv(cur_for_cli))
                if not use_right
                else None,
                "arrive_p2": _cli(arm_cli, _pose_csv(fl_arrive_p2)),
                "arrive_p3": _cli(arm_cli, _pose_csv(fl_arrive_p3)),
                "coarse": _cli(arm_cli, _pose_csv(fl_arrive_p2)),
                # 停位等精匹配更新 P2/P3 后再填；粗识别阶段不动臂、不算停
                "fine": None,
                "stop": None,
            }
            result.hover_flange = list(fl_arrive_p2)
            result.fine_flange = None
            _progress("coarse")

            def _pick_fine(
                corner_idx: int,
                target_xyz,
                label: str,
                hover_flange: list[float],
                tip_xyz: list[float],
                *,
                other_xyz,
            ):
                # 一次只认一个角：P2 到位只更新 P2；P3 到位只更新 P3
                other_xyz = list(other_xyz) if other_xyz is not None else None
                LOG.info(
                    "PHASE: LARGE single fine → 单角 P%d（不认另一角） "
                    "cam=(%.3f,%.3f,%.3f) flange=(%.3f,%.3f,%.3f) "
                    "rpy_deg=(%.1f,%.1f,%.1f) hover=+%.2f",
                    corner_idx,
                    tip_xyz[0],
                    tip_xyz[1],
                    tip_xyz[2],
                    hover_flange[0],
                    hover_flange[1],
                    hover_flange[2],
                    hover_flange[3] * 180.0 / math.pi,
                    hover_flange[4] * 180.0 / math.pi,
                    hover_flange[5] * 180.0 / math.pi,
                    hover_h,
                )
                # 仅全流程：先到位；到位后原地等识别（最多 ~10s），认到才回、才去下一角
                if not do_move:
                    LOG.info(
                        "LARGE single fine %s: plan_only → 跳过到位与精检 "
                        "（停 tip 等执行到位后再算）",
                        label,
                    )
                    return None

                from .rim_motion import _pose_close_enough  # noqa: PLC0415
                from .cameras import _bgr_corr  # noqa: PLC0415

                # 识别等待：认到立刻停；超时才报未识别。期间绝不动臂。
                detect_wait_s = float(
                    getattr(self.cfg, "large_fine_detect_wait_s", 10.0) or 10.0
                )
                # move_line 抛错但 XY 已近：仍原地识别，避免「看着到了又开下一角乱动」
                soft_xy_ok_mm = 80.0

                def _wrist_cam():
                    """grab 可能是 bound method 或 UI 的 _grab_right_wrist 函数。"""
                    cam = getattr(grab, "__self__", None)
                    if cam is not None and hasattr(cam, "flush_and_grab"):
                        return cam
                    return None

                # —— 动臂前：记下粗图 + seq；到位前绝不存 fine_P* ——
                coarse_ref = None
                if getattr(self, "_last_wrist_coarse_bgr", None) is not None:
                    coarse_ref = self._last_wrist_coarse_bgr.copy()
                elif getattr(self, "_last_wrist_bgr", None) is not None:
                    coarse_ref = self._last_wrist_bgr.copy()
                pre_seq = -1
                cam0 = _wrist_cam()
                if cam0 is not None and hasattr(cam0, "current_seq"):
                    try:
                        pre_seq = int(cam0.current_seq())
                    except Exception:
                        pre_seq = -1
                LOG.info(
                    "LARGE single fine %s: 动臂前 barrier seq=%d "
                    "（到位后必须更新的帧，禁止用粗定位缓存图）",
                    label,
                    pre_seq,
                )

                arrive_exc = None
                try:
                    arm_motion.move_line(
                        hover_flange,
                        velocity_scale=float(self.cfg.move_velocity_scale),
                    )
                except Exception as exc:  # noqa: BLE001
                    arrive_exc = exc
                    LOG.warning(
                        "LARGE single fine %s: move_line 报错（仍读 TCP，"
                        "近则原地等识别，远才放弃）: %s",
                        label,
                        exc,
                    )

                try:
                    self._wait_still(settle_s=0.5)
                except Exception:
                    pass

                try:
                    cur_fl_chk = list(arm_motion.get_flange_pose()[:6])
                except Exception as exc:  # noqa: BLE001
                    LOG.error(
                        "LARGE single fine %s: 到位后读 TCP 失败 → 跳过精检: %s",
                        label,
                        exc,
                    )
                    fine_status[label] = "tcp_read_fail"
                    return None

                dxy_mm = (
                    math.hypot(
                        cur_fl_chk[0] - hover_flange[0],
                        cur_fl_chk[1] - hover_flange[1],
                    )
                    * 1000.0
                )
                dz_mm = abs(cur_fl_chk[2] - hover_flange[2]) * 1000.0
                pose_ok = _pose_close_enough(cur_fl_chk, hover_flange)
                soft_ok = dxy_mm <= soft_xy_ok_mm

                if not pose_ok:
                    LOG.warning(
                        "LARGE single fine %s: TCP 未进硬到位 "
                        "Δxy=%.0fmm ΔZ=%.0fmm（目标Z=%.1f 实测=%.1f）"
                        "%s",
                        label,
                        dxy_mm,
                        dz_mm,
                        hover_flange[2] * 1000.0,
                        cur_fl_chk[2] * 1000.0,
                        " → 仍近，清缓存后再识别" if soft_ok else " → 太远，放弃本角",
                    )

                def _flush_after_arrive():
                    """到位后清缓存，且画面必须相对粗定位变了才开拍。"""
                    cam = _wrist_cam()
                    if cam is not None and hasattr(cam, "flush_and_grab"):
                        return cam.flush_and_grab(
                            n_drop=5,
                            pre_seq=pre_seq if pre_seq >= 0 else None,
                            ref_bgr=coarse_ref,
                            max_corr=0.90,
                            content_wait_s=3.0,
                        )
                    # UI 函数 grab：带 fresh → 内部 flush_and_grab
                    try:
                        return grab(
                            fresh=True,
                            pre_seq=pre_seq if pre_seq >= 0 else None,
                            ref_bgr=coarse_ref,
                        )
                    except TypeError:
                        pass
                    try:
                        return grab(fresh=True)
                    except TypeError:
                        pass
                    for _ in range(5):
                        try:
                            grab()
                        except Exception:
                            pass
                    return grab()

                def _grab_fresh_frame():
                    """识别循环内：只要比上一帧新的 bundle seq（不再全量 flush）。"""
                    try:
                        return grab(fresh=True)
                    except TypeError:
                        pass
                    try:
                        grab()
                    except Exception:
                        pass
                    return grab()

                def _save_arrive_snap(tag: str, title: str, bgr: np.ndarray) -> None:
                    try:
                        vis = bgr.copy()
                        cv2.putText(
                            vis,
                            title,
                            (8, 28),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 0, 255) if "FAIL" in tag else (0, 165, 255),
                            2,
                        )
                        self._last_wrist_bgr = bgr
                        if corner_idx == 2:
                            self._last_wrist_fine_right_vis = vis
                        else:
                            self._last_wrist_fine_left_vis = vis
                        self._last_wrist_pose_vis = vis
                        if save_vis_dir is not None:
                            out_w = Path(save_vis_dir) / "wrist"
                            out_w.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(
                                str(
                                    out_w
                                    / f"fine_P{corner_idx}_{tag}.jpg"
                                ),
                                vis,
                            )
                    except Exception:
                        pass

                if not soft_ok:
                    # 太远：仍先 flush 再存一张，避免把粗定位图标成 ARRIVE FAIL
                    try:
                        wb_fail, _ = _flush_after_arrive()
                        _save_arrive_snap(
                            "arrive_fail",
                            (
                                f"LARGE single fine {label} ARRIVE FAIL "
                                f"xy={dxy_mm:.0f} Z={dz_mm:.0f}mm"
                            ),
                            wb_fail,
                        )
                    except Exception:
                        pass
                    fine_status[label] = (
                        f"arrive_far_xy={dxy_mm:.0f}mm"
                        + (f" exc={arrive_exc}" if arrive_exc else "")
                    )
                    return None

                _invalidate_tf_tool()  # 确认到位后再重读活手眼

                # 到位 → 清缓存 → 丢管道旧帧 → 与粗图比内容 → 才存图/识别
                try:
                    wb0, wd0 = _flush_after_arrive()
                    self._last_wrist_bgr = wb0
                    self._last_wrist_depth = wd0
                    corr0 = (
                        _bgr_corr(coarse_ref, wb0)
                        if coarse_ref is not None
                        else -1.0
                    )
                    LOG.info(
                        "LARGE single fine %s: 到位后新帧 OK "
                        "corr_vs_coarse=%.3f shape=%s → 开始识别",
                        label,
                        corr0,
                        getattr(wb0, "shape", None),
                    )
                    if arrive_exc is not None or not pose_ok:
                        _save_arrive_snap(
                            "arrive_soft",
                            (
                                f"{label} soft-arrive xy={dxy_mm:.0f} "
                                f"Z={dz_mm:.0f}mm corr={corr0:.2f} "
                                f"→ wait detect {detect_wait_s:.0f}s"
                            ),
                            wb0,
                        )
                except Exception as exc:  # noqa: BLE001
                    LOG.error(
                        "LARGE single fine %s: 到位后取新帧失败 "
                        "（禁止用粗定位缓存图识别）: %s",
                        label,
                        exc,
                    )
                    fine_status[label] = f"flush_fail:{exc}"
                    return None

                corner_xyz = None
                used_fdet = None
                used_bgr = None
                last_wb = None
                last_reject_det = None
                last_reject_d_mm = None
                # 光心应对准角点：检出点须靠近画面中心，否则视为高度/到位虚到位
                max_center_px = 180.0
                # 与粗角点 XY 匹配：不再放宽到 2×（batch_064 P2 Δ135mm 曾被误收）
                match_ok = float(max_d)

                def _save_fine_vis_now(
                    bgr: np.ndarray,
                    det: PcbDetection | None,
                    *,
                    title: str,
                    tag: str,
                    ok: bool,
                ) -> None:
                    """角点一检出就落盘+推 UI，不等整段跑完。"""
                    try:
                        if det is not None:
                            vis = draw_pose_vis(bgr, det, title=title)
                        else:
                            vis = bgr.copy()
                            cv2.putText(
                                vis,
                                title,
                                (8, 28),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255) if not ok else (0, 200, 0),
                                2,
                            )
                        self._last_wrist_pose_vis = vis
                        # 单臂：按角点分槽，避免 left/right 命名混淆
                        if corner_idx == 2:
                            self._last_wrist_fine_right_vis = vis
                        else:
                            self._last_wrist_fine_left_vis = vis
                        if save_vis_dir is not None:
                            out_w = Path(save_vis_dir) / "wrist"
                            out_w.mkdir(parents=True, exist_ok=True)
                            # 只按角点存：fine_P2_*.jpg / fine_P3_*.jpg + pose_P2/P3
                            stem = f"fine_P{corner_idx}_{tag}"
                            cv2.imwrite(str(out_w / f"{stem}.jpg"), vis)
                            cv2.imwrite(
                                str(out_w / f"pose_P{corner_idx}.jpg"), vis
                            )
                            cv2.imwrite(
                                str(out_w / f"color_P{corner_idx}.jpg"), bgr
                            )
                            LOG.info(
                                "LARGE single fine saved wrist/%s.jpg "
                                "+ pose_P%d.jpg + color_P%d.jpg (%s)",
                                stem,
                                corner_idx,
                                corner_idx,
                                "OK" if ok else "reject/no-det",
                            )
                        _progress("fine")
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning(
                            "LARGE single fine %s save vis failed: %s",
                            label,
                            exc,
                        )

                LOG.info(
                    "LARGE single fine %s: 到位后原地等识别 ≤%.0fs "
                    "（认到立刻更新并离开；超时才报未识别；期间不动臂）",
                    label,
                    detect_wait_s,
                )
                t0 = _time.monotonic()
                fi = 0
                while True:
                    elapsed = _time.monotonic() - t0
                    if elapsed >= detect_wait_s and fi > 0:
                        break
                    fi += 1
                    wb, wd = _grab_fresh_frame()
                    last_wb = wb
                    self._last_wrist_bgr = wb
                    self._last_wrist_depth = wd
                    ee = list(arm_motion.get_flange_pose()[:6])
                    T_bc = _T_base_cam_now(ee)
                    # 原图无框 → vflip/hflip/rot180 再检，UV 映回原图后定位
                    dets, flip_used = detect_with_flip_fallback(fine_det, wb)
                    if flip_used != "orig" and dets:
                        LOG.info(
                            "LARGE single fine %s t=%.1fs frame %d: 原图无检出 → "
                            "%s 检出×%d，UV 已映回原图",
                            label,
                            elapsed,
                            fi,
                            flip_used,
                            len(dets),
                        )
                    best = None
                    best_d = 1e9
                    best_det = None
                    ih, iw = int(wb.shape[0]), int(wb.shape[1])
                    cx0, cy0 = 0.5 * iw, 0.5 * ih
                    for d in dets or []:
                        if not d.corners:
                            continue
                        u, v = d.corners[0]
                        dist_c = math.hypot(float(u) - cx0, float(v) - cy0)
                        if dist_c > max_center_px:
                            LOG.warning(
                                "LARGE single fine %s: 角点距画面中心 "
                                "%.0fpx > %.0fpx → 视为精拍高度/光轴未压住，丢弃",
                                label,
                                dist_c,
                                max_center_px,
                            )
                            continue
                        try:
                            cxy = corner_uv_to_base(
                                float(u),
                                float(v),
                                T_base_cam=T_bc,
                                K=wrist_K,
                                fixed_depth_cam_m=fixed_cam_d,
                            )
                        except Exception:
                            continue
                        dxy = math.hypot(
                            float(cxy[0]) - float(target_xyz[0]),
                            float(cxy[1]) - float(target_xyz[1]),
                        )
                        # 更靠近另一角 → 认错角，丢弃
                        if other_xyz is not None and len(other_xyz) >= 2:
                            d_other = math.hypot(
                                float(cxy[0]) - float(other_xyz[0]),
                                float(cxy[1]) - float(other_xyz[1]),
                            )
                            if d_other + 1e-6 < dxy:
                                LOG.warning(
                                    "LARGE single fine %s: 检出更靠近另一角 "
                                    "(Δthis=%.0fmm Δother=%.0fmm) → 丢弃",
                                    label,
                                    dxy * 1000.0,
                                    d_other * 1000.0,
                                )
                                continue
                        if dxy < best_d:
                            best_d = dxy
                            best = [
                                float(cxy[0]),
                                float(cxy[1]),
                                float(cxy[2]),
                            ]
                            best_det = d
                    flip_tag = "" if flip_used == "orig" else f" [{flip_used}]"
                    if best is not None and best_d <= match_ok:
                        # 识别成功 → 只更新当前这一个角点，再允许去下一角/停位
                        corner_xyz = best
                        used_fdet = best_det
                        used_bgr = wb
                        LOG.info(
                            "LARGE single fine %s UPDATE-ONLY P%d="
                            "(%.4f,%.4f,%.4f) Δ=%.1fmm t=%.1fs fixed_camZ=%.3f%s",
                            label,
                            corner_idx,
                            best[0],
                            best[1],
                            best[2],
                            best_d * 1000.0,
                            _time.monotonic() - t0,
                            fixed_cam_d,
                            flip_tag,
                        )
                        _save_fine_vis_now(
                            wb,
                            best_det,
                            title=(
                                f"ONLY P{corner_idx} {label} "
                                f"Δ={best_d * 1000.0:.0f}mm OK{flip_tag}"
                            ),
                            tag="ok",
                            ok=True,
                        )
                        fine_status[label] = "ok"
                        break
                    if best is not None:
                        last_reject_det = best_det
                        last_reject_d_mm = best_d * 1000.0
                        LOG.warning(
                            "LARGE single fine %s t=%.1fs: 检出但 Δcoarse=%.1fmm "
                            "> 匹配阈%.0fmm → 继续等（不更新、不动臂）%s",
                            label,
                            elapsed,
                            best_d * 1000.0,
                            match_ok * 1000.0,
                            flip_tag,
                        )
                        # 只周期性刷 reject 图，避免每帧狂写盘
                        if fi == 1 or fi % 5 == 0:
                            _save_fine_vis_now(
                                wb,
                                best_det,
                                title=(
                                    f"ONLY P{corner_idx} {label} "
                                    f"REJECT Δ={best_d * 1000.0:.0f}mm "
                                    f"wait {elapsed:.0f}/{detect_wait_s:.0f}s"
                                    f"{flip_tag}"
                                ),
                                tag="reject",
                                ok=False,
                            )
                    elif not dets:
                        if fi == 1 or fi % 5 == 0:
                            LOG.warning(
                                "LARGE single fine %s t=%.1fs frame %d: "
                                "原图+翻转均未检出 → 继续等",
                                label,
                                elapsed,
                                fi,
                            )
                    # 超时前继续睡一拍再抓；已成功则上面 break
                    if corner_xyz is not None:
                        break
                    remain = detect_wait_s - (_time.monotonic() - t0)
                    if remain <= 0:
                        break
                    _time.sleep(min(max(float(gap_f), 0.15), remain))

                if corner_xyz is None and last_wb is not None:
                    waited = _time.monotonic() - t0
                    if last_reject_det is not None:
                        _save_fine_vis_now(
                            last_wb,
                            last_reject_det,
                            title=(
                                f"LARGE single fine {label} P{corner_idx} "
                                f"FAIL Δ={last_reject_d_mm:.0f}mm "
                                f"after {waited:.0f}s"
                            ),
                            tag="fail",
                            ok=False,
                        )
                        fine_status[label] = (
                            f"match_reject_Δ={last_reject_d_mm:.0f}mm"
                            f"_wait={waited:.0f}s"
                        )
                    else:
                        _save_fine_vis_now(
                            last_wb,
                            None,
                            title=(
                                f"LARGE single fine {label} P{corner_idx} "
                                f"NO DET after {waited:.0f}s"
                            ),
                            tag="nodet",
                            ok=False,
                        )
                        fine_status[label] = f"no_det_wait={waited:.0f}s"
                    LOG.error(
                        "LARGE single fine %s: 等待 %.0fs 仍未有效识别 → 不更新 P%d、"
                        "不去停位（本角结束，下角才动）",
                        label,
                        waited,
                        corner_idx,
                    )
                return corner_xyz

            fine_p2 = None
            fine_p3 = None
            stop_ready = False
            stop_flange = None
            tip_stop_x = tip_stop_y = tip_stop_z = None
            fine_status: dict[str, str] = {}

            # 精：仅执行流程到位拍；只定位只给粗到达 CLI，不算停、不动臂
            if need_fine and do_move:
                # 严格串行：P2 到位→原地等识别≤10s→再 P3；认到才去下一角/停位
                fine_p2 = _pick_fine(
                    2,
                    p2_c,
                    "P2",
                    fl_arrive_p2,
                    tip_arrive_p2,
                    other_xyz=p3_c,
                )
                fine_p3 = _pick_fine(
                    3,
                    p3_c,
                    "P3",
                    fl_arrive_p3,
                    tip_arrive_p3,
                    other_xyz=p2_c if fine_p2 is None else fine_p2,
                )
                result.debug = {
                    **(result.debug or {}),
                    "grasp_mode": grasp_mode,
                    "stage": "after_large_single_fine",
                    "fine_p2": fine_p2,
                    "fine_p3": fine_p3,
                    "fine_status": dict(fine_status),
                    "fixed_cam_depth_m": fixed_cam_d,
                }
                # 精角点 = 单角检出后与粗 P2/P3 做 XY 最近邻匹配。
                # 缺一侧时用粗角点兜底（至少一侧精成功才去停位）。
                if fine_p2 is not None or fine_p3 is not None:
                    new_corners = [list(c) for c in coarse.corners_base]
                    use_p2 = (
                        list(fine_p2)
                        if fine_p2 is not None
                        else list(p2_c)
                    )
                    use_p3 = (
                        list(fine_p3)
                        if fine_p3 is not None
                        else list(p3_c)
                    )
                    new_corners[2] = use_p2
                    new_corners[3] = use_p3
                    if fine_p2 is None or fine_p3 is None:
                        LOG.warning(
                            "LARGE single: 精匹配部分成功 "
                            "P2=%s P3=%s → 缺侧用粗角点兜底算停位",
                            "精" if fine_p2 is not None else "粗兜底",
                            "精" if fine_p3 is not None else "粗兜底",
                        )
                    from .geometry import wrap_pi

                    rz = gripper_rz_large_parallel_p2p3(
                        new_corners[2],
                        new_corners[3],
                        orient_rz_offset_deg=float(
                            getattr(
                                self.cfg, "large_orient_rz_offset_deg", 0.0
                            )
                            or 0.0
                        ),
                    )
                    rz_off_deg = float(
                        getattr(
                            self.cfg, "large_single_stop_rz_offset_deg", 36.87
                        )
                        or 0.0
                    )
                    rz_final = wrap_pi(
                        float(rz) + math.radians(rz_off_deg)
                    )
                    LOG.info(
                        "LARGE single stop rz: base=%.1f° + offset=%.1f° → final=%.1f°",
                        math.degrees(float(rz)),
                        rz_off_deg,
                        math.degrees(float(rz_final)),
                    )
                    mid_x = 0.5 * (
                        float(new_corners[2][0]) + float(new_corners[3][0])
                    )
                    mid_y = 0.5 * (
                        float(new_corners[2][1]) + float(new_corners[3][1])
                    )
                    z_fine = 0.5 * (
                        float(new_corners[2][2]) + float(new_corners[3][2])
                    )
                    refined = ObjectPose(
                        x=mid_x,
                        y=mid_y,
                        z=float(z_fine),
                        rz=float(rz_final),
                        edge_yaw=math.atan2(
                            float(new_corners[3][1])
                            - float(new_corners[2][1]),
                            float(new_corners[3][0])
                            - float(new_corners[2][0]),
                        ),
                        corners_base=new_corners,
                        conf=float(coarse.conf),
                        board_id=coarse.board_id,
                        cls_id=int(self.cfg.cls_pcb_large),
                    )
                    coarse = refined
                    result.coarse = coarse
                    result.fine = refined
                    self._last_coarse_boards = [coarse]
                    # 板心：mid(P2,P3) + (width/2)*⊥板内；边长 width/length 来自配置
                    _ex = float(new_corners[3][0]) - float(new_corners[2][0])
                    _ey = float(new_corners[3][1]) - float(new_corners[2][1])
                    _n2 = math.hypot(_ex, _ey) or 1.0
                    grasp_ux, grasp_uy = _ex / _n2, _ey / _n2
                    # 两侧法向，取指向板内（四角中心）的一侧
                    nxa, nya = -grasp_uy, grasp_ux
                    nxb, nyb = grasp_uy, -grasp_ux
                    cx_board = float(
                        np.mean([float(c[0]) for c in new_corners[:4]])
                    )
                    cy_board = float(
                        np.mean([float(c[1]) for c in new_corners[:4]])
                    )
                    inward_x = cx_board - float(mid_x)
                    inward_y = cy_board - float(mid_y)
                    if nxa * inward_x + nya * inward_y >= nxb * inward_x + nyb * inward_y:
                        stop_nx, stop_ny = nxa, nya
                    else:
                        stop_nx, stop_ny = nxb, nyb
                    stop_off = float(stop_half_w)  # = width/2（或调试覆盖）
                    tip_stop_x = float(mid_x) + stop_off * float(stop_nx)
                    tip_stop_y = float(mid_y) + stop_off * float(stop_ny)
                    if stop_z_override is not None:
                        tip_stop_z = float(stop_z_override)
                    else:
                        # 角点 Z +10cm（精或粗兜底）
                        tip_stop_z = float(z_fine) + stop_z_above
                    # 停位：末端垂直向下固定 rx=-90° ry=90°；
                    # rz = ∥P2→P3 计算值 + stop_rz_offset_deg
                    stop_flange = [
                        float(tip_stop_x),
                        float(tip_stop_y),
                        float(tip_stop_z),
                        -0.5 * math.pi,  # rx = -90°
                        0.5 * math.pi,  # ry = +90°
                        float(rz_final),
                    ]
                    stop_ready = True
                    result.fine_flange = list(stop_flange)
                    result.hover_flange = list(stop_flange)
                    LOG.info(
                        "LARGE single 停位 tip=板心 mm/deg = "
                        "(%.1f, %.1f, %.1f, %.1f, %.1f, %.1f)  "
                        "mid23_mm=(%.1f, %.1f) + half_w=%.0fmm "
                        "(board %.0f×%.0fmm, |P3P2|=%.0fmm) "
                        "n̂=(%.2f,%.2f)",
                        tip_stop_x * 1000.0,
                        tip_stop_y * 1000.0,
                        tip_stop_z * 1000.0,
                        -90.0,
                        90.0,
                        math.degrees(float(rz_final)),
                        mid_x * 1000.0,
                        mid_y * 1000.0,
                        stop_off * 1000.0,
                        board_w * 1000.0,
                        board_len * 1000.0,
                        _n2 * 1000.0,
                        stop_nx,
                        stop_ny,
                    )
                    result.debug = {
                        **(result.debug or {}),
                        "fine_match": {
                            "p2": "fine" if fine_p2 is not None else "coarse_fallback",
                            "p3": "fine" if fine_p3 is not None else "coarse_fallback",
                        },
                        "stop_flange": list(stop_flange),
                        "stop_tip": [
                            tip_stop_x,
                            tip_stop_y,
                            tip_stop_z,
                        ],
                        "mid23_xy": [mid_x, mid_y],
                    }
                else:
                    LOG.warning(
                        "LARGE single: 精匹配两侧都失败 "
                        "(P2/P3 均未与粗角点匹配上) → 不算停 tip、不去停位"
                    )

            # —— 3. 可视化叠粗图（有精更新用新角点；否则只叠粗 P2/P3，不叠停）——
            pts = []
            if coarse.corners_base and len(coarse.corners_base) >= 4:
                pts.append(("P2", coarse.corners_base[2], (0, 165, 255)))
                pts.append(("P3", coarse.corners_base[3], (0, 255, 0)))
                if stop_ready:
                    mid = [
                        0.5
                        * (
                            float(coarse.corners_base[2][0])
                            + float(coarse.corners_base[3][0])
                        ),
                        0.5
                        * (
                            float(coarse.corners_base[2][1])
                            + float(coarse.corners_base[3][1])
                        ),
                        float(z_board),
                    ]
                    pts.append(("mid23", mid, (0, 255, 255)))
                    pts.append(
                        (
                            "stop",
                            [tip_stop_x, tip_stop_y, tip_stop_z],
                            (255, 0, 255),
                        )
                    )
            self._overlay_on_wrist_coarse(
                pts,
                title="LARGE single mid/stop/P2/P3 on wrist-coarse",
                base_bgr=self._last_wrist_coarse_bgr,
            )

            # —— 4. CLI / 停 tip：写入后再推 progress，保证网页能看到停位 ——
            result.debug = {
                **(result.debug or {}),
                "grasp_mode": grasp_mode,
                "arm": arm_cli,
                "arrive_p2_tip": list(tip_arrive_p2),
                "arrive_p3_tip": list(tip_arrive_p3),
                "arrive_p2_flange": list(fl_arrive_p2),
                "arrive_p3_flange": list(fl_arrive_p3),
                "stop_ready": stop_ready,
                "fine_p2": fine_p2,
                "fine_p3": fine_p3,
                "stop_flange": list(stop_flange) if stop_flange else None,
                "stop_tip": (
                    [tip_stop_x, tip_stop_y, tip_stop_z]
                    if stop_ready
                    else None
                ),
                "stop_dy_m": stop_dy,  # = half_width（板心半宽），非固定 13cm
                "board_width_m": board_w,
                "board_length_m": board_len,
                "stop_half_width_m": stop_half_w,
                "stop_z_above_board_m": stop_z_above,
                "fine_hover_height_m": hover_h,
                "print_xy": {
                    "mid_mode": "board_center_from_p2p3_wh",
                    "p2_xy": [
                        float(coarse.corners_base[2][0]),
                        float(coarse.corners_base[2][1]),
                    ]
                    if coarse.corners_base and len(coarse.corners_base) > 2
                    else [],
                    "p3_xy": [
                        float(coarse.corners_base[3][0]),
                        float(coarse.corners_base[3][1]),
                    ]
                    if coarse.corners_base and len(coarse.corners_base) > 3
                    else [],
                    "mid23_xy": (
                        [
                            0.5
                            * (
                                float(coarse.corners_base[2][0])
                                + float(coarse.corners_base[3][0])
                            ),
                            0.5
                            * (
                                float(coarse.corners_base[2][1])
                                + float(coarse.corners_base[3][1])
                            ),
                        ]
                        if coarse.corners_base
                        and len(coarse.corners_base) > 3
                        else None
                    ),
                    "arrive_p2_tip": list(tip_arrive_p2),
                    "arrive_p3_tip": list(tip_arrive_p3),
                    "tip_stop_xy": (
                        [tip_stop_x, tip_stop_y] if stop_ready else None
                    ),
                    "tip_stop_z": tip_stop_z if stop_ready else None,
                    "stop_flange_mm_deg": (
                        [
                            tip_stop_x * 1000.0,
                            tip_stop_y * 1000.0,
                            tip_stop_z * 1000.0,
                            -90.0,
                            90.0,
                            math.degrees(float(stop_flange[5])),
                        ]
                        if stop_ready and stop_flange is not None
                        else None
                    ),
                    "grasp_z_m": float(z_board),
                    "fine_hover_height_m": hover_h,
                    "stop_ready": stop_ready,
                    "board_width_m": board_w,
                    "board_length_m": board_len,
                    "stop_half_width_m": stop_half_w,
                    "stop_dy_m": stop_dy,
                },
            }

            cur_csv = _pose_csv(cur_for_cli)
            stop_csv = _pose_csv(stop_flange) if stop_ready else None
            result.debug["cli"] = {
                "arm": arm_cli,
                "current": _cli(arm_cli, cur_csv),
                "current_right": _cli(arm_cli, cur_csv) if use_right else None,
                "current_left": _cli(arm_cli, cur_csv)
                if not use_right
                else None,
                "arrive_p2": _cli(arm_cli, _pose_csv(fl_arrive_p2)),
                "arrive_p3": _cli(arm_cli, _pose_csv(fl_arrive_p3)),
                "coarse": _cli(arm_cli, _pose_csv(fl_arrive_p2)),
                "fine": _cli(arm_cli, stop_csv) if stop_ready else None,
                "stop": _cli(arm_cli, stop_csv) if stop_ready else None,
            }
            if need_fine and do_move:
                _progress("fine")

            if do_move and stop_ready and stage in ("all", "fine"):
                LOG.info(
                    "PHASE: LARGE single → 停位=板心 "
                    "mm/deg=(%.1f,%.1f,%.1f, %.1f,%.1f,%.1f) "
                    "half_w=%.0fmm (board %.0f×%.0f) · Z=board+%.0fmm",
                    tip_stop_x * 1000.0,
                    tip_stop_y * 1000.0,
                    tip_stop_z * 1000.0,
                    -90.0,
                    90.0,
                    math.degrees(float(stop_flange[5])),
                    stop_half_w * 1000.0,
                    board_w * 1000.0,
                    board_len * 1000.0,
                    stop_z_above * 1000.0,
                )
                arm_motion.move_line(
                    stop_flange,
                    velocity_scale=float(self.cfg.move_velocity_scale),
                )

            if save_vis_dir:
                self._save_stage_vis(
                    Path(save_vis_dir),
                    head_rect=None,
                    head_det=None,
                    extra_meta={
                        "grasp_mode": grasp_mode,
                        "stage": "large_single_done",
                        "stop_ready": stop_ready,
                        "stop_flange": (
                            list(stop_flange) if stop_flange else None
                        ),
                        "stop_tip": (
                            [tip_stop_x, tip_stop_y, tip_stop_z]
                            if stop_ready
                            else None
                        ),
                        "arm": arm_cli,
                    },
                    save_head_pred=False,
                )

            result.ok = True
            if plan_only:
                result.message = (
                    "large_single plan_only OK（粗到达已出；停 tip 等精匹配）"
                )
                result.debug["stage"] = "plan_only"
            elif stop_ready and stop_flange is not None:
                result.message = (
                    "large_single OK 停位mm/deg="
                    f"({tip_stop_x * 1000.0:.1f},"
                    f"{tip_stop_y * 1000.0:.1f},"
                    f"{tip_stop_z * 1000.0:.1f},"
                    f"-90.0,90.0,"
                    f"{math.degrees(float(stop_flange[5])):.1f})"
                )
                result.debug["stage"] = "done"
            else:
                st = (result.debug or {}).get("fine_status") or fine_status
                result.message = (
                    "large_single: P2/P3 精识别都未成功 → 未去停位。"
                    f" 状态 P2={st.get('P2', '?')} P3={st.get('P3', '?')}。"
                    "（到位后会原地等≤10s 认到才动；超时/太远才放弃）"
                )
                result.debug["stage"] = "fine_incomplete"
                result.debug["fine_status"] = dict(st)
            return result
        except Exception as exc:  # noqa: BLE001
            LOG.exception("LARGE single failed")
            result.ok = False
            result.message = f"large_single failed: {exc}"
            result.debug = {**(result.debug or {}), "grasp_mode": grasp_mode}
            return result
        finally:
            if own_cam is not None:
                try:
                    own_cam.close()
                except Exception:
                    pass


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
        - ``all``：头粗 →（大板：腕精 0/3 → 规划）→ 接近 →（小板腕精+示教高+收夹 / 大板确认对挤）
        - ``coarse``：头粗 → 接近（小板悬停 / 大板外开）；若 ``do_grasp`` 且大板则继续确认对挤
        - ``fine``：小板——头定 ID → 悬停 → 腕精按 ID → 示教高 → 收夹；
          大板——头粗 + 固定平面 → 腕精 mid(P0,P3) → 规划（权重未配则跳过腕精）

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
        # 大板单臂支路 B：无头，直接腕粗
        if self.force_class in ("large_single", "large-single", "single"):
            return self.run_large_single_arm(
                grab_wrist,
                settle_s=settle_s,
                do_grasp=do_grasp,
                save_vis_dir=save_vis_dir,
                stage=stage,
                plan_only=plan_only,
                on_progress=on_progress,
            )
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
            # 粗定位一出即落盘原图+预测标签（后续失败也能留标注素材）
            if save_vis_dir:
                self._save_stage_vis(
                    Path(save_vis_dir),
                    head_rect=rect,
                    head_det=head_det,
                    extra_meta={
                        "grasp_mode": grasp_mode,
                        "stage": "after_coarse_detect",
                    },
                    save_wrist_pred=False,
                )
            # —— 2. 接近（按 mode）——
            d_xy = 0.0
            d_xyz = 0.0
            if grasp_mode == "large":
                tool_z = float(self.cfg.large_tool_z_m)
                grasp_z_fixed = getattr(self.cfg, "large_grasp_z_m", None)
                if grasp_z_fixed is None:
                    raise RuntimeError("large_board.grasp_z_m 未写死")
                grasp_z_tip = float(grasp_z_fixed)
                z_src = "grasp_z_m"
                LOG.info(
                    "LARGE tipZ=写死 %.4f；粗到位=角点XY / grasp_z+%.0fcm（统一）；"
                    "腕精深度=当前相机到该高度，夹取 XY=角点+dx/±dy",
                    float(grasp_z_tip),
                    float(
                        getattr(self.cfg, "large_fine_hover_height_m", 0.35) or 0.35
                    )
                    * 100.0,
                )
                if coarse.corners_base is not None and len(coarse.corners_base) >= 4:
                    from .large_board import gripper_rz_large_parallel_p2p3, tip_to_flange

                    hover_h = float(
                        getattr(self.cfg, "large_fine_hover_height_m", 0.35) or 0.35
                    )
                    z_arrive = float(grasp_z_tip) + hover_h
                    p2c = coarse.corners_base[2]
                    p3c = coarse.corners_base[3]
                    grip_rz = gripper_rz_large_parallel_p2p3(
                        p2c,
                        p3c,
                        orient_rz_offset_deg=float(
                            getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
                        ),
                    )
                    ux = float(p2c[0] - p3c[0])
                    uy = float(p2c[1] - p3c[1])
                    nrm = math.hypot(ux, uy) or 1.0
                    left_arrive = tip_to_flange(
                        [float(p3c[0]), float(p3c[1]), z_arrive],
                        gripper_rz=grip_rz,
                        grasp_ux=ux / nrm,
                        grasp_uy=uy / nrm,
                        side="left",
                        tool_y_m=0.0,
                        tool_z_m=0.0,
                    )
                    right_arrive = tip_to_flange(
                        [float(p2c[0]), float(p2c[1]), z_arrive],
                        gripper_rz=grip_rz,
                        grasp_ux=ux / nrm,
                        grasp_uy=uy / nrm,
                        side="right",
                        tool_y_m=0.0,
                        tool_z_m=0.0,
                    )
                    _announce_targets(
                        "粗定位完成 · 下一步末端目标",
                        [("左臂 P3", left_arrive), ("右臂 P2", right_arrive)],
                    )
                    result.hover_flange = list(left_arrive)
                    result.coarse_flange = list(left_arrive)
                    result.debug = {
                        **(result.debug or {}),
                        "coarse_left": list(left_arrive),
                        "coarse_right": list(right_arrive),
                    }
                _progress("coarse")
                z_for_fine = float(grasp_z_tip)
                try:
                    coarse = self.refine_large_board_p0p3(
                        coarse,
                        grab_wrist,
                        grasp_z_m=z_for_fine,
                        plan_only=bool(plan_only),
                        save_vis_dir=save_vis_dir,
                    )
                    result.coarse = coarse
                    result.fine = coarse  # 大板「精」= 刷新后的 2/3 几何
                    result.fine_boards = list(
                        getattr(self, "_last_fine_boards", []) or []
                    )
                    result.debug = {
                        **(result.debug or {}),
                        "grasp_mode": grasp_mode,
                        "stage": "after_large_fine",
                        "live": True,
                        "wrist_fine_ok": bool(
                            getattr(self, "_large_wrist_fine_ok", False)
                        ),
                    }
                    if save_vis_dir:
                        self._save_stage_vis(
                            Path(save_vis_dir),
                            head_rect=rect,
                            head_det=head_det,
                            extra_meta={
                                "grasp_mode": grasp_mode,
                                "stage": "after_large_fine",
                            },
                        )
                except Exception as exc:  # noqa: BLE001
                    LOG.error("LARGE fine failed → 中止，不做接近: %s", exc)
                    if save_vis_dir and self._last_wrist_bgr is not None:
                        self._save_stage_vis(
                            Path(save_vis_dir),
                            head_rect=rect,
                            head_det=head_det,
                            extra_meta={
                                "grasp_mode": grasp_mode,
                                "stage": "after_large_fine_failed",
                            },
                        )
                    result.ok = False
                    result.message = str(exc)
                    result.debug = {
                        **(result.debug or {}),
                        "grasp_mode": grasp_mode,
                        "stage": "large_fine_failed",
                        "live": True,
                        "wrist_fine_ok": False,
                        "error": str(exc),
                    }
                    return result
                # tip 只来自腕精认出的 P2/P3；跳过/未识别时只保留粗定位「接近」
                if not bool(getattr(self, "_large_wrist_fine_ok", False)):
                    LOG.info(
                        "LARGE: 腕精未完成 → 不产出 tip（不伪造头粗角点）"
                    )
                    result.fine = None
                    result.debug = {
                        **(result.debug or {}),
                        "grasp_mode": grasp_mode,
                        "stage": "plan_only_coarse_only"
                        if plan_only
                        else "coarse_only_no_tip",
                        "live": True,
                        "move": False,
                        "wrist_fine_ok": False,
                        "left_grasp": None,
                        "right_grasp": None,
                        "left_approach": None,
                        "right_approach": None,
                    }
                    result.ok = True
                    result.message = (
                        "粗定位完成；tip 需腕精认出角点后才有"
                        + ("（只定位：当前腕图未认出）" if plan_only else "")
                    )
                    return result
                # 夹取 / 接近高度保持写死的 grasp_z；精定位只更新角点 XY
                app_h = getattr(self.cfg, "large_approach_hover_height_m", 0.0)
                if app_h is None:
                    app_h = 0.0
                else:
                    app_h = float(app_h)
                approach_z = (
                    float(grasp_z_tip) + app_h
                    if grasp_z_tip is not None
                    else None
                )
                LOG.info(
                    "LARGE final approach tipZ=%.4f (%s) app_hover=%.3f → approachZ=%.4f",
                    float(grasp_z_tip) if grasp_z_tip is not None else float("nan"),
                    z_src,
                    app_h,
                    float(approach_z) if approach_z is not None else float("nan"),
                )
                large_plan = plan_large_board_grasp(
                    coarse,
                    tool_y_m=float(self.cfg.large_tool_y_m),
                    tool_z_m=tool_z,
                    open_margin_m=float(
                        getattr(self.cfg, "large_open_margin_m", 0.0) or 0.0
                    ),
                    squeeze_close_m=float(self.cfg.large_squeeze_close_m),
                    hover_z_m=self.cfg.large_hover_z_m,
                    half_width_m=self.cfg.large_half_width_m,
                    grasp_z_m=grasp_z_tip,
                    tip_span_m=self.cfg.large_tip_span_m,
                    orient_edge=str(
                        getattr(self.cfg, "large_orient_edge", "parallel_p2p3")
                        or "parallel_p2p3"
                    ),
                    orient_rz_offset_deg=float(
                        getattr(self.cfg, "large_orient_rz_offset_deg", 0.0) or 0.0
                    ),
                    approach_dx_m=float(
                        getattr(self.cfg, "large_approach_dx_m", 0.15) or 0.15
                    ),
                    approach_dy_m=float(
                        getattr(self.cfg, "large_approach_dy_m", 0.08) or 0.08
                    ),
                    approach_z_m=approach_z,
                )
                self._last_large_plan = large_plan
                _announce_targets(
                    "精定位完成 · 夹取目标",
                    [
                        ("左接近", large_plan.left_approach),
                        ("右接近", large_plan.right_approach),
                        ("左夹取", large_plan.left_grasp),
                        ("右夹取", large_plan.right_grasp),
                    ],
                )
                result.debug = {
                    **(result.debug or {}),
                    "left_approach": list(large_plan.left_approach),
                    "right_approach": list(large_plan.right_approach),
                    "left_grasp": list(large_plan.left_grasp),
                    "right_grasp": list(large_plan.right_grasp),
                }
                _progress("fine")
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
                            f"tip P3/P2 +dx={large_plan.approach_dx_m:.2f} "
                            f"±dy={large_plan.approach_dy_m:.2f}"
                        ),
                        plan=large_plan,
                    )
                # 抓取 tip / 接近位 → 反向手眼投到头图
                self._overlay_grasp_on_head(
                    [
                        ("tipL", large_plan.tip_left, (0, 255, 0)),
                        ("tipR", large_plan.tip_right, (0, 165, 255)),
                        ("appL", large_plan.left_approach[:3], (80, 255, 80)),
                        ("appR", large_plan.right_approach[:3], (80, 200, 255)),
                    ],
                    title="LARGE grasp tip/approach ← inv(T_base_cam)",
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
                            f"--arm {arm} --pose={pose_csv} "
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
                        **(result.debug or {}),
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
                    _progress("fine")
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
                        # 含 Z：此前只看 XY，会导致 tipZ=681 差 3~4cm 仍当到位
                        dx = float(cur[0]) - float(goal[0])
                        dy = float(cur[1]) - float(goal[1])
                        dz = float(cur[2]) - float(goal[2])
                        return float(math.sqrt(dx * dx + dy * dy + dz * dz) * 1000.0)

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
                        **(result.debug or {}),
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
            if grasp_mode != "large":
                _announce_targets("粗定位完成 · 悬停目标", [("左臂", hover)])
                _progress("coarse")
                # 粗定位板心 → 头图
            self._overlay_grasp_on_head(
                    [
                        (
                            "coarse",
                            [float(coarse.x), float(coarse.y), float(coarse.z)],
                            (0, 255, 255),
                        )
                    ],
                    title="SMALL coarse center ← inv(T_base_cam)",
                )
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
                            self._overlay_grasp_on_head(
                                [
                                    (
                                        "coarse",
                                        [
                                            float(coarse.x),
                                            float(coarse.y),
                                            float(coarse.z),
                                        ],
                                        (0, 255, 255),
                                    ),
                                    (
                                        "grasp",
                                        [float(fine.x), float(fine.y), float(gz)],
                                        (0, 0, 255),
                                    ),
                                ],
                                title="SMALL grasp ← inv(T_base_cam)",
                            )
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
                            f"--arm {arm_side} --pose={pose_csv} "
                            f"--execute --allow-large"
                        )

                    # current_flange=左，current_flange_right=右（_read_both_tcp）
                    pose_l = result.current_flange
                    pose_r = result.current_flange_right
                    cur_left_csv = _pose_csv(pose_l)
                    cur_right_csv = _pose_csv(pose_r)
                    cur_left_cmd = (
                        f"python -m rim_python.move_to_pose --arm left "
                        f"--pose={cur_left_csv} --execute --allow-large"
                        if cur_left_csv
                        else "—"
                    )
                    cur_right_cmd = (
                        f"python -m rim_python.move_to_pose --arm right "
                        f"--pose={cur_right_csv} --execute --allow-large"
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
                self._overlay_grasp_on_head(
                    [
                        (
                            "coarse",
                            [float(coarse.x), float(coarse.y), float(coarse.z)],
                            (0, 255, 255),
                        ),
                        (
                            "grasp",
                            [float(fine.x), float(fine.y), float(grasp_z)],
                            (0, 0, 255),
                        ),
                    ],
                    title="SMALL grasp ← inv(T_base_cam)",
                )
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
