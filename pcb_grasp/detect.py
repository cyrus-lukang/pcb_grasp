"""YOLO-Pose PCB corner detection (ndarray in, corners out)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


@dataclass
class PcbDetection:
    """Four corners in image pixels: 0=TL, 1=TR, 2=BR, 3=BL (model order)."""

    corners: list[tuple[float, float]]  # len 4
    box: list[float] | None
    conf: float
    cls_id: int = 0

    @property
    def center_uv(self) -> tuple[float, float]:
        xs = [c[0] for c in self.corners]
        ys = [c[1] for c in self.corners]
        return (sum(xs) / 4.0, sum(ys) / 4.0)


class YoloPoseDetector:
    def __init__(
        self,
        weights: str | Path,
        *,
        conf: float = 0.4,
        kpt_conf: float = 0.35,
        device: int | str = 0,
        imgsz: int = 640,
        n_corners: int = 4,
        class_ids: list[int] | None = None,
    ) -> None:
        from ultralytics import YOLO

        self.model = YOLO(str(weights))
        self.conf = conf
        self.kpt_conf = kpt_conf
        self.device = device
        self.imgsz = imgsz
        self.n_corners = n_corners
        # None = keep all classes；粗定位 2cls 头模传 [0] 只留 PCB
        self.class_ids = None if class_ids is None else {int(c) for c in class_ids}

    def detect(self, image_bgr: np.ndarray) -> list[PcbDetection]:
        r = self.model.predict(
            image_bgr,
            conf=self.conf,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
            classes=sorted(self.class_ids) if self.class_ids is not None else None,
        )[0]
        out: list[PcbDetection] = []
        if r.boxes is None or len(r.boxes) == 0 or r.keypoints is None:
            return out
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        if r.boxes.cls is not None:
            clss = r.boxes.cls.cpu().numpy().astype(int)
        else:
            clss = np.zeros(len(boxes), dtype=int)
        kxy = r.keypoints.xy.cpu().numpy()
        kcf = (
            r.keypoints.conf.cpu().numpy()
            if r.keypoints.conf is not None
            else np.ones((len(kxy), kxy.shape[1]), dtype=np.float32)
        )
        for i in range(len(boxes)):
            cid = int(clss[i]) if i < len(clss) else 0
            if self.class_ids is not None and cid not in self.class_ids:
                continue
            corners: list[tuple[float, float]] = []
            ok = True
            for j in range(self.n_corners):
                if j >= kxy.shape[1] or float(kcf[i, j]) < self.kpt_conf:
                    ok = False
                    break
                corners.append((float(kxy[i, j, 0]), float(kxy[i, j, 1])))
            if not ok or len(corners) < self.n_corners:
                continue
            out.append(
                PcbDetection(
                    corners=corners,
                    box=[float(x) for x in boxes[i].tolist()],
                    conf=float(confs[i]),
                    cls_id=cid,
                )
            )
        out.sort(key=lambda d: d.conf, reverse=True)
        return out

    def detect_best(self, image_bgr: np.ndarray) -> PcbDetection | None:
        dets = self.detect(image_bgr)
        return dets[0] if dets else None


def _unmap_uv_from_flip(
    u: float, v: float, *, flip: str, h: int, w: int
) -> tuple[float, float]:
    """把翻转/旋转图上的像素坐标映回原图。"""
    if flip in ("", "orig", "none"):
        return float(u), float(v)
    if flip == "vflip":
        return float(u), float(h - 1 - v)
    if flip == "hflip":
        return float(w - 1 - u), float(v)
    if flip == "rot180":
        return float(w - 1 - u), float(h - 1 - v)
    return float(u), float(v)


def _remap_det_to_orig(
    det: PcbDetection, *, flip: str, h: int, w: int
) -> PcbDetection:
    corners = [
        _unmap_uv_from_flip(float(u), float(v), flip=flip, h=h, w=w)
        for u, v in det.corners
    ]
    box = None
    if det.box and len(det.box) >= 4:
        x1, y1, x2, y2 = (float(x) for x in det.box[:4])
        p1 = _unmap_uv_from_flip(x1, y1, flip=flip, h=h, w=w)
        p2 = _unmap_uv_from_flip(x2, y2, flip=flip, h=h, w=w)
        box = [
            min(p1[0], p2[0]),
            min(p1[1], p2[1]),
            max(p1[0], p2[0]),
            max(p1[1], p2[1]),
        ]
    return PcbDetection(
        corners=corners, box=box, conf=float(det.conf), cls_id=int(det.cls_id)
    )


def detect_with_flip_fallback(
    detector: YoloPoseDetector,
    image_bgr: np.ndarray,
    *,
    flips: Sequence[str] = ("vflip", "hflip", "rot180"),
) -> tuple[list[PcbDetection], str]:
    """原图先检；无框时再垂直/水平/180°翻转检，UV 映回原图。

    Returns:
        (dets_in_original_uv, flip_name)  flip_name 为 ``orig`` 或所用变换名。
    """
    import cv2

    dets = detector.detect(image_bgr)
    if dets:
        return dets, "orig"
    h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    for name in flips:
        if name == "vflip":
            im = cv2.flip(image_bgr, 0)
        elif name == "hflip":
            im = cv2.flip(image_bgr, 1)
        elif name == "rot180":
            im = cv2.rotate(image_bgr, cv2.ROTATE_180)
        else:
            continue
        dets_f = detector.detect(im)
        if not dets_f:
            continue
        remapped = [
            _remap_det_to_orig(d, flip=name, h=h, w=w) for d in dets_f
        ]
        remapped.sort(key=lambda d: d.conf, reverse=True)
        return remapped, name
    return [], "orig"


def draw_detection(image_bgr: np.ndarray, det: PcbDetection) -> np.ndarray:
    import cv2

    vis = image_bgr.copy()
    pts = [(int(round(u)), int(round(v))) for u, v in det.corners]
    for a, b in zip(pts, pts[1:] + pts[:1]):
        cv2.line(vis, a, b, (0, 255, 0), 2)
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 5, (0, 0, 255), -1)
        cv2.putText(
            vis, str(i), (p[0] + 4, p[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2
        )
    if det.box:
        x1, y1, x2, y2 = [int(round(v)) for v in det.box]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)
    return vis


def draw_pose_vis(
    image_bgr: np.ndarray,
    det: PcbDetection,
    *,
    title: str = "pose",
    kpt_dir: tuple[int, int] = (1, 0),
    board_id: int | None = None,
) -> np.ndarray:
    """Pose visualization: box + keypoints + direction arrow (default P1→P0)."""
    import cv2

    vis = draw_detection(image_bgr, det)
    pts = det.corners
    i0, i1 = int(kpt_dir[0]), int(kpt_dir[1])
    tag = title if board_id is None else f"{title} ID={board_id}"
    if 0 <= i0 < len(pts) and 0 <= i1 < len(pts):
        p_from = (int(round(pts[i0][0])), int(round(pts[i0][1])))
        p_to = (int(round(pts[i1][0])), int(round(pts[i1][1])))
        cv2.arrowedLine(vis, p_from, p_to, (255, 255, 0), 2, tipLength=0.25)
        dx = float(pts[i1][0] - pts[i0][0])
        dy = float(pts[i1][1] - pts[i0][1])
        ang = float(np.degrees(np.arctan2(dy, dx)))
        cv2.putText(
            vis,
            f"{tag}  P{i0}→P{i1} ang={ang:.1f}",
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
    else:
        cv2.putText(
            vis, tag, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )
    if board_id is not None:
        cx, cy = det.center_uv
        cv2.putText(
            vis,
            f"ID{board_id}",
            (int(round(cx)) - 20, int(round(cy)) - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            3,
        )
    return vis


_CLS_NAME = {0: "pcb", 1: "nopcb", 2: "pcb_large"}
# pcb=绿, nopcb=橙, 大板=青, 抓取目标=红
_CLS_COLOR = {0: (0, 220, 0), 1: (0, 165, 255), 2: (220, 200, 0)}


def draw_multi_board_vis(
    image_bgr: np.ndarray,
    dets: list[PcbDetection],
    board_ids: list[int | None],
    *,
    title: str = "boards",
    highlight_id: int | None = None,
    kp_i: int = 0,
    kp_j: int = 1,
    large_cls: int = 2,
    force_edge_for_id: tuple[int, int] | None = None,
) -> np.ndarray:
    """Draw multiple detections with board IDs + numbered corners; highlight grasp target.

    小板参考边默认 ``kp_i→kp_j``（配置多为 P0→P1）；
    大板参考边固定 **P0→P3**（与 tipL / 夹手 rz 一致）。
    ``force_edge_for_id``：对高亮目标强制某条边（如强制大板几何时）。
    """
    import cv2

    vis = image_bgr.copy()
    cv2.putText(
        vis, title, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
    )
    for det, bid in zip(dets, board_ids):
        cid = int(getattr(det, "cls_id", 0) or 0)
        cname = _CLS_NAME.get(cid, f"cls{cid}")
        if highlight_id is not None and bid == highlight_id:
            color = (0, 0, 255)
            thick = 2
        else:
            color = _CLS_COLOR.get(cid, (0, 255, 0))
            thick = 2 if cid == 1 else 1
        # 大板夹取边 = P0→P3；小板 = 配置 kp
        if (
            force_edge_for_id is not None
            and highlight_id is not None
            and bid == highlight_id
        ):
            ei, ej = int(force_edge_for_id[0]), int(force_edge_for_id[1])
        elif cid == int(large_cls):
            ei, ej = 0, 3
        else:
            ei, ej = int(kp_i), int(kp_j)
        pts = [(int(round(u)), int(round(v))) for u, v in det.corners]
        for a, b in zip(pts, pts[1:] + pts[:1]):
            cv2.line(vis, a, b, color, thick)
        for i, p in enumerate(pts):
            cv2.circle(vis, p, 5, (0, 0, 255) if i in (ei, ej) else (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(i),
                (p[0] + 4, p[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )
        if 0 <= ei < len(pts) and 0 <= ej < len(pts):
            cv2.arrowedLine(
                vis, pts[ei], pts[ej], (255, 255, 0), 2, tipLength=0.25
            )
            dx = float(pts[ej][0] - pts[ei][0])
            dy = float(pts[ej][1] - pts[ei][1])
            ang = float(np.degrees(np.arctan2(dy, dx)))
            if highlight_id is not None and bid == highlight_id:
                cv2.putText(
                    vis,
                    f"P{ei}→P{ej} img={ang:.1f}°",
                    (8, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 255),
                    2,
                )
        if det.box:
            x1, y1, x2, y2 = [int(round(v)) for v in det.box]
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 1)
        cx, cy = det.center_uv
        id_s = "?" if bid is None else str(int(bid))
        cv2.putText(
            vis,
            f"ID{id_s} {cname}",
            (int(round(cx)) - 40, int(round(cy)) - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            color,
            2,
        )
    return vis


def draw_grasp_points_on_head(
    image_bgr: np.ndarray,
    points: list[tuple[str, Sequence[float], tuple[int, int, int]]],
    *,
    K,
    T_base_cam: np.ndarray,
    title: str | None = "grasp←base (handeye)",
) -> np.ndarray:
    """把基座系抓取点经 ``inv(T_base_cam)`` 投到头相机图上并标注。

    ``points``: ``(label, xyz_base, bgr_color)``；越界/相机后方跳过。
    """
    import cv2

    from .localize import base_xyz_to_pixel

    vis = image_bgr.copy()
    h, w = vis.shape[:2]
    if title:
        cv2.putText(
            vis, title, (8, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 255), 2
        )
    for label, xyz, color in points:
        if xyz is None or len(xyz) < 3:
            continue
        uv = base_xyz_to_pixel(xyz, K=K, T_base_cam=T_base_cam)
        if uv is None:
            continue
        u, v = int(round(uv[0])), int(round(uv[1]))
        if u < -20 or v < -20 or u > w + 20 or v > h + 20:
            continue
        cv2.drawMarker(
            vis, (u, v), color, markerType=cv2.MARKER_CROSS, markerSize=22, thickness=2
        )
        cv2.circle(vis, (u, v), 8, color, 2)
        cv2.putText(
            vis,
            str(label),
            (u + 10, v - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return vis
