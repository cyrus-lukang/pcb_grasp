"""Per-run experiment batch folders for live grasp artifacts."""

from __future__ import annotations

import json
import logging
import math
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

LOG = logging.getLogger("pcb_grasp.experiment_log")

_BATCH_RE = re.compile(r"^batch_(\d+)")


def next_batch_dir(root: str | Path) -> Path:
    """Create ``root/batch_NNN_YYYYMMDD_HHMMSS`` (NNN = next index)."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    max_n = 0
    for p in root.iterdir():
        if not p.is_dir():
            continue
        m = _BATCH_RE.match(p.name)
        if m:
            max_n = max(max_n, int(m.group(1)))
    n = max_n + 1
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch = root / f"batch_{n:03d}_{stamp}"
    batch.mkdir(parents=True, exist_ok=False)
    LOG.info("experiment batch → %s", batch)
    return batch


def _write_meta(batch: Path, meta: dict[str, Any]) -> None:
    path = batch / "meta.json"
    prev: dict[str, Any] = {}
    if path.is_file():
        try:
            prev = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            prev = {}
    prev.update(meta)
    path.write_text(json.dumps(prev, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def colorize_depth_any(depth: np.ndarray, *, depth_scale: float = 0.001) -> np.ndarray:
    """Colorize depth whether float meters or uint16 mm/raw."""
    from .stereo_depth import colorize_depth

    if depth.dtype == np.float32 or depth.dtype == np.float64:
        mm = np.zeros(depth.shape[:2], dtype=np.uint16)
        valid = depth > 0.05
        mm[valid] = np.clip(depth[valid] * 1000.0, 0, 65535).astype(np.uint16)
        return colorize_depth(mm)
    # uint16: may be mm (D405 with scale 0.0001 → raw*0.0001=m → raw is 0.1mm?
    # Our wrist_depth_scale=0.0001 means raw*0.0001 = m → raw units are 0.1mm
    # For viz: convert to mm approx
    d = depth.astype(np.float32) * float(depth_scale) * 1000.0  # → mm
    mm = np.clip(d, 0, 65535).astype(np.uint16)
    return colorize_depth(mm)


_ANNOTATE_README = """# 人工标注对照

本目录在粗/精定位识别后自动生成，供人工标角点后与模型预测对比。

## 文件

| 文件 | 含义 |
|------|------|
| `coarse_image.png` | 头相机校正原图（无绘制），粗定位用 |
| `coarse_pred.json` | 模型粗定位预测（角点像素 + 基座 xyz + rz） |
| `fine_image.png` | 腕相机原图（无绘制），精定位用 |
| `fine_pred.json` | 模型精定位预测 |

同内容也在 `../head/rect.jpg` + `../head/pred.json`、`../wrist/color.jpg` + `../wrist/pred.json`。

## 角点编号（与模型一致）

`0=TL, 1=TR, 2=BR, 3=BL`（图像坐标，原点左上，u 向右，v 向下）。

## 建议人工标注格式

复制 `*_pred.json` 为 `*_gt.json`，只改 `detections[*].corners_uv`（及可选 `box_xyxy`）。
对比时：

- **角点误差**：同 board_id / 同角点 index 的像素 L2；有相机内参+外参时可抬到基座算 mm
- **rz 误差**：用人工角点按同规则重算 rz（小板∥P0→P1，大板⊥P0→P3），再与 `rz_deg` 做最小角差
"""


def _edge_angle_deg(corners_uv: list[list[float]], i: int, j: int) -> float | None:
    if len(corners_uv) <= max(i, j):
        return None
    du = float(corners_uv[j][0]) - float(corners_uv[i][0])
    dv = float(corners_uv[j][1]) - float(corners_uv[i][1])
    return float(math.degrees(math.atan2(dv, du)))


def detection_to_dict(
    det: Any,
    *,
    pose: Any | None = None,
    board_id: int | None = None,
    is_target: bool = False,
) -> dict[str, Any]:
    """Serialize one YOLO-Pose detection (+ optional localized pose) for annotation."""
    corners_uv = [[float(u), float(v)] for u, v in det.corners]
    bid = board_id
    if bid is None and pose is not None:
        bid = getattr(pose, "board_id", None)
    item: dict[str, Any] = {
        "board_id": None if bid is None else int(bid),
        "cls_id": int(getattr(det, "cls_id", 0) or 0),
        "conf": float(getattr(det, "conf", 0.0) or 0.0),
        "box_xyxy": (
            None
            if not getattr(det, "box", None)
            else [float(x) for x in det.box]
        ),
        "corners_uv": corners_uv,
        "center_uv": [float(x) for x in det.center_uv],
        "edge_uv_deg": {
            "p0_p1": _edge_angle_deg(corners_uv, 0, 1),
            "p0_p3": _edge_angle_deg(corners_uv, 0, 3),
        },
        "is_target": bool(is_target),
    }
    if pose is not None:
        item["cls_id"] = int(getattr(pose, "cls_id", item["cls_id"]) or item["cls_id"])
        item["center_base_xyz"] = [
            float(pose.x),
            float(pose.y),
            float(pose.z),
        ]
        item["rz_rad"] = float(pose.rz)
        item["rz_deg"] = float(math.degrees(pose.rz))
        item["edge_yaw_rad"] = float(pose.edge_yaw)
        item["edge_yaw_deg"] = float(math.degrees(pose.edge_yaw))
        cb = getattr(pose, "corners_base", None) or []
        item["corners_base_xyz"] = [
            [float(c[0]), float(c[1]), float(c[2])] for c in cb
        ]
        item["conf"] = float(getattr(pose, "conf", item["conf"]) or item["conf"])
    return item


def build_pred_payload(
    *,
    stage: str,
    camera: str,
    image_rel: str,
    image_bgr: np.ndarray | None,
    detections: Sequence[Any] | None,
    poses: Sequence[Any] | None = None,
    target_board_id: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build structured pred JSON (pixel labels + optional base pose / rz)."""
    h = w = 0
    if image_bgr is not None and getattr(image_bgr, "shape", None) is not None:
        h, w = int(image_bgr.shape[0]), int(image_bgr.shape[1])
    dets = list(detections or [])
    pose_list = list(poses or [])
    items: list[dict[str, Any]] = []
    for i, det in enumerate(dets):
        pose = pose_list[i] if i < len(pose_list) else None
        bid = None if pose is None else getattr(pose, "board_id", None)
        is_tgt = (
            bid is not None
            and target_board_id is not None
            and int(bid) == int(target_board_id)
        )
        items.append(
            detection_to_dict(det, pose=pose, board_id=bid, is_target=is_tgt)
        )
    # 仅有 pose、无对齐 det 时仍写出基座位姿（便于 rz 对照）
    if not items and pose_list:
        for pose in pose_list:
            bid = getattr(pose, "board_id", None)
            is_tgt = (
                bid is not None
                and target_board_id is not None
                and int(bid) == int(target_board_id)
            )
            cb = getattr(pose, "corners_base", None) or []
            items.append(
                {
                    "board_id": None if bid is None else int(bid),
                    "cls_id": int(getattr(pose, "cls_id", 0) or 0),
                    "conf": float(getattr(pose, "conf", 0.0) or 0.0),
                    "box_xyxy": None,
                    "corners_uv": None,
                    "center_uv": None,
                    "center_base_xyz": [
                        float(pose.x),
                        float(pose.y),
                        float(pose.z),
                    ],
                    "rz_rad": float(pose.rz),
                    "rz_deg": float(math.degrees(pose.rz)),
                    "edge_yaw_rad": float(pose.edge_yaw),
                    "edge_yaw_deg": float(math.degrees(pose.edge_yaw)),
                    "corners_base_xyz": [
                        [float(c[0]), float(c[1]), float(c[2])] for c in cb
                    ],
                    "is_target": bool(is_tgt),
                    "note": "pose-only（无图像角点）",
                }
            )
    payload: dict[str, Any] = {
        "stage": stage,
        "camera": camera,
        "image": image_rel,
        "image_size": [w, h],
        "corner_order": ["0=TL", "1=TR", "2=BR", "3=BL"],
        "target_board_id": target_board_id,
        "n_detections": len(items),
        "detections": items,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "usage": (
            "复制本文件为同目录 *_gt.json，只改 corners_uv；"
            "角点误差用同 index 像素差；rz 用人工角点按同规则重算后与 rz_deg 比"
        ),
    }
    if extra:
        payload.update(extra)
    return payload


def save_pred_labels(
    batch: Path,
    *,
    stage: str,
    camera: str,
    image_rel: str,
    image_bgr: np.ndarray | None,
    pred_json_rel: str,
    annotate_stem: str | None = None,
    detections: Sequence[Any] | None = None,
    poses: Sequence[Any] | None = None,
    target_board_id: int | None = None,
    extra: dict[str, Any] | None = None,
    write_annotate_png: bool = True,
) -> list[str]:
    """Write pred JSON (+ optional lossless annotate PNG copy of the original)."""
    batch = Path(batch)
    batch.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    payload = build_pred_payload(
        stage=stage,
        camera=camera,
        image_rel=image_rel,
        image_bgr=image_bgr,
        detections=detections,
        poses=poses,
        target_board_id=target_board_id,
        extra=extra,
    )
    pred_path = batch / pred_json_rel
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    written.append(str(pred_path.relative_to(batch)))

    if annotate_stem:
        ann = batch / "annotate"
        ann.mkdir(parents=True, exist_ok=True)
        readme = ann / "README.md"
        if not readme.is_file():
            readme.write_text(_ANNOTATE_README, encoding="utf-8")
            written.append("annotate/README.md")
        ann_json = ann / f"{annotate_stem}_pred.json"
        # annotate 内 image 字段改为同目录 png，便于单独拷走标注
        ann_payload = dict(payload)
        ann_payload["image"] = f"{annotate_stem}_image.png"
        ann_json.write_text(
            json.dumps(ann_payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append(str(ann_json.relative_to(batch)))
        if write_annotate_png and image_bgr is not None:
            png = ann / f"{annotate_stem}_image.png"
            cv2.imwrite(str(png), image_bgr)
            written.append(str(png.relative_to(batch)))
    LOG.info("pred labels [%s] → %s (%d det)", stage, pred_json_rel, payload["n_detections"])
    return written


def save_experiment_artifacts(
    batch: Path,
    *,
    head_pose_vis: np.ndarray | None = None,
    head_rect: np.ndarray | None = None,
    head_raw_left: np.ndarray | None = None,
    head_raw_right: np.ndarray | None = None,
    head_depth_u16_mm: np.ndarray | None = None,
    head_depth_stats: dict[str, Any] | None = None,
    wrist_pose_vis: np.ndarray | None = None,
    wrist_match_vis: np.ndarray | None = None,
    wrist_color: np.ndarray | None = None,
    wrist_depth: np.ndarray | None = None,
    wrist_depth_scale: float = 0.0001,
    meta: dict[str, Any] | None = None,
    # 粗/精预测标签（识别后保留，供人工标注对照）
    head_pred_dets: Sequence[Any] | None = None,
    head_pred_poses: Sequence[Any] | None = None,
    wrist_pred_dets: Sequence[Any] | None = None,
    wrist_pred_poses: Sequence[Any] | None = None,
    target_board_id: int | None = None,
    save_head_pred: bool = True,
    save_wrist_pred: bool = True,
) -> list[str]:
    """Save one experiment round's images/depths/pred labels. Returns written paths."""
    batch = Path(batch)
    batch.mkdir(parents=True, exist_ok=True)
    head_dir = batch / "head"
    wrist_dir = batch / "wrist"
    head_dir.mkdir(exist_ok=True)
    wrist_dir.mkdir(exist_ok=True)
    written: list[str] = []

    def _im(path: Path, img: np.ndarray | None) -> None:
        if img is None:
            return
        cv2.imwrite(str(path), img)
        written.append(str(path.relative_to(batch)))

    def _npy(path: Path, arr: np.ndarray | None) -> None:
        if arr is None:
            return
        np.save(str(path), arr)
        written.append(str(path.relative_to(batch)))

    _im(head_dir / "pose.jpg", head_pose_vis)
    _im(head_dir / "rect.jpg", head_rect)
    _im(head_dir / "raw_left.jpg", head_raw_left)
    _im(head_dir / "raw_right.jpg", head_raw_right)
    if head_depth_u16_mm is not None:
        _npy(head_dir / "depth_ffs_mm.npy", head_depth_u16_mm)
        from .stereo_depth import colorize_depth

        _im(head_dir / "depth_ffs_color.jpg", colorize_depth(head_depth_u16_mm))
        # also 16-bit png for quick view in some tools
        cv2.imwrite(str(head_dir / "depth_ffs_mm.png"), head_depth_u16_mm)
        written.append("head/depth_ffs_mm.png")
    if head_depth_stats is not None:
        (head_dir / "depth_stats.json").write_text(
            json.dumps(head_depth_stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.append("head/depth_stats.json")

    _im(wrist_dir / "pose.jpg", wrist_pose_vis)
    _im(wrist_dir / "match.jpg", wrist_match_vis)
    _im(wrist_dir / "color.jpg", wrist_color)
    if wrist_depth is not None:
        _npy(wrist_dir / "depth.npy", wrist_depth)
        _im(
            wrist_dir / "depth_color.jpg",
            colorize_depth_any(wrist_depth, depth_scale=wrist_depth_scale),
        )

    if save_head_pred and (
        head_pred_dets or head_pred_poses or head_rect is not None
    ):
        if head_pred_dets or head_pred_poses:
            written.extend(
                save_pred_labels(
                    batch,
                    stage="coarse",
                    camera="head_rect",
                    image_rel="head/rect.jpg",
                    image_bgr=head_rect,
                    pred_json_rel="head/pred.json",
                    annotate_stem="coarse",
                    detections=head_pred_dets,
                    poses=head_pred_poses,
                    target_board_id=target_board_id,
                )
            )

    if save_wrist_pred and (wrist_pred_dets or wrist_pred_poses):
        written.extend(
            save_pred_labels(
                batch,
                stage="fine",
                camera="wrist_color",
                image_rel="wrist/color.jpg",
                image_bgr=wrist_color,
                pred_json_rel="wrist/pred.json",
                annotate_stem="fine",
                detections=wrist_pred_dets,
                poses=wrist_pred_poses,
                target_board_id=target_board_id,
            )
        )

    payload = {
        "batch": batch.name,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "files": written,
    }
    if meta:
        payload.update(meta)
    _write_meta(batch, payload)
    LOG.info("saved %d artifacts → %s", len(written), batch)
    return written
