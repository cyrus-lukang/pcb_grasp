#!/usr/bin/env python3
"""头部离线定位：raw_left(cam7手眼) + raw_right(cam6双目) → YOLO → FFS → 基座位姿。

用法::

  cd /home/ubuntu/workspace/pcb_grasp
  .venv/bin/python projects/head_localize_offline/scripts/run_localize.py \\
    --capture .../batch_029_.../head \\
    --out projects/head_localize_offline/output/run_001
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[3]  # pcb_grasp/
sys.path.insert(0, str(ROOT))

from pcb_grasp.detect import YoloPoseDetector, draw_multi_board_vis  # noqa: E402
from pcb_grasp.localize import Intrinsics, head_localize  # noqa: E402
from pcb_grasp.stereo_depth import HeadStereoDepth, colorize_depth  # noqa: E402
from pcb_grasp.undistort import (  # noqa: E402
    T_base_from_rectified,
    build_maps_from_stereo_json,
)

CLS_NAME = {0: "pcb_small", 1: "nopcb", 2: "pcb_large"}


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _load_T_base_cam(handeye_yaml: Path) -> np.ndarray:
    d = _load_yaml(handeye_yaml)
    mat = d.get("matrix") or d.get("T_link_base_camera") or d.get("transform")
    if mat is None:
        raise RuntimeError(f"no matrix in {handeye_yaml}")
    T = np.asarray(mat, dtype=np.float64).reshape(4, 4)
    return T


def _find_pair(capture: Path) -> tuple[Path, Path]:
    """Prefer raw_left/raw_right; also accept left/right naming."""
    pairs = [
        ("raw_left.jpg", "raw_right.jpg"),
        ("raw_left.png", "raw_right.png"),
        ("left.jpg", "right.jpg"),
        ("left.png", "right.png"),
    ]
    for a, b in pairs:
        la, rb = capture / a, capture / b
        if la.is_file() and rb.is_file():
            return la, rb
    # single image: treat as left (handeye cam); depth needs pair
    for name in ("raw_right.jpg", "raw_left.jpg", "rect.jpg"):
        p = capture / name
        if p.is_file():
            raise RuntimeError(
                f"found {p.name} but need stereo pair raw_left+raw_right under {capture}"
            )
    raise FileNotFoundError(f"no raw_left/raw_right under {capture}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Offline head PCB localize")
    ap.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "config" / "localize.yaml",
    )
    ap.add_argument(
        "--capture",
        type=Path,
        required=True,
        help="目录，内含 raw_left.jpg + raw_right.jpg（或已有 depth_ffs_mm.npy + rect.jpg）",
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--use-saved-depth",
        action="store_true",
        help="优先用 capture 下已有 rect.jpg + depth_ffs_mm.npy，跳过重算 FFS",
    )
    ap.add_argument("--also-detect-right", action="store_true", default=True)
    args = ap.parse_args()

    cfg = _load_yaml(args.config)
    capture = args.capture.resolve()
    out = (
        args.out.resolve()
        if args.out
        else (
            Path(__file__).resolve().parents[1]
            / "output"
            / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    )
    out.mkdir(parents=True, exist_ok=True)

    handeye = Path(cfg["handeye_yaml"])
    weights = Path(cfg["pose_weights"])
    stereo_json = Path(cfg["stereo_calib_json"])
    T_base_cam = _load_T_base_cam(handeye)

    det_cfg = cfg.get("detect") or {}
    detector = YoloPoseDetector(
        weights,
        conf=float(det_cfg.get("conf", 0.4)),
        kpt_conf=float(det_cfg.get("kpt_conf", 0.35)),
        device=det_cfg.get("device", 0),
        imgsz=int(det_cfg.get("imgsz", 480)),
    )

    maps = build_maps_from_stereo_json(stereo_json)
    K = Intrinsics.from_K(maps.K_rect)
    T_head = T_base_from_rectified(T_base_cam, maps.R1)

    left_path, right_path = _find_pair(capture)
    left = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
    right = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
    if left is None or right is None:
        raise RuntimeError(f"cannot read {left_path} / {right_path}")

    depth = None
    depth_stats = None
    rect = None
    used_saved = False
    depth_npy = capture / "depth_ffs_mm.npy"
    rect_jpg = capture / "rect.jpg"
    if args.use_saved_depth and depth_npy.is_file() and rect_jpg.is_file():
        rect = cv2.imread(str(rect_jpg), cv2.IMREAD_COLOR)
        depth = np.load(str(depth_npy))
        used_saved = True
        print(f"reuse saved depth: {depth_npy}")
    else:
        est = HeadStereoDepth(
            maps=maps,
            stereo_calib_json=stereo_json,
            weights=str(cfg.get("ffs_weights") or "") or None,
            cfg=str(cfg.get("ffs_cfg") or "") or None,
        )
        rect, right_rect, depth, _valid, depth_stats = est.estimate_raw(left, right)
        cv2.imwrite(str(out / "right_rect.jpg"), right_rect)
        if depth_stats is not None:
            (out / "depth_stats.json").write_text(
                json.dumps(
                    {
                        "valid_ratio": float(depth_stats.valid_ratio),
                        "median_m": float(depth_stats.median_m),
                        "p10_m": float(depth_stats.p10_m),
                        "p90_m": float(depth_stats.p90_m),
                        "summary": depth_stats.summary(),
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

    assert rect is not None and depth is not None
    cv2.imwrite(str(out / "rect.jpg"), rect)
    np.save(out / "depth_ffs_mm.npy", depth)
    cv2.imwrite(str(out / "depth_ffs_color.jpg"), colorize_depth(depth))
    cv2.imwrite(str(out / "raw_left.jpg"), left)
    cv2.imwrite(str(out / "raw_right.jpg"), right)

    dets = detector.detect(rect)
    if not dets:
        raise RuntimeError("no PCB detected on rectified cam7 (raw_left)")

    large_cls = int((cfg.get("classes") or {}).get("pcb_large", 2))
    kp_i = int(cfg.get("kp_i", 0))
    kp_j = int(cfg.get("kp_j", 1))
    tray_z = float(cfg.get("head_tray_z_m", 0.71548))
    use_fixed_large = bool(cfg.get("large_use_fixed_grasp_z_plane", False))

    boards = []
    for i, det in enumerate(dets):
        cid = int(det.cls_id)
        is_large = cid == large_cls
        if is_large and not use_fixed_large:
            pose = head_localize(
                det,
                T_base_cam=T_head,
                K=K,
                depth=depth,
                depth_scale=0.001,
                tray_z_m=None,
                kp_i=kp_i,
                kp_j=kp_j,
                corner_z_max=1.0,
                required_corner_indices=(2, 3),
            )
        else:
            pose = head_localize(
                det,
                T_base_cam=T_head,
                K=K,
                depth=None if not is_large else depth,
                depth_scale=0.001,
                tray_z_m=None if is_large else tray_z,
                kp_i=kp_i,
                kp_j=kp_j,
                corner_z_max=1.0,
                required_corner_indices=(2, 3) if is_large else None,
            )
        boards.append(
            {
                "board_id": i + 1,
                "cls_id": cid,
                "cls": CLS_NAME.get(cid, f"cls{cid}"),
                "conf": float(det.conf),
                "box_xyxy": det.box,
                "corners_uv": [[float(u), float(v)] for u, v in det.corners],
                "center_uv": [float(det.center_uv[0]), float(det.center_uv[1])],
                "center_base_xyz": [float(pose.x), float(pose.y), float(pose.z)],
                "corners_base_xyz": [
                    [float(x) for x in c] for c in (pose.corners_base or [])
                ],
                "rz_rad": float(pose.rz),
                "rz_deg": float(math.degrees(pose.rz)),
                "edge_yaw_rad": float(pose.edge_yaw) if pose.edge_yaw is not None else None,
                "edge_yaw_deg": (
                    float(math.degrees(pose.edge_yaw))
                    if pose.edge_yaw is not None
                    else None
                ),
            }
        )

    # 选最近 PCB（非 nopcb）；大板优先展示
    pickable = [b for b in boards if b["cls_id"] != 1]
    if not pickable:
        pickable = boards
    target = min(
        pickable,
        key=lambda b: float(b["center_base_xyz"][0]) ** 2
        + float(b["center_base_xyz"][1]) ** 2,
    )
    for b in boards:
        b["is_target"] = b["board_id"] == target["board_id"]

    vis = draw_multi_board_vis(
        rect,
        dets,
        [b["board_id"] for b in boards],
        title="head offline",
        highlight_id=target["board_id"],
        kp_i=kp_i,
        kp_j=kp_j,
        large_cls=large_cls,
    )
    # overlay base xyz
    cx, cy = target["center_uv"]
    xyz = target["center_base_xyz"]
    cv2.putText(
        vis,
        f"base xyz=({xyz[0]:.3f},{xyz[1]:.3f},{xyz[2]:.3f}) m  rz={target['rz_deg']:.1f}deg",
        (8, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.imwrite(str(out / "pose.jpg"), vis)

    right_det_payload = None
    if args.also_detect_right:
        # 仅可视化：raw_right 不是手眼相机，角点不可直接用手眼反投影
        rdets = detector.detect(right)
        if rdets:
            rvis = draw_multi_board_vis(
                right,
                rdets,
                list(range(1, len(rdets) + 1)),
                title="raw_right detect ONLY (no handeye)",
                highlight_id=1,
                kp_i=kp_i,
                kp_j=kp_j,
                large_cls=large_cls,
            )
            cv2.imwrite(str(out / "raw_right_detect.jpg"), rvis)
            right_det_payload = [
                {
                    "cls_id": int(d.cls_id),
                    "cls": CLS_NAME.get(int(d.cls_id), f"cls{d.cls_id}"),
                    "conf": float(d.conf),
                    "corners_uv": [[float(u), float(v)] for u, v in d.corners],
                    "center_uv": [float(d.center_uv[0]), float(d.center_uv[1])],
                }
                for d in rdets
            ]

    payload = {
        "ok": True,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "project": str(Path(__file__).resolve().parents[1]),
        "capture": str(capture),
        "out": str(out),
        "handeye_yaml": str(handeye),
        "pose_weights": str(weights),
        "note": (
            "手眼 head_fisheye_right_s36 = portal cam7 = raw_left；"
            "raw_right=cam6 仅双目。定位在校正左目(rect)上完成。"
        ),
        "used_saved_depth": used_saved,
        "left_image": str(left_path),
        "right_image": str(right_path),
        "K_rect": {"fx": K.fx, "fy": K.fy, "cx": K.cx, "cy": K.cy},
        "T_base_cam_physical": T_base_cam.tolist(),
        "n_detections": len(boards),
        "target_board_id": target["board_id"],
        "detections": boards,
        "raw_right_detections": right_det_payload,
    }
    (out / "pred.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("=" * 60)
    print(f"out: {out}")
    print(f"handeye: {handeye.name}")
    print(f"model:   {weights.parent.name}/best.pt")
    print(f"detect on rect(cam7/raw_left): {len(boards)} board(s)")
    for b in boards:
        mark = " ← TARGET" if b["is_target"] else ""
        x, y, z = b["center_base_xyz"]
        print(
            f"  ID{b['board_id']} {b['cls']} conf={b['conf']:.3f}  "
            f"xyz=({x:.4f}, {y:.4f}, {z:.4f}) m  rz={b['rz_deg']:.2f}°{mark}"
        )
        print("    corners_base:")
        for i, c in enumerate(b["corners_base_xyz"]):
            print(f"      P{i}: ({c[0]:.4f}, {c[1]:.4f}, {c[2]:.4f})")
    if right_det_payload:
        print(f"raw_right detect-only: {len(right_det_payload)} (no base pose)")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
