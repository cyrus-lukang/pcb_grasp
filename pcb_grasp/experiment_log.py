"""Per-run experiment batch folders for live grasp artifacts."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

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
) -> list[str]:
    """Save one experiment round's images/depths. Returns list of written filenames."""
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
