"""Minimal Box / YOLO load stubs; SAM2 path needs full install separately."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Set

import numpy as np
from PIL import Image

CORNER_COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]


@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    cls: int = 0


def load_yolo(
    path: Path,
    w: int,
    h: int,
    cls_ids: Optional[Set[int]] = None,
) -> List[Box]:
    if not Path(path).is_file():
        return []
    boxes: List[Box] = []
    for line in Path(path).read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        cid = int(float(p[0]))
        if cls_ids is not None and cid not in cls_ids:
            continue
        cx, cy, bw, bh = map(float, p[1:5])
        boxes.append(
            Box(
                (cx - bw / 2) * w,
                (cy - bh / 2) * h,
                (cx + bw / 2) * w,
                (cy + bh / 2) * h,
                cls=cid,
            )
        )
    return boxes


class Sam2Segmenter:
    def __init__(self, model_id: str = "facebook/sam2.1-hiera-large"):
        raise RuntimeError(
            f"SAM2 未接入本机 stub (model={model_id}). "
            "请用 --skip-seg 只做检测,或补全 seg_filter_imu_sam2.Sam2Segmenter。"
        )

    def masks_for_boxes(self, image, boxes, points=None):
        raise NotImplementedError


def draw_mask_overlay(image, masks, results, name, out_dir):
    """No-op stub when SAM2 overlays are unavailable."""
    return
