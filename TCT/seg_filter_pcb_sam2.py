"""PCB SAM2 post-process stubs used by annotate_capture_pcb (seg path)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

CORNER_START = "tl"
CORNER_START_SEQ = {
    "tl": (0, 1, 2, 3),
    "tr": (1, 2, 3, 0),
    "br": (2, 3, 0, 1),
    "bl": (3, 0, 1, 2),
}


@dataclass
class SegResult:
    keep: bool
    corners: Optional[List[Tuple[float, float]]] = None
    reason: str = ""


def refine_poor_masks(seg, image, boxes, masks, score_thresh=0.88):
    return masks, []


def strip_white_support(mask, rgb):
    return mask


def expand_mask_with_metal(mask, rgb, box=None):
    return mask


def analyze_pcb(
    mask,
    min_blob_ratio,
    max_aspect,
    med_area,
    min_area_ratio,
    min_rect_fill,
    rgb=None,
    box=None,
    cls=0,
    order_by_metal=False,
) -> SegResult:
    return SegResult(keep=False, corners=None, reason="sam2_stub")
