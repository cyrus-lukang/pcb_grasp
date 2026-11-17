"""Minimal helpers shared with annotate_capture_pcb (YOLO dump / area)."""
from __future__ import annotations


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def to_yolo(cid, box, w, h, score=None):
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    s = f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
    if score is not None:
        s += f" {score:.4f}"
    return s
