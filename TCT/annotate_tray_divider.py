"""Tray detect helpers for annotate_capture_pcb --inside-tray."""
from __future__ import annotations

from annotate import ClassConfig

TRAY_CC = ClassConfig(
    name="tray",
    prompts=["black tray", "grid tray", "compartment tray"],
    threshold=0.30,
)


def pick_tray(dets, W, H, min_frac=0.04):
    """Pick largest tray-like det; fallback to full image."""
    img_area = float(W * H)
    best = None
    best_a = 0.0
    for d in dets:
        x1, y1, x2, y2 = d.bbox
        a = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if a < min_frac * img_area:
            continue
        if a > best_a:
            best_a = a
            best = [float(x1), float(y1), float(x2), float(y2)]
    return best if best is not None else [0.0, 0.0, float(W), float(H)]
