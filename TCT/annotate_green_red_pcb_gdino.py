"""Green/red PCB color filters + NMS helpers for annotate_capture_pcb."""
from __future__ import annotations

import cv2
import numpy as np


def green_frac(rgb: np.ndarray, box, h_lo=25, s_min=20, v_min=20) -> float:
    """框内 HSV 绿色像素占比。真 PCB 通常 >=0.1,白框误检接近 0。"""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (h_lo, s_min, v_min), (100, 255, 255))
    return float((m > 0).mean())


def red_frac(rgb: np.ndarray, box, s_min=40, v_min=40) -> float:
    """框内 HSV 红色像素占比(含 H 绕 0 的两端)。"""
    H, W = rgb.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(rgb[y1:y2, x1:x2], cv2.COLOR_RGB2HSV)
    m1 = cv2.inRange(hsv, (0, s_min, v_min), (10, 255, 255))
    m2 = cv2.inRange(hsv, (160, s_min, v_min), (180, 255, 255))
    return float(((m1 > 0) | (m2 > 0)).mean())


def nms(boxes, scores, thr=0.4):
    if not boxes:
        return []
    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    idx = scores.argsort()[::-1]
    keep = []
    while len(idx):
        i = int(idx[0])
        keep.append(i)
        if len(idx) == 1:
            break
        rest = idx[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        ua = ((boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) +
              (boxes[rest, 2] - boxes[rest, 0]) *
              (boxes[rest, 3] - boxes[rest, 1]) - inter)
        idx = rest[(inter / np.maximum(ua, 1e-6)) < thr]
    return keep


def _area(box):
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def containment_frac(inner, outer) -> float:
    xx1 = max(inner[0], outer[0])
    yy1 = max(inner[1], outer[1])
    xx2 = min(inner[2], outer[2])
    yy2 = min(inner[3], outer[3])
    inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
    a = _area(inner)
    return inter / a if a > 1e-6 else 0.0


def drop_outer_boxes(items, contain_thr=0.7):
    """大框套小框时去掉大框。items: (tag, box, score)"""
    n = len(items)
    drop = set()
    for i in range(n):
        if i in drop:
            continue
        _, bi, _ = items[i]
        ai = _area(bi)
        for j in range(n):
            if i == j or j in drop:
                continue
            _, bj, _ = items[j]
            aj = _area(bj)
            if ai <= aj:
                continue
            if containment_frac(bj, bi) >= contain_thr:
                drop.add(i)
                break
    return [items[k] for k in range(n) if k not in drop]


def cross_class_nms(items, thr=0.45):
    """跨绿/红类 NMS: IoU 高时保留分数更高者。items: (tag, box, score)"""
    if len(items) <= 1:
        return items
    order = sorted(range(len(items)), key=lambda i: items[i][2], reverse=True)
    keep = []
    suppressed = set()
    for i in order:
        if i in suppressed:
            continue
        keep.append(i)
        _, bi, _ = items[i]
        ai = _area(bi)
        for j in order:
            if j == i or j in suppressed:
                continue
            _, bj, _ = items[j]
            xx1 = max(bi[0], bj[0])
            yy1 = max(bi[1], bj[1])
            xx2 = min(bi[2], bj[2])
            yy2 = min(bi[3], bj[3])
            inter = max(0.0, xx2 - xx1) * max(0.0, yy2 - yy1)
            ua = ai + _area(bj) - inter
            if ua > 0 and inter / ua >= thr:
                suppressed.add(j)
    return [items[i] for i in keep]
