"""Coarse/fine PCB board ID: sort by base distance, match wrist→head."""

from __future__ import annotations

import logging
import math
from typing import Sequence

from .geometry import ObjectPose

LOG = logging.getLogger("pcb_grasp.board_id")

CLS_NAME = {0: "pcb", 1: "nopcb", 2: "pcb_large"}


def cls_name(cls_id: int | None) -> str:
    if cls_id is None:
        return "?"
    return CLS_NAME.get(int(cls_id), f"cls{int(cls_id)}")


def dist_to_base(p: ObjectPose, *, metric: str = "xy") -> float:
    """Distance from board center to base origin."""
    if metric == "xyz":
        return float(math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z))
    return float(math.hypot(p.x, p.y))


def pose_xy_dist(a: ObjectPose, b: ObjectPose) -> float:
    return float(math.hypot(a.x - b.x, a.y - b.y))


def assign_board_ids_by_distance(
    poses: Sequence[ObjectPose],
    *,
    metric: str = "xy",
) -> list[ObjectPose]:
    """Sort nearest→farthest to base; assign board_id = 1,2,3,…"""
    if not poses:
        return []
    ranked = sorted(
        enumerate(poses),
        key=lambda it: dist_to_base(it[1], metric=metric),
    )
    out: list[ObjectPose] = []
    for new_id, (_old_i, p) in enumerate(ranked, start=1):
        q = ObjectPose(
            x=p.x,
            y=p.y,
            z=p.z,
            rz=p.rz,
            edge_yaw=p.edge_yaw,
            corners_base=[list(c) for c in p.corners_base],
            conf=p.conf,
            board_id=int(new_id),
            cls_id=int(getattr(p, "cls_id", 0) or 0),
        )
        out.append(q)
        LOG.info(
            "COARSE board_id=%d %s dist_%s=%.3fm xyz=(%.3f,%.3f,%.3f) conf=%.2f",
            new_id,
            cls_name(q.cls_id),
            metric,
            dist_to_base(q, metric=metric),
            q.x,
            q.y,
            q.z,
            q.conf,
        )
    return out


def match_fine_to_coarse(
    fine_poses: Sequence[ObjectPose],
    coarse_boards: Sequence[ObjectPose],
    *,
    max_dist_m: float = 0.10,
) -> list[ObjectPose]:
    """Assign each fine pose the nearest coarse board_id (within max_dist_m)."""
    if not coarse_boards:
        return []
    used_coarse: set[int] = set()
    # Greedy: nearest fine↔coarse pairs first (avoids two fines claiming same id)
    pairs: list[tuple[float, int, int]] = []
    for fi, f in enumerate(fine_poses):
        for ci, c in enumerate(coarse_boards):
            d = pose_xy_dist(f, c)
            pairs.append((d, fi, ci))
    pairs.sort(key=lambda t: t[0])

    fine_to_id: dict[int, int] = {}
    for d, fi, ci in pairs:
        if fi in fine_to_id or ci in used_coarse:
            continue
        if d > float(max_dist_m):
            continue
        cid = coarse_boards[ci].board_id
        if cid is None:
            continue
        fine_to_id[fi] = int(cid)
        used_coarse.add(ci)

    out: list[ObjectPose] = []
    for fi, f in enumerate(fine_poses):
        bid = fine_to_id.get(fi)
        # 继承粗匹配板的类别；无匹配则保留腕部自身 cls
        cls = int(getattr(f, "cls_id", 0) or 0)
        if bid is not None:
            for c in coarse_boards:
                if c.board_id == bid:
                    cls = int(getattr(c, "cls_id", cls) or cls)
                    break
        q = ObjectPose(
            x=f.x,
            y=f.y,
            z=f.z,
            rz=f.rz,
            edge_yaw=f.edge_yaw,
            corners_base=[list(c) for c in f.corners_base],
            conf=f.conf,
            board_id=bid,
            cls_id=cls,
        )
        out.append(q)
        if bid is None:
            LOG.warning(
                "FINE unmatched xyz=(%.3f,%.3f,%.3f) (max_dist=%.0fmm)",
                f.x,
                f.y,
                f.z,
                max_dist_m * 1000.0,
            )
        else:
            LOG.info(
                "FINE matched → board_id=%d %s xyz=(%.3f,%.3f,%.3f)",
                bid,
                cls_name(cls),
                f.x,
                f.y,
                f.z,
            )
    return out


def select_board(
    boards: Sequence[ObjectPose],
    target_id: int = 1,
) -> ObjectPose:
    for b in boards:
        if b.board_id == int(target_id):
            return b
    ids = [b.board_id for b in boards]
    raise RuntimeError(
        f"target board_id={target_id} not found among {ids}"
    )


def select_grasp_board(
    boards: Sequence[ObjectPose],
    target_id: int = 1,
    *,
    pcb_cls: int = 0,
    large_cls: int | None = 2,
    prefer_large: bool = True,
) -> ObjectPose:
    """选抓取目标：有大板时优先大板；否则取第 target_id 近的小板（跳过 nopcb）。"""
    if prefer_large and large_cls is not None:
        larges = [
            b
            for b in boards
            if int(getattr(b, "cls_id", 0) or 0) == int(large_cls)
        ]
        if larges:
            larges_sorted = sorted(larges, key=lambda b: int(b.board_id or 999))
            idx = int(target_id) - 1
            if 0 <= idx < len(larges_sorted):
                LOG.info(
                    "prefer large board_id=%s among %d large",
                    larges_sorted[idx].board_id,
                    len(larges_sorted),
                )
                return larges_sorted[idx]
            return larges_sorted[0]

    pcbs = [b for b in boards if int(getattr(b, "cls_id", 0) or 0) == int(pcb_cls)]
    if not pcbs:
        LOG.warning("no pcb (cls=%d) among boards; fall back to all", pcb_cls)
        return select_board(boards, target_id)
    pcbs_sorted = sorted(pcbs, key=lambda b: int(b.board_id or 999))
    idx = int(target_id) - 1
    if idx < 0:
        idx = 0
    if idx >= len(pcbs_sorted):
        # 只抓能识别到的：目标 ID 超出已检出小板数 → 用最远一块（仍是认出的）
        LOG.warning(
            "target pcb index=%d > n_pcb=%d ids=%s；改用已识别的 id=%s",
            target_id,
            len(pcbs_sorted),
            [b.board_id for b in pcbs_sorted],
            pcbs_sorted[-1].board_id,
        )
        return pcbs_sorted[-1]
    return pcbs_sorted[idx]
