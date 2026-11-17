"""UI 复位位姿：记录当前法兰 XYZ，姿态固定 rpy=(-90°, 0°, 0°)。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Sequence

DEFAULT_FILE = Path(__file__).resolve().parents[1] / "config" / "reset_pose.json"

# 复位原点姿态（度 / 弧度）
RESET_RPY_DEG = (-90.0, 0.0, 0.0)
RESET_RPY_RAD = tuple(math.radians(d) for d in RESET_RPY_DEG)


def make_reset_pose(xyz: Sequence[float]) -> list[float]:
    """当前 XYZ + 固定 rpy=(-90°,0°,0°)。"""
    return [
        float(xyz[0]),
        float(xyz[1]),
        float(xyz[2]),
        float(RESET_RPY_RAD[0]),
        float(RESET_RPY_RAD[1]),
        float(RESET_RPY_RAD[2]),
    ]


def save_reset_pose(
    xyzrpy: Sequence[float],
    path: Path | str | None = None,
    *,
    arm: str = "A",
) -> Path:
    """写入复位位：只用传入位姿的 XYZ，rpy 强制 (-90,0,0)。"""
    out = Path(path) if path else DEFAULT_FILE
    if len(xyzrpy) < 3:
        raise ValueError("复位位姿至少需要 xyz")
    pose = make_reset_pose(xyzrpy[:3])
    payload = {
        "xyzrpy_m_rad": pose,
        "xyz_m": pose[:3],
        "rpy_rad": list(pose[3:6]),
        "rpy_deg": list(RESET_RPY_DEG),
        "arm": str(arm),
        "frame": "base_link · arm_*_linkTCP",
        "note": "「记为复位位」只记当前 XYZ；姿态固定 rpy=(-90°,0°,0°)",
        "fixed_rpy_deg": list(RESET_RPY_DEG),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return out


def load_reset_pose(path: Path | str | None = None) -> list[float] | None:
    """读复位位；始终用固定 rpy=(-90,0,0)，兼容旧文件。"""
    p = Path(path) if path else DEFAULT_FILE
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    pose = data.get("xyzrpy_m_rad") or data.get("pose")
    if not pose or len(pose) < 3:
        return None
    return make_reset_pose(pose[:3])


def reset_pose_brief(path: Path | str | None = None) -> dict | None:
    p = Path(path) if path else DEFAULT_FILE
    pose = load_reset_pose(p)
    if pose is None:
        return None
    return {
        "xyz_m": [round(pose[0], 4), round(pose[1], 4), round(pose[2], 4)],
        "rpy_deg": [round(d, 1) for d in RESET_RPY_DEG],
        "file": str(p),
    }
