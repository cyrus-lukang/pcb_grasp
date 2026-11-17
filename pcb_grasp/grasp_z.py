"""Teach / load fixed grasp flange Z (meters, Link_Base)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

LOG = logging.getLogger("pcb_grasp.grasp_z")

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GRASP_Z_FILE = ROOT / "config" / "grasp_z.json"


def load_grasp_z_m(path: str | Path | None = None) -> float | None:
    p = Path(path) if path else DEFAULT_GRASP_Z_FILE
    if not p.is_file():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    if data.get("grasp_z_m") is None:
        return None
    return float(data["grasp_z_m"])


def save_grasp_z_m(
    z_m: float,
    path: str | Path | None = None,
    *,
    meta: dict[str, Any] | None = None,
) -> Path:
    p = Path(path) if path else DEFAULT_GRASP_Z_FILE
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "grasp_z_m": float(z_m),
        "unit": "m",
        "frame": "Link_Base flange Z",
        "note": "精定位 XY(+rz) 到位后降到此固定高度，再收夹爪",
    }
    if meta:
        payload["meta"] = meta
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    LOG.info("saved grasp_z_m=%.4f m (%.1f mm) → %s", z_m, z_m * 1000.0, p)
    return p
