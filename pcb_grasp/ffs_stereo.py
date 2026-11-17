"""Fast-FoundationStereo helpers（本项目内自带，不依赖 wzx/tray_pipeline）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_FFS_ROOT = Path("/home/ubuntu/workspace/perception/models/Fast-FoundationStereo")


def _ensure_ffs_path(ffs_root: Path) -> None:
    root = Path(ffs_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Fast-FoundationStereo 不存在: {root}")
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)


def load_stereo_model(
    weights: Path,
    cfg: Path,
    iters: int,
    max_disp: int,
    *,
    ffs_root: Path = DEFAULT_FFS_ROOT,
) -> Any:
    if not Path(weights).is_file():
        raise FileNotFoundError(f"找不到 Fast-FoundationStereo 权重: {weights}")
    if not Path(cfg).is_file():
        raise FileNotFoundError(f"找不到 Fast-FoundationStereo 配置: {cfg}")
    _ensure_ffs_path(Path(ffs_root))

    import torch
    import yaml as yamllib
    from omegaconf import OmegaConf

    from core.foundation_stereo import FoundationStereo  # noqa: F401

    with Path(cfg).open() as handle:
        model_cfg = yamllib.safe_load(handle)
    model_args = OmegaConf.create(model_cfg)
    model_args.valid_iters = iters
    model_args.max_disp = max_disp

    torch.autograd.set_grad_enabled(False)
    model = torch.load(Path(weights), map_location="cpu", weights_only=False)
    model.args.valid_iters = model_args.valid_iters
    model.args.max_disp = model_args.max_disp
    try:
        model.args.normalize = True
    except Exception:
        model.args["normalize"] = True
    model.cuda().eval()
    return model


def stereo_depth(
    left_bgr,
    right_bgr,
    fx: float,
    baseline_mm: float,
    *,
    iters: int,
    max_disp: int,
    model: Any,
    ffs_root: Path = DEFAULT_FFS_ROOT,
) -> tuple[np.ndarray, np.ndarray]:
    """已矫正左右图 → uint16 毫米深度 + valid mask。"""
    _ensure_ffs_path(Path(ffs_root))
    import torch
    from core.utils.utils import InputPadder
    from Utils import AMP_DTYPE

    left = cv2.cvtColor(left_bgr, cv2.COLOR_BGR2RGB)
    right = cv2.cvtColor(right_bgr, cv2.COLOR_BGR2RGB)
    height, width = left.shape[:2]
    t0 = torch.as_tensor(left).cuda().float()[None].permute(0, 3, 1, 2)
    t1 = torch.as_tensor(right).cuda().float()[None].permute(0, 3, 1, 2)
    padder = InputPadder(t0.shape, divis_by=32, force_square=False)
    t0, t1 = padder.pad(t0, t1)
    with torch.amp.autocast("cuda", enabled=True, dtype=AMP_DTYPE):
        disp = model.forward(
            t0, t1, iters=iters, test_mode=True, optimize_build_volume="pytorch1"
        )
    disp = padder.unpad(disp.float()).data.cpu().numpy().reshape(height, width)
    disp = np.clip(disp, 0, None).astype(np.float32)
    xx = np.arange(width, dtype=np.float32)[None, :]
    disp[(xx - disp) < 0] = np.inf
    depth_mm = fx * baseline_mm / disp
    valid = np.isfinite(depth_mm) & (depth_mm > 50.0) & (depth_mm < 20000.0)
    depth = np.zeros((height, width), np.uint16)
    depth[valid] = np.clip(np.rint(depth_mm[valid]), 1, 65535).astype(np.uint16)
    return depth, valid
