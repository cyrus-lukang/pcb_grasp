#!/usr/bin/env python3
"""pcb_grasp 全流程网页控制台。

相机与机械臂分开：
  - 只连相机 → 看头摄视野 +「识别抓取点」（不动臂）
  - 连上机械臂 → 大板到接近位（tip 左右各外开）→「内收 5mm」步进夹近（无自动合拢）

  ./run_ui.sh
  open http://127.0.0.1:8770
"""

from __future__ import annotations

import json
import logging
import math
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcb_grasp.config import load_config
from pcb_grasp.experiment_log import next_batch_dir

HOST = "127.0.0.1"
PORT = 8770
UI_DIR = Path(__file__).resolve().parent
INDEX = UI_DIR / "index.html"

LOG = logging.getLogger("pcb_grasp.ui")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

_lock = threading.Lock()
_cfg = load_config(ROOT / "config" / "grasp_config.yaml")
_job: dict = {
    "running": False,
    "ok": None,
    "message": "",
    "result": None,
    "batch_dir": "",
    "error": "",
    "awaiting_confirm": False,
    "confirm_message": "",
    "confirm_plan": None,
    "confirm_board_id": None,
    "inward_steps": 0,
}
_robot: dict = {
    "connecting": False,
    "connected": False,
    "error": "",
    "ip": "",
    "mode": "",
}
_camera: dict = {
    "connecting": False,
    "connected": False,
    "error": "",
    "mode": "",  # head | head+wrist
}
_dual = None
_motion = None
_head_cam = None
_wrist_cam = None
_preview_pipe = None
_preview_jpeg: bytes = b""
_wrist_preview_jpeg: bytes = b""
_preview_meta: dict = {"ok": False, "error": "", "ts": 0.0, "n_det": 0}
_confirm_jpeg: bytes = b""
_confirm_event = threading.Event()
_confirm_answer: bool | None = None
# 大板接近后：网页「内收 5mm」步进（沿 grasp 法向），不再一次合拢
_inward_ctx: dict | None = None
_INWARD_STEP_M = 0.005
_cam_lock = threading.Lock()
# 本轮作业开始时的左右臂法兰位姿（复位用）
_session_home_left: list[float] | None = None
_session_home_right: list[float] | None = None


def _record_session_home() -> dict:
    """作业启动时记下当前左右法兰，供「复位」回到起点。"""
    global _session_home_left, _session_home_right
    left = right = None
    with _lock:
        motion, dual = _motion, _dual
    if motion is not None:
        try:
            left = [float(x) for x in motion.get_flange_pose()[:6]]
        except Exception as exc:
            LOG.warning("record session home left failed: %s", exc)
    if dual is not None:
        try:
            rm = getattr(dual, "right", None)
            if rm is not None:
                right = [float(x) for x in rm.get_flange_pose()[:6]]
        except Exception as exc:
            LOG.warning("record session home right failed: %s", exc)
    with _lock:
        _session_home_left = left
        _session_home_right = right
    LOG.info(
        "session home recorded L=%s R=%s",
        None if left is None else tuple(round(v, 4) for v in left[:3]),
        None if right is None else tuple(round(v, 4) for v in right[:3]),
    )
    return {
        "left": left,
        "right": right,
        "has_home": left is not None or right is not None,
    }


def _go_session_home() -> dict:
    """双臂回到本轮作业起点；清 job 状态便于重新定位/抓取。"""
    global _session_home_left, _session_home_right, _inward_ctx
    global _preview_jpeg, _confirm_jpeg, _wrist_preview_jpeg
    with _lock:
        if _job.get("running"):
            raise ValueError("任务进行中，请先等结束或急停")
        left = list(_session_home_left) if _session_home_left else None
        right = list(_session_home_right) if _session_home_right else None
        dual, motion = _dual, _motion
    if left is None and right is None:
        raise ValueError("尚未记录作业起点（请先跑一次「全流程」）")
    if motion is None and dual is None:
        raise ValueError("请先「连接机械臂」")
    scale = float(getattr(_cfg, "move_velocity_scale", 0.25))
    moved = []
    if dual is not None and left is not None and right is not None and hasattr(dual, "move_both"):
        LOG.info("session home → move_both")
        dual.move_both(left, right, velocity_scale=scale, parallel=True)
        moved = ["left", "right"]
    else:
        if left is not None and motion is not None:
            LOG.info("session home → left %s", [round(v, 4) for v in left[:3]])
            try:
                motion.move_line(left, velocity_scale=scale, keep_orientation=False, direct=False)
            except TypeError:
                motion.move_line(left, velocity_scale=scale)
            moved.append("left")
        if right is not None and dual is not None:
            rm = getattr(dual, "right", None)
            if rm is not None:
                LOG.info("session home → right %s", [round(v, 4) for v in right[:3]])
                try:
                    rm.move_line(right, velocity_scale=scale, keep_orientation=False, direct=False)
                except TypeError:
                    rm.move_line(right, velocity_scale=scale)
                moved.append("right")
    with _lock:
        _inward_ctx = None
        _preview_jpeg = b""
        _confirm_jpeg = b""
        _wrist_preview_jpeg = b""
        _job.update(
            {
                "running": False,
                "ok": None,
                "message": "已复位到作业起点，可刷新后重新定位/抓取",
                "result": None,
                "error": "",
                "awaiting_confirm": False,
                "confirm_message": "",
                "confirm_plan": None,
                "confirm_board_id": None,
                "inward_steps": 0,
                "has_wrist_vis": False,
            }
        )
    return {
        "ok": True,
        "moved": moved,
        "message": "已复位到作业起点，请刷新页面后重新「只定位/全流程」",
        "home": {
            "left": left,
            "right": right,
        },
    }


def _robot_snapshot() -> dict:
    with _lock:
        return dict(_robot)


def _set_robot(**kwargs) -> None:
    with _lock:
        _robot.update(kwargs)


def _camera_snapshot() -> dict:
    with _lock:
        return dict(_camera)


def _set_camera(**kwargs) -> None:
    with _lock:
        _camera.update(kwargs)


def _job_snapshot() -> dict:
    with _lock:
        return dict(_job)


def _reset_pose_info() -> dict | None:
    try:
        from pcb_grasp.reset_pose import reset_pose_brief

        return reset_pose_brief()
    except Exception:
        return None


def _encode_jpeg(bgr, *, max_w: int = 1280, quality: int = 80) -> bytes:
    import cv2

    img = bgr
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / float(w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _close_motion() -> None:
    global _dual, _motion
    if _dual is not None:
        try:
            _dual.close()
        except Exception:
            pass
        _dual = None
        _motion = None
    elif _motion is not None:
        try:
            _motion.close()
        except Exception:
            pass
        _motion = None
    _set_robot(connected=False, connecting=False, mode="")


def _close_cams() -> None:
    global _head_cam, _wrist_cam, _preview_pipe
    if _wrist_cam is not None:
        try:
            _wrist_cam.close()
        except Exception:
            pass
        _wrist_cam = None
    if _head_cam is not None:
        try:
            _head_cam.close()
        except Exception:
            pass
        _head_cam = None
    _preview_pipe = None
    _set_camera(connected=False, connecting=False, mode="", error="")


def _ensure_head_cam():
    global _head_cam
    if _head_cam is None:
        from pcb_grasp.cameras import open_head_camera

        src = getattr(_cfg, "cam_head_source", "portal")
        LOG.info("opening head camera source=%s", src)
        _head_cam = open_head_camera(_cfg)
    return _head_cam


def _ensure_wrist_cam():
    global _wrist_cam
    if _wrist_cam is None:
        from pcb_grasp.cameras import open_wrist_camera

        src = str(getattr(_cfg, "cam_wrist_source", "portal") or "portal")
        LOG.info(
            "opening wrist source=%s portal_id=%s",
            src,
            getattr(_cfg, "cam_portal_wrist_id", 1),
        )
        _wrist_cam = open_wrist_camera(_cfg)
    return _wrist_cam


def _grab_head():
    return _ensure_head_cam().grab()


def _grab_wrist():
    return _ensure_wrist_cam().grab()


def _connect_camera() -> dict:
    """Open head (required) and wrist (optional). Independent of robot."""
    global _head_cam, _wrist_cam
    _set_camera(connecting=True, connected=False, error="", mode="")
    try:
        with _cam_lock:
            # 先关掉旧 watch，避免 portal「too many watches」
            if _wrist_cam is not None:
                try:
                    _wrist_cam.close()
                except Exception:
                    pass
                _wrist_cam = None
            if _head_cam is not None:
                try:
                    _head_cam.close()
                except Exception:
                    pass
                _head_cam = None

            src = str(getattr(_cfg, "cam_head_source", "portal"))
            wrist_src = str(getattr(_cfg, "cam_wrist_source", "portal") or "portal")
            portal = str(getattr(_cfg, "cam_portal", ""))
            _portalish = (
                "portal",
                "watch",
                "watch_portal",
                "rim",
                "bundle",
                "frame_bundle",
            )
            both_portal = src.lower() in _portalish and wrist_src.lower() in (
                *_portalish,
                "305",
                "gemini",
            )
            if both_portal:
                from pcb_grasp.cameras import open_portal_head_and_wrist

                bundle = str(getattr(_cfg, "rim_frame_bundle", "10.3.8.7:50066"))
                LOG.info(
                    "opening shared frame_bundle head+wrist @ %s",
                    bundle,
                )
                _head_cam, _wrist_cam = open_portal_head_and_wrist(_cfg)
                mode = f"head:bundle@{bundle}+wrist"
            else:
                head = _ensure_head_cam()
                mode = f"head:{src}"
                if src.lower() in _portalish:
                    bundle = str(getattr(_cfg, "rim_frame_bundle", portal or ""))
                    mode = f"head:bundle@{bundle}"
                try:
                    _ensure_wrist_cam()
                    mode = mode + "+wrist"
                except Exception as exc:
                    LOG.warning("wrist camera skipped: %s", exc)
                _ = head
        _set_camera(connecting=False, connected=True, error="", mode=mode)
        LOG.info("camera connected mode=%s", mode)
    except Exception as exc:
        LOG.warning("camera connect failed: %s", exc)
        with _cam_lock:
            try:
                if _wrist_cam is not None:
                    _wrist_cam.close()
            except Exception:
                pass
            _wrist_cam = None
            try:
                if _head_cam is not None:
                    _head_cam.close()
            except Exception:
                pass
            _head_cam = None
        _set_camera(
            connecting=False,
            connected=False,
            error=str(exc) or "相机连接失败",
            mode="",
        )
    return _camera_snapshot()


def _connect_camera_async() -> None:
    def _worker() -> None:
        snap = _camera_snapshot()
        if snap.get("connecting") or snap.get("connected"):
            return
        _connect_camera()

    threading.Thread(target=_worker, name="camera-connect", daemon=True).start()


def _ensure_preview_pipe():
    global _preview_pipe
    if _preview_pipe is None:
        from pcb_grasp.pipeline import GraspPipeline

        _preview_pipe = GraspPipeline(cfg=_cfg, motion=None)
    return _preview_pipe


def _grab_preview(*, detect: bool = True) -> tuple[bytes, dict]:
    """Grab head FOV (rectified left) with optional YOLO overlay."""
    global _preview_jpeg, _preview_meta
    import cv2
    from pcb_grasp.detect import draw_multi_board_vis

    if not _camera_snapshot().get("connected"):
        raise RuntimeError("相机未连接，请先点「连接相机」")

    with _cam_lock:
        pipe = _ensure_preview_pipe()
        maps = pipe._ensure_head_maps()  # noqa: SLF001
        left, _right = _grab_head()
        rect = maps.undistort_left(left)
        n_det = 0
        vis = rect.copy()
        if detect:
            try:
                # 头预览只加载头模，避免小板腕模缺失拖垮预览
                if pipe._head_det is None:  # noqa: SLF001
                    pipe._ensure_detectors()  # noqa: SLF001
                assert pipe._head_det is not None  # noqa: SLF001
                dets = pipe._head_det.detect(rect)  # noqa: SLF001
                n_det = len(dets)
                if dets:
                    ids = list(range(1, n_det + 1))
                    vis = draw_multi_board_vis(
                        rect,
                        dets,
                        ids,
                        title=f"HEAD FOV detect ×{n_det}",
                        highlight_id=1,
                    )
                else:
                    cv2.putText(
                        vis,
                        "HEAD FOV (no detect)",
                        (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 200, 255),
                        2,
                    )
            except Exception as exc:
                LOG.warning("preview detect: %s", exc)
                cv2.putText(
                    vis,
                    "HEAD FOV (detect err)",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
        else:
            cv2.putText(
                vis, "HEAD FOV", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2
            )
        jpeg = _encode_jpeg(vis)
        meta = {"ok": True, "error": "", "ts": time.time(), "n_det": n_det}
        with _lock:
            _preview_jpeg = jpeg
            _preview_meta = dict(meta)
        return jpeg, meta


def _robot_endpoint_label(cfg) -> str:
    backend = str(getattr(cfg, "motion_backend", "rim") or "rim").lower()
    if backend == "tj":
        return str(cfg.tj_robot_ip)
    return str(getattr(cfg, "rim_frame_bundle", "10.3.8.7:50066"))


def _connect_robot() -> dict:
    """Try real robot connect. Never raises to caller — updates _robot status."""
    global _dual, _motion, _cfg
    cfg = _cfg
    endpoint = _robot_endpoint_label(cfg)
    backend = str(getattr(cfg, "motion_backend", "rim") or "rim").lower()
    _set_robot(
        connecting=True,
        connected=False,
        error="",
        ip=endpoint,
        mode="",
    )
    try:
        cfg.dry_run = False
        with _lock:
            old_dual, old_motion = _dual, _motion
            _dual = None
            _motion = None
        if old_dual is not None:
            try:
                old_dual.close()
            except Exception:
                pass
        elif old_motion is not None:
            try:
                old_motion.close()
            except Exception:
                pass

        from pcb_grasp.robot_factory import connect_real_motion

        motion, dual = connect_real_motion(cfg, record_grasp_z=False)
        mode = (
            f"{backend}-dual"
            if dual is not None
            else f"{backend}-single"
        )
        with _lock:
            _dual = dual
            _motion = motion
        _set_robot(
            connecting=False,
            connected=True,
            error="",
            ip=endpoint,
            mode=mode,
        )
        LOG.info("robot connected %s @ %s", mode, endpoint)
    except Exception as exc:
        LOG.warning("robot connect failed: %s", exc)
        with _lock:
            _dual = None
            _motion = None
        _set_robot(
            connecting=False,
            connected=False,
            error=str(exc) or "连接失败",
            ip=_robot_endpoint_label(_cfg),
            mode="",
        )
    return _robot_snapshot()


def _dry_motion():
    from pcb_grasp.dual_tj_motion import DryRunDualMotion
    from pcb_grasp.motion import DryRunMotion

    motion = DryRunMotion(initial_pose=_cfg.home_pose)
    dual = DryRunDualMotion()
    dual.connect()
    return motion, dual


def _pose_payload(pose) -> dict | None:
    if pose is None:
        return None
    return {
        "xyz": [float(x) for x in pose.xyz],
        "rz": float(pose.rz),
        "cls_id": int(getattr(pose, "cls_id", 0) or 0),
        "board_id": getattr(pose, "board_id", None),
        "conf": float(getattr(pose, "conf", 0.0) or 0.0),
    }


def _result_payload(result, batch_dir: Path) -> dict:
    coarse = _pose_payload(result.coarse)
    fine = _pose_payload(result.fine)
    dxy_mm = None
    if coarse is not None and fine is not None:
        dx = (fine["xyz"][0] - coarse["xyz"][0]) * 1000.0
        dy = (fine["xyz"][1] - coarse["xyz"][1]) * 1000.0
        dxy_mm = {
            "dx": round(dx, 1),
            "dy": round(dy, 1),
            "hypot": round((dx * dx + dy * dy) ** 0.5, 1),
        }
    def _mm_deg(p):
        if not p or len(p) < 6:
            return None
        return [
            round(float(p[0]) * 1000.0, 1),
            round(float(p[1]) * 1000.0, 1),
            round(float(p[2]) * 1000.0, 1),
            round(math.degrees(float(p[3])), 1),
            round(math.degrees(float(p[4])), 1),
            round(math.degrees(float(p[5])), 1),
        ]

    cur = getattr(result, "current_flange", None)
    cur_r = getattr(result, "current_flange_right", None)
    dbg = result.debug or {}
    cli = dbg.get("cli") or {}
    left_app = dbg.get("left_approach")
    right_app = dbg.get("right_approach")
    coarse_left = dbg.get("coarse_left")
    coarse_right = dbg.get("coarse_right")
    left_grasp = dbg.get("left_grasp")
    right_grasp = dbg.get("right_grasp")
    return {
        "ok": bool(result.ok),
        "message": result.message,
        "batch_dir": str(batch_dir),
        "grasp_mode": dbg.get("grasp_mode", "small"),
        "target_board_id": result.target_board_id,
        "current_flange": cur,
        "current_flange_right": cur_r,
        "hover_flange": result.hover_flange,
        "fine_flange": result.fine_flange,
        "current_xyzrpy_mm_deg": _mm_deg(cur),
        "current_right_xyzrpy_mm_deg": _mm_deg(cur_r),
        "coarse_xyzrpy_mm_deg": _mm_deg(result.hover_flange),
        "fine_xyzrpy_mm_deg": _mm_deg(result.fine_flange),
        # 大板双臂末端（接近 / 合拢 tip）
        "coarse_left": coarse_left,
        "coarse_right": coarse_right,
        "coarse_left_xyzrpy_mm_deg": _mm_deg(coarse_left),
        "coarse_right_xyzrpy_mm_deg": _mm_deg(coarse_right),
        "left_approach": left_app,
        "right_approach": right_app,
        "left_grasp": left_grasp,
        "right_grasp": right_grasp,
        "left_approach_xyzrpy_mm_deg": _mm_deg(left_app),
        "right_approach_xyzrpy_mm_deg": _mm_deg(right_app),
        "left_grasp_xyzrpy_mm_deg": _mm_deg(left_grasp),
        "right_grasp_xyzrpy_mm_deg": _mm_deg(right_grasp),
        "cli": {
            "current": cli.get("current"),
            "current_left": cli.get("current_left") or cli.get("current"),
            "current_right": cli.get("current_right"),
            "coarse": cli.get("coarse"),
            "fine": cli.get("fine"),
            "arm": dbg.get("arm") or cli.get("arm"),
            "left_approach": cli.get("left_approach"),
            "right_approach": cli.get("right_approach"),
            "left_grasp": cli.get("left_grasp"),
            "right_grasp": cli.get("right_grasp"),
        },
        "coarse": coarse,
        "fine": fine,
        "dxy_mm": dxy_mm,
        "debug": result.debug,
    }


def _large_confirm(info: dict) -> bool:
    """接近后阻塞：网页可反复「内收 5mm」，点「结束」才返回（不再自动合拢）。"""
    global _confirm_jpeg, _confirm_answer, _inward_ctx
    vis = info.get("vis")
    jpeg = b""
    if vis is not None:
        try:
            jpeg = _encode_jpeg(vis, max_w=1600, quality=85)
        except Exception as exc:
            LOG.warning("confirm jpeg: %s", exc)
    plan = info.get("plan") or {}
    ux = float(plan.get("grasp_ux", 0.0) or 0.0)
    uy = float(plan.get("grasp_uy", 0.0) or 0.0)
    if math.hypot(ux, uy) < 1e-9:
        # 兜底：按用户举例的 ±Y 内收
        ux, uy = 0.0, 1.0
    # 进会话时读一次双臂位姿缓存，后续内收不再每步双读 TCP
    left_pose = right_pose = None
    with _lock:
        dual0 = _dual
    if dual0 is not None:
        try:
            left_pose, right_pose = dual0.get_flange_poses()
        except Exception as exc:  # noqa: BLE001
            LOG.warning("seed inward poses failed: %s", exc)
    with _lock:
        _confirm_jpeg = jpeg
        _confirm_answer = None
        _inward_ctx = {
            "ux": ux,
            "uy": uy,
            "step_m": float(_INWARD_STEP_M),
            "n": 0,
            "busy": False,
            "board_id": info.get("board_id"),
            "plan": plan,
            "left_pose": None if left_pose is None else list(left_pose[:6]),
            "right_pose": None if right_pose is None else list(right_pose[:6]),
        }
        step_mm = _INWARD_STEP_M * 1000.0
        _job.update(
            {
                "awaiting_confirm": True,
                "confirm_message": str(
                    info.get("message")
                    or (
                        f"已到接近位。点「内收 {step_mm:.0f}mm」逐步夹近；"
                        "点「结束」退出（无自动合拢）"
                    )
                ),
                "confirm_plan": plan,
                "confirm_board_id": info.get("board_id"),
                "inward_steps": 0,
                "message": f"大板接近位 · 可内收（每步 {step_mm:.0f}mm）…",
            }
        )
        _confirm_event.clear()
    LOG.info(
        "awaiting large-board inward board_id=%s u=(%.3f,%.3f) step=%.0fmm",
        info.get("board_id"),
        ux,
        uy,
        _INWARD_STEP_M * 1000.0,
    )
    ok_wait = _confirm_event.wait(timeout=600.0)
    with _lock:
        n_steps = int((_inward_ctx or {}).get("n") or 0)
        _job.update(
            {
                "awaiting_confirm": False,
                "confirm_message": "",
                "message": (
                    f"内收会话结束（共 {n_steps} 步）"
                    if ok_wait
                    else "内收会话超时（停在当前位）"
                ),
            }
        )
        _confirm_answer = None
        # 保留 _inward_ctx 方向，结束后仍可用主栏「内收」再步进
    if not ok_wait:
        LOG.warning("large inward session timeout")
    return False  # 永不触发自动合拢


def _do_inward_step() -> dict:
    """双臂沿法向各内收一步：左 +u、右 −u。并行 jog，用缓存位姿少读 TCP。"""
    global _inward_ctx
    with _lock:
        if _inward_ctx is None:
            raise RuntimeError("尚无大板接近会话（请先全流程到接近位）")
        if _inward_ctx.get("busy"):
            raise RuntimeError("上一步内收还在动，稍后再点")
        _inward_ctx["busy"] = True
        ctx = dict(_inward_ctx)
        dual = _dual
    if dual is None:
        with _lock:
            if _inward_ctx is not None:
                _inward_ctx["busy"] = False
        raise RuntimeError("机械臂未连接")
    ux = float(ctx["ux"])
    uy = float(ctx["uy"])
    step = float(ctx.get("step_m") or _INWARD_STEP_M)
    nrm = math.hypot(ux, uy)
    if nrm < 1e-9:
        with _lock:
            if _inward_ctx is not None:
                _inward_ctx["busy"] = False
        raise RuntimeError("内收方向无效")
    ux, uy = ux / nrm, uy / nrm
    pl = ctx.get("left_pose")
    pr = ctx.get("right_pose")
    if pl is None or pr is None or len(pl) < 6 or len(pr) < 6:
        try:
            pl, pr = dual.get_flange_poses()
        except Exception as exc:  # noqa: BLE001
            with _lock:
                if _inward_ctx is not None:
                    _inward_ctx["busy"] = False
            raise RuntimeError(f"读双臂位姿失败: {exc}") from exc
    left6 = [
        float(pl[0]) + ux * step,
        float(pl[1]) + uy * step,
        float(pl[2]),
        float(pl[3]),
        float(pl[4]),
        float(pl[5]),
    ]
    right6 = [
        float(pr[0]) - ux * step,
        float(pr[1]) - uy * step,
        float(pr[2]),
        float(pr[3]),
        float(pr[4]),
        float(pr[5]),
    ]
    # 短距并行点动：偏快、短超时，靠缓存位姿连续步进（丝滑感）
    max_linear_vel = 0.20
    jog_timeout_s = 5.0
    LOG.info(
        "INWARD jog#%s Δ=%.0fmm vmax=%.2f u=(%.3f,%.3f) L→(%.3f,%.3f) R→(%.3f,%.3f)",
        int(ctx.get("n") or 0) + 1,
        step * 1000.0,
        max_linear_vel,
        ux,
        uy,
        left6[0],
        left6[1],
        right6[0],
        right6[1],
    )
    try:
        if hasattr(dual, "jog_both"):
            result = dual.jog_both(
                left6,
                right6,
                max_linear_vel=max_linear_vel,
                timeout_s=jog_timeout_s,
            )
        else:
            result = dual.move_both(
                left6, right6, velocity_scale=0.15, parallel=True
            )
    finally:
        with _lock:
            if _inward_ctx is not None:
                _inward_ctx["busy"] = False
    with _lock:
        if _inward_ctx is not None:
            _inward_ctx["n"] = int(_inward_ctx.get("n") or 0) + 1
            # 用目标位姿作下一步缓存（避免每步双读 TCP）
            _inward_ctx["left_pose"] = list(left6)
            _inward_ctx["right_pose"] = list(right6)
            n_done = int(_inward_ctx["n"])
        else:
            n_done = 1
        _job["inward_steps"] = n_done
        _job["message"] = (
            f"内收 ×{n_done}（每步 {step*1000:.0f}mm · 双臂并行）"
            + ("" if result.ok else f" · {result.message}")
        )
        if _job.get("awaiting_confirm"):
            _job["confirm_message"] = (
                f"已内收 {n_done} 步。可继续「内收」或「结束内收」。"
            )
    return {
        "ok": bool(result.ok),
        "n": n_done,
        "step_mm": step * 1000.0,
        "ux": ux,
        "uy": uy,
        "left": [round(float(x), 5) for x in (result.left or left6)[:6]],
        "right": [round(float(x), 5) for x in (result.right or right6)[:6]],
        "message": _job.get("message") or "",
    }


def _run_job(opts: dict) -> None:
    global _job, _preview_jpeg, _confirm_jpeg, _wrist_preview_jpeg
    from pcb_grasp.pipeline import GraspPipeline

    batch_dir = next_batch_dir(Path(opts.get("vis") or (ROOT / "output" / "live_vis")))
    plan_only = bool(opts.get("plan_only"))
    with _lock:
        _wrist_preview_jpeg = b""
        _job.update(
            {
                "running": True,
                "ok": None,
                "message": "识别抓取点…" if plan_only else "running",
                "result": None,
                "batch_dir": str(batch_dir),
                "error": "",
                "awaiting_confirm": False,
                "confirm_message": "",
                "confirm_plan": None,
                "confirm_board_id": None,
                "has_wrist_vis": False,
            }
        )
    dry_pair = None
    try:
        if not _camera_snapshot().get("connected"):
            raise RuntimeError("相机未连接，请先「连接相机」")

        want_real = bool(opts.get("real")) and not plan_only
        robot = _robot_snapshot()
        if want_real:
            if not robot.get("connected"):
                raise RuntimeError("运动到接近位需要先「连接机械臂」")
            with _lock:
                motion, dual = _motion, _dual
            if motion is None:
                raise RuntimeError("当前没有连接机械臂")
            # 全流程一开始记下当前左右位姿，供网页「复位」
            _record_session_home()
        elif plan_only and robot.get("connected") and _motion is not None:
            # 识别抓取点：不动臂，但用真实 TCP 做腕部精定位（腕可见时）
            with _lock:
                motion, dual = _motion, _dual
        else:
            motion, dual = _dry_motion()
            dry_pair = (motion, dual)
            # plan_only 未连臂：临时 Rim 只读 TCP，供腕部眼在手外参
            if plan_only:
                try:
                    from pcb_grasp.rim_motion import RimMotion

                    tmp = RimMotion(
                        frame_bundle=str(
                            getattr(_cfg, "rim_frame_bundle", None)
                            or "10.3.8.7:50066"
                        ),
                        arm=str(getattr(_cfg, "tj_arm", "A") or "A"),
                        dry_run=False,
                    )
                    tmp.connect()
                    motion = tmp
                    dry_pair = (tmp, dual)
                    LOG.info("plan_only: RimMotion 只读 TCP for wrist fine")
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("plan_only: 读 TCP 失败，腕部精定位可能跳过: %s", exc)

        force = opts.get("force_class")
        if force in ("", "auto", None):
            force = None
        is_large_single = str(force or "") in (
            "large_single",
            "large-single",
            "single",
        )
        pipe = GraspPipeline(
            cfg=_cfg,
            motion=motion,
            dual_motion=dual,
            force_class=force,
        )

        # 真机抓取 / 识别抓取点 / 精定位 / 大板单臂：开腕部
        stage_opt = str(opts.get("stage") or "all")
        need_wrist = (
            is_large_single
            or (want_real and stage_opt in ("all", "fine"))
            or (plan_only and stage_opt in ("all", "fine"))
        )
        if need_wrist and _camera_snapshot().get("mode") == "head":
            with _cam_lock:
                try:
                    _ensure_wrist_cam()
                    _set_camera(mode="head+wrist")
                except Exception as exc:
                    LOG.warning("wrist open for grasp/plan: %s", exc)

        large_cb = None
        if (
            want_real
            and bool(opts.get("confirm_large", True))
            and not plan_only
            and not is_large_single
        ):
            large_cb = _large_confirm

        def _on_progress(partial, phase: str) -> None:
            """粗/精定位一出就写进 job.result，并推头/腕识别图到网页。"""
            global _preview_jpeg, _wrist_preview_jpeg
            try:
                payload = _result_payload(partial, batch_dir)
                payload["ok"] = None  # 进行中
                def _xyz_mm(p):
                    if not p or len(p) < 3:
                        return None
                    return f"({p[0]*1000:.1f},{p[1]*1000:.1f},{p[2]*1000:.1f})"

                if phase == "coarse":
                    left = _xyz_mm(payload.get("coarse_left") or payload.get("hover_flange"))
                    right = _xyz_mm(payload.get("coarse_right"))
                    if left and right:
                        msg = (
                            f"粗定位完成 ID={payload.get('target_board_id')} "
                            f"左目标mm={left} 右目标mm={right}"
                        )
                    elif left:
                        msg = (
                            f"粗定位完成 ID={payload.get('target_board_id')} "
                            f"目标末端mm={left}"
                        )
                    else:
                        msg = f"粗定位完成 ID={payload.get('target_board_id')}"
                elif phase == "fine":
                    la = _xyz_mm(payload.get("left_approach"))
                    ra = _xyz_mm(payload.get("right_approach"))
                    lg = _xyz_mm(payload.get("left_grasp"))
                    rg = _xyz_mm(payload.get("right_grasp"))
                    if la and ra:
                        msg = (
                            f"精定位完成 接近 左{la} 右{ra}"
                            + (f" 夹取 左{lg} 右{rg}" if lg and rg else "")
                        )
                    else:
                        d = payload.get("dxy_mm") or {}
                        msg = (
                            f"精定位完成 Δxy=({d.get('dx', '?')},{d.get('dy', '?')})mm "
                            f"|Δ|={d.get('hypot', '?')}mm"
                        )
                else:
                    msg = f"进度: {phase}"
                # 头/腕识别图：粗（含大板单臂腕粗）与精都推；有图就写
                if getattr(pipe, "_last_head_pose_vis", None) is not None:
                    try:
                        jpeg = _encode_jpeg(pipe._last_head_pose_vis)  # noqa: SLF001
                        with _lock:
                            _preview_jpeg = jpeg
                    except Exception:
                        pass
                if getattr(pipe, "_last_wrist_pose_vis", None) is not None:
                    try:
                        wjpeg = _encode_jpeg(pipe._last_wrist_pose_vis)  # noqa: SLF001
                        with _lock:
                            _wrist_preview_jpeg = wjpeg
                    except Exception:
                        pass
                payload["message"] = msg
                with _lock:
                    _job.update(
                        {
                            "running": True,
                            "message": msg,
                            "result": payload,
                            "batch_dir": str(batch_dir),
                            "has_wrist_vis": bool(_wrist_preview_jpeg),
                        }
                    )
                LOG.info("ui progress [%s] %s", phase, msg)
            except Exception as exc:  # noqa: BLE001
                LOG.warning("ui progress publish failed: %s", exc)

        with _cam_lock:
            result = pipe.run_live(
                _grab_head,
                _grab_wrist,
                settle_s=float(opts.get("settle") or _cfg.live_settle_s),
                do_grasp=bool(opts.get("do_grasp", True)) and want_real and not plan_only,
                save_vis_dir=batch_dir,
                stage=str(opts.get("stage") or "all"),
                large_confirm=large_cb,
                plan_only=plan_only,
                on_progress=_on_progress,
            )

        if getattr(pipe, "_last_head_pose_vis", None) is not None:
            try:
                jpeg = _encode_jpeg(pipe._last_head_pose_vis)  # noqa: SLF001
                with _lock:
                    _preview_jpeg = jpeg
                    if plan_only:
                        _confirm_jpeg = jpeg
                    _preview_meta.clear()
                    _preview_meta.update(
                        {
                            "ok": True,
                            "error": "",
                            "ts": time.time(),
                            "n_det": -1,
                        }
                    )
            except Exception:
                pass
        if getattr(pipe, "_last_wrist_pose_vis", None) is not None:
            try:
                wjpeg = _encode_jpeg(pipe._last_wrist_pose_vis)  # noqa: SLF001
                with _lock:
                    _wrist_preview_jpeg = wjpeg
            except Exception:
                pass

        payload = _result_payload(result, batch_dir)
        try:
            (batch_dir / "result.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        except Exception:
            pass
        with _lock:
            _job.update(
                {
                    "running": False,
                    "ok": payload["ok"],
                    "message": payload["message"],
                    "result": payload,
                    "batch_dir": str(batch_dir),
                    "error": "" if payload["ok"] else payload["message"],
                    "awaiting_confirm": False,
                }
            )
    except Exception as exc:
        LOG.exception("ui job failed")
        with _lock:
            _job.update(
                {
                    "running": False,
                    "ok": False,
                    "message": str(exc),
                    "error": traceback.format_exc(),
                    "batch_dir": str(batch_dir),
                    "awaiting_confirm": False,
                }
            )
        _confirm_event.set()
    finally:
        if dry_pair is not None:
            _m, d = dry_pair
            for obj in (_m, d):
                if obj is None:
                    continue
                try:
                    obj.close()
                except Exception:
                    pass


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("[ui]", fmt % args)

    def _send(self, code, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_jpeg(self, data: bytes, *, cache: bool = False):
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        if not cache:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self):
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/api/config":
            self._send(
                200,
                {
                    "ok": True,
                    "classes": {
                        "pcb_small": _cfg.cls_pcb_small,
                        "nopcb": _cfg.cls_nopcb,
                        "pcb_large": _cfg.cls_pcb_large,
                    },
                    "large_board_enable": _cfg.large_board_enable,
                    "hover_height_m": _cfg.hover_height_m,
                    "grasp_z_m": _cfg.grasp_z_m,
                    "target_board_id": _cfg.target_board_id,
                    "open_margin_m": _cfg.large_open_margin_m,
                    "squeeze_close_m": _cfg.large_squeeze_close_m,
                    "squeeze_force_n": _cfg.large_squeeze_force_n,
                    "half_width_m": _cfg.large_half_width_m,
                    "robot_ip": _robot_endpoint_label(_cfg),
                    "motion_backend": str(getattr(_cfg, "motion_backend", "rim")),
                    "robot": _robot_snapshot(),
                    "camera": _camera_snapshot(),
                    "reset_pose": _reset_pose_info(),
                },
            )
            return
        if path == "/api/reset_pose":
            info = _reset_pose_info()
            self._send(
                200,
                {"ok": True, "reset_pose": info, "has_reset": info is not None},
            )
            return
        if path == "/api/status":

            job = _job_snapshot()
            with _lock:
                preview = dict(_preview_meta)
            self._send(
                200,
                {
                    "ok": True,
                    **job,
                    "robot": _robot_snapshot(),
                    "camera": _camera_snapshot(),
                    "preview": preview,
                },
            )
            return
        if path == "/api/robot":
            self._send(200, {"ok": True, "robot": _robot_snapshot()})
            return
        if path == "/api/camera":
            self._send(200, {"ok": True, "camera": _camera_snapshot()})
            return
        if path == "/api/wrist.jpg":
            with _lock:
                jpeg = _wrist_preview_jpeg
            if jpeg:
                self._send_jpeg(jpeg)
                return
            # 尚无精定位结果图时：尝试拉左腕实时画面
            try:
                import cv2

                with _cam_lock:
                    bgr, _dep = _grab_wrist()
                vis = bgr.copy()
                cv2.putText(
                    vis,
                    "WRIST live (await fine result)",
                    (8, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )
                self._send_jpeg(_encode_jpeg(vis))
                return
            except Exception as exc:
                LOG.warning("wrist live preview failed: %s", exc)
                self._send(
                    404,
                    {
                        "ok": False,
                        "error": f"暂无腕部图（精定位后才有结果；实时预览失败: {exc}）",
                    },
                )
                return
        if path == "/api/head.jpg":
            job = _job_snapshot()
            if job.get("running") and not job.get("awaiting_confirm"):
                with _lock:
                    jpeg = _preview_jpeg
                if jpeg:
                    self._send_jpeg(jpeg)
                    return
                self._send(503, {"ok": False, "error": "任务占用相机，暂无预览帧"})
                return
            if job.get("awaiting_confirm"):
                with _lock:
                    jpeg = _confirm_jpeg or _preview_jpeg
                if jpeg:
                    self._send_jpeg(jpeg)
                    return
            try:
                qs = parsed.query or ""
                detect = "nodetect" not in qs
                jpeg, _meta = _grab_preview(detect=detect)
                self._send_jpeg(jpeg)
            except Exception as exc:
                LOG.warning("head preview failed: %s", exc)
                with _lock:
                    _preview_meta.clear()
                    _preview_meta.update(
                        {
                            "ok": False,
                            "error": str(exc),
                            "ts": time.time(),
                            "n_det": 0,
                        }
                    )
                self._send(500, {"ok": False, "error": str(exc)})
            return
        if path == "/api/confirm.jpg":
            with _lock:
                jpeg = _confirm_jpeg or _preview_jpeg
            if not jpeg:
                self.send_error(404, "no confirm image")
                return
            self._send_jpeg(jpeg)
            return
        if path in ("/", "/index.html"):
            body = INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            # 打开页面只尝试连相机，不连机械臂
            snap = _camera_snapshot()
            if not snap.get("connected") and not snap.get("connecting"):
                _connect_camera_async()
            return
        self.send_error(404)

    def do_POST(self):
        global _confirm_answer, _cfg, _preview_pipe, _confirm_jpeg
        try:
            data = self._read_json()
            if self.path == "/api/connect_camera":
                snap = _camera_snapshot()
                if snap.get("connecting"):
                    self._send(200, {"ok": True, "camera": snap, "message": "正在连接相机…"})
                    return
                if snap.get("connected"):
                    self._send(200, {"ok": True, "camera": snap, "message": "相机已连接"})
                    return
                cam = _connect_camera()
                self._send(
                    200,
                    {
                        "ok": True,
                        "camera": cam,
                        "message": "相机已连接" if cam.get("connected") else "相机连接失败",
                    },
                )
                return
            if self.path == "/api/connect":
                snap = _robot_snapshot()
                if snap.get("connecting"):
                    self._send(200, {"ok": True, "robot": snap, "message": "正在连接机械臂…"})
                    return
                robot = _connect_robot()
                self._send(
                    200,
                    {
                        "ok": True,
                        "robot": robot,
                        "message": "机械臂已连接" if robot.get("connected") else "机械臂连接失败",
                    },
                )
                return
            if self.path == "/api/inward":
                out = _do_inward_step()
                self._send(200, out)
                return
            if self.path == "/api/confirm":
                job = _job_snapshot()
                if not job.get("awaiting_confirm"):
                    self._send(200, {"ok": False, "error": "当前不在大板内收会话"})
                    return
                # 仅「结束」会话；不再有合拢
                with _lock:
                    _confirm_answer = False
                _confirm_event.set()
                self._send(
                    200,
                    {
                        "ok": True,
                        "confirmed": False,
                        "message": "已结束内收会话",
                    },
                )
                return
            if self.path == "/api/run":
                with _lock:
                    if _job.get("running"):
                        raise ValueError("已有任务在跑")
                plan_only = bool(data.get("plan_only"))
                want_real = bool(data.get("real")) and not plan_only
                if not _camera_snapshot().get("connected"):
                    raise ValueError("请先「连接相机」")
                if want_real and not _robot_snapshot().get("connected"):
                    raise ValueError("请先「连接机械臂」")
                opts = {
                    "real": want_real,
                    "plan_only": plan_only,
                    "stage": data.get("stage") or "all",
                    "force_class": data.get("force_class") or None,
                    "do_grasp": bool(data.get("do_grasp", True)),
                    "settle": data.get("settle"),
                    "vis": data.get("vis") or str(ROOT / "output" / "live_vis"),
                    "confirm_large": bool(data.get("confirm_large", True)),
                }
                t = threading.Thread(
                    target=_run_job, args=(opts,), name="grasp-ui-job", daemon=True
                )
                t.start()
                self._send(
                    200,
                    {
                        "ok": True,
                        "message": "识别抓取点…" if plan_only else "已启动抓取",
                    },
                )
                return
            if self.path == "/api/session_home":
                action = str(data.get("action") or "go").strip().lower()
                if action in ("record", "capture", "save"):
                    info = _record_session_home()
                    if not info.get("has_home"):
                        raise ValueError("记录失败：读不到法兰位姿")
                    self._send(
                        200,
                        {
                            "ok": True,
                            "action": "record",
                            "home": info,
                            "message": "已记录当前为作业起点",
                        },
                    )
                    return
                out = _go_session_home()
                self._send(200, out)
                return
            if self.path == "/api/estop":
                with _lock:
                    dual = _dual
                    if _job.get("awaiting_confirm"):
                        _confirm_answer = False
                        _confirm_event.set()
                if dual is not None and hasattr(dual, "estop"):
                    dual.estop()
                self._send(200, {"ok": True})
                return
            if self.path == "/api/gripper":
                # {"open": true} 张开；{"open": false} 收夹
                with _lock:
                    motion = _motion
                if motion is None:
                    raise ValueError("请先「连接机械臂」")
                want_open = bool(data.get("open", True))
                motion.set_gripper(want_open)
                self._send(
                    200,
                    {
                        "ok": True,
                        "open": want_open,
                        "message": "夹爪已张开" if want_open else "夹爪已收夹",
                    },
                )
                return
            if self.path == "/api/reset_pose":
                # action=capture：当前末端记为复位位；action=go：运动到已存复位位
                action = str(data.get("action") or "go").strip().lower()
                with _lock:
                    motion = _motion
                    busy = bool(_job.get("running"))
                if motion is None:
                    raise ValueError("请先「连接机械臂」")
                if busy:
                    raise ValueError("抓取任务进行中，请先等结束或急停")
                from pcb_grasp.reset_pose import (
                    load_reset_pose,
                    reset_pose_brief,
                    save_reset_pose,
                )

                if action in ("capture", "save", "set"):
                    cur = [float(x) for x in motion.get_flange_pose()[:6]]
                    arm = str(getattr(_cfg, "tj_arm", "A"))
                    # 只记当前 XYZ；姿态固定 rpy=(-90°,0°,0°)
                    path = save_reset_pose(cur, arm=arm)
                    brief = reset_pose_brief(path)
                    LOG.info(
                        "reset pose captured xyz=(%.4f,%.4f,%.4f) fixed rpy_deg=(-90,0,0) → %s",
                        cur[0],
                        cur[1],
                        cur[2],
                        path,
                    )
                    self._send(
                        200,
                        {
                            "ok": True,
                            "action": "capture",
                            "reset_pose": brief,
                            "message": (
                                "已记为复位位 "
                                f"xyz=({cur[0]:.3f},{cur[1]:.3f},{cur[2]:.3f}) "
                                "rpy_deg=(-90.0, 0.0, 0.0)"
                            ),
                        },
                    )
                    return
                if action in ("go", "home", "reset"):
                    pose = load_reset_pose()
                    if pose is None:
                        raise ValueError("尚未「记为复位位」，请先记录当前末端位姿")
                    scale = float(getattr(_cfg, "move_velocity_scale", 0.25))
                    LOG.info(
                        "RESET go xyz=(%.4f,%.4f,%.4f) rpy_deg=(-90.0, 0.0, 0.0) "
                        "(lift→orient→xy→down)",
                        pose[0],
                        pose[1],
                        pose[2],
                    )
                    if hasattr(motion, "move_reset"):
                        motion.move_reset(pose, velocity_scale=scale, lift_m=0.08)
                    else:
                        try:
                            motion.move_line(
                                pose,
                                velocity_scale=scale,
                                keep_orientation=False,
                                direct=False,
                            )
                        except TypeError:
                            motion.move_line(pose, velocity_scale=scale)
                    after = [float(x) for x in motion.get_flange_pose()[:6]]
                    self._send(
                        200,
                        {
                            "ok": True,
                            "action": "go",
                            "reset_pose": reset_pose_brief(),
                            "after": {
                                "xyz_m": [
                                    round(after[0], 4),
                                    round(after[1], 4),
                                    round(after[2], 4),
                                ],
                                "rpy_deg": [
                                    round(after[3] * 180.0 / math.pi, 1),
                                    round(after[4] * 180.0 / math.pi, 1),
                                    round(after[5] * 180.0 / math.pi, 1),
                                ],
                            },
                            "message": "复位完成",
                        },
                    )
                    return
                raise ValueError(f"未知 action={action}（用 capture / go）")
            if self.path == "/api/reload_config":
                _cfg = load_config(ROOT / "config" / "grasp_config.yaml")
                _preview_pipe = None
                self._send(200, {"ok": True})
                return
            self._send(404, {"ok": False, "error": "未知接口"})
        except Exception as exc:
            self._send(400, {"ok": False, "error": str(exc)})


def main() -> int:
    print(f"pcb_grasp UI: http://{HOST}:{PORT}")
    print(
        f"相机与机械臂分开：先连相机看抓取点（可不连臂）；"
        f"大板接近/抓取 backend={getattr(_cfg, 'motion_backend', 'rim')} "
        f"endpoint={_robot_endpoint_label(_cfg)}"
    )
    # 只自动连相机，不自动连臂
    _connect_camera_async()
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        _close_motion()
        _close_cams()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
