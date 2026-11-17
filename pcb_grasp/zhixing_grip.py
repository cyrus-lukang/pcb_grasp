"""知行夹爪（Marvin RS485）— 臂走 RIM 时夹爪仍走 TJ/Marvin。

位置语义与 ``verify_zhixing_gripper`` 一致：0=合，1=开；收夹常用 0.4。
"""

from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

LOG = logging.getLogger("pcb_grasp.zhixing_grip")

TJ_ROOT = Path("/home/ubuntu/workspace/TJ_demo")
TJ_DEMO = TJ_ROOT / "DEMO_PYTHON"


def _ensure_path() -> None:
    for p in (TJ_ROOT, TJ_DEMO):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


class ZhixingGripHandle:
    """Hold Marvin + RS485 pump + ZhiXingGripper."""

    def __init__(self) -> None:
        self.robot: Any = None
        self.rx: Any = None
        self.grip: Any = None
        self.arm: str = "A"
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.grip is not None

    def close(self) -> None:
        if self.rx is not None:
            try:
                self.rx.stop()
            except Exception:  # noqa: BLE001
                pass
        self.rx = None
        self.grip = None
        self.robot = None


def connect_zhixing(
    *,
    robot_ip: str,
    arm: str = "A",
    com_ch: int = 2,
) -> ZhixingGripHandle | None:
    """连 Marvin 只开夹爪通道；失败返回 None（不抛）。"""
    _ensure_path()
    arm_u = str(arm).strip().upper()
    if arm_u in ("L", "LEFT"):
        arm_u = "A"
    elif arm_u in ("R", "RIGHT"):
        arm_u = "B"
    h = ZhixingGripHandle()
    h.arm = arm_u
    try:
        from SDK_PYTHON.fx_robot import Marvin_Robot  # type: ignore
        from verify_zhixing_gripper import (  # type: ignore
            MarvinModbus,
            Rs485RxPump,
            ZhiXingGripper,
        )
    except Exception as exc:  # noqa: BLE001
        LOG.warning("zhixing import failed: %s", exc)
        return None

    class _Adapter:
        def __init__(self, robot: Any, lock: threading.Lock) -> None:
            self._robot = robot
            self._lock = lock

        def clear_485_cache(self, arm: str):
            with self._lock:
                return self._robot.clear_ch_data(arm)

        def set_485_data(self, arm: str, data, size_int: int, com: int):
            if isinstance(data, str):
                payload = bytes(int(x, 16) for x in data.split())
            else:
                payload = bytes(data)
            with self._lock:
                sent = self._robot.set_ch_data(arm, payload, len(payload), com)
            ok = sent is not None and int(sent) > 0
            return ok, sent

        def get_485_data(self, arm: str, com: int):
            with self._lock:
                result = self._robot.get_ch_data(arm, com)
            tag = result[0]
            receive_hex = result[1]
            if isinstance(receive_hex, (bytes, bytearray)):
                receive_hex = " ".join(f"{b:02X}" for b in receive_hex)
            return tag, receive_hex or ""

    try:
        robot = Marvin_Robot()
        if robot.connect(str(robot_ip)) == 0:
            LOG.warning("zhixing Marvin connect failed @ %s", robot_ip)
            return None
        h.robot = robot
        adapter = _Adapter(robot, h._lock)
        rx = Rs485RxPump(adapter, arm_u, int(com_ch))
        rx.start()
        grip = ZhiXingGripper(MarvinModbus(adapter, rx, arm_u, int(com_ch)))
        if not grip.initialize():
            rx.stop()
            LOG.warning("zhixing enable failed arm=%s", arm_u)
            return None
        h.rx = rx
        h.grip = grip
        LOG.info(
            "Zhixing gripper ready via Marvin @ %s arm=%s com=%s",
            robot_ip,
            arm_u,
            com_ch,
        )
        return h
    except Exception as exc:  # noqa: BLE001
        LOG.warning("zhixing setup failed: %s", exc)
        h.close()
        return None


def move_zhixing(
    handle: ZhixingGripHandle,
    *,
    position: float,
    speed: float,
    torque: float,
) -> bool:
    if handle is None or handle.grip is None:
        return False
    try:
        ok = bool(handle.grip.move(float(position), float(speed), float(torque)))
        LOG.info(
            "zhixing move arm=%s pos=%.3f speed=%.0f torque=%.0f ok=%s",
            handle.arm,
            float(position),
            float(speed),
            float(torque),
            ok,
        )
        return ok
    except Exception as exc:  # noqa: BLE001
        LOG.warning("zhixing move failed: %s", exc)
        return False
