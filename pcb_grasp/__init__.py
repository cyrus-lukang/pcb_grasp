"""PCB coarse (head) + fine (wrist) grasp localization & motion."""

from .geometry import (
    base_y_angle_to_gripper_rz,
    compute_flange_target,
    pose_xyz_rpy_to_mat4,
    rz_from_line_xy,
)

__all__ = [
    "GraspPipeline",
    "GraspResult",
    "base_y_angle_to_gripper_rz",
    "compute_flange_target",
    "pose_xyz_rpy_to_mat4",
    "rz_from_line_xy",
]


def __getattr__(name: str):
    if name in {"GraspPipeline", "GraspResult"}:
        from .pipeline import GraspPipeline, GraspResult

        return {"GraspPipeline": GraspPipeline, "GraspResult": GraspResult}[name]
    raise AttributeError(name)
