"""Canonical F2 visual colors and link-to-material mapping."""

import re

F2_VISUAL_MATERIAL_COLORS = {
    "white": (2.0, 2.0, 2.0),
    "dark": (0.0, 0.0, 0.0),
    "aluminum": (0.32818532, 0.32691818, 0.32691818),
    "red": (1.0, 0.0, 0.0),
    "blue": (0.0, 0.0, 1.0),
}


def classify_f2_visual_material(prim_path: str) -> str | None:
    """Return the canonical URDF material for an F2 visual prim."""

    path = prim_path.lower()
    if "/visuals/" not in path:
        return None
    if "end_effector_" in path:
        return "red"
    if "lidar_r_link" in path:
        return "blue"
    if "d455" in path or "/camera_head_link/" in path:
        return "aluminum"
    if (
        "wrist_d401_base" in path
        or "gripper_" in path
        or "base_mobile_assy" in path
        or "wheel_drive_link" in path
        or "rear_wheel_steer_link" in path
    ):
        return "dark"
    if re.search(r"/arm_[lr]_link[1-7](?:/|$)", path):
        return "dark"
    if "lift_link" in path or "wheel_steer_link" in path:
        return "aluminum"
    if (
        "body_arm_assy" in path
        or "/head_link1/" in path
        or "/head_link2/" in path
    ):
        return "white"
    return None
