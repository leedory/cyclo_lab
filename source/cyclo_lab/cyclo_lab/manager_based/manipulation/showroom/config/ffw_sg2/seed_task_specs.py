"""Seed-demonstration task specs for SG2 showroom recording.

Add one ``SeedTaskSpec`` here for each showroom seed-data task, then register
its env cfg in ``__init__.py``.  The env cfg code stays shared; task-specific
choices such as target object, policy cameras, and showroom USD are data.
"""

from __future__ import annotations

from dataclasses import dataclass

from cyclo_lab.assets.environments.robotis_showroom import ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE


@dataclass(frozen=True)
class SeedTaskSpec:
    """Data needed to instantiate one seed-demonstration task."""

    task_id: str
    env_name: str
    instruction: str
    target_object: str
    target_side: str
    environment_usd_path: str
    policy_cameras: tuple[str, ...] = ("cam_head", "cam_wrist_left", "cam_wrist_right")
    robot_profile: str = DEFAULT_SG2_CAMERA_PROFILE
    control_hz: float = 15.0
    physics_hz: float = 30.0
    camera_hz: float = 15.0
    episode_length_s: float = 120.0


SEED_TASK_SPECS: dict[str, SeedTaskSpec] = {
    "000458": SeedTaskSpec(
        task_id="000458",
        env_name="Cyclo-Real-Showroom-Pick-Peanut-FFW-SG2-v0",
        instruction="Pick up the Peanut Mix with right gripper.",
        target_object="peanut_mix_bag",
        target_side="right",
        environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    ),
}


TASK_000458_SPEC = SEED_TASK_SPECS["000458"]
