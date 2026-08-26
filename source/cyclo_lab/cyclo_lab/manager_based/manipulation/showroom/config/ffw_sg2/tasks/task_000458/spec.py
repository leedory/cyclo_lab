"""Identity and platform choices for showroom Task000458."""

from __future__ import annotations


from cyclo_lab.assets.environments.robotis_showroom import ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ..common import ShowroomTaskSpec
TASK_000458_TARGET_OBJECT = "peanut_mix_bag_02"

TASK_000458_SPEC = ShowroomTaskSpec(
    task_id="000458",
    env_name="Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
    instruction=(
        f"Take {TASK_000458_TARGET_OBJECT} out of the shelf with the right gripper."
    ),
    target_object=TASK_000458_TARGET_OBJECT,
    target_side="right",
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
)
