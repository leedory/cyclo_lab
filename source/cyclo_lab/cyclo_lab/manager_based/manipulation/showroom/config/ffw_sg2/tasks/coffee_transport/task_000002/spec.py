"""Identity for temporary horizontal coffee transport Task000002."""

from cyclo_lab.assets.environments.robotis_showroom import ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ...common import ShowroomTaskSpec


TASK_000002_SPEC = ShowroomTaskSpec(
    task_id="000002",
    env_name="Cyclo-Real-Showroom-Task000002-FFW-SG2-v0",
    instruction=(
        "Return the mobile base to the left cabinet, then place the right, center, "
        "and left orange coffee cans on the matching right cabinet shelf."
    ),
    target_object="coffee_can_right",
    target_side="right",
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
)
