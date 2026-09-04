"""Identity for temporary vertical coffee transport Task000001."""

from cyclo_lab.assets.environments.robotis_showroom import ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ...common import ShowroomTaskSpec


TASK_000001_SPEC = ShowroomTaskSpec(
    task_id="000001",
    env_name="Cyclo-Real-Showroom-Task000001-FFW-SG2-v0",
    instruction=(
        "Return the lift to home, then place the right, center, and left orange "
        "coffee cans on the lower shelf of kolbjorn_cabinet_02."
    ),
    target_object="coffee_can_right",
    target_side="right",
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
)
