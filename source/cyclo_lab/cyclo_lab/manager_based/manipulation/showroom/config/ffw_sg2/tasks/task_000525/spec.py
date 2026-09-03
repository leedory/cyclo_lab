"""Identity and approved scene choices for showroom Task000525."""

from __future__ import annotations

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
)
from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ..common import ShowroomTaskSpec
from .arrangement import TASK000525_TARGET_OBJECT
from .layout import TASK000525_CAN_NAMES


TASK_000525_COFFEE_OBJECTS = TASK000525_CAN_NAMES
TASK_000525_TARGET_OBJECT = TASK000525_TARGET_OBJECT
TASK_000525_TARGET_SIDE = "region_conditioned"

TASK_000525_SOURCE_CABINET_PRIM_SUFFIX = (
    "/RobotisShowroom/robotis_showroom/kolbjorn_cabinet_02"
)
TASK_000525_DESTINATION_TABLE_PRIM_SUFFIX = (
    "/RobotisShowroom/robotis_showroom/central_dining_set"
)
TASK_000525_ROBOT_USD_PATH = (
    f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_SG2_softgripper.usd"
)


TASK_000525_SPEC = ShowroomTaskSpec(
    task_id="000525",
    env_name="Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
    instruction=(
        f"Pick up {TASK_000525_TARGET_OBJECT} from kolbjorn_cabinet_02, "
        "drive to central_dining_set, and place the "
        "can on the ivory mat on the table."
    ),
    target_object=TASK_000525_TARGET_OBJECT,
    target_side=TASK_000525_TARGET_SIDE,
    environment_usd_path=ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    robot_profile=DEFAULT_SG2_CAMERA_PROFILE,
)
