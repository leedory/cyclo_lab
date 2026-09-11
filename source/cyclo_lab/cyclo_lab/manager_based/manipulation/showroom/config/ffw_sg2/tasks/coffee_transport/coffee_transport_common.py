"""Shared geometry, asset configuration, and success checks for coffee tasks."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg

from cyclo_lab.assets.environments.robotis_showroom import (
    read_robotis_showroom_object_placements,
)
from cyclo_lab.assets.object import COFFEE_CAN_CFG, CYCLO_LAB_OBJECT_ASSETS_DATA_DIR

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


COFFEE_CAN_NAMES = ("coffee_can_right", "coffee_can_left")
COFFEE_CAN_SIDES = {
    "coffee_can_right": "right",
    "coffee_can_left": "left",
}
COFFEE_CAN_RADIUS_M = 0.0335
COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M = 0.045

# Measured from kolbjorn_cabinet_02 collision supports.
SOURCE_UPPER_SUPPORT_Z_M = 1.305128932
DESTINATION_LOWER_SUPPORT_Z_M = 1.017119288
SHELF_LEVEL_DELTA_Z_M = DESTINATION_LOWER_SUPPORT_Z_M - SOURCE_UPPER_SUPPORT_Z_M
CABINET_RIGHT_DELTA_Y_M = 0.815072039386852
LIFT_TRAVEL_COMMAND_M = -0.28
ROBOT_SPAWN_XY_RANDOM_M = 0.02
ROBOT_SPAWN_YAW_RANDOM_RAD = 0.017453292519943295
COFFEE_TRANSPORT_HOME_POS = (-1.47138, 0.80011813, 0.0)
COFFEE_TRANSPORT_HOME_ROT = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class CenterSquare:
    """Closed, world-aligned range for a can root center on the upper shelf."""

    name: str
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    @property
    def default_position_m(self) -> tuple[float, float, float]:
        return (
            (self.x_min_m + self.x_max_m) / 2.0,
            (self.y_min_m + self.y_max_m) / 2.0,
            SOURCE_UPPER_SUPPORT_Z_M + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
        )

    def translated(self, *, dz_m: float = 0.0, dy_m: float = 0.0) -> "CenterSquare":
        return CenterSquare(
            name=self.name,
            x_min_m=self.x_min_m,
            x_max_m=self.x_max_m,
            y_min_m=self.y_min_m + dy_m,
            y_max_m=self.y_max_m + dy_m,
        )


# The recovered outer markers were 0.15 m wide. The full 0.12 m depth reached
# too close to the cabinet back for a body grasp. Keep its robot-side half;
# lateral sampling remains 0.12 m wide and the nominal seed lies on the rear
# boundary of the reviewed reachable region.
SOURCE_SPAWN_SQUARES = (
    CenterSquare("coffee_can_right", -2.20137, -2.14137, 0.93394, 1.05394),
    CenterSquare("coffee_can_left", -2.20137, -2.14137, 0.52042, 0.64042),
)


def make_orange_coffee_can_cfg(square: CenterSquare):
    """Create one independently simulated orange can at a reviewed center."""

    cfg = deepcopy(COFFEE_CAN_CFG)
    cfg.prim_path = f"{{ENV_REGEX_NS}}/{square.name}"
    cfg.spawn.variants = {"appearance": "orange"}
    cfg.spawn.semantic_tags = [
        ("class", "coffee_can"),
        ("instance", square.name),
        ("appearance", "orange"),
    ]
    cfg.init_state.pos = square.default_position_m
    cfg.init_state.rot = (1.0, 0.0, 0.0, 0.0)
    return cfg


def iter_static_task_jelly_bag_cfgs():
    """Yield twelve Task458-default jelly bags as static showroom furniture."""

    jelly_placements = [
        placement
        for placement in read_robotis_showroom_object_placements()
        if placement[1] == "jelly_bag"
    ][:12]
    if len(jelly_placements) != 12:
        raise RuntimeError(
            f"Expected at least twelve authored jelly bags, found {len(jelly_placements)}"
        )
    for object_name, _, pos, rot in jelly_placements:
        yield object_name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{CYCLO_LAB_OBJECT_ASSETS_DATA_DIR}/object/jelly_bag_convexhull.usd",
                collision_props=sim_utils.CollisionPropertiesCfg(
                    contact_offset=0.003,
                    rest_offset=0.0,
                ),
                semantic_tags=[("class", "jelly_bag"), ("instance", object_name)],
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )


def coffee_transport_success(
    env: ManagerBasedEnv,
    *,
    destination: str,
    position_tolerance_m: float = 0.006,
) -> torch.Tensor:
    """Require both cans to lie in their translated source-square region."""

    if destination == "down":
        dz_m, dy_m = SHELF_LEVEL_DELTA_Z_M, 0.0
    elif destination == "right":
        dz_m, dy_m = 0.0, CABINET_RIGHT_DELTA_Y_M
    else:
        raise ValueError(f"Unknown coffee-transport destination: {destination!r}")
    success = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for square in SOURCE_SPAWN_SQUARES:
        position = env.scene[square.name].data.root_pos_w
        origin = env.scene.env_origins
        target = square.translated(dz_m=dz_m, dy_m=dy_m)
        inside = (
            (position[:, 0] >= target.x_min_m + origin[:, 0] - position_tolerance_m)
            & (position[:, 0] <= target.x_max_m + origin[:, 0] + position_tolerance_m)
            & (position[:, 1] >= target.y_min_m + origin[:, 1] - position_tolerance_m)
            & (position[:, 1] <= target.y_max_m + origin[:, 1] + position_tolerance_m)
            & (
                torch.abs(
                    position[:, 2]
                    - (
                        SOURCE_UPPER_SUPPORT_Z_M
                        + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
                        + dz_m
                        + origin[:, 2]
                    )
                )
                <= position_tolerance_m
            )
        )
        success &= inside
    return success
