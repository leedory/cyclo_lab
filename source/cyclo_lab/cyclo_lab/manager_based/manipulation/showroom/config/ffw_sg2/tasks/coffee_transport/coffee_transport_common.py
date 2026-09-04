"""Shared geometry, reset events, and success checks for temporary coffee tasks."""

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


COFFEE_CAN_NAMES = ("coffee_can_left", "coffee_can_center", "coffee_can_right")
COFFEE_CAN_SIDES = {
    "coffee_can_left": "left",
    "coffee_can_center": "right",
    "coffee_can_right": "right",
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


# The recovered markers were 0.15 m wide.  A 0.12 m square retains their
# centers but leaves positive can-body clearance at the near/front shelf edge.
SOURCE_SPAWN_SQUARES = (
    CenterSquare("coffee_can_left", -2.26137, -2.14137, 0.52042, 0.64042),
    CenterSquare("coffee_can_center", -2.26137, -2.14137, 0.72682, 0.84682),
    CenterSquare("coffee_can_right", -2.26137, -2.14137, 0.93394, 1.05394),
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
    """Yield ten Task458-default jelly bags as static showroom furniture."""

    jelly_placements = [
        placement
        for placement in read_robotis_showroom_object_placements()
        if placement[1] == "jelly_bag"
    ][:10]
    if len(jelly_placements) != 10:
        raise RuntimeError(f"Expected at least ten authored jelly bags, found {len(jelly_placements)}")
    for object_name, _, pos, rot in jelly_placements:
        yield object_name, AssetBaseCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{object_name}",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{CYCLO_LAB_OBJECT_ASSETS_DATA_DIR}/object/jelly_bag_convexhull.usd",
                collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.003, rest_offset=0.0),
                semantic_tags=[("class", "jelly_bag"), ("instance", object_name)],
            ),
            init_state=AssetBaseCfg.InitialStateCfg(pos=pos, rot=rot),
        )


def _sample_and_write_source_cans(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Reset every can independently in its named, non-overlapping square."""

    for square in SOURCE_SPAWN_SQUARES:
        asset = env.scene[square.name]
        sample = torch.rand((len(env_ids), 2), device=asset.device)
        root_pose = asset.data.default_root_state[env_ids, :7].clone()
        root_pose[:, 0] = square.x_min_m + sample[:, 0] * (square.x_max_m - square.x_min_m) + env.scene.env_origins[env_ids, 0]
        root_pose[:, 1] = square.y_min_m + sample[:, 1] * (square.y_max_m - square.y_min_m) + env.scene.env_origins[env_ids, 1]
        root_pose[:, 2] = SOURCE_UPPER_SUPPORT_Z_M + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M + env.scene.env_origins[env_ids, 2]
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)

        asset.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=asset.device), env_ids=env_ids)

def _sample_robot_home_root_pose(env: ManagerBasedEnv, env_ids: torch.Tensor) -> torch.Tensor:
    """Sample one episode-local HOME pose around the recovered USD root."""

    robot = env.scene["robot"]
    home_pose = robot.data.default_root_state[env_ids, :7].clone()
    home_pose[:, :3] += env.scene.env_origins[env_ids]
    translation = (2.0 * torch.rand((len(env_ids), 2), device=robot.device) - 1.0) * ROBOT_SPAWN_XY_RANDOM_M
    home_pose[:, :2] += translation
    yaw = (2.0 * torch.rand(len(env_ids), device=robot.device) - 1.0) * ROBOT_SPAWN_YAW_RANDOM_RAD
    half_yaw = yaw / 2.0
    yaw_w = half_yaw.cos()
    yaw_z = half_yaw.sin()
    w, x, y, z = (home_pose[:, index].clone() for index in range(3, 7))
    home_pose[:, 3] = w * yaw_w - z * yaw_z
    home_pose[:, 4] = x * yaw_w + y * yaw_z
    home_pose[:, 5] = y * yaw_w - x * yaw_z
    home_pose[:, 6] = z * yaw_w + w * yaw_z
    return home_pose


def _write_episode_robot_spawn(env: ManagerBasedEnv, env_ids: torch.Tensor, *, route_y_max_m: float) -> None:
    """Keep a sampled HOME, then optionally start at a rightward route point."""

    robot = env.scene["robot"]
    home_pose = _sample_robot_home_root_pose(env, env_ids)
    if not hasattr(env, "_coffee_transport_episode_home_root_pose_w"):
        env._coffee_transport_episode_home_root_pose_w = torch.zeros(
            (env.num_envs, 7), device=robot.device, dtype=home_pose.dtype
        )
    env._coffee_transport_episode_home_root_pose_w[env_ids] = home_pose
    spawn_pose = home_pose.clone()
    if route_y_max_m:
        spawn_pose[:, 1] += torch.rand(len(env_ids), device=robot.device) * route_y_max_m
    robot.write_root_pose_to_sim(spawn_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros((len(env_ids), 6), device=robot.device), env_ids=env_ids)


def reset_vertical_coffee_transport(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Spawn cans on the upper shelf and the lift uniformly in [-0.28, 0] m."""

    _sample_and_write_source_cans(env, env_ids)
    _write_episode_robot_spawn(env, env_ids, route_y_max_m=0.0)
    robot = env.scene["robot"]
    lift_id = robot.joint_names.index("lift_joint")
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, lift_id] = LIFT_TRAVEL_COMMAND_M * torch.rand(len(env_ids), device=robot.device)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)


def reset_horizontal_coffee_transport(env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
    """Spawn cans on the upper shelf and the base between its two cabinet poses."""

    _sample_and_write_source_cans(env, env_ids)
    _write_episode_robot_spawn(env, env_ids, route_y_max_m=CABINET_RIGHT_DELTA_Y_M)


def coffee_transport_success(env: ManagerBasedEnv, *, destination: str, position_tolerance_m: float = 0.006) -> torch.Tensor:
    """Require all three cans to lie in their translated source-square region."""

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
            & (torch.abs(position[:, 2] - (SOURCE_UPPER_SUPPORT_Z_M + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M + dz_m + origin[:, 2])) <= position_tolerance_m)
        )
        success &= inside
    return success
