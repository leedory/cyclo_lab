"""Reset events shared by the coffee-transport task family."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import TYPE_CHECKING

import torch

from .coffee_transport_common import (
    COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
    SOURCE_SPAWN_SQUARES,
    SOURCE_UPPER_SUPPORT_Z_M,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_coffee_can_visual_yaw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: Sequence[str],
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Rotate only each orange can's visual mesh around its local Z axis.

    The shared coffee-can visual is an axisymmetric lathe mesh. Rotating it
    changes the label/texture direction without changing the rigid root pose,
    velocities, or collision proxies used by either transport task.
    """

    yaw_min, yaw_max = yaw_range_rad
    if not (-math.pi <= yaw_min <= yaw_max <= math.pi):
        raise ValueError(
            "Coffee-transport visual yaw range must be ordered and inside "
            f"[-pi, pi], got {yaw_range_rad}."
        )
    if len(env_ids) == 0 or not object_names:
        return

    from pxr import UsdGeom

    samples = torch.rand(
        (len(env_ids), len(object_names)), device=env.device, dtype=torch.float32
    )
    yaw_rad = yaw_min + samples * (yaw_max - yaw_min)
    yaw_deg = torch.rad2deg(yaw_rad)

    env_id_values = env_ids.detach().cpu().tolist()
    yaw_rad_values = yaw_rad.detach().cpu().tolist()
    yaw_deg_values = yaw_deg.detach().cpu().tolist()
    sample_cache = getattr(env, "_coffee_transport_coffee_visual_yaw", {})
    for sample_index, env_id in enumerate(env_id_values):
        env_prim_path = env.scene.env_prim_paths[env_id]
        for object_index, object_name in enumerate(object_names):
            visual_path = f"{env_prim_path}/{object_name}/Visual/SharedMesh"
            visual_prim = env.scene.stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid() or not visual_prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(
                    "Coffee-transport coffee-can visual mesh is missing at "
                    f"{visual_path}."
                )

            xformable = UsdGeom.Xformable(visual_prim)
            rotate_op = next(
                (
                    op
                    for op in xformable.GetOrderedXformOps()
                    if op.GetOpName() == "xformOp:rotateZ"
                ),
                None,
            )
            if rotate_op is None:
                rotate_op = xformable.AddRotateZOp(
                    precision=UsdGeom.XformOp.PrecisionFloat
                )
            sampled_rad = float(yaw_rad_values[sample_index][object_index])
            sampled_deg = float(yaw_deg_values[sample_index][object_index])
            rotate_op.Set(sampled_deg)
            sample_cache.setdefault(int(env_id), {})[object_name] = {
                "rad": sampled_rad,
                "deg": sampled_deg,
            }

    env._coffee_transport_coffee_visual_yaw = sample_cache


def randomize_coffee_can_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    object_names: tuple[str, ...],
) -> None:
    """Reset each can independently inside its named non-overlapping square."""

    squares = {square.name: square for square in SOURCE_SPAWN_SQUARES}
    for object_name in object_names:
        square = squares[object_name]
        asset = env.scene[object_name]
        sample = torch.rand((len(env_ids), 2), device=asset.device)
        root_pose = asset.data.default_root_state[env_ids, :7].clone()
        root_pose[:, 0] = (
            square.x_min_m
            + sample[:, 0] * (square.x_max_m - square.x_min_m)
            + env.scene.env_origins[env_ids, 0]
        )
        root_pose[:, 1] = (
            square.y_min_m
            + sample[:, 1] * (square.y_max_m - square.y_min_m)
            + env.scene.env_origins[env_ids, 1]
        )
        root_pose[:, 2] = (
            SOURCE_UPPER_SUPPORT_Z_M
            + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
            + env.scene.env_origins[env_ids, 2]
        )
        asset.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(
            torch.zeros((len(env_ids), 6), device=asset.device), env_ids=env_ids
        )


def reset_episode_home_and_base_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    home_enabled: bool,
    depth_x_max_m: float,
    lateral_y_max_m: float,
    yaw_max_rad: float,
    route_enabled: bool = False,
    route_offset_range_m: tuple[float, float] = (0.0, 0.0),
) -> None:
    """Cache HOME first, then spawn on the HOME-relative Task000002 route."""

    robot = env.scene["robot"]
    home_pose = robot.data.default_root_state[env_ids, :7].clone()
    home_pose[:, :3] += env.scene.env_origins[env_ids]
    if home_enabled:
        translation = 2.0 * torch.rand((len(env_ids), 2), device=robot.device) - 1.0
        home_pose[:, 0] += translation[:, 0] * depth_x_max_m
        home_pose[:, 1] += translation[:, 1] * lateral_y_max_m
        yaw = (
            (2.0 * torch.rand(len(env_ids), device=robot.device) - 1.0)
            * yaw_max_rad
        )
        half_yaw = yaw / 2.0
        yaw_w = half_yaw.cos()
        yaw_z = half_yaw.sin()
        w, x, y, z = (home_pose[:, index].clone() for index in range(3, 7))
        home_pose[:, 3] = w * yaw_w - z * yaw_z
        home_pose[:, 4] = x * yaw_w + y * yaw_z
        home_pose[:, 5] = y * yaw_w - x * yaw_z
        home_pose[:, 6] = z * yaw_w + w * yaw_z

    if not hasattr(env, "_coffee_transport_episode_home_root_pose_w"):
        env._coffee_transport_episode_home_root_pose_w = torch.zeros(
            (env.num_envs, 7), device=robot.device, dtype=home_pose.dtype
        )
    env._coffee_transport_episode_home_root_pose_w[env_ids] = home_pose

    spawn_pose = home_pose.clone()
    if route_enabled:
        lower, upper = route_offset_range_m
        route_sample = torch.rand(len(env_ids), device=robot.device)
        spawn_pose[:, 1] += lower + route_sample * (upper - lower)
    robot.write_root_pose_to_sim(spawn_pose, env_ids=env_ids)
    robot.write_root_velocity_to_sim(
        torch.zeros((len(env_ids), 6), device=robot.device), env_ids=env_ids
    )


def randomize_lift_start(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    *,
    position_range_m: tuple[float, float],
) -> None:
    """Sample only the Task000001 lift joint; all other joints keep reset values."""

    robot = env.scene["robot"]
    lift_id = robot.joint_names.index("lift_joint")
    joint_pos = robot.data.joint_pos[env_ids].clone()
    lower, upper = position_range_m
    sample = torch.rand(len(env_ids), device=robot.device)
    joint_pos[:, lift_id] = lower + sample * (upper - lower)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(
        joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids
    )
