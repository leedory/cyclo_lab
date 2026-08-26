"""Reset events and recorded pose provenance for peanut take-out generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


SHELF_SUPPORT_COLLIDER_SUFFIX = (
    "RobotisShowroom/robotis_showroom/kolbjorn_cabinet_1/"
    "CollisionProxies/C03_LATERAL"
)


def _env_ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)


def refresh_shelf_support_collider(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    collider_suffix: str = SHELF_SUPPORT_COLLIDER_SUFFIX,
) -> None:
    """Notify PhysX of the authored shelf collider without disabling it."""
    from isaacsim.core.utils.stage import get_current_stage
    from pxr import UsdPhysics

    stage = get_current_stage()
    for env_id in _env_ids(env, env_ids).detach().cpu().tolist():
        path = f"/World/envs/env_{env_id}/{collider_suffix}"
        prim = stage.GetPrimAtPath(path)
        if not prim.IsValid() or prim.IsInstanceProxy():
            raise RuntimeError(f"shelf support collider is missing or instanced: {path}")
        collision = UsdPhysics.CollisionAPI(prim)
        attribute = collision.GetCollisionEnabledAttr()
        if attribute.Get() is not True:
            raise RuntimeError(f"shelf support collider must start enabled: {path}")
        # Clearing exposes the inherited True value and sends the USD change
        # notice needed by PhysX. No simulation step occurs before Set(True).
        attribute.Clear()
        if attribute.Get() is not True:
            raise RuntimeError(f"shelf support collider inheritance is not enabled: {path}")
        collision.CreateCollisionEnabledAttr().Set(True)


def randomize_target_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    lateral_y_max_m: float,
    yaw_max_rad: float,
    asset_cfg: SceneEntityCfg,
) -> None:
    """Sample target Y/yaw continuously while preserving authored X/Z."""
    ids = _env_ids(env, env_ids)
    target = env.scene[asset_cfg.name]
    samples = torch.rand((len(ids), 2), device=target.device) * 2.0 - 1.0
    lateral_y = samples[:, 0] * lateral_y_max_m
    yaw = samples[:, 1] * yaw_max_rad

    root_pose = target.data.default_root_state[ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[ids]
    root_pose[:, 1] += lateral_y
    zero = torch.zeros_like(yaw)
    yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, yaw)
    root_pose[:, 3:7] = math_utils.quat_mul(yaw_quat, root_pose[:, 3:7])
    target.write_root_pose_to_sim(root_pose, env_ids=ids)
    target.write_root_velocity_to_sim(
        torch.zeros((len(ids), 6), dtype=root_pose.dtype, device=target.device),
        env_ids=ids,
    )

    if not hasattr(env, "_task458_requested_target_pose"):
        env._task458_requested_target_pose = torch.zeros(
            (env.num_envs, 2), dtype=root_pose.dtype, device=env.device
        )
    env._task458_requested_target_pose[ids] = torch.stack(
        (lateral_y, yaw), dim=-1
    )


def randomize_robot_root(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    depth_x_max_m: float,
    lateral_y_max_m: float,
    yaw_max_rad: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample robot X/Y/yaw inside the configured box."""
    ids = _env_ids(env, env_ids)
    robot = env.scene[asset_cfg.name]
    samples = torch.rand((len(ids), 3), device=robot.device) * 2.0 - 1.0
    delta_x = samples[:, 0] * depth_x_max_m
    delta_y = samples[:, 1] * lateral_y_max_m
    delta_yaw = samples[:, 2] * yaw_max_rad

    root_pose = robot.data.default_root_state[ids, :7].clone()
    root_pose[:, :3] += env.scene.env_origins[ids]
    root_pose[:, 0] += delta_x
    root_pose[:, 1] += delta_y
    zero = torch.zeros_like(delta_yaw)
    yaw_quat = math_utils.quat_from_euler_xyz(zero, zero, delta_yaw)
    root_pose[:, 3:7] = math_utils.quat_mul(yaw_quat, root_pose[:, 3:7])
    robot.write_root_pose_to_sim(root_pose, env_ids=ids)
    robot.write_root_velocity_to_sim(
        torch.zeros((len(ids), 6), dtype=root_pose.dtype, device=robot.device),
        env_ids=ids,
    )

    if not hasattr(env, "_task458_requested_robot_root"):
        env._task458_requested_robot_root = torch.zeros(
            (env.num_envs, 3), dtype=root_pose.dtype, device=env.device
        )
    env._task458_requested_robot_root[ids] = torch.stack(
        (delta_x, delta_y, delta_yaw), dim=-1
    )


def requested_target_y_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    return getattr(
        env,
        "_task458_requested_target_pose",
        torch.zeros((env.num_envs, 2), device=env.device),
    ).clone()


def requested_robot_root_xy_yaw(env: ManagerBasedEnv) -> torch.Tensor:
    return getattr(
        env,
        "_task458_requested_robot_root",
        torch.zeros((env.num_envs, 3), device=env.device),
    ).clone()


def realized_robot_root_xy_yaw(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    current = robot.data.root_pose_w
    default = robot.data.default_root_state[:, :7].clone()
    default[:, :3] += env.scene.env_origins
    delta_quat = math_utils.quat_mul(
        current[:, 3:7], math_utils.quat_inv(default[:, 3:7])
    )
    _, _, yaw = math_utils.euler_xyz_from_quat(delta_quat)
    return torch.stack(
        (current[:, 0] - default[:, 0], current[:, 1] - default[:, 1], yaw),
        dim=-1,
    )


def realized_target_y_yaw(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    target = env.scene[object_cfg.name]
    current = target.data.root_pose_w
    default = target.data.default_root_state[:, :7].clone()
    default[:, :3] += env.scene.env_origins
    delta_quat = math_utils.quat_mul(
        current[:, 3:7], math_utils.quat_inv(default[:, 3:7])
    )
    _, _, yaw = math_utils.euler_xyz_from_quat(delta_quat)
    return torch.stack((current[:, 1] - default[:, 1], yaw), dim=-1)


def _latch_on_episode_start(
    env: ManagerBasedEnv,
    value: torch.Tensor,
    cache_name: str,
) -> torch.Tensor:
    cached = getattr(env, cache_name, None)
    if (
        cached is None
        or cached.shape != value.shape
        or cached.device != value.device
        or cached.dtype != value.dtype
    ):
        cached = value.clone()
    else:
        reset_mask = env.episode_length_buf == 0
        cached[reset_mask] = value[reset_mask]
    setattr(env, cache_name, cached)
    return cached.clone()


def reset_latched_target_y_yaw(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return _latch_on_episode_start(
        env,
        realized_target_y_yaw(env, object_cfg),
        "_task458_reset_target_y_yaw",
    )


def reset_latched_robot_root_xy_yaw(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return _latch_on_episode_start(
        env,
        realized_robot_root_xy_yaw(env, asset_cfg),
        "_task458_reset_robot_root_xy_yaw",
    )
