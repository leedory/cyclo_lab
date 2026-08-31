"""Observation helpers for the canonical SG2 showroom environment."""

from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import image, last_action, time_out
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils import math as math_utils


def eef_pose(
    env: ManagerBasedEnv,
    eef_cfg: SceneEntityCfg = SceneEntityCfg("eef"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return an end-effector pose in the robot-root coordinate frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    eef: FrameTransformer = env.scene[eef_cfg.name]
    position, quaternion = math_utils.subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        eef.data.target_pos_w[:, 0, :],
        eef.data.target_quat_w[:, 0, :],
    )
    return torch.cat((position, quaternion), dim=1)


def eef_pose_world(
    env: ManagerBasedEnv,
    eef_cfg: SceneEntityCfg = SceneEntityCfg("eef"),
) -> torch.Tensor:
    """Return the TCP pose in the simulation world frame."""

    eef: FrameTransformer = env.scene[eef_cfg.name]
    return torch.cat(
        (eef.data.target_pos_w[:, 0, :], eef.data.target_quat_w[:, 0, :]), dim=1
    )


def asset_root_pose_world(env: ManagerBasedEnv, asset_name: str) -> torch.Tensor:
    """Return an articulation or rigid object's root pose in world WXYZ."""

    asset = env.scene[asset_name]
    return torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=1)


def joint_pos_name(env: ManagerBasedEnv, joint_names: tuple[str, ...], asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos[:, joint_ids]


def joint_pos_target_name(
    env: ManagerBasedEnv,
    joint_names: tuple[str, ...],
    asset_name: str = "robot",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    joint_ids = [asset.joint_names.index(name) for name in joint_names]
    return asset.data.joint_pos_target[:, joint_ids]


def base_twist(env: ManagerBasedEnv, asset_name: str = "robot") -> torch.Tensor:
    asset: Articulation = env.scene[asset_name]
    return torch.cat([asset.data.root_lin_vel_b[:, 0:2], asset.data.root_ang_vel_b[:, 2:3]], dim=-1)
