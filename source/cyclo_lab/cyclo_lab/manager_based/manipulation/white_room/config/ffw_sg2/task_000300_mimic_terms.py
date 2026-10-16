"""MimicGen signals for the upright Task000300 bottle and shallow tray."""

import torch

from cyclo_lab.assets.environments.white_room_layout import TABLE_TOP_Z
from cyclo_lab.assets.environments.white_room_task000300 import (
    COSMETIC_BODY_WIDTH_M,
    COSMETIC_TOTAL_LENGTH_M,
    ORANGE_TRAY_CORNER_RADIUS_M,
    ORANGE_TRAY_FLOOR_M,
    ORANGE_TRAY_SIZE_XY_M,
    ORANGE_TRAY_WALL_M,
)

from .contact_failure import invalid_contact


TASK000300_TRAY_CENTER_TOLERANCE_M = 0.050


def reset_metric_state(env, env_ids=None) -> None:
    if not hasattr(env, "_task000300_grasp_latched"):
        env._task000300_grasp_latched = torch.zeros(
            env.num_envs, dtype=torch.bool, device=env.device
        )
        env._task000300_release_latched = torch.zeros_like(env._task000300_grasp_latched)
    ids = (
        torch.arange(env.num_envs, device=env.device)
        if env_ids is None
        else torch.as_tensor(env_ids, device=env.device, dtype=torch.long).reshape(-1)
    )
    env._task000300_grasp_latched[ids] = False
    env._task000300_release_latched[ids] = False
    env._task000300_mimic_metrics = None


def _metrics(env, side: str) -> dict[str, torch.Tensor]:
    if side not in ("left", "right"):
        raise ValueError(f"Unsupported Task000300 Mimic side: {side}")
    if not hasattr(env, "_task000300_grasp_latched"):
        reset_metric_state(env)
    step = int(env.common_step_counter)
    cached = getattr(env, "_task000300_mimic_metrics", None)
    if cached is not None and cached["step"] == step and cached["side"] == side:
        return cached

    bottle = env.scene["navy_block"]
    tray = env.scene["white_basket"]
    robot = env.scene["robot"]
    gripper_index = robot.joint_names.index(f"gripper_{side[0]}_joint1")
    gripper_position = robot.data.joint_pos[:, gripper_index]

    # The bottle starts upright; count a grasp only after it rises from the table.
    lifted = bottle.data.root_pos_w[:, 2] >= (
        TABLE_TOP_Z + COSMETIC_TOTAL_LENGTH_M / 2.0 + 0.018
    )
    env._task000300_grasp_latched |= (gripper_position >= 0.5) & lifted

    delta = bottle.data.root_pos_w - tray.data.root_pos_w
    # Require all four body footprint corners to fit inside the rounded inner rim.
    # Checking only X/Y bounds would accept bottles over the newly cut-out corners.
    quat = bottle.data.root_quat_w
    yaw = torch.atan2(
        2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
        1.0 - 2.0 * (quat[:, 2].square() + quat[:, 3].square()),
    )
    half_body = COSMETIC_BODY_WIDTH_M / 2.0
    corners = torch.tensor(
        ((-half_body, -half_body), (-half_body, half_body),
         (half_body, -half_body), (half_body, half_body)),
        dtype=delta.dtype,
        device=delta.device,
    )
    corner_x = delta[:, 0, None] + yaw.cos()[:, None] * corners[:, 0] - yaw.sin()[:, None] * corners[:, 1]
    corner_y = delta[:, 1, None] + yaw.sin()[:, None] * corners[:, 0] + yaw.cos()[:, None] * corners[:, 1]
    center_x = ORANGE_TRAY_SIZE_XY_M[0] / 2.0 - ORANGE_TRAY_CORNER_RADIUS_M
    center_y = ORANGE_TRAY_SIZE_XY_M[1] / 2.0 - ORANGE_TRAY_CORNER_RADIUS_M
    inner_radius = ORANGE_TRAY_CORNER_RADIUS_M - ORANGE_TRAY_WALL_M
    outside_x = (corner_x.abs() - center_x).clamp_min(0.0)
    outside_y = (corner_y.abs() - center_y).clamp_min(0.0)
    inside_xy = ((outside_x.square() + outside_y.square()) <= inner_radius**2).all(dim=1)
    floor_center_z = ORANGE_TRAY_FLOOR_M + COSMETIC_TOTAL_LENGTH_M / 2.0
    supported_height = (delta[:, 2] - floor_center_z).abs() <= 0.0125
    upright = 1.0 - 2.0 * (quat[:, 1].square() + quat[:, 2].square()) >= 0.95
    inside_tray = inside_xy & supported_height & upright
    centered_in_tray = (
        torch.linalg.vector_norm(delta[:, :2], dim=1)
        <= TASK000300_TRAY_CENTER_TOLERANCE_M
    )
    placed_in_tray = inside_tray & centered_in_tray

    env._task000300_release_latched |= (
        env._task000300_grasp_latched
        & (gripper_position <= 0.2)
        & placed_in_tray
    )
    # ObservationManager probes term shapes before the environment's first
    # reset.  The runtime close gate is initialized during that reset, so use
    # an all-false placeholder only for this pre-reset shape probe.
    clean_close = getattr(
        env,
        "_task000300_close_permitted",
        torch.zeros_like(env._task000300_grasp_latched),
    )
    clean_release = getattr(
        env,
        "_task000300_release_permitted",
        torch.zeros_like(env._task000300_grasp_latched),
    )
    metrics = {
        "step": step,
        "side": side,
        "block_lifted": env._task000300_grasp_latched.clone(),
        "released_in_basket": env._task000300_release_latched.clone(),
        "success": (
            placed_in_tray
            & clean_close
            & clean_release
            & env._task000300_release_latched
            & ~invalid_contact(env)
        ),
    }
    env._task000300_mimic_metrics = metrics
    return metrics


def block_lifted(env, side: str) -> torch.Tensor:
    return _metrics(env, side)["block_lifted"]


def released_in_basket(env, side: str) -> torch.Tensor:
    return _metrics(env, side)["released_in_basket"]


def task_success(env, side: str) -> torch.Tensor:
    return _metrics(env, side)["success"]
