"""Task success for taking the target packet out of the shelf.

Success is intentionally task-level, not a conservative collision estimate:
the packet must still be held, its rear face must be outside the shelf, and
neighboring packets must remain approximately where they started. The packet
may rotate or settle inside the closed gripper; its orientation is not a
success condition.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def _env_ids(env, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)


def _target_local_bounds(bbox_cache, target_prim) -> tuple[torch.Tensor, torch.Tensor]:
    """Return collision bounds in the target rigid-body frame."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    world_to_target = UsdGeom.XformCache().GetLocalToWorldTransform(target_prim).GetInverse()
    points = []
    for prim in Usd.PrimRange(target_prim):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        aligned = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum, maximum = aligned.GetMin(), aligned.GetMax()
        for x in (float(minimum[0]), float(maximum[0])):
            for y in (float(minimum[1]), float(maximum[1])):
                for z in (float(minimum[2]), float(maximum[2])):
                    point = world_to_target.Transform(Gf.Vec3d(x, y, z))
                    points.append([float(point[index]) for index in range(3)])
    if not points:
        raise RuntimeError(f"target has no collision geometry: {target_prim.GetPath()}")
    values = torch.tensor(points)
    return values.amin(dim=0), values.amax(dim=0)


def initialize_takeout_geometry(env, target_name: str, shelf_prim_suffix: str) -> None:
    """Cache the shelf front and target bounds used by the success metric."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        useExtentsHint=True,
    )
    shelf_front_x = []
    target_centers = []
    target_half_extents = []
    for env_path in env.scene.env_prim_paths:
        shelf = stage.GetPrimAtPath(f"{env_path}{shelf_prim_suffix}")
        target = stage.GetPrimAtPath(f"{env_path}/{target_name}")
        if not shelf.IsValid() or not target.IsValid():
            raise RuntimeError(f"missing shelf or target below {env_path}")
        shelf_front_x.append(
            float(bbox_cache.ComputeWorldBound(shelf).ComputeAlignedRange().GetMax()[0])
        )
        minimum, maximum = _target_local_bounds(bbox_cache, target)
        target_centers.append(((minimum + maximum) * 0.5).tolist())
        target_half_extents.append(((maximum - minimum) * 0.5).tolist())

    env._task458_shelf_front_x = torch.tensor(shelf_front_x, device=env.device)
    env._task458_target_local_center = torch.tensor(target_centers, device=env.device)
    env._task458_target_local_half_extent = torch.tensor(
        target_half_extents, device=env.device
    )


def initialize_metric_buffers(env, target_name: str) -> None:
    """Allocate per-environment buffers without overwriting active episodes."""
    count = env.num_envs
    env._task458_last_tcp_relative_pos = torch.full(
        (count, 3), float("nan"), device=env.device
    )
    env._task458_grasp_stable_count = torch.zeros(
        count, dtype=torch.long, device=env.device
    )
    env._task458_grasp_latched = torch.zeros(
        count, dtype=torch.bool, device=env.device
    )
    env._task458_release_latched = torch.zeros(
        count, dtype=torch.bool, device=env.device
    )
    env._task458_takeout_metrics = None
    env._task458_neighbor_baselines = {
        name: asset.data.root_pose_w.clone()
        for name, asset in env.scene.rigid_objects.items()
        if name != target_name
    }


def reset_takeout_metric_state(env, env_ids, target_name: str) -> None:
    """Reset metric history only for the environments that restarted."""
    if not hasattr(env, "_task458_shelf_front_x"):
        initialize_takeout_geometry(
            env,
            target_name=target_name,
            shelf_prim_suffix=env.cfg.shelf_prim_suffix,
        )
    if not hasattr(env, "_task458_last_tcp_relative_pos"):
        initialize_metric_buffers(env, target_name)

    ids = _env_ids(env, env_ids)
    env._task458_last_tcp_relative_pos[ids] = float("nan")
    env._task458_grasp_stable_count[ids] = 0
    env._task458_grasp_latched[ids] = False
    env._task458_release_latched[ids] = False
    for name, baseline in env._task458_neighbor_baselines.items():
        baseline[ids] = env.scene[name].data.root_pose_w[ids].clone()
    env._task458_takeout_metrics = None


def _takeout_metrics(
    env,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
    eef_cfg: SceneEntityCfg,
    gripper_joint_name: str,
    gripper_close_threshold: float,
    tcp_envelope_min_m: tuple[float, float, float],
    tcp_envelope_max_m: tuple[float, float, float],
    relative_motion_tolerance_m: float,
    stable_control_steps: int,
    shelf_front_clearance_m: float,
    neighbor_translation_tolerance_m: float,
    neighbor_rotation_tolerance_rad: float,
    neighbor_baseline_settle_steps: int,
    shelf_prim_suffix: str,
) -> dict[str, torch.Tensor]:
    if not hasattr(env, "_task458_shelf_front_x"):
        initialize_takeout_geometry(env, object_cfg.name, shelf_prim_suffix)
    if not hasattr(env, "_task458_last_tcp_relative_pos"):
        initialize_metric_buffers(env, object_cfg.name)

    step = int(env.common_step_counter)
    cached = getattr(env, "_task458_takeout_metrics", None)
    if cached is not None and cached["step"] == step:
        return cached

    target = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    eef = env.scene[eef_cfg.name]
    tcp_pos = eef.data.target_pos_w[:, 0, :]
    tcp_quat = eef.data.target_quat_w[:, 0, :]
    relative_pos, _ = math_utils.subtract_frame_transforms(
        tcp_pos,
        tcp_quat,
        target.data.root_pos_w,
        target.data.root_quat_w,
    )

    gripper_index = robot.joint_names.index(gripper_joint_name)
    gripper_closed = robot.data.joint_pos[:, gripper_index] >= gripper_close_threshold
    gripper_open = ~gripper_closed
    lower = torch.tensor(tcp_envelope_min_m, device=env.device)
    upper = torch.tensor(tcp_envelope_max_m, device=env.device)
    inside_gripper = torch.logical_and(
        relative_pos >= lower, relative_pos <= upper
    ).all(dim=-1)

    previous = env._task458_last_tcp_relative_pos
    previous_valid = torch.isfinite(previous).all(dim=-1)
    relative_motion = torch.linalg.vector_norm(
        relative_pos - torch.nan_to_num(previous), dim=-1
    )
    stable_now = (
        gripper_closed
        & inside_gripper
        & previous_valid
        & (relative_motion <= relative_motion_tolerance_m)
    )
    env._task458_grasp_stable_count = torch.where(
        stable_now,
        env._task458_grasp_stable_count + 1,
        torch.zeros_like(env._task458_grasp_stable_count),
    )
    env._task458_last_tcp_relative_pos = relative_pos.clone()
    held_current = env._task458_grasp_stable_count >= stable_control_steps
    env._task458_grasp_latched |= held_current

    rotation = math_utils.matrix_from_quat(target.data.root_quat_w)
    target_center = target.data.root_pos_w + math_utils.quat_apply(
        target.data.root_quat_w, env._task458_target_local_center
    )
    projected_half_x = (
        torch.abs(rotation[:, 0, :]) * env._task458_target_local_half_extent
    ).sum(dim=-1)
    target_rear_x = target_center[:, 0] - projected_half_x
    target_outside_shelf = target_rear_x >= (
        env._task458_shelf_front_x + shelf_front_clearance_m
    )

    neighbors_static = torch.ones(
        env.num_envs, dtype=torch.bool, device=env.device
    )
    max_neighbor_translation = torch.zeros(env.num_envs, device=env.device)
    max_neighbor_rotation = torch.zeros(env.num_envs, device=env.device)
    settling = env.episode_length_buf <= neighbor_baseline_settle_steps
    for name, baseline in env._task458_neighbor_baselines.items():
        current = env.scene[name].data.root_pose_w
        baseline[settling] = current[settling].clone()
        translation = torch.linalg.vector_norm(
            current[:, :3] - baseline[:, :3], dim=-1
        )
        quat_dot = torch.abs(
            (current[:, 3:7] * baseline[:, 3:7]).sum(dim=-1)
        ).clamp(max=1.0)
        rotation_error = 2.0 * torch.acos(quat_dot)
        max_neighbor_translation = torch.maximum(
            max_neighbor_translation, translation
        )
        max_neighbor_rotation = torch.maximum(max_neighbor_rotation, rotation_error)
        neighbors_static &= (
            translation <= neighbor_translation_tolerance_m
        ) & (rotation_error <= neighbor_rotation_tolerance_rad)

    task_success_value = held_current & target_outside_shelf & neighbors_static
    # This is a MimicGen subtask boundary signal, not the task-level success
    # criterion. Keep it scoped to the target/gripper behavior so failed
    # neighbor-static checks do not prevent source trajectory segmentation.
    released_after_takeout = (
        env._task458_grasp_latched
        & target_outside_shelf
        & gripper_open
    )
    env._task458_release_latched |= released_after_takeout
    metrics = {
        "step": step,
        "grasp_stable": env._task458_grasp_latched.clone(),
        "released_after_takeout": env._task458_release_latched.clone(),
        "held_current": held_current,
        "gripper_open": gripper_open,
        "target_outside_shelf": target_outside_shelf,
        "neighbors_static": neighbors_static,
        "task_success": task_success_value,
        "tcp_relative_motion": relative_motion,
        "target_rear_x": target_rear_x,
        "shelf_front_x": env._task458_shelf_front_x,
        "max_neighbor_translation": max_neighbor_translation,
        "max_neighbor_rotation": max_neighbor_rotation,
    }
    env._task458_takeout_metrics = metrics
    return metrics


def grasp_stable(env, metric_params) -> torch.Tensor:
    """Latched grasp signal used only to split the Mimic source trajectory."""
    return _takeout_metrics(env, **metric_params)["grasp_stable"]


def released_after_takeout(env, metric_params) -> torch.Tensor:
    """Latched release signal used only to split the Mimic source trajectory."""
    return _takeout_metrics(env, **metric_params)["released_after_takeout"]


def held_current(env, metric_params) -> torch.Tensor:
    return _takeout_metrics(env, **metric_params)["held_current"]


def target_outside_shelf(env, metric_params) -> torch.Tensor:
    return _takeout_metrics(env, **metric_params)["target_outside_shelf"]


def neighbors_static(env, metric_params) -> torch.Tensor:
    return _takeout_metrics(env, **metric_params)["neighbors_static"]


def task_success(env, metric_params) -> torch.Tensor:
    return _takeout_metrics(env, **metric_params)["task_success"]


def takeout_success(env, metric_params) -> torch.Tensor:
    return task_success(env, metric_params)
