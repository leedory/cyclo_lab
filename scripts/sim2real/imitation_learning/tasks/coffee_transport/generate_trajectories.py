"""Generate randomized two-can coffee transport trajectories from one v3 seed."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-id", choices=("000001", "000002"), required=True)
parser.add_argument("--input-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--num-successes", type=int, default=10)
parser.add_argument("--max-attempts", type=int, default=100)
parser.add_argument("--seed", type=int, default=20260907)
parser.add_argument("--keep-failures", action="store_true")
parser.add_argument("--grasp-correction-can-x-m", type=float, default=0.0)
parser.add_argument("--grasp-correction-can-z-m", type=float, default=0.0)
parser.add_argument("--grasp-correction-blend-steps", type=int, default=30)
parser.add_argument(
    "--experimental-realized-initial-pose",
    action="store_true",
    help=(
        "Deprecated no-op retained for v12 probe command compatibility; "
        "commanded-joint hold is now the production default."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers.recorder_manager import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.coffee_transport_common import (
    CABINET_RIGHT_DELTA_Y_M,
    COFFEE_CAN_NAMES,
    DESTINATION_LOWER_SUPPORT_Z_M,
    LIFT_TRAVEL_COMMAND_M,
    SOURCE_SPAWN_SQUARES,
    SOURCE_UPPER_SUPPORT_Z_M,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.grasp_tracking import (
    CloseSegment,
    infer_close_segment,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.quality_contract import (
    DEFAULT_BODY_CARRY_QUALITY,
    DEFAULT_PLACEMENT_QUALITY,
    evaluate_body_carry,
    evaluate_placement,
    upright_cosine_from_wxyz,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.trajectory_timing import (
    scene_pose_at_pre_step,
)
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_PUBLISHED_JOINT_NAMES


ACTIVE = (("coffee_can_right", "right"), ("coffee_can_left", "left"))
EEF_KEYS = {"left": "left_eef_pose", "right": "right_eef_pose"}
CAMERA_NAMES = ("cam_head", "cam_wrist_left", "cam_wrist_right")
GRIPPER_INDEX = {"left": 7, "right": 15}
LIFT_INDEX = 18
POSE_BLEND_STEPS = 12
PRE_RECORD_SETTLE_MAX_STEPS = 120
PRE_RECORD_SETTLE_STABLE_STEPS = 10
PRE_RECORD_ARM_VELOCITY_TOLERANCE_RADPS = 0.05
PRE_RECORD_ARM_STEP_TOLERANCE_RAD = 0.0005
PRE_RECORD_EEF_POSITION_STEP_TOLERANCE_M = 0.0005
PRE_RECORD_EEF_ORIENTATION_STEP_TOLERANCE_RAD = 0.005
INITIAL_HOLD_AUDIT_STEPS = 10
NORMALIZE_STABLE_STEPS = 8
AXIS_STABLE_STEPS = 8
BASE_SPEED_MAX_MPS = 0.10
BASE_YAW_SPEED_MAX_RADPS = 0.12
BASE_ENDPOINT_XY_TOLERANCE_M = 0.020
BASE_ENDPOINT_YAW_TOLERANCE_RAD = math.radians(1.0)
SUCCESS_CRITERION_ID = "coffee_two_can_carry_and_placement_quality_gate_v3"
GENERATION_METHODS = {
    "000001": "coffee_two_can_endpoint_lift_targets_pending_validation_v13",
    "000002": "coffee_two_can_production_realized_initial_commanded_joint_hold_v12",
}
INITIAL_NORMALIZATION_CONTRACT_IDS = {
    "000001": "coffee_realized_pose_endpoint_lift_targets_v13",
    "000002": "coffee_production_realized_pose_commanded_joint_hold_v12",
}
GENERATION_PROFILE_IDS = {
    "000001": "coffee_physical_trajectory_generation_endpoint_lift_targets_v13",
    "000002": "coffee_physical_trajectory_generation_commanded_joint_hold_v12",
}
PRODUCTION_APPROVED_BY_TASK = {"000001": False, "000002": True}
TASK_AXIS_ACTION_CONTRACTS = {
    "000001": "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1",
    "000002": "base_body_velocity_feedback_v1",
}
COMMON_PRODUCTION_PROVENANCE = {
    "coffee_can_collision_profile_id": "cylinder_r33p5_h90_v1",
    "coffee_can_collision_asset_sha256": (
        "fe340f411e93fd3fd54bc06b1b4ed1e6dec24214179e69ae5dd6ce1dd50d4a82"
    ),
    "coffee_can_collision_12gon_applied": False,
}
PRODUCTION_PROVENANCE_BY_TASK = {
    "000001": {
        **COMMON_PRODUCTION_PROVENANCE,
        "production_promotion_id": "coffee_endpoint_lift_targets_v13_pending_validation",
        "production_promotion_report": "",
        "production_validation_status": "pending_new_physical_trajectory_validation",
        "supersedes_generation_contract_id": (
            "coffee_production_realized_pose_commanded_joint_hold_v12"
        ),
        "superseded_contract_issue": "intermediate_absolute_lift_targets",
    },
    "000002": {
        **COMMON_PRODUCTION_PROVENANCE,
        "production_promotion_id": "coffee_commanded_joint_hold_v12_promoted_20260908",
        "production_promotion_report": (
            "agents/experiments/2026-09-07_coffee_transport_final450/reports/"
            "INITIAL_HOLD_V12_PRODUCTION_PROMOTION_KR.md"
        ),
        "production_probe_task000001_seed": 2026100104,
        "production_probe_task000001_hdf5_sha256": (
            "1e78df53dabe323e2fa42ae51887ab2540cded697b9a6069bfa0aa01780fdad1"
        ),
        "production_probe_task000001_audit_sha256": (
            "52338902ac4e57a52a67dd912e654e0af80726bda58de42e22aa74278e183967"
        ),
        "production_probe_task000002_seed": 2026103001,
        "production_probe_task000002_hdf5_sha256": (
            "cb7c8bf8c0a17983201d237fac56fbb137d31cd658c2ecd76f91d10d224408f9"
        ),
        "production_probe_task000002_audit_sha256": (
            "10b8dbd5192f9493eebbd8c8bf62992e071a1fbf9fd2e08f58d725fb5b57f9eb"
        ),
    },
}
INITIAL_HOLD_ARM_MAX_RAD = 0.010
INITIAL_HOLD_TASK000001_ROOT_XY_MAX_M = 0.003
REALIZED_SETTLE_ROOT_LINEAR_VELOCITY_MAX_MPS = 0.005
REALIZED_SETTLE_ROOT_ANGULAR_VELOCITY_MAX_RADPS = 0.02
REALIZED_SETTLE_CAN_LINEAR_VELOCITY_MAX_MPS = 0.005
REALIZED_SETTLE_CAN_ANGULAR_VELOCITY_MAX_RADPS = 0.02


@dataclass(frozen=True)
class SourceEpisode:
    path: str
    sha256: str
    actions: np.ndarray
    joint_pos: np.ndarray
    eef: dict[str, np.ndarray]
    root_pose: np.ndarray
    initial_root_pose: np.ndarray
    can_pose: dict[str, np.ndarray]
    initial_can_pose: dict[str, np.ndarray]
    attrs: dict[str, object]


@dataclass
class AttemptMetrics:
    initial: dict[str, float]
    cycles: dict[str, dict[str, float]]


def _jsonable(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_lift_action_endpoints(actions: np.ndarray, *, context: str) -> None:
    values = np.asarray(actions)[:, LIFT_INDEX]
    endpoint_distance = np.minimum(
        np.abs(values),
        np.abs(values - LIFT_TRAVEL_COMMAND_M),
    )
    invalid = np.flatnonzero(endpoint_distance > 1.0e-6)
    if len(invalid):
        frame = int(invalid[0])
        raise ValueError(
            f"{context} lift action at frame {frame} is {values[frame]:.9f}; "
            "Task000001 requires only absolute endpoint targets 0.0 or -0.28 m"
        )


def load_source(path: str, expected_task_id: str) -> SourceEpisode:
    source_path = os.path.abspath(path)
    source_sha256 = file_sha256(source_path)
    with h5py.File(path, "r") as source:
        data = source["data"]
        demos = sorted(data.keys())
        if demos != ["demo_0"]:
            raise ValueError(f"coffee v3 seed must contain only demo_0, got {demos}")
        group = data["demo_0"]
        attrs = {key: _jsonable(value) for key, value in group.attrs.items()}
        checks = {
            "coffee_task_id": expected_task_id,
            "coffee_seed_mode": (
                "two_operator_pickups_tact_immediate_g_automatic_place_v4"
                if expected_task_id == "000001"
                else "two_operator_pickups_tact_immediate_g_automatic_place_v3"
            ),
            "coffee_can_order": "coffee_can_right,coffee_can_left",
            "coffee_arm_order": "right,left",
        }
        for key, expected in checks.items():
            if attrs.get(key) != expected:
                raise ValueError(
                    f"seed {key}={attrs.get(key)!r}; required {expected!r}"
                )
        if not bool(attrs.get("success")) or int(attrs.get("coffee_completed_cycles", 0)) != 2:
            raise ValueError("seed must be accepted and contain two completed cycles")
        actions = np.asarray(group["actions"], dtype=np.float32)
        expected_width = 19 if expected_task_id == "000001" else 22
        if actions.shape[1] != expected_width:
            raise ValueError(f"seed action width must be {expected_width}, got {actions.shape}")
        if expected_task_id == "000001":
            validate_lift_action_endpoints(actions, context="seed")
        return SourceEpisode(
            path=source_path,
            sha256=source_sha256,
            actions=actions,
            joint_pos=np.asarray(group["obs/joint_pos"], dtype=np.float32),
            eef={
                side: np.asarray(group[f"obs/{key}"], dtype=np.float32)
                for side, key in EEF_KEYS.items()
            },
            root_pose=np.asarray(group["states/articulation/robot/root_pose"], dtype=np.float32),
            initial_root_pose=np.asarray(
                group["initial_state/articulation/robot/root_pose"][0],
                dtype=np.float32,
            ),
            can_pose={
                name: np.asarray(group[f"states/rigid_object/{name}/root_pose"], dtype=np.float32)
                for name in COFFEE_CAN_NAMES
            },
            initial_can_pose={
                name: np.asarray(
                    group[f"initial_state/rigid_object/{name}/root_pose"][0],
                    dtype=np.float32,
                )
                for name in COFFEE_CAN_NAMES
            },
            attrs=attrs,
        )


def quat_yaw_wxyz(quaternion: torch.Tensor) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def pose_lerp(first: torch.Tensor, second: torch.Tensor, alpha: float) -> torch.Tensor:
    weight = min(max(float(alpha), 0.0), 1.0)
    position = (1.0 - weight) * first[:3] + weight * second[:3]
    quaternion = math_utils.quat_slerp(first[3:], second[3:], weight)
    return torch.cat((position, quaternion))


def current_root_pose(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return torch.cat((robot.data.root_pos_w[0], robot.data.root_quat_w[0]))


def capture_initial_snapshot(env) -> dict[str, object]:
    robot = env.scene["robot"]
    lift_id = robot.joint_names.index("lift_joint")
    return {
        "robot_spawn": current_root_pose(env).detach().clone(),
        "robot_home": env._coffee_transport_episode_home_root_pose_w[0].detach().clone(),
        "lift": float(robot.data.joint_pos[0, lift_id].item()),
        "head": robot.data.joint_pos[0, [robot.joint_names.index(name) for name in FFW_SG2_PUBLISHED_JOINT_NAMES[16:18]]].detach().clone(),
        "cans": {
            name: torch.cat((env.scene[name].data.root_pos_w[0], env.scene[name].data.root_quat_w[0])).detach().clone()
            for name in COFFEE_CAN_NAMES
        },
    }


def snapshot_metadata(prefix: str, snapshot: dict[str, object]) -> dict[str, float]:
    spawn = snapshot["robot_spawn"]
    home = snapshot["robot_home"]
    values = {
        f"{prefix}_robot_spawn_x_m": float(spawn[0].item()),
        f"{prefix}_robot_spawn_y_m": float(spawn[1].item()),
        f"{prefix}_robot_spawn_yaw_rad": quat_yaw_wxyz(spawn[3:]),
        f"{prefix}_robot_home_x_m": float(home[0].item()),
        f"{prefix}_robot_home_y_m": float(home[1].item()),
        f"{prefix}_robot_home_yaw_rad": quat_yaw_wxyz(home[3:]),
        f"{prefix}_lift_spawn_m": float(snapshot["lift"]),
    }
    for name, pose in snapshot["cans"].items():
        values[f"{prefix}_{name}_x_m"] = float(pose[0].item())
        values[f"{prefix}_{name}_y_m"] = float(pose[1].item())
        values[f"{prefix}_{name}_yaw_rad"] = quat_yaw_wxyz(pose[3:])
    return values


def current_eef_pose_root(env, side: str) -> torch.Tensor:
    return env.obs_buf["policy"][EEF_KEYS[side]][0].detach().clone()


def current_eef_pose_world(env, side: str) -> torch.Tensor:
    sensor = env.scene[f"{side}_eef"]
    return torch.cat(
        (sensor.data.target_pos_w[0, 0], sensor.data.target_quat_w[0, 0])
    )


def source_eef_world(source: SourceEpisode, side: str, step: int, device: str) -> torch.Tensor:
    root = torch.as_tensor(
        scene_pose_at_pre_step(source.initial_root_pose, source.root_pose, step),
        device=device,
    )
    relative = torch.as_tensor(source.eef[side][step], device=device)
    position, quaternion = math_utils.combine_frame_transforms(
        root[:3], root[3:], relative[:3], relative[3:]
    )
    return torch.cat((position, quaternion))


def source_can_pose_at_pre_step(
    source: SourceEpisode, can_name: str, step: int, device: str
) -> torch.Tensor:
    return torch.as_tensor(
        scene_pose_at_pre_step(
            source.initial_can_pose[can_name], source.can_pose[can_name], step
        ),
        device=device,
    )


def translated_source_eef_root(
    env,
    source: SourceEpisode,
    side: str,
    step: int,
    translation_world: torch.Tensor,
) -> torch.Tensor:
    world = source_eef_world(source, side, step, env.device)
    target_position = world[:3] + translation_world
    root = current_root_pose(env)
    position, quaternion = math_utils.subtract_frame_transforms(
        root[:3], root[3:], target_position, world[3:]
    )
    return torch.cat((position, quaternion))


def source_pose_root(source: SourceEpisode, side: str, step: int, device: str) -> torch.Tensor:
    return torch.as_tensor(source.eef[side][step], device=device).clone()


def source_action(env, source: SourceEpisode, step: int) -> torch.Tensor:
    return torch.as_tensor(source.actions[step], dtype=torch.float32, device=env.device)


def build_action(
    env,
    source: SourceEpisode,
    step: int,
    left_pose: torch.Tensor,
    right_pose: torch.Tensor,
    *,
    base_velocity: torch.Tensor | None = None,
    lift_target: float | None = None,
    gripper_override: dict[str, float] | None = None,
) -> torch.Tensor:
    source_command = source_action(env, source, step)
    width = 19 if args_cli.task_id == "000001" else 22
    action = torch.zeros((1, width), dtype=torch.float32, device=env.device)
    action[0, 0:7] = left_pose
    action[0, 7] = source_command[7]
    action[0, 8:15] = right_pose
    action[0, 15] = source_command[15]
    action[0, 16:18] = source_command[16:18]
    action[0, 18] = source_command[18] if lift_target is None else lift_target
    if gripper_override:
        for side, value in gripper_override.items():
            action[0, GRIPPER_INDEX[side]] = value
    if width == 22 and base_velocity is not None:
        action[0, 19:22] = base_velocity
    if not bool(torch.isfinite(action).all().item()):
        raise ValueError("generated action contains non-finite values")
    return action


def step_checked(env, action: torch.Tensor):
    observation, _, terminated, truncated, _ = env.step(action)
    if bool(terminated[0].item()) or bool(truncated[0].item()):
        raise RuntimeError("environment terminated before the generated trajectory completed")
    return observation



def refresh_camera_observations(env, updates: int = 4) -> None:
    """Refresh reset camera buffers without advancing physics or actions."""

    if updates < 1:
        raise ValueError("camera refresh updates must be positive")
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    env.scene.write_data_to_sim()
    for _ in range(updates):
        for camera_name in CAMERA_NAMES:
            env.scene.sensors[camera_name].reset(env_ids)
        env.sim.render()
        for camera_name in CAMERA_NAMES:
            _ = env.scene.sensors[camera_name].data.output["rgb"]
    env.obs_buf = env.observation_manager.compute(update_history=True)

def refresh_eef_observations(env) -> None:
    """Refresh lazy FrameTransformer buffers after direct state restoration."""

    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    for side in ("left", "right"):
        sensor = env.scene[f"{side}_eef"]
        sensor.reset(env_ids)
        sensor.update(0.0, force_recompute=True)
        _ = sensor.data.target_pos_w
    env.obs_buf = env.observation_manager.compute(update_history=True)


def hold_current_action(env, source: SourceEpisode, step: int, *, lift_target=None, base_velocity=None):
    return build_action(
        env,
        source,
        step,
        current_eef_pose_root(env, "left"),
        current_eef_pose_root(env, "right"),
        lift_target=lift_target,
        base_velocity=base_velocity,
    )


def hold_source_pose_action(env, source: SourceEpisode, step: int):
    return build_action(
        env,
        source,
        step,
        source_pose_root(source, "left", step, env.device),
        source_pose_root(source, "right", step, env.device),
    )


def settle_episode_before_recording(
    env, source: SourceEpisode, task_id: str
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    """Settle arm dynamics off-record, then restore the sampled task state."""

    robot = env.scene["robot"]
    arm_names = (
        *FFW_SG2_PUBLISHED_JOINT_NAMES[:7],
        *FFW_SG2_PUBLISHED_JOINT_NAMES[8:15],
    )
    arm_ids = [robot.joint_names.index(name) for name in arm_names]
    head_ids = [
        robot.joint_names.index(name)
        for name in FFW_SG2_PUBLISHED_JOINT_NAMES[16:18]
    ]
    lift_id = robot.joint_names.index("lift_joint")
    sampled_root_pose = current_root_pose(env).detach().clone()
    sampled_lift = robot.data.joint_pos[0, lift_id].detach().clone()
    sampled_head = robot.data.joint_pos[0, head_ids].detach().clone()
    sampled_can_poses = {
        name: torch.cat(
            (
                env.scene[name].data.root_pos_w[0],
                env.scene[name].data.root_quat_w[0],
            )
        )
        .detach()
        .clone()
        for name in COFFEE_CAN_NAMES
    }
    held_eef = {
        side: current_eef_pose_root(env, side)
        for side in ("left", "right")
    }
    initial_arm = robot.data.joint_pos[0, arm_ids].detach().clone()
    previous_arm = initial_arm.clone()
    previous_eef = {side: pose.clone() for side, pose in held_eef.items()}
    stable = 0
    max_arm_drift = 0.0
    final_arm_velocity = float("inf")
    final_arm_step = float("inf")
    final_eef_position_step = float("inf")
    final_eef_orientation_step = float("inf")

    for step in range(1, PRE_RECORD_SETTLE_MAX_STEPS + 1):
        action = build_action(
            env,
            source,
            0,
            held_eef["left"],
            held_eef["right"],
            lift_target=float(sampled_lift.item()),
            base_velocity=(
                torch.zeros(3, dtype=torch.float32, device=env.device)
                if task_id == "000002"
                else None
            ),
        )
        action[0, 16:18] = sampled_head
        step_checked(env, action)

        arm = robot.data.joint_pos[0, arm_ids].detach().clone()
        final_arm_velocity = float(
            torch.max(torch.abs(robot.data.joint_vel[0, arm_ids])).item()
        )
        final_arm_step = float(torch.max(torch.abs(arm - previous_arm)).item())
        max_arm_drift = max(
            max_arm_drift, float(torch.max(torch.abs(arm - initial_arm)).item())
        )
        eef_position_steps = []
        eef_orientation_steps = []
        current_eef = {}
        for side in ("left", "right"):
            pose = current_eef_pose_root(env, side)
            current_eef[side] = pose
            eef_position_steps.append(
                float(
                    torch.linalg.vector_norm(
                        pose[:3] - previous_eef[side][:3]
                    ).item()
                )
            )
            dot = torch.clamp(
                torch.abs(torch.dot(pose[3:], previous_eef[side][3:])),
                max=1.0,
            )
            eef_orientation_steps.append(float((2.0 * torch.acos(dot)).item()))
        final_eef_position_step = max(eef_position_steps)
        final_eef_orientation_step = max(eef_orientation_steps)
        is_stable = (
            final_arm_velocity <= PRE_RECORD_ARM_VELOCITY_TOLERANCE_RADPS
            and final_arm_step <= PRE_RECORD_ARM_STEP_TOLERANCE_RAD
            and final_eef_position_step
            <= PRE_RECORD_EEF_POSITION_STEP_TOLERANCE_M
            and final_eef_orientation_step
            <= PRE_RECORD_EEF_ORIENTATION_STEP_TOLERANCE_RAD
        )
        stable = stable + 1 if is_stable else 0
        previous_arm = arm
        previous_eef = current_eef
        if stable >= PRE_RECORD_SETTLE_STABLE_STEPS:
            break
    else:
        raise RuntimeError(
            "arm pre-record settle gate failed: "
            f"velocity={final_arm_velocity:.6f} rad/s "
            f"joint_step={final_arm_step:.6f} rad "
            f"eef_position_step={final_eef_position_step:.6f} m "
            f"eef_orientation_step={final_eef_orientation_step:.6f} rad"
        )

    root_drift_before_restore = float(
        torch.linalg.vector_norm(
            current_root_pose(env)[:2] - sampled_root_pose[:2]
        ).item()
    )
    can_drift_before_restore = max(
        float(
            torch.linalg.vector_norm(
                env.scene[name].data.root_pos_w[0] - sampled_can_poses[name][:3]
            ).item()
        )
        for name in COFFEE_CAN_NAMES
    )

    # Keep the settled arm posture, but restore every randomized task coordinate
    # that the hidden physics pre-roll was not allowed to alter.
    restored_joint_pos = robot.data.joint_pos[0].detach().clone()
    restored_joint_pos[head_ids] = sampled_head
    restored_joint_pos[lift_id] = sampled_lift
    robot.write_joint_state_to_sim(
        restored_joint_pos.unsqueeze(0),
        torch.zeros_like(restored_joint_pos).unsqueeze(0),
    )
    robot.set_joint_position_target(restored_joint_pos.unsqueeze(0))
    robot.write_root_pose_to_sim(sampled_root_pose.unsqueeze(0))
    robot.write_root_velocity_to_sim(
        torch.zeros((1, 6), dtype=torch.float32, device=env.device)
    )
    for name, pose in sampled_can_poses.items():
        can = env.scene[name]
        can.write_root_pose_to_sim(pose.unsqueeze(0))
        can.write_root_velocity_to_sim(
            torch.zeros((1, 6), dtype=torch.float32, device=env.device)
        )
    env.scene.write_data_to_sim()
    env.sim.forward()
    refresh_eef_observations(env)

    restored_root_error = float(
        torch.linalg.vector_norm(
            current_root_pose(env)[:2] - sampled_root_pose[:2]
        ).item()
    )
    restored_lift_error = abs(
        float(robot.data.joint_pos[0, lift_id].item() - sampled_lift.item())
    )
    restored_can_error = max(
        float(
            torch.linalg.vector_norm(
                env.scene[name].data.root_pos_w[0] - sampled_can_poses[name][:3]
            ).item()
        )
        for name in COFFEE_CAN_NAMES
    )
    if max(restored_root_error, restored_lift_error, restored_can_error) > 1.0e-5:
        raise RuntimeError(
            "pre-record sampled state restoration failed: "
            f"root={restored_root_error:.8f} lift={restored_lift_error:.8f} "
            f"can={restored_can_error:.8f}"
        )

    metrics = {
        "preroll_steps": float(step),
        "preroll_arm_joint_max_drift_rad": max_arm_drift,
        "preroll_final_arm_velocity_max_radps": final_arm_velocity,
        "preroll_final_arm_step_max_rad": final_arm_step,
        "preroll_final_eef_position_step_max_m": final_eef_position_step,
        "preroll_final_eef_orientation_step_max_rad": (
            final_eef_orientation_step
        ),
        "preroll_root_xy_drift_before_restore_m": root_drift_before_restore,
        "preroll_can_translation_drift_before_restore_m": (
            can_drift_before_restore
        ),
        "preroll_restored_root_xy_error_m": restored_root_error,
        "preroll_restored_lift_error_m": restored_lift_error,
        "preroll_restored_can_translation_error_m": restored_can_error,
    }
    return metrics, {side: pose.clone() for side, pose in held_eef.items()}


def settle_restored_episode_as_realized_initial(
    env,
    source: SourceEpisode,
    task_id: str,
    hold_eef_pose: dict[str, torch.Tensor],
    requested: dict[str, object],
) -> dict[str, float]:
    """Second hidden settle; adopt only realized poses inside reviewed envelopes."""

    robot = env.scene["robot"]
    arm_names = (*FFW_SG2_PUBLISHED_JOINT_NAMES[:7], *FFW_SG2_PUBLISHED_JOINT_NAMES[8:15])
    arm_ids = [robot.joint_names.index(name) for name in arm_names]
    previous_arm = robot.data.joint_pos[0, arm_ids].detach().clone()
    previous_eef = {side: current_eef_pose_root(env, side) for side in ("left", "right")}
    stable = 0
    root_linear_speed = root_angular_speed = float("inf")
    can_linear_speed = can_angular_speed = float("inf")

    for step in range(1, PRE_RECORD_SETTLE_MAX_STEPS + 1):
        action = build_action(
            env, source, 0, hold_eef_pose["left"], hold_eef_pose["right"],
            lift_target=float(requested["lift"]),
            base_velocity=(torch.zeros(3, dtype=torch.float32, device=env.device) if task_id == "000002" else None),
        )
        action[0, 16:18] = requested["head"]
        step_checked(env, action)
        arm = robot.data.joint_pos[0, arm_ids].detach().clone()
        arm_velocity = float(torch.max(torch.abs(robot.data.joint_vel[0, arm_ids])).item())
        arm_step = float(torch.max(torch.abs(arm - previous_arm)).item())
        eef_position_step = 0.0
        eef_orientation_step = 0.0
        current_eef = {}
        for side in ("left", "right"):
            pose = current_eef_pose_root(env, side)
            current_eef[side] = pose
            eef_position_step = max(eef_position_step, float(torch.linalg.vector_norm(pose[:3] - previous_eef[side][:3]).item()))
            dot = torch.clamp(torch.abs(torch.dot(pose[3:], previous_eef[side][3:])), max=1.0)
            eef_orientation_step = max(eef_orientation_step, float((2.0 * torch.acos(dot)).item()))
        root_linear_speed = float(torch.linalg.vector_norm(robot.data.root_lin_vel_w[0]).item())
        root_angular_speed = float(torch.linalg.vector_norm(robot.data.root_ang_vel_w[0]).item())
        can_linear_speed = max(float(torch.linalg.vector_norm(env.scene[name].data.root_lin_vel_w[0]).item()) for name in COFFEE_CAN_NAMES)
        can_angular_speed = max(float(torch.linalg.vector_norm(env.scene[name].data.root_ang_vel_w[0]).item()) for name in COFFEE_CAN_NAMES)
        is_stable = (
            arm_velocity <= PRE_RECORD_ARM_VELOCITY_TOLERANCE_RADPS
            and arm_step <= PRE_RECORD_ARM_STEP_TOLERANCE_RAD
            and eef_position_step <= PRE_RECORD_EEF_POSITION_STEP_TOLERANCE_M
            and eef_orientation_step <= PRE_RECORD_EEF_ORIENTATION_STEP_TOLERANCE_RAD
            and root_linear_speed <= REALIZED_SETTLE_ROOT_LINEAR_VELOCITY_MAX_MPS
            and root_angular_speed <= REALIZED_SETTLE_ROOT_ANGULAR_VELOCITY_MAX_RADPS
            and can_linear_speed <= REALIZED_SETTLE_CAN_LINEAR_VELOCITY_MAX_MPS
            and can_angular_speed <= REALIZED_SETTLE_CAN_ANGULAR_VELOCITY_MAX_RADPS
        )
        stable = stable + 1 if is_stable else 0
        previous_arm = arm
        previous_eef = current_eef
        if stable >= PRE_RECORD_SETTLE_STABLE_STEPS:
            break
    else:
        raise RuntimeError("realized-pose second settle did not stabilize")

    refresh_eef_observations(env)
    actual_spawn = current_root_pose(env).detach().clone()
    requested_spawn = requested["robot_spawn"]
    requested_home = requested["robot_home"]
    yaw_delta = math.atan2(
        math.sin(quat_yaw_wxyz(actual_spawn[3:]) - quat_yaw_wxyz(requested_spawn[3:])),
        math.cos(quat_yaw_wxyz(actual_spawn[3:]) - quat_yaw_wxyz(requested_spawn[3:])),
    )
    realized_home = requested_home.clone()
    realized_home[:2] += actual_spawn[:2] - requested_spawn[:2]
    realized_home_yaw = quat_yaw_wxyz(requested_home[3:]) + yaw_delta
    realized_home[3:] = torch.tensor(
        [math.cos(realized_home_yaw / 2.0), 0.0, 0.0, math.sin(realized_home_yaw / 2.0)],
        dtype=realized_home.dtype, device=realized_home.device,
    )
    env._coffee_transport_episode_home_root_pose_w[0] = realized_home
    realized = capture_initial_snapshot(env)

    nominal_home = robot.data.default_root_state[0, :7].detach().clone()
    nominal_home[:3] += env.scene.env_origins[0]
    home = realized["robot_home"]
    yaw_error = math.atan2(
        math.sin(quat_yaw_wxyz(home[3:]) - quat_yaw_wxyz(nominal_home[3:])),
        math.cos(quat_yaw_wxyz(home[3:]) - quat_yaw_wxyz(nominal_home[3:])),
    )
    if (
        abs(float(home[0] - nominal_home[0])) > 0.020
        or abs(float(home[1] - nominal_home[1])) > 0.020
        or abs(yaw_error) > math.radians(1.0)
    ):
        raise RuntimeError("realized robot HOME is outside reviewed +/-2 cm, +/-1 deg envelope")
    for square in SOURCE_SPAWN_SQUARES:
        pose = realized["cans"][square.name]
        if not (square.x_min_m <= float(pose[0]) <= square.x_max_m and square.y_min_m <= float(pose[1]) <= square.y_max_m):
            raise RuntimeError(f"realized {square.name} is outside its reviewed half-depth source region")

    lift = float(realized["lift"])
    if not -0.28 <= lift <= 0.0:
        raise RuntimeError(f"realized lift {lift:.6f} m is outside [-0.28, 0.0]")

    metrics = {
        **snapshot_metadata("requested", requested),
        **snapshot_metadata("realized", realized),
        "realized_second_settle_steps": float(step),
        "realized_second_settle_root_linear_speed_mps": root_linear_speed,
        "realized_second_settle_root_angular_speed_radps": root_angular_speed,
        "realized_second_settle_can_linear_speed_mps": can_linear_speed,
        "realized_second_settle_can_angular_speed_radps": can_angular_speed,
    }
    return metrics


def validate_initial_hold(task_id: str, metrics: dict[str, object]) -> None:
    arm_delta = metrics["normalization_first10_arm_joint_max_delta_rad"]
    if arm_delta > INITIAL_HOLD_ARM_MAX_RAD:
        raise RuntimeError(f"initial arm hold gate failed: {arm_delta:.6f} rad")
    if task_id == "000001":
        root_delta = metrics["normalization_first10_root_xy_max_delta_m"]
        if root_delta > INITIAL_HOLD_TASK000001_ROOT_XY_MAX_M:
            raise RuntimeError(f"initial root hold gate failed: {root_delta:.6f} m")


def normalize_episode_start(
    env,
    source: SourceEpisode,
    task_id: str,
    hold_eef_pose: dict[str, torch.Tensor],
) -> dict[str, object]:
    robot = env.scene["robot"]
    arm_names = (
        *FFW_SG2_PUBLISHED_JOINT_NAMES[:7],
        *FFW_SG2_PUBLISHED_JOINT_NAMES[8:15],
    )
    arm_ids = [robot.joint_names.index(name) for name in arm_names]
    initial_arm_joint_pos = robot.data.joint_pos[0, arm_ids].detach().clone()
    initial_root_xy = current_root_pose(env)[:2].detach().clone()
    initial_eef_pose = {
        side: hold_eef_pose[side].detach().clone() for side in ("left", "right")
    }
    normalization_steps = 0
    first10_arm_joint_max_delta_rad = 0.0
    first10_root_xy_max_delta_m = 0.0
    arm_hold_terms = []
    arm_hold_targets: dict[str, torch.Tensor] = {}
    arm_hold_start_actual: dict[str, torch.Tensor] = {}

    def step_and_measure(action: torch.Tensor) -> None:
        nonlocal normalization_steps
        nonlocal first10_arm_joint_max_delta_rad
        nonlocal first10_root_xy_max_delta_m
        step_checked(env, action)
        normalization_steps += 1
        if normalization_steps <= INITIAL_HOLD_AUDIT_STEPS:
            arm_delta = torch.max(
                torch.abs(robot.data.joint_pos[0, arm_ids] - initial_arm_joint_pos)
            )
            root_delta = torch.linalg.vector_norm(
                current_root_pose(env)[:2] - initial_root_xy
            )
            first10_arm_joint_max_delta_rad = max(
                first10_arm_joint_max_delta_rad, float(arm_delta.item())
            )
            first10_root_xy_max_delta_m = max(
                first10_root_xy_max_delta_m, float(root_delta.item())
            )

    def metrics() -> dict[str, object]:
        return {
            "normalization_steps": float(normalization_steps),
            "normalization_external_joint_hold_steps": float(
                normalization_steps
            ),
            "normalization_external_joint_hold_left_target_rad_json": json.dumps(
                arm_hold_targets["left"][0].detach().cpu().tolist()
            ),
            "normalization_external_joint_hold_right_target_rad_json": json.dumps(
                arm_hold_targets["right"][0].detach().cpu().tolist()
            ),
            "normalization_external_joint_hold_left_start_actual_rad_json": json.dumps(
                arm_hold_start_actual["left"][0].detach().cpu().tolist()
            ),
            "normalization_external_joint_hold_right_start_actual_rad_json": json.dumps(
                arm_hold_start_actual["right"][0].detach().cpu().tolist()
            ),
            "normalization_first10_arm_joint_max_delta_rad": (
                first10_arm_joint_max_delta_rad
            ),
            "normalization_first10_root_xy_max_delta_m": (
                first10_root_xy_max_delta_m
            ),
        }

    try:
        for side, term_name in (
            ("left", "arm_l_action"),
            ("right", "arm_r_action"),
        ):
            term = env.action_manager.get_term(term_name)
            begin_hold = getattr(term, "begin_external_joint_position_hold", None)
            if begin_hold is None:
                raise TypeError(
                    f"{term_name} does not support external joint-position hold"
                )
            target, start_actual = begin_hold()
            arm_hold_terms.append(term)
            expected_shape = (env.num_envs, 7)
            if target.shape != expected_shape or start_actual.shape != expected_shape:
                raise RuntimeError(
                    f"{term_name} hold target/actual must be {expected_shape}, "
                    f"got {tuple(target.shape)}/{tuple(start_actual.shape)}"
                )
            arm_hold_targets[side] = target
            arm_hold_start_actual[side] = start_actual

        if task_id == "000001":
            lift_id = robot.joint_names.index("lift_joint")
            initial_lift = float(robot.data.joint_pos[0, lift_id].item())
            stable = 0
            for _ in range(120):
                measured = float(robot.data.joint_pos[0, lift_id].item())
                target = 0.0
                # Preserve the EEF command stream even though joint hold owns the arms.
                lift_delta = target - initial_lift
                poses = {
                    side: pose.clone() for side, pose in initial_eef_pose.items()
                }
                for pose in poses.values():
                    pose[2] += lift_delta
                step_and_measure(
                    build_action(
                        env,
                        source,
                        0,
                        poses["left"],
                        poses["right"],
                        lift_target=target,
                    ),
                )
                stable = stable + 1 if abs(measured) <= 0.004 else 0
                if (
                    stable >= NORMALIZE_STABLE_STEPS
                    and normalization_steps >= INITIAL_HOLD_AUDIT_STEPS
                ):
                    return metrics()
            raise RuntimeError("lift did not normalize to 0 m")

        home = env._coffee_transport_episode_home_root_pose_w[0]
        stable = 0
        for _ in range(240):
            root = current_root_pose(env)
            world_error = home[:2] - root[:2]
            yaw_error = quat_yaw_wxyz(home[3:]) - quat_yaw_wxyz(root[3:])
            yaw_error = math.atan2(math.sin(yaw_error), math.cos(yaw_error))
            cos_yaw = math.cos(quat_yaw_wxyz(root[3:]))
            sin_yaw = math.sin(quat_yaw_wxyz(root[3:]))
            body = torch.tensor(
                [
                    cos_yaw * world_error[0] + sin_yaw * world_error[1],
                    -sin_yaw * world_error[0] + cos_yaw * world_error[1],
                ],
                dtype=torch.float32,
                device=env.device,
            )
            speed = float(torch.linalg.vector_norm(body).item())
            if speed > BASE_SPEED_MAX_MPS:
                body *= BASE_SPEED_MAX_MPS / speed
            velocity = torch.cat(
                (
                    body,
                    torch.tensor(
                        [
                            max(
                                -BASE_YAW_SPEED_MAX_RADPS,
                                min(BASE_YAW_SPEED_MAX_RADPS, yaw_error),
                            )
                        ],
                        device=env.device,
                    ),
                )
            )
            step_and_measure(
                build_action(
                    env,
                    source,
                    0,
                    initial_eef_pose["left"],
                    initial_eef_pose["right"],
                    base_velocity=velocity,
                ),
            )
            good = (
                float(torch.linalg.vector_norm(world_error).item())
                <= BASE_ENDPOINT_XY_TOLERANCE_M
                and abs(yaw_error) <= BASE_ENDPOINT_YAW_TOLERANCE_RAD
            )
            stable = stable + 1 if good else 0
            if (
                stable >= NORMALIZE_STABLE_STEPS
                and normalization_steps >= INITIAL_HOLD_AUDIT_STEPS
            ):
                return metrics()
        raise RuntimeError(
            "mobile base did not normalize to episode HOME: "
            f"root_xy={current_root_pose(env)[:2].tolist()} "
            f"home_xy={home[:2].tolist()}"
        )
    finally:
        for term in reversed(arm_hold_terms):
            term.end_external_joint_position_hold()

def infer_axis_end(source: SourceEpisode, cycle: int, task_id: str) -> int:
    home = int(source.attrs[f"cycle_{cycle}_home_stable_step"])
    release = int(source.attrs[f"cycle_{cycle}_release_step"])
    if task_id == "000001":
        values = source.actions[:, LIFT_INDEX]
        changed = np.flatnonzero(np.abs(np.diff(values)) > 1.0e-5) + 1
        candidates = changed[(changed > home) & (changed < release)]
    else:
        speed = np.linalg.norm(source.actions[:, 19:22], axis=1)
        candidates = np.flatnonzero((speed > 1.0e-4) & (np.arange(len(speed)) > home) & (np.arange(len(speed)) < release))
    if len(candidates) == 0:
        raise ValueError(f"cycle {cycle}: cannot infer destination-axis boundary")
    return int(candidates[-1])


def drive_base_to(
    env,
    source: SourceEpisode,
    step: int,
    target_pose: torch.Tensor,
    *,
    passive_side: str | None = None,
    passive_pose: torch.Tensor | None = None,
    gripper_override: dict[str, float] | None = None,
) -> None:
    stable = 0
    for _ in range(300):
        root = current_root_pose(env)
        world_error = target_pose[:2] - root[:2]
        current_yaw = quat_yaw_wxyz(root[3:])
        target_yaw = quat_yaw_wxyz(target_pose[3:])
        yaw_error = math.atan2(math.sin(target_yaw - current_yaw), math.cos(target_yaw - current_yaw))
        cos_yaw, sin_yaw = math.cos(current_yaw), math.sin(current_yaw)
        body = torch.stack(
            (
                cos_yaw * world_error[0] + sin_yaw * world_error[1],
                -sin_yaw * world_error[0] + cos_yaw * world_error[1],
            )
        )
        speed = float(torch.linalg.vector_norm(body).item())
        if speed > BASE_SPEED_MAX_MPS:
            body *= BASE_SPEED_MAX_MPS / speed
        velocity = torch.cat(
            (
                body,
                torch.tensor(
                    [max(-BASE_YAW_SPEED_MAX_RADPS, min(BASE_YAW_SPEED_MAX_RADPS, yaw_error))],
                    device=env.device,
                ),
            )
        )
        poses = {
            side: source_pose_root(source, side, step, env.device)
            for side in ("left", "right")
        }
        if passive_side is not None and passive_pose is not None:
            poses[passive_side] = passive_pose
        step_checked(
            env,
            build_action(
                env,
                source,
                step,
                poses["left"],
                poses["right"],
                base_velocity=velocity,
                gripper_override=gripper_override,
            ),
        )
        good = float(torch.linalg.vector_norm(world_error).item()) <= BASE_ENDPOINT_XY_TOLERANCE_M and abs(yaw_error) <= BASE_ENDPOINT_YAW_TOLERANCE_RAD
        stable = stable + 1 if good else 0
        if stable >= AXIS_STABLE_STEPS:
            return
    raise RuntimeError("mobile base failed to reach the requested route endpoint")


def execute_seed_range(
    env,
    source: SourceEpisode,
    start: int,
    end: int,
    *,
    active_side: str | None = None,
    translation_world: torch.Tensor | None = None,
    blend_to_root: tuple[int, int] | None = None,
    passive_side: str | None = None,
    passive_pose: torch.Tensor | None = None,
    gripper_override: dict[str, float] | None = None,
) -> None:
    for step in range(start, end + 1):
        poses = {
            side: source_pose_root(source, side, step, env.device)
            for side in ("left", "right")
        }
        if passive_side is not None and passive_pose is not None:
            poses[passive_side] = passive_pose
        if active_side is not None and translation_world is not None:
            transformed = translated_source_eef_root(
                env, source, active_side, step, translation_world
            )
            if blend_to_root is None:
                poses[active_side] = transformed
            else:
                blend_start, blend_end = blend_to_root
                alpha = (step - blend_start) / max(1, blend_end - blend_start)
                poses[active_side] = pose_lerp(
                    transformed,
                    source_pose_root(source, active_side, step, env.device),
                    alpha,
                )
        step_checked(
            env,
            build_action(
                env,
                source,
                step,
                poses["left"],
                poses["right"],
                gripper_override=gripper_override,
            ),
        )


def interpolate_active_pose(
    env,
    source: SourceEpisode,
    side: str,
    step: int,
    target: torch.Tensor,
    *,
    gripper_value: float,
    passive_pose: torch.Tensor,
) -> None:
    initial = current_eef_pose_root(env, side)
    passive = "left" if side == "right" else "right"
    for index in range(1, POSE_BLEND_STEPS + 1):
        poses = {
            passive: passive_pose,
            side: pose_lerp(initial, target, index / POSE_BLEND_STEPS),
        }
        step_checked(
            env,
            build_action(
                env,
                source,
                step,
                poses["left"],
                poses["right"],
                gripper_override={side: gripper_value},
            ),
        )


def _orientation_error_rad(first: torch.Tensor, second: torch.Tensor) -> float:
    return float(
        math_utils.quat_error_magnitude(first.unsqueeze(0), second.unsqueeze(0))[0].item()
    )


def execute_replayed_grasp(
    env,
    source: SourceEpisode,
    *,
    can_name: str,
    side: str,
    start_step: int,
    close: CloseSegment,
    translation_world: torch.Tensor,
    correction_world: torch.Tensor,
    passive_side: str,
    passive_pose: torch.Tensor,
) -> dict[str, float]:
    metrics = {
        "close_onset_step": float(close.onset_step),
        "close_frame_count": float(close.grasp_step - close.onset_step + 1),
        "open_gripper_value": close.open_value,
        "closed_gripper_value": close.closed_value,
    }

    # Preserve synchronized v1 motion while gradually compensating the measured
    # systematic TCP tracking offset before and during the recorded close ramp.
    correction_start = max(
        start_step, close.onset_step - args_cli.grasp_correction_blend_steps
    )
    for step in range(start_step, close.grasp_step + 1):
        alpha = (step - correction_start) / max(
            1, close.grasp_step - correction_start
        )
        alpha = min(max(alpha, 0.0), 1.0)
        execute_seed_range(
            env,
            source,
            step,
            step,
            active_side=side,
            translation_world=translation_world + alpha * correction_world,
            passive_side=passive_side,
            passive_pose=passive_pose,
        )

    # These measurements describe the realized grasp but never delay or reject it.
    # Both relative translations are EEF poses expressed in the can-local frame.
    source_can = source_can_pose_at_pre_step(
        source, can_name, close.grasp_step, env.device
    )
    source_eef = source_eef_world(source, side, close.grasp_step, env.device)
    source_relative_position, source_relative_quaternion = (
        math_utils.subtract_frame_transforms(
            source_can[:3], source_can[3:], source_eef[:3], source_eef[3:]
        )
    )
    can = env.scene[can_name]
    measured_eef = current_eef_pose_world(env, side)
    measured_relative_position, measured_relative_quaternion = (
        math_utils.subtract_frame_transforms(
            can.data.root_pos_w[0],
            can.data.root_quat_w[0],
            measured_eef[:3],
            measured_eef[3:],
        )
    )
    translation_error = measured_relative_position - source_relative_position
    metrics.update(
        {
            "diagnostic_eef_can_distance_m": float(
                torch.linalg.vector_norm(
                    measured_eef[:3] - can.data.root_pos_w[0]
                ).item()
            ),
            "diagnostic_can_local_eef_x_m": float(measured_relative_position[0].item()),
            "diagnostic_can_local_eef_y_m": float(measured_relative_position[1].item()),
            "diagnostic_can_local_eef_z_m": float(measured_relative_position[2].item()),
            "diagnostic_source_can_local_eef_x_m": float(source_relative_position[0].item()),
            "diagnostic_source_can_local_eef_y_m": float(source_relative_position[1].item()),
            "diagnostic_source_can_local_eef_z_m": float(source_relative_position[2].item()),
            "diagnostic_can_local_error_x_m": float(translation_error[0].item()),
            "diagnostic_can_local_error_y_m": float(translation_error[1].item()),
            "diagnostic_can_local_error_z_m": float(translation_error[2].item()),
            "diagnostic_orientation_error_rad": _orientation_error_rad(
                measured_relative_quaternion, source_relative_quaternion
            ),
        }
    )
    return metrics


def carry_quality(
    env,
    source: SourceEpisode,
    cycle: int,
    can_name: str,
    side: str,
    initial_pose: torch.Tensor,
    *,
    passive_side: str,
    passive_pose: torch.Tensor,
    closed_value: float,
) -> tuple[bool, dict[str, float], str]:
    home_step = int(source.attrs[f"cycle_{cycle}_home_stable_step"])
    robot = env.scene["robot"]
    joint_ids = torch.tensor(
        [robot.joint_names.index(name) for name in FFW_SG2_PUBLISHED_JOINT_NAMES],
        device=env.device,
    )
    source_can = source_can_pose_at_pre_step(source, can_name, home_step, env.device)
    source_eef = source_eef_world(source, side, home_step, env.device)
    source_distance = float(torch.linalg.vector_norm(source_eef[:3] - source_can[:3]).item())
    stable = 0
    last = None
    for _ in range(60):
        poses = {
            active_side: source_pose_root(source, active_side, home_step, env.device)
            for active_side in ("left", "right")
        }
        poses[passive_side] = passive_pose
        step_checked(
            env,
            build_action(
                env,
                source,
                home_step,
                poses["left"],
                poses["right"],
                gripper_override={side: closed_value},
            ),
        )
        can = env.scene[can_name]
        position = can.data.root_pos_w[0]
        velocity = can.data.root_vel_w[0]
        eef = current_eef_pose_world(env, side)
        measured = robot.data.joint_pos[0, joint_ids]
        expected = torch.as_tensor(source.joint_pos[home_step], device=env.device)
        posture = torch.cat((measured[:7], measured[8:15], measured[16:19]))
        expected_posture = torch.cat((expected[:7], expected[8:15], expected[16:19]))
        metrics = {
            "eef_object_distance_m": float(torch.linalg.vector_norm(eef[:3] - position).item()),
            "source_eef_object_distance_m": source_distance,
            "tcp_z_relative_to_can_m": float((eef[2] - position[2]).item()),
            "can_displacement_m": float(torch.linalg.vector_norm(position - initial_pose[:3]).item()),
            "home_joint_max_error": float(torch.max(torch.abs(posture - expected_posture)).item()),
            "can_linear_speed_mps": float(torch.linalg.vector_norm(velocity[:3]).item()),
            "can_angular_speed_radps": float(torch.linalg.vector_norm(velocity[3:]).item()),
        }
        decision = evaluate_body_carry(**metrics, require_stability=False)
        stable = stable + 1 if decision.passed else 0
        last = (decision, metrics)
        if stable >= DEFAULT_BODY_CARRY_QUALITY.stable_steps:
            return True, metrics, ""
    assert last is not None
    return False, last[1], ",".join(last[0].failures)


def placement_quality(env, source: SourceEpisode, cycle: int, can_name: str, side: str, task_id: str) -> tuple[bool, dict[str, float], str]:
    placed_step = int(source.attrs[f"cycle_{cycle}_placed_step"])
    square = next(item for item in SOURCE_SPAWN_SQUARES if item.name == can_name)
    dy = 0.0 if task_id == "000001" else CABINET_RIGHT_DELTA_Y_M
    expected_z = DESTINATION_LOWER_SUPPORT_Z_M + 0.045 if task_id == "000001" else SOURCE_UPPER_SUPPORT_Z_M + 0.045
    stable = 0
    last = None
    for _ in range(DEFAULT_PLACEMENT_QUALITY.stable_steps + 12):
        step_checked(env, hold_current_action(env, source, placed_step))
        can = env.scene[can_name]
        position = can.data.root_pos_w[0]
        quaternion = can.data.root_quat_w[0]
        velocity = can.data.root_vel_w[0]
        eef = current_eef_pose_world(env, side)
        metrics = {
            "center_x_m": float(position[0].item()),
            "center_y_m": float(position[1].item()),
            "origin_z_m": float(position[2].item()),
            "expected_origin_z_m": expected_z,
            "upright_cosine": upright_cosine_from_wxyz(tuple(float(v) for v in quaternion)),
            "can_linear_speed_mps": float(torch.linalg.vector_norm(velocity[:3]).item()),
            "can_angular_speed_radps": float(torch.linalg.vector_norm(velocity[3:]).item()),
            "active_eef_distance_m": float(torch.linalg.vector_norm(eef[:3] - position).item()),
            "center_region_xyxy_m": (square.x_min_m, square.x_max_m, square.y_min_m + dy, square.y_max_m + dy),
        }
        decision = evaluate_placement(**metrics)
        stable = stable + 1 if decision.passed else 0
        last = (decision, metrics)
        if stable >= DEFAULT_PLACEMENT_QUALITY.stable_steps:
            metrics.pop("center_region_xyxy_m")
            return True, metrics, ""
    assert last is not None
    last[1].pop("center_region_xyxy_m")
    return False, last[1], ",".join(last[0].failures)


def execute_cycle(env, source: SourceEpisode, cycle: int, task_id: str, initial_pose: torch.Tensor) -> tuple[bool, dict[str, float], str]:
    can_name, side = ACTIVE[cycle]
    passive_side = "left" if side == "right" else "right"
    start = int(source.attrs[f"cycle_{cycle}_operator_enabled_step"])
    grasp = int(source.attrs[f"cycle_{cycle}_grasp_step"])
    g_step = int(source.attrs[f"cycle_{cycle}_g_step"])
    home = int(source.attrs[f"cycle_{cycle}_home_stable_step"])
    placed = int(source.attrs[f"cycle_{cycle}_placed_step"])
    complete = int(source.attrs[f"cycle_{cycle}_complete_step"])
    axis_end = infer_axis_end(source, cycle, task_id)
    close = infer_close_segment(
        source.actions[:, GRIPPER_INDEX[side]],
        operator_start_step=start,
        grasp_step=grasp,
        g_step=g_step,
    )
    passive_pose = source_pose_root(source, passive_side, start, env.device)

    sampled_initial = initial_pose
    source_initial = torch.as_tensor(source.initial_can_pose[can_name], device=env.device)
    source_delta = sampled_initial[:3] - source_initial[:3]
    correction_local = torch.tensor(
        [
            args_cli.grasp_correction_can_x_m,
            0.0,
            args_cli.grasp_correction_can_z_m,
        ],
        dtype=torch.float32,
        device=env.device,
    )
    correction_world = math_utils.quat_apply(sampled_initial[3:], correction_local)
    # The recorded EEF world path already contains the lift/base destination
    # displacement. Only the randomized source-position residual is retargeted.
    destination_delta = source_delta.clone()

    grasp_metrics = execute_replayed_grasp(
        env,
        source,
        can_name=can_name,
        side=side,
        start_step=start,
        close=close,
        translation_world=source_delta,
        correction_world=correction_world,
        passive_side=passive_side,
        passive_pose=passive_pose,
    )
    closed_override = {side: close.closed_value}
    for step in range(grasp + 1, g_step + 1):
        correction_weight = 1.0 - (step - grasp) / max(1, g_step - grasp)
        execute_seed_range(
            env,
            source,
            step,
            step,
            active_side=side,
            translation_world=source_delta + correction_weight * correction_world,
            passive_side=passive_side,
            passive_pose=passive_pose,
            gripper_override=closed_override,
        )
    execute_seed_range(
        env,
        source,
        g_step + 1,
        home,
        active_side=side,
        translation_world=source_delta,
        blend_to_root=(g_step + 1, home),
        passive_side=passive_side,
        passive_pose=passive_pose,
        gripper_override=closed_override,
    )
    carry_ok, carry_metrics, carry_reason = carry_quality(
        env,
        source,
        cycle,
        can_name,
        side,
        sampled_initial,
        passive_side=passive_side,
        passive_pose=passive_pose,
        closed_value=close.closed_value,
    )
    metrics = {
        **{f"grasp_{key}": value for key, value in grasp_metrics.items()},
        **{f"carry_{key}": value for key, value in carry_metrics.items()},
    }
    if not carry_ok:
        return False, metrics, f"{can_name}:carry:{carry_reason}"

    if task_id == "000001":
        execute_seed_range(
            env,
            source,
            home + 1,
            axis_end,
            gripper_override=closed_override,
        )
    else:
        destination_root = env._coffee_transport_episode_home_root_pose_w[0].clone()
        destination_root[1] += CABINET_RIGHT_DELTA_Y_M
        drive_base_to(
            env,
            source,
            home,
            destination_root,
            passive_side=passive_side,
            passive_pose=passive_pose,
            gripper_override=closed_override,
        )

    first_place_target = translated_source_eef_root(
        env, source, side, axis_end + 1, destination_delta
    )
    interpolate_active_pose(
        env,
        source,
        side,
        axis_end + 1,
        first_place_target,
        gripper_value=close.closed_value,
        passive_pose=source_pose_root(source, passive_side, axis_end + 1, env.device),
    )
    execute_seed_range(
        env,
        source,
        axis_end + 1,
        placed,
        active_side=side,
        translation_world=destination_delta,
    )
    placement_ok, placement_metrics, placement_reason = placement_quality(
        env, source, cycle, can_name, side, task_id
    )
    metrics.update(
        {f"placement_{key}": value for key, value in placement_metrics.items()}
    )
    if not placement_ok:
        return False, metrics, f"{can_name}:placement:{placement_reason}"

    root_target = source_pose_root(source, side, placed, env.device)
    open_value = float(source.actions[placed, GRIPPER_INDEX[side]])
    interpolate_active_pose(
        env,
        source,
        side,
        placed,
        root_target,
        gripper_value=open_value,
        passive_pose=source_pose_root(source, passive_side, placed, env.device),
    )
    if task_id == "000001":
        execute_seed_range(env, source, placed + 1, complete)
    else:
        drive_base_to(
            env,
            source,
            placed,
            env._coffee_transport_episode_home_root_pose_w[0],
        )
        execute_seed_range(env, source, complete - 5, complete)
    return True, metrics, ""


def initial_metrics(env, task_id: str) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    root = current_root_pose(env).detach().clone()
    home = env._coffee_transport_episode_home_root_pose_w[0].detach().clone()
    values = {
        "robot_spawn_x_m": float(root[0].item()),
        "robot_spawn_y_m": float(root[1].item()),
        "robot_spawn_yaw_rad": quat_yaw_wxyz(root[3:]),
        "robot_home_x_m": float(home[0].item()),
        "robot_home_y_m": float(home[1].item()),
        "robot_home_yaw_rad": quat_yaw_wxyz(home[3:]),
    }
    poses = {}
    for name in COFFEE_CAN_NAMES:
        pose = torch.cat((env.scene[name].data.root_pos_w[0], env.scene[name].data.root_quat_w[0])).detach().clone()
        poses[name] = pose
        values[f"{name}_x_m"] = float(pose[0].item())
        values[f"{name}_y_m"] = float(pose[1].item())
        values[f"{name}_yaw_rad"] = quat_yaw_wxyz(pose[3:])
    robot = env.scene["robot"]
    lift_id = robot.joint_names.index("lift_joint")
    values["lift_spawn_m"] = float(robot.data.joint_pos[0, lift_id].item())
    values["base_route_spawn_offset_m"] = (
        float(root[1].item() - home[1].item()) if task_id == "000002" else 0.0
    )
    return values, poses


def set_episode_result(env, source: SourceEpisode, success: bool, failure_reason: str, metrics: AttemptMetrics, task_id: str, attempt: int, episode_seed: int) -> None:
    episode = env.recorder_manager.get_episode(0)
    episode.success = bool(success)
    metadata = dict(getattr(episode, "metadata", {}) or {})
    metadata.update(
        {
            "success": bool(success),
            "failure_reason": "" if success else failure_reason,
            "task_id": task_id,
            "task_env_name": f"Cyclo-Real-Showroom-Task{task_id}-FFW-SG2-v0",
            "source_demo_id": "demo_0",
            "source_seed_mode": source.attrs["coffee_seed_mode"],
            "generation_seed": episode_seed,
            "generation_attempt": attempt,
            "generation_method": GENERATION_METHODS[task_id],
            "success_criterion_id": SUCCESS_CRITERION_ID,
            "initial_normalization_contract_id": INITIAL_NORMALIZATION_CONTRACT_IDS[task_id],
            "generation_profile_id": GENERATION_PROFILE_IDS[task_id],
            "production_approved": PRODUCTION_APPROVED_BY_TASK[task_id],
            "task_axis_action_contract": TASK_AXIS_ACTION_CONTRACTS[task_id],
            "source_hdf5": source.path,
            "source_hdf5_sha256": source.sha256,
            "grasp_correction_can_x_m": args_cli.grasp_correction_can_x_m,
            "grasp_correction_can_z_m": args_cli.grasp_correction_can_z_m,
            "grasp_correction_blend_steps": args_cli.grasp_correction_blend_steps,
            "coffee_phase_contract_version": int(source.attrs["coffee_phase_contract_version"]),
        }
    )
    metadata.update(PRODUCTION_PROVENANCE_BY_TASK[task_id])
    metadata.update(metrics.initial)
    for can_name, cycle_metrics in metrics.cycles.items():
        metadata.update({f"quality_{can_name}_{key}": value for key, value in cycle_metrics.items()})
    episode.metadata = metadata


def configure_dataset_metadata(env, task_id: str, source: SourceEpisode) -> None:
    handler = getattr(env.recorder_manager, "_dataset_file_handler", None)
    data = getattr(handler, "_hdf5_data_group", None)
    if data is None:
        return
    eef_names = [
        "left_eef_x", "left_eef_y", "left_eef_z", "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz", "gripper_l_joint1",
        "right_eef_x", "right_eef_y", "right_eef_z", "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz", "gripper_r_joint1",
        "head_joint1", "head_joint2", "lift_joint",
    ]
    if task_id == "000002":
        eef_names.extend(("linear_x", "linear_y", "angular_z"))
    metadata = {
        "schema_version": "cyclo_lab_hdf5_v1",
        "task_id": task_id,
        "task_env_name": f"Cyclo-Real-Showroom-Task{task_id}-FFW-SG2-v0",
        "dataset_origin": "coffee_transport_physical_trajectory_generation",
        "source_hdf5": source.path,
        "source_hdf5_sha256": source.sha256,
        "action_names": json.dumps(eef_names),
        "action_semantics": "pre_step_dual_eef_pose_wxyz_grippers_head_lift_plus_optional_body_velocity",
        "observation_semantics": "pre_step",
        "scene_state_semantics": "post_step",
        "control_hz": 15.0,
        "camera_hz": 15.0,
        "generation_method": GENERATION_METHODS[task_id],
        "success_criterion_id": SUCCESS_CRITERION_ID,
        "initial_normalization_contract_id": INITIAL_NORMALIZATION_CONTRACT_IDS[task_id],
        "articulation_joint_names": json.dumps(env.scene["robot"].joint_names),
        "preroll_recording_policy": "excluded_then_record_post_reset_after_realized_second_settle",
        "generation_profile_id": GENERATION_PROFILE_IDS[task_id],
        "production_approved": PRODUCTION_APPROVED_BY_TASK[task_id],
        "task_axis_action_contract": TASK_AXIS_ACTION_CONTRACTS[task_id],
    }
    metadata.update(PRODUCTION_PROVENANCE_BY_TASK[task_id])
    for key, value in metadata.items():
        data.attrs[key] = value


def main() -> None:
    if os.path.exists(args_cli.output_file):
        raise FileExistsError(f"refusing existing output: {args_cli.output_file}")
    source = load_source(args_cli.input_file, args_cli.task_id)
    env_name = (
        f"Cyclo-Real-Showroom-Task{args_cli.task_id}-Trajectory-Generation-FFW-SG2-v0"
    )
    env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.rerender_on_reset = True
    env_cfg.recorders.dataset_export_dir_path = os.path.dirname(os.path.abspath(args_cli.output_file))
    env_cfg.recorders.dataset_filename = os.path.basename(args_cli.output_file)
    env_cfg.recorders.dataset_export_mode = (
        DatasetExportMode.EXPORT_ALL
        if args_cli.keep_failures
        else DatasetExportMode.EXPORT_SUCCEEDED_ONLY
    )
    env = gym.make(env_name, cfg=env_cfg).unwrapped
    configure_dataset_metadata(env, args_cli.task_id, source)

    successes = 0
    attempt = 0
    while successes < args_cli.num_successes and attempt < args_cli.max_attempts:
        attempt += 1
        episode_seed = args_cli.seed + attempt - 1
        torch.manual_seed(episode_seed)
        env.reset(seed=episode_seed)
        requested_snapshot = capture_initial_snapshot(env)
        preroll_metrics, hold_eef_pose = settle_episode_before_recording(
            env, source, args_cli.task_id
        )
        try:
            realized_initial_metrics = settle_restored_episode_as_realized_initial(
                env, source, args_cli.task_id, hold_eef_pose, requested_snapshot
            )
        except Exception as error:
            print(
                f"[Coffee Generation] REJECT pre-record task={args_cli.task_id} "
                f"attempt={attempt} seed={episode_seed}: {error}"
            )
            continue
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
        env.recorder_manager.reset(env_ids)
        env.recorder_manager.record_post_reset(env_ids)
        refresh_camera_observations(env)
        initial, initial_poses = initial_metrics(env, args_cli.task_id)
        initial.update(preroll_metrics)
        initial.update(realized_initial_metrics)
        attempt_metrics = AttemptMetrics(initial=initial, cycles={})
        success = True
        reason = ""
        try:
            normalization_metrics = normalize_episode_start(
                env, source, args_cli.task_id, hold_eef_pose
            )
            attempt_metrics.initial.update(normalization_metrics)
            validate_initial_hold(args_cli.task_id, normalization_metrics)
            for cycle, (can_name, _) in enumerate(ACTIVE):
                cycle_ok, cycle_metrics, cycle_reason = execute_cycle(
                    env,
                    source,
                    cycle,
                    args_cli.task_id,
                    initial_poses[can_name],
                )
                attempt_metrics.cycles[can_name] = cycle_metrics
                if not cycle_ok:
                    success = False
                    reason = cycle_reason
                    break
        except Exception as error:
            success = False
            reason = f"{type(error).__name__}:{error}"
        set_episode_result(
            env,
            source,
            success,
            reason,
            attempt_metrics,
            args_cli.task_id,
            attempt,
            episode_seed,
        )
        env.recorder_manager.export_episodes([0])
        if success:
            successes += 1
            print(
                f"[Coffee Generation] PASS task={args_cli.task_id} "
                f"success={successes}/{args_cli.num_successes} attempt={attempt} seed={episode_seed}",
                flush=True,
            )
        else:
            print(
                f"[Coffee Generation] REJECT task={args_cli.task_id} "
                f"attempt={attempt} seed={episode_seed} reason={reason} "
                f"initial={attempt_metrics.initial} metrics={attempt_metrics.cycles}",
                flush=True,
            )

    env.close()
    if successes < args_cli.num_successes:
        raise RuntimeError(
            f"generated only {successes}/{args_cli.num_successes} successes in {attempt} attempts"
        )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback

        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
