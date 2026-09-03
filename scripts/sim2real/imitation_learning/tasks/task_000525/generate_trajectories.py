# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Generate Task525 trajectories with a local locomanipulation state machine."""

"""Launch Isaac Sim Simulator first."""


import argparse
import os

from isaaclab.app import AppLauncher

# Launch Isaac Lab
parser = argparse.ArgumentParser(description="Locomanipulation SDG")
parser.add_argument("--task", type=str, help="The Isaac Lab locomanipulation SDG task to load for data generation.")
parser.add_argument("--dataset", type=str, help="The static manipulation dataset recorded via teleoperation.")
parser.add_argument("--output_file", type=str, help="The file name for the generated output dataset.")
parser.add_argument(
    "--lift_step",
    type=int,
    help=(
        "The step index in the input recording where the robot is ready to lift the object.  Aka, where the grasp is"
        " finished."
    ),
)
parser.add_argument(
    "--navigate_step",
    type=int,
    help=(
        "The step index in the input recording where the robot is ready to navigate.  Aka, where it has finished"
        " lifting the object"
    ),
)
parser.add_argument(
    "--place_step",
    type=int,
    default=None,
    help=(
        "First place frame in a continuous pick/navigation/place recording. "
        "When set, recorded navigation frames are skipped and drop-off is "
        "retargeted from the recorded base frame."
    ),
)
parser.add_argument(
    "--active_side",
    choices=("episode", "left", "right"),
    default="episode",
    help=(
        "Manipulating hand. The production default reads the side from each "
        "Task525 seed episode; left/right are strict debugging overrides."
    ),
)
parser.add_argument("--demo", type=str, default=None, help="The demo in the input dataset to use.")
parser.add_argument("--num_runs", type=int, default=1, help="The number of trajectories to generate.")
parser.add_argument(
    "--seed",
    type=int,
    default=20260901,
    help="Base seed for reproducible source-demo selection and per-attempt reset randomization.",
)
parser.add_argument(
    "--successful_runs_only",
    action="store_true",
    help="Count and export only trajectories that pass the environment generation quality gate.",
)
parser.add_argument(
    "--max_attempts",
    type=int,
    default=0,
    help="Maximum attempts for success-only generation; 0 uses four times num_runs.",
)
parser.add_argument(
    "--draw_visualization", type=bool, default=False, help="Draw the occupancy map and path planning visualization."
)
parser.add_argument(
    "--angular_gain",
    type=float,
    default=2.0,
    help=(
        "The angular gain to use for determining an angular control velocity when driving the robot during navigation."
    ),
)
parser.add_argument(
    "--linear_gain",
    type=float,
    default=1.0,
    help="The linear gain to use for determining the linear control velocity when driving the robot during navigation.",
)
parser.add_argument(
    "--linear_max", type=float, default=1.0, help="The maximum linear control velocity allowable during navigation."
)
parser.add_argument(
    "--angular_max", type=float, default=1.0, help="The maximum angular control velocity allowable during navigation."
)
parser.add_argument(
    "--navigation_mode",
    choices=("path_heading", "fixed_yaw_holonomic"),
    default="path_heading",
    help=(
        "Use upstream path-heading control or keep the chassis at a fixed yaw "
        "and command body-frame vx/vy along the planned path."
    ),
)
parser.add_argument(
    "--navigation_yaw",
    type=float,
    default=0.0,
    help="World yaw held by fixed_yaw_holonomic navigation.",
)
parser.add_argument(
    "--initial_yaw_counterclockwise",
    action="store_true",
    help=(
        "Force the initial fixed-yaw alignment to use positive angular-z "
        "(counterclockwise), then use shortest-angle corrections after alignment."
    ),
)
parser.add_argument(
    "--distance_threshold",
    type=float,
    default=0.2,
    help="The distance threshold in meters to perform state transitions between navigation and manipulation tasks.",
)
parser.add_argument(
    "--following_offset",
    type=float,
    default=0.6,
    help=(
        "The target point offset distance used for local path following during navigation.  A larger value will result"
        " in smoother trajectories, but may cut path corners."
    ),
)
parser.add_argument(
    "--angle_threshold",
    type=float,
    default=0.2,
    help=(
        "The angle threshold in radians to determine when the robot can move forward or transition between navigation"
        " and manipulation tasks."
    ),
)
parser.add_argument(
    "--approach_distance",
    type=float,
    default=0.5,
    help="An offset distance added to the destination to allow a buffer zone for reliably approaching the goal.",
)
parser.add_argument(
    "--randomize_placement",
    type=bool,
    default=True,
    help="Whether or not to randomize the placement of fixtures in the scene upon environment initialization.",
)
parser.add_argument(
    "--enable_pinocchio",
    action="store_true",
    default=False,
    help="Enable Pinocchio.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.enable_pinocchio:
    # Import pinocchio before AppLauncher to force the use of the version installed by IsaacLab and not the one installed by Isaac Sim
    # pinocchio is required by the Pink IK controllers and the GR1T2 retargeter
    import pinocchio  # noqa: F401

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import enum
import gymnasium as gym
import json
import random
import time
import torch

import omni.kit
import omni.kit.app

# ``path_utils`` imports Isaac Sim's Dijkstra implementation.  The minimal
# headless experience does not enable this optional extension automatically.
_extension_manager = omni.kit.app.get_app().get_extension_manager()
if not _extension_manager.is_extension_enabled("isaacsim.replicator.mobility_gen"):
    _extension_manager.set_extension_enabled_immediate("isaacsim.replicator.mobility_gen", True)

from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab.managers.recorder_manager import DatasetExportMode

import isaaclab_mimic.locomanipulation_sdg.envs  # noqa: F401
from isaaclab_mimic.locomanipulation_sdg.data_classes import LocomanipulationSDGOutputData
from isaaclab_mimic.locomanipulation_sdg.envs.locomanipulation_sdg_env import LocomanipulationSDGEnv
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import (
    OccupancyMap,
    merge_occupancy_maps,
    occupancy_map_add_to_stage,
)
from isaaclab_mimic.locomanipulation_sdg.path_utils import ParameterizedPath, plan_path
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.arrangement import (
    CoffeeArrangement,
    TASK000525_REGION_KEYS,
    TASK000525_TARGET_OBJECT,
    manipulation_side_for_region,
    validate_region_key,
)
from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose, place_randomly
from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul, transform_relative_pose

from isaaclab_tasks.utils import parse_env_cfg


class LocomanipulationSDGDataGenerationState(enum.IntEnum):
    """States for the locomanipulation SDG data generation state machine."""

    GRASP_OBJECT = 0
    """Robot grasps object at start position"""

    LIFT_OBJECT = 1
    """Robot lifts object while stationary"""

    NAVIGATE = 2
    """Robot navigates to approach position with object"""

    APPROACH = 3
    """Robot approaches final goal position"""

    DROP_OFF_OBJECT = 4
    """Robot places object at end position"""

    DONE = 5
    """Task completed"""


TASK000525_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M = 0.005
"""Maximum stationary-base root motion allowed before navigation starts."""


@configclass
class LocomanipulationSDGControlConfig:
    """Configuration for navigation control parameters."""

    angular_gain: float = 2.0
    """Proportional gain for angular velocity control"""

    linear_gain: float = 1.0
    """Proportional gain for linear velocity control"""

    linear_max: float = 1.0
    """Maximum allowed linear velocity (m/s)"""

    angular_max: float = 1.0
    """Maximum allowed angular velocity (rad/s)"""

    navigation_mode: str = "path_heading"
    """Navigation controller: path heading or fixed-yaw holonomic"""

    navigation_yaw: float = 0.0
    """World yaw maintained by fixed-yaw holonomic navigation"""

    initial_yaw_counterclockwise: bool = False
    """Force the first fixed-yaw alignment to use positive angular-z"""

    initial_yaw_alignment_pending: bool = False
    """Runtime latch cleared after the first fixed-yaw alignment"""

    distance_threshold: float = 0.1
    """Distance threshold for state transitions (m)"""

    following_offset: float = 0.6
    """Look-ahead distance for path following (m)"""

    angle_threshold: float = 0.2
    """Angular threshold for orientation control (rad)"""

    approach_distance: float = 1.0
    """Buffer distance from final goal (m)"""


def compute_navigation_velocity(
    current_pose: torch.Tensor, target_xy: torch.Tensor, config: LocomanipulationSDGControlConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute linear and angular velocities for navigation control.

    Args:
        current_pose: Current robot pose [x, y, yaw]
        target_xy: Target position [x, y]
        config: Navigation control configuration

    Returns:
        Tuple of (linear_velocity, angular_velocity)
    """
    current_xy = current_pose[:2]
    current_yaw = current_pose[2]

    # Compute position and orientation errors
    delta_xy = target_xy - current_xy
    delta_distance = torch.sqrt(torch.sum(delta_xy**2))

    target_yaw = torch.arctan2(delta_xy[1], delta_xy[0])
    delta_yaw = target_yaw - current_yaw
    # Normalize angle to [-π, π]
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    # Compute control commands
    angular_velocity = torch.clip(
        config.angular_gain * delta_yaw, -config.angular_max, config.angular_max
    )
    linear_velocity = torch.clip(config.linear_gain * delta_distance, 0.0, config.linear_max) / (
        1 + torch.abs(angular_velocity)
    )

    return linear_velocity, angular_velocity


def compute_fixed_yaw_holonomic_velocity(
    current_pose: torch.Tensor, target_xy: torch.Tensor, config: LocomanipulationSDGControlConfig
) -> tuple[torch.Tensor, torch.Tensor]:
    """Follow a world-frame path with body vx/vy while holding chassis yaw.

    The robot rotates in place first.  It then translates holonomically, so a
    long lateral leg changes wheel steering rather than chassis orientation.
    """

    target_yaw = current_pose.new_tensor(config.navigation_yaw)
    shortest_yaw_error = target_yaw - current_pose[2]
    shortest_yaw_error = (shortest_yaw_error + torch.pi) % (2 * torch.pi) - torch.pi
    yaw_error = shortest_yaw_error
    if config.initial_yaw_counterclockwise and config.initial_yaw_alignment_pending:
        if torch.abs(shortest_yaw_error) <= config.angle_threshold:
            config.initial_yaw_alignment_pending = False
            yaw_error = shortest_yaw_error.new_zeros(())
        else:
            # Use [0, 2pi) only for the initial alignment. After reaching the
            # target-yaw tolerance, shortest-angle corrections avoid an
            # unnecessary full revolution following a small overshoot.
            yaw_error = torch.remainder(target_yaw - current_pose[2], 2 * torch.pi)
    angular_velocity = torch.clip(
        config.angular_gain * yaw_error, -config.angular_max, config.angular_max
    )
    if torch.abs(yaw_error) > config.angle_threshold:
        return current_pose.new_zeros(2), angular_velocity

    world_error = target_xy - current_pose[:2]
    cos_yaw = torch.cos(current_pose[2])
    sin_yaw = torch.sin(current_pose[2])
    body_error = torch.stack((
        cos_yaw * world_error[0] + sin_yaw * world_error[1],
        -sin_yaw * world_error[0] + cos_yaw * world_error[1],
    ))
    body_velocity = config.linear_gain * body_error
    speed = torch.linalg.vector_norm(body_velocity)
    if speed > config.linear_max:
        body_velocity = body_velocity * (config.linear_max / speed)
    return body_velocity, angular_velocity


def infer_grasp_close_step(
    input_episode_data: EpisodeData,
    active_side: str,
    search_end_step: int,
    stable_steps: int = 3,
) -> int:
    """Infer the last object-relative frame from the gripper-close plateau.

    Task525's G marker is intentionally pressed only after the can is clear of
    the cabinet, so it is too late to split grasp from pull-out. The action is
    the pre-step command (it leads the resulting observation); the first stable
    fully-closed command is therefore retained as the final grasp frame.
    """

    gripper_index = 7 if active_side == "left" else 15
    end = max(1, int(search_end_step))
    values = []
    for step in range(end + 1):
        action = input_episode_data.get_action(step)
        if action is None:
            break
        values.append(float(action[gripper_index].item()))
    if len(values) < stable_steps + 1:
        return int(search_end_step)

    values_tensor = torch.tensor(values, dtype=torch.float32)
    open_value = torch.median(values_tensor[: min(10, len(values))])
    lower_value = torch.min(values_tensor)
    upper_value = torch.max(values_tensor)
    closed_value = (
        upper_value
        if torch.abs(upper_value - open_value) >= torch.abs(lower_value - open_value)
        else lower_value
    )
    excursion = closed_value - open_value
    if torch.abs(excursion) < 0.1:
        return int(search_end_step)

    close_threshold = open_value + 0.9 * excursion
    closed = values_tensor >= close_threshold if excursion > 0 else values_tensor <= close_threshold
    for step in range(0, len(values) - stable_steps + 1):
        if bool(torch.all(closed[step : step + stable_steps]).item()):
            return step
    return int(search_end_step)


def resolve_grasp_boundary_step(
    input_episode_data: EpisodeData,
    episode_metadata: dict,
    active_side: str,
    lift_step: int,
) -> tuple[int, str]:
    """Resolve Task525's grasp/pull-out boundary without treating G as grasp."""

    grasp_step = episode_metadata.get("grasp_step")
    if grasp_step is not None:
        try:
            grasp_step = int(grasp_step)
        except (TypeError, ValueError):
            grasp_step = None
    if grasp_step is not None and 0 <= grasp_step < lift_step:
        return grasp_step, "explicit grasp_step"

    inferred = infer_grasp_close_step(input_episode_data, active_side, lift_step)
    if inferred < lift_step:
        return inferred, "inferred stable gripper-close"
    return lift_step, "recorded lift_step fallback"



def infer_dropoff_replay_step(
    input_episode_data: EpisodeData,
    navigate_step: int,
    place_step: int,
    lift_joint_index: int = 18,
    command_delta_threshold_m: float = 0.005,
) -> int:
    """Include the recorded lift descent that precedes the place marker.

    A continuous Task525 seed spends most navigation frames driving the base,
    which the generator intentionally replans instead of replaying. The final
    part of that skipped interval, however, lowers the lift by about 0.30 m.
    Starting directly at ``place_step`` turns that descent into a one-frame EEF
    jump and releases the can away from the mat.
    """

    baseline_action = input_episode_data.get_action(navigate_step)
    if baseline_action is None:
        return int(place_step)
    baseline = float(baseline_action[lift_joint_index].item())
    for step in range(navigate_step + 1, place_step + 1):
        action = input_episode_data.get_action(step)
        if action is None:
            break
        delta = abs(float(action[lift_joint_index].item()) - baseline)
        if delta >= command_delta_threshold_m:
            return max(navigate_step + 1, step - 1)
    return int(place_step)

def load_and_transform_recording_data(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    reference_pose: torch.Tensor,
    target_pose: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load recording data and transform hand targets to current reference frame.

    Args:
        env: The locomanipulation SDG environment
        input_episode_data: Input episode data from static manipulation
        recording_step: Current step in the recording
        reference_pose: Original reference pose for the hand targets
        target_pose: Current target pose to transform to

    Returns:
        Tuple of transformed (left_hand_pose, right_hand_pose)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    if recording_item is None:
        return None, None

    left_hand_pose = transform_relative_pose(recording_item.left_hand_pose_target, reference_pose, target_pose)[0]
    right_hand_pose = transform_relative_pose(recording_item.right_hand_pose_target, reference_pose, target_pose)[0]

    return left_hand_pose, right_hand_pose


def build_navigation_scene(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    approach_distance: float,
    randomize_placement: bool = True,
) -> tuple[OccupancyMap, RelativePose, RelativePose]:
    """Build the static map and goals without binding a path start pose."""

    occupancy_map = merge_occupancy_maps([
        OccupancyMap.make_empty(start=(-7, -7), end=(7, 7), resolution=0.05),
        env.get_start_fixture().get_occupancy_map(),
    ])

    if randomize_placement:
        fixtures = [env.get_end_fixture()] + env.get_obstacle_fixtures()
        for fixture in fixtures:
            place_randomly(fixture, occupancy_map.buffered_meters(1.0))
            occupancy_map = merge_occupancy_maps(
                [occupancy_map, fixture.get_occupancy_map()]
            )

    initial_state = env.load_input_data(input_episode_data, 0)
    base_goal = RelativePose(
        relative_pose=transform_mul(
            transform_inv(initial_state.fixture_pose), initial_state.base_pose
        ),
        parent=env.get_end_fixture(),
    )
    base_goal_approach = RelativePose(
        relative_pose=torch.tensor(
            [-approach_distance, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
        ),
        parent=base_goal,
    )
    return occupancy_map, base_goal, base_goal_approach


def plan_navigation_path(
    env: LocomanipulationSDGEnv,
    occupancy_map: OccupancyMap,
    base_goal_approach: RelativePose,
) -> ParameterizedPath:
    """Plan from the robot's current root and reject malformed paths."""

    points = plan_path(
        start=env.get_base(),
        end=base_goal_approach,
        occupancy_map=occupancy_map.buffered_meters(0.15),
    )
    if (
        not isinstance(points, torch.Tensor)
        or points.ndim != 2
        or points.shape[0] < 2
        or points.shape[1] != 2
        or not bool(torch.isfinite(points).all().item())
    ):
        shape = tuple(points.shape) if isinstance(points, torch.Tensor) else None
        raise RuntimeError(f"Task525 planner returned an invalid path: shape={shape}")
    return ParameterizedPath(points)


TASK000525_RECORDED_PATH_POINTS = 64


def path_points_for_recording(path_helper: ParameterizedPath) -> torch.Tensor:
    """Pad planner points to a fixed frame-aligned HDF5 tensor shape."""

    points = path_helper.points
    if points.shape[0] > TASK000525_RECORDED_PATH_POINTS:
        raise RuntimeError(
            "Task525 path has more waypoints than the recording contract: "
            f"{points.shape[0]} > {TASK000525_RECORDED_PATH_POINTS}"
        )
    padding = points[-1:].expand(
        TASK000525_RECORDED_PATH_POINTS - points.shape[0], -1
    )
    return torch.cat((points, padding), dim=0)


def navigation_path_metrics(
    prefix: str,
    path_helper: ParameterizedPath,
    elapsed_s: float,
) -> dict[str, float]:
    """Return compact planner provenance for episode quality metadata."""

    points = path_helper.points
    length = torch.linalg.vector_norm(points[1:] - points[:-1], dim=-1).sum()
    return {
        f"{prefix}_planning_elapsed_ms": 1000.0 * float(elapsed_s),
        f"{prefix}_path_waypoints": float(points.shape[0]),
        f"{prefix}_path_length_m": float(length.item()),
    }


def blend_pose(pose_a: torch.Tensor, pose_b: torch.Tensor, alpha: float) -> torch.Tensor:
    """Blend two WXYZ poses while preserving unit-quaternion interpolation."""

    weight = min(max(float(alpha), 0.0), 1.0)
    position = (1.0 - weight) * pose_a[:3] + weight * pose_b[:3]
    quaternion = math_utils.quat_slerp(pose_a[3:], pose_b[3:], weight)
    return torch.cat((position, quaternion), dim=-1)


def handle_grasp_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    lift_step: int,
    output_data: LocomanipulationSDGOutputData,
    active_side: str,
    source_object_reference: torch.Tensor | None = None,
    target_object_reference: torch.Tensor | None = None,
) -> tuple[int, LocomanipulationSDGDataGenerationState]:
    """Handle the GRASP_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        lift_step: Step to transition to lift phase
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)

    # Set control targets - robot stays stationary during grasping
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.GRASP_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])

    # Retarget against fixed episode-start object frames. Using the live can
    # pose here creates positive feedback as soon as contact moves the can.
    source_object_reference = (
        recording_item.object_pose if source_object_reference is None else source_object_reference
    )
    target_object_reference = (
        env.get_object().get_pose() if target_object_reference is None else target_object_reference
    )
    if active_side == "left":
        output_data.left_hand_pose_target = transform_relative_pose(
            recording_item.left_hand_pose_target, source_object_reference, target_object_reference
        )[0]
        output_data.right_hand_pose_target = transform_relative_pose(
            recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
        )[0]
    else:
        output_data.left_hand_pose_target = transform_relative_pose(
            recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
        )[0]
        output_data.right_hand_pose_target = transform_relative_pose(
            recording_item.right_hand_pose_target, source_object_reference, target_object_reference
        )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Update state

    next_recording_step = recording_step + 1
    next_state = (
        LocomanipulationSDGDataGenerationState.LIFT_OBJECT
        if next_recording_step > lift_step
        else LocomanipulationSDGDataGenerationState.GRASP_OBJECT
    )

    return next_recording_step, next_state


def handle_lift_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    navigate_step: int,
    output_data: LocomanipulationSDGOutputData,
    active_side: str,
    source_object_reference: torch.Tensor | None = None,
    target_object_reference: torch.Tensor | None = None,
    lift_start_step: int | None = None,
) -> tuple[int, LocomanipulationSDGDataGenerationState]:
    """Handle the LIFT_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        navigate_step: Step to transition to navigation phase
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)

    # Set control targets - robot stays stationary during lifting
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.LIFT_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.tensor([0.0, 0.0, 0.0])

    # Replay the demonstrated pull-out shape from the randomized grasp pose,
    # then smoothly remove that object-frame offset before navigation so the
    # active arm reaches the demonstrated robot-relative carry/home pose.
    left_base_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    right_base_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    source_object_reference = (
        recording_item.object_pose if source_object_reference is None else source_object_reference
    )
    target_object_reference = (
        env.get_object().get_pose() if target_object_reference is None else target_object_reference
    )
    start_step = recording_step if lift_start_step is None else lift_start_step
    blend_alpha = (recording_step - start_step) / max(1, navigate_step - start_step)
    if active_side == "left":
        left_object_target = transform_relative_pose(
            recording_item.left_hand_pose_target, source_object_reference, target_object_reference
        )[0]
        output_data.left_hand_pose_target = blend_pose(left_object_target, left_base_target, blend_alpha)
        output_data.right_hand_pose_target = right_base_target
    else:
        right_object_target = transform_relative_pose(
            recording_item.right_hand_pose_target, source_object_reference, target_object_reference
        )[0]
        output_data.left_hand_pose_target = left_base_target
        output_data.right_hand_pose_target = blend_pose(right_object_target, right_base_target, blend_alpha)
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Update state
    next_recording_step = recording_step + 1
    next_state = (
        LocomanipulationSDGDataGenerationState.NAVIGATE
        if next_recording_step > navigate_step
        else LocomanipulationSDGDataGenerationState.LIFT_OBJECT
    )

    return next_recording_step, next_state


def handle_navigate_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_path_helper: ParameterizedPath,
    base_goal_approach: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
) -> LocomanipulationSDGDataGenerationState:
    """Handle the NAVIGATE state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_path_helper: Parameterized path for navigation
        base_goal_approach: Approach pose goal
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Next state
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    current_pose = env.get_base().get_pose_2d()[0]

    # Find target point along path using pure pursuit algorithm
    _, nearest_path_length, _, _ = base_path_helper.find_nearest(current_pose[:2])
    target_xy = base_path_helper.get_point_by_distance(distance=nearest_path_length + config.following_offset)

    # Compute navigation velocities
    if config.navigation_mode == "fixed_yaw_holonomic":
        body_velocity, angular_velocity = compute_fixed_yaw_holonomic_velocity(current_pose, target_xy, config)
    else:
        linear_velocity, angular_velocity = compute_navigation_velocity(current_pose, target_xy, config)
        body_velocity = torch.stack((linear_velocity, linear_velocity.new_zeros(())))

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.NAVIGATE)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.stack((body_velocity[0], body_velocity[1], angular_velocity))

    # Keep both hands robot-root-relative, matching the preceding carry state.
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Check if close enough to approach goal to transition
    goal_xy = base_goal_approach.get_pose_2d()[0, :2]
    distance_to_goal = torch.sqrt(torch.sum((current_pose[:2] - goal_xy) ** 2))

    return (
        LocomanipulationSDGDataGenerationState.APPROACH
        if distance_to_goal < config.distance_threshold
        else LocomanipulationSDGDataGenerationState.NAVIGATE
    )


def handle_approach_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_goal: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
) -> LocomanipulationSDGDataGenerationState:
    """Handle the APPROACH state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_goal: Final goal pose
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Next state
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    current_pose = env.get_base().get_pose_2d()[0]

    # Navigate directly to final goal position
    goal_xy = base_goal.get_pose_2d()[0, :2]
    if config.navigation_mode == "fixed_yaw_holonomic":
        body_velocity, angular_velocity = compute_fixed_yaw_holonomic_velocity(current_pose, goal_xy, config)
    else:
        linear_velocity, angular_velocity = compute_navigation_velocity(current_pose, goal_xy, config)
        body_velocity = torch.stack((linear_velocity, linear_velocity.new_zeros(())))

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.APPROACH)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.stack((body_velocity[0], body_velocity[1], angular_velocity))

    # Keep both hands robot-root-relative through APPROACH.
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, recording_item.base_pose, env.get_base().get_pose()
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Check if close enough to final goal to start drop-off
    distance_to_goal = torch.sqrt(torch.sum((current_pose[:2] - goal_xy) ** 2))

    return (
        LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT
        if distance_to_goal < config.distance_threshold
        else LocomanipulationSDGDataGenerationState.APPROACH
    )


def handle_drop_off_state(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    recording_step: int,
    base_goal: RelativePose,
    config: LocomanipulationSDGControlConfig,
    output_data: LocomanipulationSDGOutputData,
    recorded_base_reference: bool = False,
) -> tuple[int, LocomanipulationSDGDataGenerationState | None]:
    """Handle the DROP_OFF_OBJECT state logic.

    Args:
        env: The environment
        input_episode_data: Input episode data
        recording_step: Current recording step
        base_goal: Final goal pose
        config: Navigation control configuration
        output_data: Output data to populate

    Returns:
        Tuple of (next_recording_step, next_state)
    """
    recording_item = env.load_input_data(input_episode_data, recording_step)
    if recording_item is None:
        return recording_step, None

    # Compute orientation control to face target orientation
    current_pose = env.get_base().get_pose_2d()[0]
    target_pose = base_goal.get_pose_2d()[0]
    current_yaw = current_pose[2]
    target_yaw = target_pose[2]
    delta_yaw = target_yaw - current_yaw
    delta_yaw = (delta_yaw + torch.pi) % (2 * torch.pi) - torch.pi

    angular_velocity = torch.clip(
        config.angular_gain * delta_yaw, -config.angular_max, config.angular_max
    )
    if config.navigation_mode == "fixed_yaw_holonomic":
        # Navigation already reached and held the requested yaw.  Do not add a
        # final chassis turn before place.
        angular_velocity = delta_yaw.new_zeros(())
    linear_velocity = 0.0  # Stay in place while orienting

    # Set control targets
    output_data.data_generation_state = int(LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT)
    output_data.recording_step = recording_step
    output_data.base_velocity_target = torch.stack((
        angular_velocity.new_tensor(linear_velocity),
        angular_velocity.new_zeros(()),
        angular_velocity,
    ))

    # A continuous mobile seed recorded place targets at its final base pose.
    # A static seed instead uses the original fixture-relative behavior.
    if recorded_base_reference:
        source_reference = recording_item.base_pose
        target_reference = env.get_base().get_pose()
    else:
        source_reference = recording_item.fixture_pose
        target_reference = env.get_end_fixture().get_pose()
    output_data.left_hand_pose_target = transform_relative_pose(
        recording_item.left_hand_pose_target, source_reference, target_reference
    )[0]
    output_data.right_hand_pose_target = transform_relative_pose(
        recording_item.right_hand_pose_target, source_reference, target_reference
    )[0]
    output_data.left_hand_joint_positions_target = recording_item.left_hand_joint_positions_target
    output_data.right_hand_joint_positions_target = recording_item.right_hand_joint_positions_target

    # Continue playback if orientation is within threshold
    next_recording_step = (
        recording_step + 1
        if config.navigation_mode == "fixed_yaw_holonomic" or abs(delta_yaw) < config.angle_threshold
        else recording_step
    )

    return next_recording_step, LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT


def populate_output_data(
    env: LocomanipulationSDGEnv,
    output_data: LocomanipulationSDGOutputData,
    base_goal: RelativePose,
    base_goal_approach: RelativePose,
    base_path: torch.Tensor,
) -> None:
    """Populate remaining output data fields.

    Args:
        env: The environment
        output_data: Output data to populate
        base_goal: Final goal pose
        base_goal_approach: Approach goal pose
        base_path: Planned navigation path
    """
    output_data.base_pose = env.get_base().get_pose()
    output_data.object_pose = env.get_object().get_pose()
    output_data.start_fixture_pose = env.get_start_fixture().get_pose()
    output_data.end_fixture_pose = env.get_end_fixture().get_pose()
    output_data.base_goal_pose = base_goal.get_pose()
    output_data.base_goal_approach_pose = base_goal_approach.get_pose()
    output_data.base_path = base_path

    # Collect obstacle poses
    obstacle_poses = []
    for obstacle in env.get_obstacle_fixtures():
        obstacle_poses.append(obstacle.get_pose())
    if obstacle_poses:
        output_data.obstacle_fixture_poses = torch.cat(obstacle_poses, dim=0)[None, :]
    else:
        output_data.obstacle_fixture_poses = torch.empty((1, 0, 7))  # Empty tensor with correct shape


def replay(
    env: LocomanipulationSDGEnv,
    input_episode_data: EpisodeData,
    lift_step: int,
    navigate_step: int,
    active_side: str,
    place_step: int | None = None,
    draw_visualization: bool = False,
    angular_gain: float = 2.0,
    linear_gain: float = 1.0,
    linear_max: float = 1.0,
    angular_max: float = 1.0,
    navigation_mode: str = "path_heading",
    navigation_yaw: float = 0.0,
    initial_yaw_counterclockwise: bool = False,
    distance_threshold: float = 0.1,
    following_offset: float = 0.6,
    angle_threshold: float = 0.2,
    approach_distance: float = 1.0,
    randomize_placement: bool = True,
    episode_seed: int | None = None,
) -> tuple[bool, str, dict[str, float]]:
    """Replay a locomanipulation SDG episode with state machine control.

    This function implements a state machine for locomanipulation SDG, where the robot:
    1. Grasps an object at the start position
    2. Lifts the object while stationary
    3. Navigates with the object to an approach position
    4. Approaches the final goal position
    5. Places the object at the end position

    Args:
        env: The locomanipulation SDG environment
        input_episode_data: Static manipulation episode data to replay
        lift_step: Recording step where lifting phase begins
        navigate_step: Recording step where navigation phase begins
        active_side: Manipulation arm selected by the seed region contract
        draw_visualization: Whether to visualize occupancy map and path
        angular_gain: Proportional gain for angular velocity control
        linear_gain: Proportional gain for linear velocity control
        linear_max: Maximum linear velocity (m/s)
        angular_max: Maximum angular velocity (rad/s)
        navigation_mode: Path-heading or fixed-yaw holonomic controller
        navigation_yaw: Fixed world yaw for holonomic navigation
        initial_yaw_counterclockwise: Force the initial fixed-yaw turn counterclockwise
        distance_threshold: Distance threshold for state transitions (m)
        following_offset: Look-ahead distance for path following (m)
        angle_threshold: Angular threshold for orientation control (rad)
        approach_distance: Buffer distance from final goal (m)
        randomize_placement: Whether to randomize obstacle placement
    """

    if active_side not in ("left", "right"):
        raise ValueError(f"Task525 active_side must be left or right, got {active_side!r}")

    # Initialize environment to starting state
    env.reset_to(
        state=input_episode_data.get_initial_state(),
        env_ids=torch.tensor([0]),
        seed=episode_seed,
        is_relative=True,
    )

    # Capture fixed retarget frames after reset. The target frame must not
    # follow physics contact during grasp/pull-out.
    initial_recording_item = env.load_input_data(input_episode_data, 0)
    source_object_reference = initial_recording_item.object_pose.detach().clone()
    target_object_reference = env.get_object().get_pose().detach().clone()

    # Create navigation control configuration
    config = LocomanipulationSDGControlConfig(
        angular_gain=angular_gain,
        linear_gain=linear_gain,
        linear_max=linear_max,
        angular_max=angular_max,
        navigation_mode=navigation_mode,
        navigation_yaw=navigation_yaw,
        initial_yaw_counterclockwise=initial_yaw_counterclockwise,
        initial_yaw_alignment_pending=initial_yaw_counterclockwise,
        distance_threshold=distance_threshold,
        following_offset=following_offset,
        angle_threshold=angle_threshold,
        approach_distance=approach_distance,
    )

    # Build the scene once, then preflight a path from the reset root.  This
    # rejects invalid randomized starts before manipulation work is performed.
    occupancy_map, base_goal, base_goal_approach = build_navigation_scene(
        env, input_episode_data, approach_distance, randomize_placement
    )
    preflight_started = time.perf_counter()
    try:
        base_path_helper = plan_navigation_path(
            env, occupancy_map, base_goal_approach
        )
    except Exception as error:
        return (
            False,
            f"navigation_preflight: {type(error).__name__}: {error}",
            {},
        )
    planning_metrics = navigation_path_metrics(
        "preflight", base_path_helper, time.perf_counter() - preflight_started
    )
    recorded_base_path = path_points_for_recording(base_path_helper)

    # Visualize occupancy map and path if requested
    if draw_visualization:
        occupancy_map_add_to_stage(
            occupancy_map,
            stage=omni.usd.get_context().get_stage(),
            path="/OccupancyMap",
            z_offset=0.01,
            draw_path=base_path_helper.points,
        )

    # Initialize state machine
    output_data = LocomanipulationSDGOutputData()
    current_state = LocomanipulationSDGDataGenerationState.GRASP_OBJECT
    recording_step = 0
    dropoff_cursor_initialized = False
    dropoff_replay_step = (
        infer_dropoff_replay_step(input_episode_data, navigate_step, place_step)
        if place_step is not None
        else None
    )
    if dropoff_replay_step is not None and dropoff_replay_step != place_step:
        print(
            "[Task525] Replaying pre-place lift descent from source step "
            f"{dropoff_replay_step} before place marker {place_step}."
        )
    carry_gate_checked = False
    initial_object_pose = target_object_reference.detach().clone()
    initial_robot_root_pose = env.get_base().get_pose().detach().clone()
    max_pre_navigation_root_xy_displacement = 0.0
    last_reported_state = None

    # Main simulation loop with state machine
    while simulation_app.is_running() and not simulation_app.is_exiting():

        if current_state != last_reported_state:
            print(f"Current state: {current_state.name}, Recording step: {recording_step}")
            last_reported_state = current_state

        if not carry_gate_checked:
            current_robot_root_pose = env.get_base().get_pose().detach()
            current_root_xy_displacement = float(
                torch.linalg.vector_norm(
                    current_robot_root_pose[0, :2] - initial_robot_root_pose[0, :2]
                ).item()
            )
            max_pre_navigation_root_xy_displacement = max(
                max_pre_navigation_root_xy_displacement,
                current_root_xy_displacement,
            )

        if current_state == LocomanipulationSDGDataGenerationState.NAVIGATE and not carry_gate_checked:
            carry_gate_checked = True
            carry_ok, failure_reason, carry_metrics = env.evaluate_task525_carry_checkpoint(
                input_episode_data,
                navigate_step,
                initial_object_pose,
            )
            root_delta_xy = current_robot_root_pose[0, :2] - initial_robot_root_pose[0, :2]
            root_metrics = {
                "carry_root_dx_m": float(root_delta_xy[0].item()),
                "carry_root_dy_m": float(root_delta_xy[1].item()),
                "carry_root_xy_displacement_m": current_root_xy_displacement,
                "carry_root_xy_max_displacement_m": max_pre_navigation_root_xy_displacement,
                "carry_root_xy_limit_m": TASK000525_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M,
            }
            carry_metrics = {**carry_metrics, **root_metrics}
            if not carry_ok:
                return (
                    False,
                    failure_reason,
                    {**planning_metrics, **carry_metrics},
                )
            if (
                max_pre_navigation_root_xy_displacement
                > TASK000525_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M
            ):
                return (
                    False,
                    "carry_checkpoint: robot root moved while base command was zero "
                    "(possible arm-cabinet contact)",
                    {**planning_metrics, **carry_metrics},
                )

            # Replan from the measured root after the allowed sub-5 mm physics
            # settling, immediately before navigation.
            navigation_plan_started = time.perf_counter()
            try:
                base_path_helper = plan_navigation_path(
                    env, occupancy_map, base_goal_approach
                )
            except Exception as error:
                return (
                    False,
                    f"navigation_replan: {type(error).__name__}: {error}",
                    {**planning_metrics, **carry_metrics},
                )
            planning_metrics.update(
                navigation_path_metrics(
                    "navigation_entry",
                    base_path_helper,
                    time.perf_counter() - navigation_plan_started,
                )
            )
            recorded_base_path = path_points_for_recording(base_path_helper)
            config.initial_yaw_alignment_pending = (
                config.initial_yaw_counterclockwise
            )

        # Execute state-specific logic using helper functions
        if current_state == LocomanipulationSDGDataGenerationState.GRASP_OBJECT:
            recording_step, current_state = handle_grasp_state(
                env,
                input_episode_data,
                recording_step,
                lift_step,
                output_data,
                active_side,
                source_object_reference,
                target_object_reference,
            )

        elif current_state == LocomanipulationSDGDataGenerationState.LIFT_OBJECT:
            recording_step, current_state = handle_lift_state(
                env,
                input_episode_data,
                recording_step,
                navigate_step,
                output_data,
                active_side,
                source_object_reference,
                target_object_reference,
                lift_step + 1,
            )

        elif current_state == LocomanipulationSDGDataGenerationState.NAVIGATE:
            current_state = handle_navigate_state(
                env, input_episode_data, recording_step, base_path_helper, base_goal_approach, config, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.APPROACH:
            current_state = handle_approach_state(
                env, input_episode_data, recording_step, base_goal, config, output_data
            )

        elif current_state == LocomanipulationSDGDataGenerationState.DROP_OFF_OBJECT:
            if dropoff_replay_step is not None and not dropoff_cursor_initialized:
                recording_step = dropoff_replay_step
                dropoff_cursor_initialized = True
            recording_step, next_state = handle_drop_off_state(
                env,
                input_episode_data,
                recording_step,
                base_goal,
                config,
                output_data,
                recorded_base_reference=place_step is not None,
            )
            if next_state is None:  # End of episode data
                break
            current_state = next_state

        # Populate additional output data fields
        populate_output_data(
            env,
            output_data,
            base_goal,
            base_goal_approach,
            recorded_base_path,
        )

        # Attach output data to environment for recording
        env._locomanipulation_sdg_output_data = output_data

        # Build and execute action
        action = env.build_action_vector(
            base_velocity_target=output_data.base_velocity_target,
            left_hand_joint_positions_target=output_data.left_hand_joint_positions_target,
            right_hand_joint_positions_target=output_data.right_hand_joint_positions_target,
            left_hand_pose_target=output_data.left_hand_pose_target,
            right_hand_pose_target=output_data.right_hand_pose_target,
        )

        env.step(action)

    final_ok, final_reason, final_metrics = env.evaluate_task525_final_checkpoint()
    return final_ok, final_reason, {**planning_metrics, **carry_metrics, **final_metrics}

def _metadata_text(value: object, key: str, demo: str) -> str:
    """Normalize a required scalar HDF5 episode attribute."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    text = str(value)
    if not text:
        raise ValueError(f"{demo}: empty {key} episode metadata")
    return text


def task525_source_groups(input_handler) -> dict[str, list[str]]:
    """Group seed episodes by A-D target region and validate their arm labels."""

    data_group = input_handler._hdf5_data_group
    dataset_target = _metadata_text(
        data_group.attrs.get("target_object_name", ""),
        "target_object_name",
        "/data",
    )
    if dataset_target != TASK000525_TARGET_OBJECT:
        raise ValueError(
            f"Task525 requires target {TASK000525_TARGET_OBJECT}, got {dataset_target!r}"
        )
    groups = {region_key: [] for region_key in TASK000525_REGION_KEYS}
    for demo in input_handler.get_episode_names():
        metadata = dict(data_group[demo].attrs)
        if "task525_target_region" not in metadata:
            raise ValueError(f"{demo}: missing task525_target_region metadata")
        if "task525_manipulation_side" not in metadata:
            raise ValueError(f"{demo}: missing task525_manipulation_side metadata")
        episode_target = _metadata_text(
            metadata.get("target_object_name", ""),
            "target_object_name",
            demo,
        )
        if episode_target != TASK000525_TARGET_OBJECT:
            raise ValueError(
                f"{demo}: Task525 requires target {TASK000525_TARGET_OBJECT}, "
                f"got {episode_target!r}"
            )
        region = validate_region_key(
            _metadata_text(metadata["task525_target_region"], "target region", demo)
        )
        try:
            region_to_object = json.loads(
                _metadata_text(
                    metadata.get("task525_region_to_object", ""),
                    "task525_region_to_object",
                    demo,
                )
            )
        except json.JSONDecodeError as error:
            raise ValueError(
                f"{demo}: invalid task525_region_to_object JSON"
            ) from error
        if not isinstance(region_to_object, dict):
            raise ValueError(f"{demo}: task525_region_to_object must be an object")
        try:
            CoffeeArrangement(region, region_to_object)
        except ValueError as error:
            raise ValueError(
                f"{demo}: invalid orange-target arrangement: {error}"
            ) from error
        side = _metadata_text(
            metadata["task525_manipulation_side"], "manipulation side", demo
        )
        expected_side = manipulation_side_for_region(region)
        if side != expected_side:
            raise ValueError(
                f"{demo}: region {region} requires {expected_side}, got {side}"
            )
        groups[region].append(demo)
    missing = [region for region, demos in groups.items() if not demos]
    if missing:
        raise ValueError(f"Task525 seed dataset is missing regions: {missing}")
    return groups


def select_balanced_source_demo(
    groups: dict[str, list[str]],
    selection_index: int,
) -> str:
    """Cycle A-D and then cycle demos within each region."""

    region_index = selection_index % len(TASK000525_REGION_KEYS)
    block_index = selection_index // len(TASK000525_REGION_KEYS)
    region = TASK000525_REGION_KEYS[region_index]
    demos = groups[region]
    return demos[block_index % len(demos)]




def set_generation_episode_result(
    env: LocomanipulationSDGEnv,
    success: bool,
    failure_reason: str,
    metrics: dict[str, float],
) -> None:
    """Attach quality-gate status to the current recorder episode."""

    episode = env.recorder_manager.get_episode(0)
    episode.success = bool(success)
    metadata = dict(getattr(episode, "metadata", {}) or {})
    metadata.update({
        "success": bool(success),
        "success_criterion_id": "task525_generation_quality_gate_v2",
        "failure_reason": "" if success else failure_reason,
    })
    metadata.update({f"quality_{key}": float(value) for key, value in metrics.items()})
    metadata.update(env.task525_episode_metadata())
    episode.metadata = metadata


if __name__ == "__main__":

    with torch.no_grad():

        random.seed(args_cli.seed)
        torch.manual_seed(args_cli.seed)

        # Create environment
        if args_cli.task is not None:
            env_name = args_cli.task.split(":")[-1]
        if env_name is None:
            raise ValueError("Task/env name was not specified nor found in the dataset.")

        env_cfg = parse_env_cfg(env_name, device=args_cli.device, num_envs=1)
        env_cfg.sim.device = "cpu"
        env_cfg.seed = args_cli.seed
        env_cfg.recorders.dataset_export_dir_path = os.path.dirname(args_cli.output_file)
        env_cfg.recorders.dataset_filename = os.path.basename(args_cli.output_file)
        if args_cli.successful_runs_only:
            env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY

        env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

        # Load input data
        input_dataset_file_handler = HDF5DatasetFileHandler()
        input_dataset_file_handler.open(args_cli.dataset)
        source_groups = task525_source_groups(input_dataset_file_handler)

        successful_runs = 0
        attempts = 0
        max_attempts = args_cli.max_attempts or max(args_cli.num_runs, 4 * args_cli.num_runs)
        while attempts < max_attempts and (
            successful_runs < args_cli.num_runs if args_cli.successful_runs_only else attempts < args_cli.num_runs
        ):
            attempts += 1
            episode_seed = args_cli.seed + attempts - 1

            if args_cli.demo is None:
                selection_index = (
                    successful_runs if args_cli.successful_runs_only else attempts - 1
                )
                demo = select_balanced_source_demo(source_groups, selection_index)
            else:
                demo = args_cli.demo

            input_episode_data = input_dataset_file_handler.load_episode(demo, args_cli.device)

            # Isaac Lab's EpisodeData loader exposes seed/success but not
            # arbitrary HDF5 episode attributes.  The Task525 streaming
            # recorder stores lift/navigate/place markers as those attributes,
            # so read them from the selected source group explicitly.
            episode_metadata = dict(input_dataset_file_handler._hdf5_data_group[demo].attrs)
            target_region = validate_region_key(
                _metadata_text(
                    episode_metadata.get("task525_target_region", ""),
                    "task525_target_region",
                    demo,
                )
            )
            metadata_side = _metadata_text(
                episode_metadata.get("task525_manipulation_side", ""),
                "task525_manipulation_side",
                demo,
            )
            expected_side = manipulation_side_for_region(target_region)
            if metadata_side != expected_side:
                raise ValueError(
                    f"{demo}: region {target_region} requires {expected_side}, "
                    f"got {metadata_side}"
                )
            active_side = (
                metadata_side
                if args_cli.active_side == "episode"
                else args_cli.active_side
            )
            if active_side != metadata_side:
                raise ValueError(
                    f"{demo}: --active_side={active_side} conflicts with "
                    f"episode metadata {metadata_side}"
                )
            env.set_task525_episode_context(
                target_region=target_region,
                manipulation_side=active_side,
                source_demo=demo,
            )

            lift_step = args_cli.lift_step
            navigate_step = args_cli.navigate_step
            place_step = args_cli.place_step
            if lift_step is None and "lift_step" in episode_metadata:
                lift_step = int(episode_metadata["lift_step"])
            if navigate_step is None and "navigate_step" in episode_metadata:
                navigate_step = int(episode_metadata["navigate_step"])
            if place_step is None and "place_step" in episode_metadata:
                place_step = int(episode_metadata["place_step"])
            if lift_step is None or navigate_step is None:
                raise ValueError(
                    "lift_step and navigate_step are required, either as CLI values or episode metadata."
                )
            original_lift_step = lift_step
            lift_step, lift_step_source = resolve_grasp_boundary_step(
                input_episode_data,
                episode_metadata,
                active_side,
                lift_step,
            )
            if lift_step != original_lift_step:
                print(
                    "[Task525] Replaced late G/lift marker "
                    f"{original_lift_step} with grasp boundary {lift_step} "
                    f"({lift_step_source})."
                )
            if lift_step < 0 or navigate_step <= lift_step:
                raise ValueError(
                    f"Expected 0 <= lift_step < navigate_step, got {lift_step}, {navigate_step}."
                )
            if place_step is not None and place_step <= navigate_step:
                raise ValueError(
                    f"Expected navigate_step < place_step, got {navigate_step}, {place_step}."
                )

            success, failure_reason, quality_metrics = replay(
                env=env,
                input_episode_data=input_episode_data,
                lift_step=lift_step,
                navigate_step=navigate_step,
                place_step=place_step,
                active_side=active_side,
                draw_visualization=args_cli.draw_visualization,
                angular_gain=args_cli.angular_gain,
                linear_gain=args_cli.linear_gain,
                linear_max=args_cli.linear_max,
                angular_max=args_cli.angular_max,
                navigation_mode=args_cli.navigation_mode,
                navigation_yaw=args_cli.navigation_yaw,
                initial_yaw_counterclockwise=args_cli.initial_yaw_counterclockwise,
                distance_threshold=args_cli.distance_threshold,
                following_offset=args_cli.following_offset,
                angle_threshold=args_cli.angle_threshold,
                approach_distance=args_cli.approach_distance,
                randomize_placement=args_cli.randomize_placement,
                episode_seed=episode_seed,
            )
            quality_metrics["generation_seed"] = float(episode_seed)
            set_generation_episode_result(
                env,
                success,
                failure_reason,
                quality_metrics,
            )
            # Export before an environment reset can recompute and overwrite
            # the explicit Task525 quality result from termination terms.
            env.recorder_manager.export_episodes([0])
            if success:
                successful_runs += 1
                print(
                    f"[Task525 Quality] PASS {successful_runs}/{args_cli.num_runs} "
                    f"after {attempts} attempt(s): {quality_metrics}"
                )
            else:
                print(
                    f"[Task525 Quality] REJECT attempt {attempts}: "
                    f"{failure_reason}; metrics={quality_metrics}"
                )

        insufficient_successes = (
            args_cli.successful_runs_only and successful_runs < args_cli.num_runs
        )
        env.close()
        simulation_app.close()
        if insufficient_successes:
            raise RuntimeError(
                f"Generated only {successful_runs}/{args_cli.num_runs} successful trajectories "
                f"within {attempts} attempts."
            )
