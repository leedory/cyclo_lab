"""Online Dijkstra navigation used while recording a Task000525 seed demo.

The recorder owns the phase transition, while this module owns only the
collision-aware base motion between the operator's pick/carry and place
segments.  It deliberately does not touch the A3 left/right arm topic mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import OccupancyMap, merge_occupancy_maps
from isaaclab_mimic.locomanipulation_sdg.path_utils import ParameterizedPath, plan_path
from isaaclab_mimic.locomanipulation_sdg.scene_utils import HasPose2d

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_JOINT_POSITION_LIMITS,
    FFW_SG2_LIFT_JOINT_NAME,
    SG2_SWERVE_WHEEL_JOINTS,
)

from .locomanipulation_sdg_contract import (
    CANDIDATE_BASE_GOAL_XYYAW,
    SHOWROOM_STATIC_OBSTACLE_AABBS,
    STATIC_MAP_PREFILL_BUFFER_M,
    UPSTREAM_FINAL_BUFFER_M,
    wrap_to_pi,
)


@dataclass(frozen=True)
class Task525OnlineDijkstraCfg:
    """Conservative controller limits for the simulated carrying segment."""

    linear_gain: float = 1.0
    angular_gain: float = 2.0
    linear_max: float = 0.10
    angular_max: float = 0.25
    initial_yaw_counterclockwise: bool = True
    following_offset: float = 0.20
    distance_threshold: float = 0.05
    angle_threshold: float = 0.05
    off_path_replan_distance: float = 0.25
    max_replans: int = 2
    max_navigation_seconds: float = 90.0
    max_return_home_seconds: float = 20.0
    return_home_blend_seconds: float = 2.0
    # A grasped rigid can can keep a gripper a few degrees away from its
    # position target even when the carry posture is visibly still. Grippers
    # remain held, but are not used to decide whether the arm is safe to
    # begin base motion.
    home_joint_tolerance_rad: float = 0.08
    home_settle_steps: int = 10
    lift_lower_distance_m: float = 0.30
    lift_lower_blend_seconds: float = 2.0
    max_lift_lower_seconds: float = 10.0
    lift_position_tolerance_m: float = 0.01
    lift_settle_steps: int = 10
    settle_wheel_speed_norm: float = 0.10
    settle_steps: int = 10
    goal_xyyaw: tuple[float, float, float] = CANDIDATE_BASE_GOAL_XYYAW


class _FixedPose2d(HasPose2d):
    """Minimal upstream planner pose wrapper with a CPU [x, y, yaw] tensor."""

    def __init__(self, pose_2d: torch.Tensor):
        self._pose_2d = pose_2d.detach().to(dtype=torch.float32, device="cpu").reshape(1, 3)

    def get_pose_2d(self) -> torch.Tensor:
        return self._pose_2d


def make_task525_planning_map() -> OccupancyMap:
    """Return the reviewed static showroom map inflated for the SG2 footprint."""

    maps = []
    for _name, min_x, min_y, max_x, max_y in SHOWROOM_STATIC_OBSTACLE_AABBS:
        boundary = np.asarray(
            ((min_x, min_y), (max_x, min_y), (max_x, max_y), (min_x, max_y)),
            dtype=np.float64,
        )
        maps.append(OccupancyMap.from_occupancy_boundary(boundary, resolution=0.05))
    # Match the checked offline route: task-local prefill plus the upstream
    # final 0.15 m buffer totals the deployed 0.44 m footprint plus margin.
    return merge_occupancy_maps(maps).buffered_meters(
        STATIC_MAP_PREFILL_BUFFER_M + UPSTREAM_FINAL_BUFFER_M
    )


class Task525OnlineDijkstraNavigator:
    """Return a carrying arm home, navigate, then await an explicit place handoff."""

    IDLE = "idle"
    RETURNING_HOME = "returning_home"
    NAVIGATING = "navigating"
    SETTLING = "settling"
    LOWERING_LIFT = "lowering_lift"
    WAITING_FOR_PLACE_ACTIVATION = "waiting_for_place_activation"
    COMPLETE = "complete"
    FAILED = "failed"

    def __init__(self, env, cfg: Task525OnlineDijkstraCfg, control_hz: float):
        if control_hz <= 0.0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        self.env = env
        self.cfg = cfg
        self.control_hz = float(control_hz)
        self.planning_map = make_task525_planning_map()
        self.status = self.IDLE
        self.failure_reason: str | None = None
        self.path: ParameterizedPath | None = None
        self.frozen_arm_action: torch.Tensor | None = None
        self.return_start_arm_action: torch.Tensor | None = None
        self.return_home_steps = 0
        self.home_settled_steps = 0
        self.lift_lower_steps = 0
        self.lift_settled_steps = 0
        self.lift_lower_start_position: float | None = None
        self.lift_lower_target_position: float | None = None
        self.navigation_steps = 0
        self.settled_steps = 0
        self.replan_count = 0
        self.last_pose_2d: torch.Tensor | None = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_wheel_speed_norm = 0.0
        self._initial_yaw_alignment_pending = False

        robot = self.env.scene["robot"]
        joint_ids, joint_names = robot.find_joints(
            list(SG2_SWERVE_WHEEL_JOINTS), preserve_order=True
        )
        if len(joint_ids) != len(SG2_SWERVE_WHEEL_JOINTS):
            raise RuntimeError(f"Failed to resolve SG2 wheel joints: {joint_names}")
        self._wheel_joint_ids = joint_ids
        arm_joint_ids, arm_joint_names = robot.find_joints(
            list(FFW_SG2_ACTION_JOINT_NAMES), preserve_order=True
        )
        if len(arm_joint_ids) != len(FFW_SG2_ACTION_JOINT_NAMES):
            raise RuntimeError(f"Failed to resolve SG2 action joints: {arm_joint_names}")
        self._arm_joint_ids = arm_joint_ids
        self._lift_action_index = FFW_SG2_ACTION_JOINT_NAMES.index(FFW_SG2_LIFT_JOINT_NAME)
        self._home_settle_action_indices = [
            index
            for index, name in enumerate(FFW_SG2_ACTION_JOINT_NAMES)
            if name not in ("gripper_l_joint1", "gripper_r_joint1")
        ]

    @property
    def active(self) -> bool:
        return self.status in (
            self.RETURNING_HOME,
            self.NAVIGATING,
            self.SETTLING,
            self.LOWERING_LIFT,
        )

    @property
    def holding_arm(self) -> bool:
        """Whether autonomous safety still owns the 19D upper-body target."""

        return self.status in (
            self.RETURNING_HOME,
            self.NAVIGATING,
            self.SETTLING,
            self.LOWERING_LIFT,
            self.WAITING_FOR_PLACE_ACTIVATION,
        )

    @property
    def awaiting_place_activation(self) -> bool:
        return self.status == self.WAITING_FOR_PLACE_ACTIVATION

    @property
    def completed(self) -> bool:
        return self.status == self.COMPLETE

    def reset(self) -> None:
        self.status = self.IDLE
        self.failure_reason = None
        self.path = None
        self.frozen_arm_action = None
        self.return_start_arm_action = None
        self.return_home_steps = 0
        self.home_settled_steps = 0
        self.lift_lower_steps = 0
        self.lift_settled_steps = 0
        self.lift_lower_start_position = None
        self.lift_lower_target_position = None
        self.navigation_steps = 0
        self.settled_steps = 0
        self.replan_count = 0
        self.last_pose_2d = None
        self.last_command = (0.0, 0.0, 0.0)
        self.last_wheel_speed_norm = 0.0
        self._initial_yaw_alignment_pending = False

    def _root_pose_2d(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        position = robot.data.root_pos_w[0].detach().to(dtype=torch.float32, device="cpu")
        quat = robot.data.root_quat_w[0].detach().to(dtype=torch.float32, device="cpu")
        w, x, y, z = (float(value) for value in quat)
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return torch.tensor((float(position[0]), float(position[1]), yaw), dtype=torch.float32)

    def _wheel_speed_norm(self) -> float:
        robot = self.env.scene["robot"]
        return float(
            torch.linalg.vector_norm(robot.data.joint_vel[:, self._wheel_joint_ids], dim=1)
            .amax()
            .item()
        )

    def _plan_from(self, current_pose: torch.Tensor) -> None:
        goal = torch.tensor(self.cfg.goal_xyyaw, dtype=torch.float32)
        path = plan_path(
            start=_FixedPose2d(current_pose),
            end=_FixedPose2d(goal),
            occupancy_map=self.planning_map,
        )
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 2:
            raise RuntimeError(f"Dijkstra returned an invalid path shape {tuple(path.shape)}")
        self.path = ParameterizedPath(path.to(dtype=torch.float32, device="cpu"))

    def start(self, action: torch.Tensor, *, home_arm_action: torch.Tensor | None = None) -> bool:
        """Preflight at G, return to carrying-home, then replan and navigate.

        ``home_arm_action`` is the recorder-supplied absolute 19D Task525
        init/home target. The recorder changes only the explicitly contracted
        head and current closed right-gripper entries before passing it here.
        """

        if self.active:
            return False
        if action.shape[-1] != 22:
            raise ValueError(f"Task525 online navigation requires 22D actions, got {tuple(action.shape)}")
        self.reset()
        current_pose = self._root_pose_2d()
        try:
            self._plan_from(current_pose)
        except Exception as exc:
            self.status = self.FAILED
            self.failure_reason = f"Dijkstra planning failed: {type(exc).__name__}: {exc}"
            return False
        if home_arm_action is None:
            home_arm_action = action[:, :19]
        if home_arm_action.shape != action[:, :19].shape:
            raise ValueError(
                "Task525 carrying-home target must have the same [N,19] shape as the action prefix, "
                f"got {tuple(home_arm_action.shape)}."
            )
        self.return_start_arm_action = self._measured_arm_action()
        self.frozen_arm_action = home_arm_action.detach().clone()
        # The showroom starts near pi and docks at yaw zero. The shortest
        # signed angle is ambiguous at exactly 180 degrees, so make the first
        # alignment deterministic for operators: positive angular-z / CCW.
        self._initial_yaw_alignment_pending = (
            self.cfg.initial_yaw_counterclockwise
            and abs(wrap_to_pi(self.cfg.goal_xyyaw[2] - float(current_pose[2])))
            > self.cfg.angle_threshold
        )
        self.status = self.RETURNING_HOME
        return True

    def _begin_navigation(self) -> bool:
        """Replan from the settled current root immediately before base motion."""

        current_pose = self._root_pose_2d()
        try:
            self._plan_from(current_pose)
        except Exception as exc:
            self.status = self.FAILED
            self.failure_reason = (
                "Dijkstra navigation-entry replan failed: "
                f"{type(exc).__name__}: {exc}"
            )
            return False
        self.replan_count = 0
        self._initial_yaw_alignment_pending = (
            self.cfg.initial_yaw_counterclockwise
            and abs(wrap_to_pi(self.cfg.goal_xyyaw[2] - float(current_pose[2])))
            > self.cfg.angle_threshold
        )
        self.status = self.NAVIGATING
        print("[Task525 AutoNav] Replanned from the settled root; navigation started.")
        return True

    def enable_place_control(self) -> bool:
        """Release the autonomous arm hold after a fresh right-leader tact command."""

        if not self.awaiting_place_activation:
            return False
        self.status = self.COMPLETE
        return True

    def _home_is_settled(self) -> bool:
        if self.frozen_arm_action is None:
            raise RuntimeError("Task525 home check requires a frozen arm target.")
        robot = self.env.scene["robot"]
        current = robot.data.joint_pos[:, self._arm_joint_ids]
        target = self.frozen_arm_action.to(device=current.device, dtype=current.dtype)
        error = torch.abs(current - target)
        error = error[:, self._home_settle_action_indices]
        return bool(torch.all(error <= self.cfg.home_joint_tolerance_rad).item())

    def _home_error_summary(self) -> tuple[str, float]:
        """Return the largest measured carrying-home error for operator diagnostics."""

        if self.frozen_arm_action is None:
            raise RuntimeError("Task525 home diagnostic requires a frozen arm target.")
        current = self._measured_arm_action()
        target = self.frozen_arm_action.to(device=current.device, dtype=current.dtype)
        errors = torch.abs(current - target)[0]
        settle_errors = errors[self._home_settle_action_indices]
        local_index = int(torch.argmax(settle_errors).item())
        index = self._home_settle_action_indices[local_index]
        return FFW_SG2_ACTION_JOINT_NAMES[index], float(errors[index].item())

    def _measured_arm_action(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        return robot.data.joint_pos[:, self._arm_joint_ids].detach().clone()

    def _return_home_arm_action(self, result: torch.Tensor) -> torch.Tensor:
        """Blend from the measured grasp pose to home to avoid a contact impulse."""

        if self.frozen_arm_action is None or self.return_start_arm_action is None:
            raise RuntimeError("Task525 return-home blend is missing an arm target.")
        blend_steps = max(1, int(self.cfg.return_home_blend_seconds * self.control_hz))
        alpha = min(1.0, self.return_home_steps / blend_steps)
        start = self.return_start_arm_action.to(device=result.device, dtype=result.dtype)
        target = self.frozen_arm_action.to(device=result.device, dtype=result.dtype)
        return start + alpha * (target - start)

    def _begin_lift_lowering(self) -> None:
        """Lower the settled lift once, then keep that lower target for place."""

        if self.frozen_arm_action is None:
            raise RuntimeError("Task525 lift lowering requires a frozen arm target.")
        current = self._measured_arm_action()
        start = float(current[0, self._lift_action_index].item())
        lower_limit, _upper_limit = FFW_SG2_JOINT_POSITION_LIMITS[FFW_SG2_LIFT_JOINT_NAME]
        target = max(lower_limit, start - self.cfg.lift_lower_distance_m)
        self.lift_lower_start_position = start
        self.lift_lower_target_position = target
        self.frozen_arm_action[:, self._lift_action_index] = target
        self.lift_lower_steps = 0
        self.lift_settled_steps = 0
        self.status = self.LOWERING_LIFT
        print(
            "[Task525 AutoNav] Base settled. Lowering lift "
            f"from {start:.3f} m to {target:.3f} m before place handoff."
        )

    def _lift_is_settled(self) -> bool:
        if self.lift_lower_target_position is None:
            raise RuntimeError("Task525 lift check is missing its target.")
        current = self._measured_arm_action()[:, self._lift_action_index]
        return bool(
            torch.all(torch.abs(current - self.lift_lower_target_position) <= self.cfg.lift_position_tolerance_m).item()
        )

    def _lift_lower_arm_action(self, result: torch.Tensor) -> torch.Tensor:
        if (
            self.frozen_arm_action is None
            or self.lift_lower_start_position is None
            or self.lift_lower_target_position is None
        ):
            raise RuntimeError("Task525 lift-lowering blend is missing its target.")
        output = self.frozen_arm_action.to(device=result.device, dtype=result.dtype).clone()
        blend_steps = max(1, int(self.cfg.lift_lower_blend_seconds * self.control_hz))
        alpha = min(1.0, self.lift_lower_steps / blend_steps)
        output[:, self._lift_action_index] = self.lift_lower_start_position + alpha * (
            self.lift_lower_target_position - self.lift_lower_start_position
        )
        return output

    def _replan_if_needed(self, current_pose: torch.Tensor) -> None:
        assert self.path is not None
        _nearest, _distance, _segment, lateral_error = self.path.find_nearest(current_pose[:2])
        if float(lateral_error) <= self.cfg.off_path_replan_distance:
            return
        if self.replan_count >= self.cfg.max_replans:
            self.status = self.FAILED
            self.failure_reason = (
                f"Base departed {float(lateral_error):.3f} m from the Dijkstra path "
                f"after {self.replan_count} replans."
            )
            return
        try:
            self._plan_from(current_pose)
        except Exception as exc:
            self.status = self.FAILED
            self.failure_reason = f"Dijkstra replan failed: {type(exc).__name__}: {exc}"
            return
        self.replan_count += 1
        print(f"[Task525 AutoNav] Replanned from {float(lateral_error):.3f} m path deviation.")

    def _fixed_yaw_command(self, current_pose: torch.Tensor) -> tuple[float, float, float]:
        assert self.path is not None
        goal_x, goal_y, goal_yaw = self.cfg.goal_xyyaw
        shortest_yaw_error = wrap_to_pi(goal_yaw - float(current_pose[2]))
        if self._initial_yaw_alignment_pending:
            # [0, 2pi): follow the positive-z (counterclockwise) arc for the
            # first 180-degree showroom turn instead of letting the pi tie
            # choose a direction based on floating-point sign noise.
            yaw_error = (goal_yaw - float(current_pose[2])) % math.tau
            if yaw_error <= self.cfg.angle_threshold:
                self._initial_yaw_alignment_pending = False
                yaw_error = 0.0
        else:
            yaw_error = shortest_yaw_error
        angular_velocity = max(
            -self.cfg.angular_max,
            min(self.cfg.angular_gain * yaw_error, self.cfg.angular_max),
        )
        if abs(yaw_error) > self.cfg.angle_threshold:
            return 0.0, 0.0, angular_velocity

        _nearest, path_distance, _segment, _lateral_error = self.path.find_nearest(current_pose[:2])
        target_xy = self.path.get_point_by_distance(
            distance=float(path_distance) + self.cfg.following_offset
        )
        world_error = target_xy - current_pose[:2]
        yaw = float(current_pose[2])
        cos_yaw, sin_yaw = math.cos(yaw), math.sin(yaw)
        body_x = cos_yaw * float(world_error[0]) + sin_yaw * float(world_error[1])
        body_y = -sin_yaw * float(world_error[0]) + cos_yaw * float(world_error[1])
        speed = math.hypot(body_x, body_y) * self.cfg.linear_gain
        if speed > self.cfg.linear_max:
            scale = self.cfg.linear_max / speed
            body_x *= scale
            body_y *= scale
        else:
            body_x *= self.cfg.linear_gain
            body_y *= self.cfg.linear_gain

        distance_to_goal = math.hypot(float(current_pose[0]) - goal_x, float(current_pose[1]) - goal_y)
        if distance_to_goal <= self.cfg.distance_threshold:
            self.status = self.SETTLING
            return 0.0, 0.0, 0.0
        return body_x, body_y, angular_velocity

    def step(self) -> tuple[float, float, float]:
        """Advance the controller once; caller writes this into action[:, 19:22]."""

        if self.status in (self.IDLE, self.WAITING_FOR_PLACE_ACTIVATION, self.COMPLETE, self.FAILED):
            return 0.0, 0.0, 0.0
        if self.status == self.RETURNING_HOME:
            self.return_home_steps += 1
            if self.return_home_steps > int(self.cfg.max_return_home_seconds * self.control_hz):
                joint_name, max_error = self._home_error_summary()
                self.status = self.FAILED
                self.failure_reason = (
                    f"Carrying-home motion exceeded {self.cfg.max_return_home_seconds:.1f} seconds; "
                    f"largest remaining error is {joint_name}={max_error:.3f} "
                    f"(tolerance {self.cfg.home_joint_tolerance_rad:.3f}). "
                    "Lift the can clear of the cabinet before G, then retry with R."
                )
                return 0.0, 0.0, 0.0
            if self._home_is_settled():
                self.home_settled_steps += 1
            else:
                self.home_settled_steps = 0
            if self.home_settled_steps >= self.cfg.home_settle_steps:
                self._begin_navigation()
            elif self.return_home_steps % max(1, int(self.control_hz)) == 0:
                joint_name, max_error = self._home_error_summary()
                print(
                    "[Task525 AutoNav] Waiting for carrying-home pose: "
                    f"largest error {joint_name}={max_error:.3f} "
                    f"(tolerance {self.cfg.home_joint_tolerance_rad:.3f})."
                )
            return 0.0, 0.0, 0.0
        if self.status == self.LOWERING_LIFT:
            self.lift_lower_steps += 1
            if self.lift_lower_steps > int(self.cfg.max_lift_lower_seconds * self.control_hz):
                self.status = self.FAILED
                self.failure_reason = (
                    f"Lift lowering exceeded {self.cfg.max_lift_lower_seconds:.1f} seconds."
                )
                return 0.0, 0.0, 0.0
            if self._lift_is_settled():
                self.lift_settled_steps += 1
            else:
                self.lift_settled_steps = 0
            if self.lift_settled_steps >= self.cfg.lift_settle_steps:
                self.status = self.WAITING_FOR_PLACE_ACTIVATION
            return 0.0, 0.0, 0.0
        self.navigation_steps += 1
        if self.navigation_steps > int(self.cfg.max_navigation_seconds * self.control_hz):
            previous_status = self.status
            self.status = self.FAILED
            self.failure_reason = (
                f"Dijkstra navigation exceeded {self.cfg.max_navigation_seconds:.1f} seconds "
                f"while {previous_status}; last wheel norm={self.last_wheel_speed_norm:.3f} rad/s."
            )
            return 0.0, 0.0, 0.0

        if self.status == self.SETTLING:
            self.last_wheel_speed_norm = self._wheel_speed_norm()
            if self.last_wheel_speed_norm <= self.cfg.settle_wheel_speed_norm:
                self.settled_steps += 1
            else:
                self.settled_steps = 0
            if self.settled_steps >= self.cfg.settle_steps:
                self._begin_lift_lowering()
            return 0.0, 0.0, 0.0

        current_pose = self._root_pose_2d()
        self.last_pose_2d = current_pose
        self._replan_if_needed(current_pose)
        if self.status == self.FAILED:
            return 0.0, 0.0, 0.0
        self.last_command = self._fixed_yaw_command(current_pose)
        return self.last_command

    def apply(self, action: torch.Tensor) -> torch.Tensor:
        """Freeze carry joints and replace only the base tail while auto-nav is active."""

        if action.shape[-1] != 22:
            raise ValueError(f"Task525 online navigation requires 22D actions, got {tuple(action.shape)}")
        result = action.clone()
        result[:, 19:22] = result.new_zeros((result.shape[0], 3))
        if not self.holding_arm:
            return result
        if self.frozen_arm_action is None:
            raise RuntimeError("Online navigation is active without a frozen arm target.")
        if self.status == self.RETURNING_HOME:
            result[:, :19] = self._return_home_arm_action(result)
        elif self.status == self.LOWERING_LIFT:
            result[:, :19] = self._lift_lower_arm_action(result)
        else:
            result[:, :19] = self.frozen_arm_action.to(device=result.device, dtype=result.dtype)
        if self.active:
            result[:, 19:22] = result.new_tensor(self.step())
        return result
