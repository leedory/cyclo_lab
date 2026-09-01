"""Always-active FFW-SG2 bridge over ROS2-compatible Zenoh topics."""

from __future__ import annotations

import threading
import time
from collections.abc import Sequence

import torch

from cyclo_lab.robot_specs.ffw.sg2 import (
    BASE_BODY,
    BASE_FRAME,
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_JOINT_POSITION_LIMITS,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_ENABLE_TOPIC,
    JOINT_STATES_TOPIC,
    ODOM_FRAME,
    ODOM_TOPIC,
    SIMULATION_RESET_TOPIC,
    TF_TOPIC,
)
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers
from cyclo_lab.runtime.transport.ros2_zenoh import (
    EMPTY,
    JOINT_TRAJECTORY,
    TWIST,
    UINT8,
    close_endpoints,
    create_subscriber,
    ros_domain_id,
)


DEFAULT_CMD_VEL_TIMEOUT_SECONDS = 0.1
DEFAULT_ACTIVATION_BLEND_SECONDS = 0.5
TRAJECTORY_COMMAND_GROUPS = ("left_arm", "right_arm", "head", "lift")
TRAJECTORY_COMMAND_JOINT_NAMES = {
    "left_arm": (*FFW_SG2_LEFT_ARM_JOINT_NAMES, *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES),
    "right_arm": (*FFW_SG2_RIGHT_ARM_JOINT_NAMES, *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES),
    "head": FFW_SG2_HEAD_JOINT_NAMES,
    "lift": FFW_SG2_LIFT_JOINT_NAMES,
}


class FFWSG2TopicBridge:
    """Translate SG2 command topics into Isaac Lab actions and publish simulation state."""

    requires_activation = True

    def __init__(
        self,
        env,
        *,
        camera_publish_hz: float | None = None,
        publish_odometry_tf: bool = True,
        subscribe_reset: bool = False,
        cmd_vel_timeout: float = DEFAULT_CMD_VEL_TIMEOUT_SECONDS,
        active_trajectory_groups: Sequence[str] | None = None,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.domain_id = ros_domain_id()
        self.publish_odometry_tf = publish_odometry_tf
        self.cmd_vel_timeout = float(cmd_vel_timeout)
        self.joint_names = list(FFW_SG2_ACTION_JOINT_NAMES)
        self.published_joint_names = list(FFW_SG2_PUBLISHED_JOINT_NAMES)
        self.total_action_dim = env.action_manager.total_action_dim
        self.include_base_action = (
            "base_action" in env.action_manager.active_terms
            and self.total_action_dim == len(self.joint_names) + 3
        )
        if self.total_action_dim not in (len(self.joint_names), len(self.joint_names) + 3):
            raise ValueError(
                f"FFW-SG2 bridge expects 19D or 22D actions, got {self.total_action_dim}D."
            )

        requested_groups = (
            TRAJECTORY_COMMAND_GROUPS
            if active_trajectory_groups is None
            else tuple(active_trajectory_groups)
        )
        unknown_groups = set(requested_groups) - set(TRAJECTORY_COMMAND_GROUPS)
        if unknown_groups:
            raise ValueError(
                "Unknown FFW-SG2 trajectory command groups: "
                f"{sorted(unknown_groups)}"
            )
        if len(requested_groups) != len(set(requested_groups)):
            raise ValueError("FFW-SG2 trajectory command groups must be unique.")
        self.active_trajectory_groups = tuple(requested_groups)

        self._lock = threading.Lock()
        self._reset_requested = threading.Event()
        self._target_joint_state: dict[str, float] | None = None
        self._trajectory_commands: dict[str, dict[str, float] | None] = {
            label: None for label in TRAJECTORY_COMMAND_GROUPS
        }
        # Monotonic arrival counters let a task-local controller distinguish a
        # newly toggled A3 leader from a command cached before an autonomous
        # safety hold.  They do not alter the original left/right topic map.
        self._trajectory_command_generation = {
            label: 0 for label in self._trajectory_commands
        }
        # A3 publishes UInt8(2) once per tact press. Unlike a trajectory
        # arrival count, this is an edge event and is safe for an explicit
        # task-local handoff after autonomous motion.
        self._right_arm_tact_generation = 0
        self._latest_cmd_vel = (0.0, 0.0, 0.0)
        self._last_cmd_vel_time = 0.0
        self._activation_blend_anchor: dict[str, float] | None = None
        self._activation_blend_start_time = 0.0
        self._activation_blend_duration = 0.0
        self._closed = False

        self.subscribers = [
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS[label],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg, command_label=label: self._on_joint_trajectory(command_label, msg),
            )
            for label in TRAJECTORY_COMMAND_GROUPS
        ]
        self.subscribers.append(
            create_subscriber(
                topic=FFW_SG2_RIGHT_ARM_ENABLE_TOPIC,
                msg_type=UINT8,
                callback=self._on_right_arm_enable,
            )
        )
        if self.include_base_action:
            self.subscribers.append(
                create_subscriber(
                    topic=FFW_SG2_ACTION_TOPICS["mobile"],
                    msg_type=TWIST,
                    callback=self._on_cmd_vel,
                )
            )
        if subscribe_reset:
            self.subscribers.append(
                create_subscriber(
                    topic=SIMULATION_RESET_TOPIC,
                    msg_type=EMPTY,
                    callback=self._on_reset,
                )
            )

        self.state_publisher = ArticulationStatePublisher(
            self.robot,
            joint_names=self.published_joint_names,
            joint_states_topic=JOINT_STATES_TOPIC,
            base_frame=BASE_FRAME,
            base_body=BASE_BODY,
            odom_topic=ODOM_TOPIC if publish_odometry_tf else None,
            tf_topic=TF_TOPIC if publish_odometry_tf else None,
            odom_frame=ODOM_FRAME,
        )

        self.camera_publishers = CompressedCameraPublishers(
            env.scene,
            FFW_SG2_CAMERA_TOPICS,
            camera_publish_hz,
        )
        self.publishers = [*self.state_publisher.publishers, *self.camera_publishers.endpoints]

        print(
            f"[Zenoh ROS2] FFW-SG2 topic bridge ready: {self.total_action_dim}D "
            f"({'joint+base' if self.include_base_action else 'joint-only'}), "
            f"active trajectory groups={self.active_trajectory_groups}, "
            f"ROS_DOMAIN_ID={self.domain_id}"
        )

    def _clamp_joint(self, name: str, value: float) -> float:
        lower, upper = FFW_SG2_JOINT_POSITION_LIMITS[name]
        return min(max(float(value), lower), upper)

    def _on_joint_trajectory(self, label: str, msg) -> None:
        if msg is None or not msg.points:
            return
        command = {
            name: self._clamp_joint(name, position)
            for name, position in zip(msg.joint_names, msg.points[-1].positions)
            if name in FFW_SG2_JOINT_POSITION_LIMITS
        }
        if not command:
            return
        with self._lock:
            cached = self._trajectory_commands[label] or {}
            cached.update(command)
            self._trajectory_commands[label] = cached
            self._trajectory_command_generation[label] += 1

    def trajectory_command_generation(self, label: str) -> int:
        """Return the count of valid trajectory messages received for one control group."""

        if label not in self._trajectory_command_generation:
            raise KeyError(f"Unknown FFW-SG2 trajectory group: {label}")
        with self._lock:
            return self._trajectory_command_generation[label]

    def _on_right_arm_enable(self, msg) -> None:
        """Record only a real A3 right-tact toggle, never a stream update."""

        if msg is None or int(getattr(msg, "data", -1)) != 2:
            return
        with self._lock:
            self._right_arm_tact_generation += 1

    def right_arm_tact_generation(self) -> int:
        """Return the count of A3 right-tact presses observed by this bridge."""

        with self._lock:
            return self._right_arm_tact_generation

    def _on_cmd_vel(self, msg) -> None:
        if msg is None:
            return
        with self._lock:
            self._latest_cmd_vel = (
                float(msg.linear.x),
                float(msg.linear.y),
                float(msg.angular.z),
            )
            self._last_cmd_vel_time = time.monotonic()

    def _on_reset(self, _msg) -> None:
        self._reset_requested.set()

    def consume_reset_request(self) -> bool:
        """Return and clear the reset request without resetting from the callback thread."""
        if not self._reset_requested.is_set():
            return False
        self._reset_requested.clear()
        return True

    def _read_current_joint_state(self) -> dict[str, float]:
        positions = self.robot.data.joint_pos[0].detach().cpu().tolist()
        name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        return {
            name: self._clamp_joint(name, positions[name_to_index[name]])
            for name in self.joint_names
            if name in name_to_index
        }

    def _read_default_joint_state(self) -> dict[str, float]:
        """Return the environment's configured absolute reset targets."""

        positions = self.robot.data.default_joint_pos[0].detach().cpu().tolist()
        name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        return {
            name: self._clamp_joint(name, positions[name_to_index[name]])
            for name in self.joint_names
            if name in name_to_index
        }

    def _joint_targets(self) -> dict[str, float]:
        with self._lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
            for label in self.active_trajectory_groups:
                command = self._trajectory_commands[label]
                if command:
                    self._target_joint_state.update(command)
            targets = dict(self._target_joint_state)
            anchor = self._activation_blend_anchor
            if anchor is None:
                return targets

            duration = self._activation_blend_duration
            elapsed = time.monotonic() - self._activation_blend_start_time
            alpha = 1.0 if duration <= 0.0 else min(max(elapsed / duration, 0.0), 1.0)
            if alpha >= 1.0:
                self._activation_blend_anchor = None
                return targets
            return {
                name: anchor[name] + alpha * (targets[name] - anchor[name])
                for name in self.joint_names
            }

    def _current_cmd_vel(self) -> tuple[float, float, float]:
        with self._lock:
            command = self._latest_cmd_vel
            last_command_time = self._last_cmd_vel_time
        if last_command_time == 0.0:
            return 0.0, 0.0, 0.0
        if self.cmd_vel_timeout > 0.0 and time.monotonic() - last_command_time > self.cmd_vel_timeout:
            return 0.0, 0.0, 0.0
        return command

    def _make_action(
        self,
        targets: dict[str, float],
        base_command: tuple[float, float, float],
    ) -> torch.Tensor:
        values = [targets[name] for name in self.joint_names]
        if self.include_base_action:
            values.extend(base_command)
        return torch.tensor(values, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def get_action(self) -> torch.Tensor:
        """Return the latest absolute joint targets and optional base velocity."""
        return self._make_action(self._joint_targets(), self._current_cmd_vel())

    def get_hold_action(self) -> torch.Tensor:
        """Hold the current joint pose and command zero base velocity."""
        with self._lock:
            if self._target_joint_state is None:
                self._target_joint_state = self._read_current_joint_state()
            targets = dict(self._target_joint_state)
        return self._make_action(targets, (0.0, 0.0, 0.0))

    def get_measured_joint_hold_action(self) -> torch.Tensor:
        """Hold the articulation's measured joint pose, not a cached leader target.

        Autonomous task transitions use this to avoid snapping a contacted
        gripper to an older A3 command when ownership changes hands.
        """

        return self._make_action(self._read_current_joint_state(), (0.0, 0.0, 0.0))

    def begin_control_activation(
        self,
        transition_seconds: float = DEFAULT_ACTIVATION_BLEND_SECONDS,
    ) -> None:
        """Blend from the settled simulated pose into the absolute live command.

        Leader topics are subscribed while control is inactive, so a cached
        position can differ slightly from the settled simulated pose. Starting
        from the current articulation state avoids an impulse when B enables
        those cached absolute commands.
        """
        if transition_seconds < 0.0:
            raise ValueError("SG2 activation transition must be non-negative.")
        current = self._read_current_joint_state()
        targets = dict(current)
        blend_anchor = dict(current)
        if len(self.active_trajectory_groups) != len(TRAJECTORY_COMMAND_GROUPS):
            defaults = self._read_default_joint_state()
            inactive_groups = set(TRAJECTORY_COMMAND_GROUPS) - set(self.active_trajectory_groups)
            for label in inactive_groups:
                for name in TRAJECTORY_COMMAND_JOINT_NAMES[label]:
                    if name in defaults:
                        targets[name] = defaults[name]
                        # The recorder was already commanding this absolute
                        # reset target before B. Excluding inactive joints
                        # from the activation blend keeps that command
                        # continuous across the ownership transition.
                        blend_anchor[name] = defaults[name]
        with self._lock:
            self._target_joint_state = targets
            self._activation_blend_anchor = blend_anchor
            self._activation_blend_start_time = time.monotonic()
            self._activation_blend_duration = float(transition_seconds)

    def publish_observations(self) -> None:
        self.state_publisher.publish_all()
        self.camera_publishers.publish()

    def clear_command_cache(self) -> None:
        with self._lock:
            self._target_joint_state = None
            for label in self._trajectory_commands:
                self._trajectory_commands[label] = None
                self._trajectory_command_generation[label] = 0
            self._right_arm_tact_generation = 0
            self._latest_cmd_vel = (0.0, 0.0, 0.0)
            self._last_cmd_vel_time = 0.0
            self._activation_blend_anchor = None
            self._activation_blend_start_time = 0.0
            self._activation_blend_duration = 0.0
        self.state_publisher.reset_odom_origin()

    def reset(self) -> None:
        self._reset_requested.clear()
        self.clear_command_cache()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        close_endpoints(self.subscribers)
        close_endpoints(self.publishers)
        print("[Zenoh ROS2] FFW-SG2 topic bridge closed.")
