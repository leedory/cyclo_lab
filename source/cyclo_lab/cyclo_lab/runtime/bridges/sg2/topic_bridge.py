"""Always-active FFW-SG2 bridge over ROS2-compatible Zenoh topics."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence

import torch

from cyclo_lab.robot_specs.ffw.sg2 import (
    BASE_BODY,
    BASE_FRAME,
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_ARM_ENABLE_TOPICS,
    FFW_SG2_ACTION_TOPICS,
    FFW_SG2_CAMERA_TOPICS,
    FFW_SG2_FOOT_SWITCH_MIDDLE_PEDAL_TOPIC,
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_JOINT_POSITION_LIMITS,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    JOINT_STATES_TOPIC,
    ODOM_FRAME,
    ODOM_TOPIC,
    SIMULATION_RESET_TOPIC,
    TF_TOPIC,
)
from cyclo_lab.runtime.publishers.articulation_state_publisher import ArticulationStatePublisher
from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers
from cyclo_lab.runtime.transport.ros2_zenoh import (
    BOOL,
    EMPTY,
    JOINT_TRAJECTORY,
    TWIST,
    UINT8,
    close_endpoints,
    create_publisher,
    create_subscriber,
    ros_domain_id,
)


DEFAULT_CMD_VEL_TIMEOUT_SECONDS = 0.1
DEFAULT_ACTIVATION_BLEND_SECONDS = 20.0
DEFAULT_CONTINUOUS_TRACKING_TIME_CONSTANT_SECONDS = 0.4
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
    hold_reset_joint_targets = False

    def __init__(
        self,
        env,
        *,
        camera_publish_hz: float | None = None,
        publish_odometry_tf: bool = True,
        subscribe_reset: bool = False,
        cmd_vel_timeout: float = DEFAULT_CMD_VEL_TIMEOUT_SECONDS,
        active_trajectory_groups: Sequence[str] | None = None,
        hold_reset_joint_targets: bool = False,
    ) -> None:
        self.env = env
        self.robot = env.scene["robot"]
        self.domain_id = ros_domain_id()
        self.publish_odometry_tf = publish_odometry_tf
        self.cmd_vel_timeout = float(cmd_vel_timeout)
        self.hold_reset_joint_targets = bool(hold_reset_joint_targets)
        full_joint_names = list(FFW_SG2_ACTION_JOINT_NAMES)
        self.published_joint_names = list(FFW_SG2_PUBLISHED_JOINT_NAMES)
        self.total_action_dim = env.action_manager.total_action_dim
        if self.total_action_dim == 16:
            self.joint_names = full_joint_names[:16]
            self.include_base_action = False
        elif self.total_action_dim == len(full_joint_names):
            self.joint_names = full_joint_names
            self.include_base_action = False
        elif self.total_action_dim == len(full_joint_names) + 3:
            self.joint_names = full_joint_names
            self.include_base_action = "base_action" in env.action_manager.active_terms
        else:
            raise ValueError(
                f"FFW-SG2 bridge expects 16D, 19D, or 22D actions, got {self.total_action_dim}D."
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
        self._activation_blocked_commands: dict[str, dict[str, float]] = {}
        # Monotonic arrival counters let a task-local controller distinguish a
        # newly toggled A3 leader from a command cached before an autonomous
        # safety hold.  They do not alter the original left/right topic map.
        self._trajectory_command_generation = {
            label: 0 for label in self._trajectory_commands
        }
        # A3 publishes UInt8(2) once per tact press. Unlike a trajectory
        # arrival count, this is an edge event and is safe for an explicit
        # task-local handoff after autonomous motion.
        self._arm_tact_generation = {"left": 0, "right": 0}
        # ROS/Zenoh callbacks run outside the simulation loop. Defer reading
        # articulation state and starting the blend until get_action().
        self._leader_activation_requested = threading.Event()
        self._latest_cmd_vel = (0.0, 0.0, 0.0)
        self._last_cmd_vel_time = 0.0
        self._activation_blend_anchor: dict[str, float] | None = None
        self._activation_blend_start_time = 0.0
        self._activation_blend_duration = 0.0
        self._smoothed_joint_targets: dict[str, float] | None = None
        self._tracking_filter_last_time = 0.0
        self._closed = False

        self.subscribers = [
            create_subscriber(
                topic=FFW_SG2_ACTION_TOPICS[label],
                msg_type=JOINT_TRAJECTORY,
                callback=lambda msg, command_label=label: self._on_joint_trajectory(command_label, msg),
            )
            for label in TRAJECTORY_COMMAND_GROUPS
        ]
        self.subscribers.extend(
            create_subscriber(
                topic=topic,
                msg_type=UINT8,
                callback=lambda msg, side=side: self._on_arm_enable(side, msg),
            )
            for side, topic in FFW_SG2_ARM_ENABLE_TOPICS.items()
        )
        self.subscribers.append(
            create_subscriber(
                topic=FFW_SG2_FOOT_SWITCH_MIDDLE_PEDAL_TOPIC,
                msg_type=BOOL,
                callback=self._on_middle_foot_switch,
            )
        )
        self.arm_enable_publishers = {
            side: create_publisher(topic, UINT8)
            for side, topic in FFW_SG2_ARM_ENABLE_TOPICS.items()
        }
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
        self.publishers = [
            *self.state_publisher.publishers,
            *self.camera_publishers.endpoints,
            *self.arm_enable_publishers.values(),
        ]

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
            if command == self._activation_blocked_commands.get(label):
                return
            self._activation_blocked_commands.pop(label, None)
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

    def _on_arm_enable(self, side: str, msg) -> None:
        """Record only a real A3 arm-tact toggle, never a stream update."""

        if side not in self._arm_tact_generation:
            raise KeyError(f"Unknown FFW-SG2 arm side: {side}")
        if msg is None or int(getattr(msg, "data", -1)) != 2:
            return
        with self._lock:
            self._arm_tact_generation[side] += 1
        self._leader_activation_requested.set()

    def arm_tact_generation(self, side: str) -> int:
        """Return the count of A3 tact presses observed for one arm."""

        if side not in self._arm_tact_generation:
            raise KeyError(f"Unknown FFW-SG2 arm side: {side}")
        with self._lock:
            return self._arm_tact_generation[side]

    def _on_middle_foot_switch(self, msg) -> None:
        """Start the leader blend on the middle-pedal press edge."""

        if msg is not None and bool(getattr(msg, "data", False)):
            self._leader_activation_requested.set()

    def left_arm_tact_generation(self) -> int:
        return self.arm_tact_generation("left")

    def right_arm_tact_generation(self) -> int:
        return self.arm_tact_generation("right")

    def set_arm_enabled(self, side: str, enabled: bool) -> None:
        """Set the A3 arm mode explicitly instead of relying on toggle history."""

        if side not in self.arm_enable_publishers:
            raise KeyError(f"Unknown FFW-SG2 arm side: {side}")
        # A3 canonical commands are 0=IDLE, 1=TELEOP, and 2=toggle.
        self.arm_enable_publishers[side].publish(data=1 if enabled else 0)

    def set_all_arms_idle(self) -> None:
        """Put both A3 arms in a known IDLE mode for autonomous ownership."""

        for side in ("left", "right"):
            self.set_arm_enabled(side, False)

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

    def latch_reset_joint_targets(self) -> None:
        """Keep this episode's reset targets as the inactive control command."""
        if not self.hold_reset_joint_targets:
            return
        positions = self.robot.data.joint_pos_target[0].detach().cpu().tolist()
        name_to_index = {name: index for index, name in enumerate(self.robot.data.joint_names)}
        targets = {
            name: self._clamp_joint(name, positions[name_to_index[name]])
            for name in self.joint_names
            if name in name_to_index
        }
        missing = set(self.joint_names) - set(targets)
        if missing:
            raise RuntimeError(f"SG2 reset targets missing joints: {sorted(missing)}")
        with self._lock:
            self._target_joint_state = targets
            self._smoothed_joint_targets = dict(targets)
            self._tracking_filter_last_time = time.monotonic()

    def _joint_targets(self) -> dict[str, float]:
        with self._lock:
            if self._target_joint_state is None:
                if self.hold_reset_joint_targets:
                    raise RuntimeError("SG2 reset targets were not latched before control")
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
            t = 1.0 if duration <= 0.0 else min(max(elapsed / duration, 0.0), 1.0)
            alpha = 6.0 * t**5 - 15.0 * t**4 + 10.0 * t**3
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

    def _smooth_joint_targets(self, targets: dict[str, float]) -> dict[str, float]:
        """Continuously low-pass all leader-controlled joint targets."""

        now = time.monotonic()
        with self._lock:
            previous = self._smoothed_joint_targets
            last_time = self._tracking_filter_last_time
            if previous is None or last_time <= 0.0:
                smoothed = dict(targets)
            else:
                elapsed = max(now - last_time, 0.0)
                tau = DEFAULT_CONTINUOUS_TRACKING_TIME_CONSTANT_SECONDS
                alpha = 1.0 if tau <= 0.0 else -math.expm1(-elapsed / tau)
                smoothed = {
                    name: previous.get(name, target) + alpha * (target - previous.get(name, target))
                    for name, target in targets.items()
                }
            self._smoothed_joint_targets = smoothed
            self._tracking_filter_last_time = now
            return dict(smoothed)

    def get_action(self) -> torch.Tensor:
        """Return continuously smoothed joint targets and optional base velocity."""
        if self._leader_activation_requested.is_set():
            self._leader_activation_requested.clear()
            self._begin_leader_activation_blend()
        targets = self._smooth_joint_targets(self._joint_targets())
        return self._make_action(targets, self._current_cmd_vel())

    def _begin_leader_activation_blend(
        self,
        transition_seconds: float = DEFAULT_ACTIVATION_BLEND_SECONDS,
    ) -> None:
        """Blend from the measured pose after a tact or foot-pedal activation."""

        if transition_seconds < 0.0:
            raise ValueError("SG2 leader transition must be non-negative.")
        current = self._read_current_joint_state()
        now = time.monotonic()
        with self._lock:
            self._activation_blend_anchor = dict(current)
            self._activation_blend_start_time = now
            self._activation_blend_duration = float(transition_seconds)
            self._smoothed_joint_targets = dict(current)
            self._tracking_filter_last_time = now

    def get_hold_action(self) -> torch.Tensor:
        """Hold latched reset targets or the current pose with zero base velocity."""
        with self._lock:
            if self._target_joint_state is None:
                if self.hold_reset_joint_targets:
                    raise RuntimeError("SG2 reset targets were not latched before hold")
                self._target_joint_state = self._read_current_joint_state()
            targets = dict(self._target_joint_state)
            self._smoothed_joint_targets = dict(targets)
            self._tracking_filter_last_time = time.monotonic()
        return self._make_action(targets, (0.0, 0.0, 0.0))

    def get_measured_joint_hold_action(self) -> torch.Tensor:
        """Hold the articulation's measured joint pose, not a cached leader target.

        Autonomous task transitions use this to avoid snapping a contacted
        gripper to an older A3 command when ownership changes hands.
        """

        targets = self._read_current_joint_state()
        with self._lock:
            self._smoothed_joint_targets = dict(targets)
            self._tracking_filter_last_time = time.monotonic()
        return self._make_action(targets, (0.0, 0.0, 0.0))

    def begin_control_activation(
        self,
        transition_seconds: float = DEFAULT_ACTIVATION_BLEND_SECONDS,
    ) -> None:
        """Enable live commands and blend them from the current reset/hold pose.

        WhiteRoom discards cached arm commands and blends only commands received
        after activation. Other SG2 tasks blend from the measured articulation pose.
        """
        if transition_seconds < 0.0:
            raise ValueError("SG2 activation transition must be non-negative.")
        if self.hold_reset_joint_targets:
            with self._lock:
                if self._target_joint_state is None:
                    raise RuntimeError("SG2 reset targets were not latched before activation")
                targets = dict(self._target_joint_state)
            blend_anchor = dict(targets)
        else:
            current = self._read_current_joint_state()
            targets = dict(current)
            blend_anchor = dict(current)
        if (
            not self.hold_reset_joint_targets
            and len(self.active_trajectory_groups) != len(TRAJECTORY_COMMAND_GROUPS)
        ):
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
        now = time.monotonic()
        with self._lock:
            self._target_joint_state = targets
            if self.hold_reset_joint_targets:
                for label in ("left_arm", "right_arm"):
                    self._trajectory_commands[label] = None
            # Hold head/lift until input changes from its pre-B value.
            self._activation_blocked_commands.clear()
            for label in ("head", "lift"):
                command = self._trajectory_commands[label]
                if command and label in self.active_trajectory_groups:
                    self._activation_blocked_commands[label] = dict(command)
                    self._trajectory_commands[label] = None
            self._activation_blend_anchor = blend_anchor
            self._activation_blend_start_time = now
            self._activation_blend_duration = float(transition_seconds)
            self._smoothed_joint_targets = dict(targets)
            self._tracking_filter_last_time = now

    def publish_observations(self) -> bool:
        """Publish state and return whether a due camera batch succeeded."""

        self.state_publisher.publish_all()
        return self.camera_publishers.publish()

    def clear_command_cache(self) -> None:
        self._leader_activation_requested.clear()
        with self._lock:
            self._target_joint_state = None
            for label in self._trajectory_commands:
                self._trajectory_commands[label] = None
                self._trajectory_command_generation[label] = 0
            self._activation_blocked_commands.clear()
            for side in self._arm_tact_generation:
                self._arm_tact_generation[side] = 0
            self._latest_cmd_vel = (0.0, 0.0, 0.0)
            self._last_cmd_vel_time = 0.0
            self._activation_blend_anchor = None
            self._activation_blend_start_time = 0.0
            self._activation_blend_duration = 0.0
            self._smoothed_joint_targets = None
            self._tracking_filter_last_time = 0.0
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
