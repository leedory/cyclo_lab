"""Online G-key automation for two-can coffee seed acquisition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import torch

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_JOINT_POSITION_LIMITS,
)

from .coffee_transport_common import (
    CABINET_RIGHT_DELTA_Y_M,
    COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
    LIFT_TRAVEL_COMMAND_M,
    SHELF_LEVEL_DELTA_Z_M,
    SOURCE_SPAWN_SQUARES,
    SOURCE_UPPER_SUPPORT_Z_M,
)
from .generation_contract import COFFEE_CAN_CYCLES, COFFEE_TRANSPORT_PHASES
from .quality_contract import (
    DEFAULT_PLACEMENT_QUALITY,
    evaluate_placement,
    upright_cosine_from_wxyz,
)


class SeedAutomationState(str, Enum):
    IDLE = "idle"
    OPERATOR_PICK_AND_EXTRACT = "operator_pick_and_extract"
    WAITING_FOR_OPERATOR_TACT = "waiting_for_operator_tact"
    RETURNING_HOME = "returning_home"
    AXIS_TO_DESTINATION = "axis_to_destination"
    INVERSE_CARRY_CLOSED = "inverse_carry_closed"
    RELEASE = "release"
    INVERSE_APPROACH_OPEN = "inverse_approach_open"
    VALIDATE_PLACEMENT = "validate_placement"
    AXIS_TO_SOURCE = "axis_to_source"
    TASK_COMPLETE = "task_complete"
    FAILED = "failed"


@dataclass(frozen=True)
class CoffeeSeedAutomationCfg:
    control_hz: float = 15.0
    return_home_blend_seconds: float = 2.0
    return_home_timeout_seconds: float = 20.0
    home_joint_tolerance_rad: float = 0.08
    home_stable_steps: int = 10
    operator_arm_max_delta_rad: float = 0.08
    operator_gripper_max_delta_rad: float = 0.12
    lift_timeout_seconds: float = 8.0
    lift_tolerance_m: float = 0.006
    base_linear_gain: float = 1.5
    base_angular_gain: float = 2.0
    base_linear_max_mps: float = 0.10
    base_angular_max_radps: float = 0.25
    base_position_tolerance_m: float = 0.010
    base_yaw_tolerance_rad: float = math.radians(1.0)
    base_timeout_seconds: float = 15.0
    axis_stable_steps: int = 5
    release_dwell_seconds: float = 0.5
    placement_return_wait_seconds: float = 1.0
    post_release_home_timeout_seconds: float = 5.0
    post_release_home_stable_steps: int = 5
    minimum_operator_frames: int = 15
    held_can_distance_abort_m: float = 0.140


class CoffeeTransportSeedAutomation:
    """Keep teleoperation to pickup/extract and own every post-G action."""

    def __init__(
        self,
        env,
        *,
        task_id: str,
        cfg: CoffeeSeedAutomationCfg = CoffeeSeedAutomationCfg(),
    ) -> None:
        if task_id not in ("000001", "000002"):
            raise ValueError(f"Unsupported coffee transport task: {task_id}")
        self.env = env
        self.task_id = task_id
        self.cfg = cfg
        self.robot = env.scene["robot"]
        self._joint_ids = torch.tensor(
            [self.robot.joint_names.index(name) for name in FFW_SG2_ACTION_JOINT_NAMES],
            dtype=torch.long,
            device=env.device,
        )
        self._action_index = {
            name: index for index, name in enumerate(FFW_SG2_ACTION_JOINT_NAMES)
        }
        self._head_indices = [
            self._action_index["head_joint1"],
            self._action_index["head_joint2"],
        ]
        self._lift_index = self._action_index["lift_joint"]
        self._arm_indices = {
            side: [
                index
                for index, name in enumerate(FFW_SG2_ACTION_JOINT_NAMES)
                if name.startswith(f"arm_{side[0]}_")
                or name == f"gripper_{side[0]}_joint1"
            ]
            for side in ("left", "right")
        }
        self._gripper_index = {
            side: self._action_index[f"gripper_{side[0]}_joint1"]
            for side in ("left", "right")
        }
        self._home_settle_indices = [
            index
            for index in range(19)
            if index not in self._gripper_index.values()
        ]
        self.reset()

    @property
    def active(self) -> bool:
        return self.state not in (
            SeedAutomationState.IDLE,
            SeedAutomationState.OPERATOR_PICK_AND_EXTRACT,
            SeedAutomationState.TASK_COMPLETE,
            SeedAutomationState.FAILED,
        )

    @property
    def task_complete(self) -> bool:
        return self.state is SeedAutomationState.TASK_COMPLETE

    @property
    def failed(self) -> bool:
        return self.state is SeedAutomationState.FAILED

    @property
    def operator_enabled(self) -> bool:
        return self.state is SeedAutomationState.OPERATOR_PICK_AND_EXTRACT

    @property
    def awaiting_operator_tact(self) -> bool:
        return self.state is SeedAutomationState.WAITING_FOR_OPERATOR_TACT

    def enable_operator_cycle(self, *, recorded_step_index: int) -> bool:
        """Transfer the active arm immediately after a fresh arm-specific tact."""

        if not self.awaiting_operator_tact:
            return False
        cycle = self.current_cycle
        assert cycle is not None
        assert self._held_action is not None
        anchor = self._held_action.detach().clone()
        self._operator_previous_action = anchor
        self._operator_actions = [anchor]
        self._operator_start_step = int(recorded_step_index)
        self._markers[f"cycle_{cycle.order}_tact_step"] = int(recorded_step_index)
        self._markers[f"cycle_{cycle.order}_operator_enabled_step"] = int(
            recorded_step_index
        )
        self._enter(SeedAutomationState.OPERATOR_PICK_AND_EXTRACT)
        print(
            f"[CoffeeSeed] {cycle.arm.title()} tact received; control is enabled "
            f"for {cycle.can_name}. Pick, extract, then press G."
        )
        return True

    @property
    def current_cycle(self):
        if self.cycle_index >= len(COFFEE_CAN_CYCLES):
            return None
        return COFFEE_CAN_CYCLES[self.cycle_index]

    def reset(self) -> None:
        self.state = SeedAutomationState.IDLE
        self.cycle_index = 0
        self.failure_reason = ""
        self._g_requested = False
        self._operator_actions: list[torch.Tensor] = []
        self._operator_previous_action: torch.Tensor | None = None
        self._operator_start_step: int | None = None
        self._operator_gripper_input_baseline: float | None = None
        self._markers: dict[str, int] = {}
        self._nominal_action: torch.Tensor | None = None
        self._held_action: torch.Tensor | None = None
        self._safe_hold_action: torch.Tensor | None = None
        self._return_start_action: torch.Tensor | None = None
        self._carry_to_home_actions: list[torch.Tensor] = []
        self._closed_gripper_value = 0.0
        self._carry_reverse: list[torch.Tensor] = []
        self._approach_reverse: list[torch.Tensor] = []
        self._replay_index = 0
        self._state_steps = 0
        self._stable_steps = 0
        self._axis_start_value = 0.0
        self._axis_source_value = 0.0
        self._axis_destination_value = 0.0

    def begin_episode(self, *, recorded_step_index: int = 0) -> None:
        self.reset()
        self._nominal_action = self.robot.data.default_joint_pos[
            :, self._joint_ids
        ].detach().clone()
        self._held_action = self._nominal_action.detach().clone()
        self._markers["normalize_start_step"] = int(recorded_step_index)
        self._markers["normalize_complete_step"] = int(recorded_step_index)
        self._begin_operator_cycle(recorded_step_index)

    def request_g(self) -> None:
        if self.state is not SeedAutomationState.OPERATOR_PICK_AND_EXTRACT:
            print(f"[CoffeeSeed] G ignored while state={self.state.value}.")
            return
        self._g_requested = True

    def metadata(self) -> dict[str, object]:
        return {
            "coffee_seed_mode": (
                "two_operator_pickups_tact_immediate_g_automatic_place_v4"
                if self.task_id == "000001"
                else "two_operator_pickups_tact_immediate_g_automatic_place_v3"
            ),
            "coffee_task_id": self.task_id,
            "coffee_completed_cycles": self.cycle_index,
            "coffee_failure_reason": self.failure_reason,
            "coffee_phase_contract_version": 7 if self.task_id == "000001" else 6,
            "coffee_task_axis_action_contract": (
                "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1"
                if self.task_id == "000001"
                else "base_body_velocity_feedback_v1"
            ),
            "coffee_phase_count": len(COFFEE_TRANSPORT_PHASES),
            "coffee_operator_arm_max_delta_rad_per_step": self.cfg.operator_arm_max_delta_rad,
            "coffee_operator_gripper_max_delta_rad_per_step": self.cfg.operator_gripper_max_delta_rad,
            **self._markers,
        }

    def final_placement_failures(self) -> dict[str, tuple[str, ...]]:
        """Re-evaluate every can before the recorder accepts ``N``."""

        failures: dict[str, tuple[str, ...]] = {}
        for cycle in COFFEE_CAN_CYCLES:
            decision = self._placement_decision(cycle)
            if not decision.passed:
                failures[cycle.can_name] = decision.failures
                square = next(
                    item for item in SOURCE_SPAWN_SQUARES
                    if item.name == cycle.can_name
                ).translated(
                    dz_m=SHELF_LEVEL_DELTA_Z_M
                    if self.task_id == "000001"
                    else 0.0,
                    dy_m=CABINET_RIGHT_DELTA_Y_M
                    if self.task_id == "000002"
                    else 0.0,
                )
                position = (
                    self.env.scene[cycle.can_name].data.root_pos_w[0]
                    - self.env.scene.env_origins[0]
                )
                tolerance = (
                    DEFAULT_PLACEMENT_QUALITY.center_position_tolerance_m
                )
                print(
                    f"[CoffeeSeed] {cycle.can_name} final center "
                    f"xy=({float(position[0]):.5f}, "
                    f"{float(position[1]):.5f}) m; "
                    f"accepted x=[{square.x_min_m - tolerance:.5f}, "
                    f"{square.x_max_m + tolerance:.5f}], "
                    f"y=[{square.y_min_m - tolerance:.5f}, "
                    f"{square.y_max_m + tolerance:.5f}] m."
                )
        return failures

    def apply(self, action: torch.Tensor, *, recorded_step_index: int) -> torch.Tensor:
        if action.ndim != 2 or action.shape[0] != 1:
            raise ValueError(
                "Coffee seed automation currently requires one environment action."
            )
        expected_dim = 19 if self.task_id == "000001" else 22
        if action.shape[1] != expected_dim:
            raise ValueError(
                f"Task{self.task_id} requires {expected_dim}D actions, "
                f"got {tuple(action.shape)}."
            )
        if self.state is SeedAutomationState.IDLE:
            return action
        if self.state in (SeedAutomationState.FAILED, SeedAutomationState.TASK_COMPLETE):
            return self._safe_hold(action)
        if self.state is SeedAutomationState.WAITING_FOR_OPERATOR_TACT:
            return self._automatic_joint_action(action)

        if self.state is SeedAutomationState.OPERATOR_PICK_AND_EXTRACT:
            result = self._mask_operator_action(action)
            self._record_operator_frame(result)
            if self._g_requested:
                self._g_requested = False
                self._accept_g_or_fail(recorded_step_index)
                if self.failed:
                    return self._safe_hold(result)
            if self.state is SeedAutomationState.OPERATOR_PICK_AND_EXTRACT:
                return result
            action = result

        if self.state is SeedAutomationState.RETURNING_HOME:
            return self._apply_return_home(action, recorded_step_index)

        if self.state is SeedAutomationState.AXIS_TO_DESTINATION:
            if not self._held_can_guard_passes():
                return self._safe_hold(action)
            return self._apply_axis_motion(action, recorded_step_index)
        if self.state is SeedAutomationState.AXIS_TO_SOURCE:
            return self._apply_axis_motion(action, recorded_step_index)
        if self.state is SeedAutomationState.INVERSE_CARRY_CLOSED:
            if not self._held_can_guard_passes():
                return self._safe_hold(action)
            return self._apply_replay(
                action,
                closed=True,
                recorded_step_index=recorded_step_index,
            )
        if self.state is SeedAutomationState.RELEASE:
            return self._apply_release(action, recorded_step_index)
        if self.state is SeedAutomationState.INVERSE_APPROACH_OPEN:
            return self._apply_replay(action, closed=False, recorded_step_index=recorded_step_index)
        if self.state is SeedAutomationState.VALIDATE_PLACEMENT:
            return self._apply_placement_gate(action, recorded_step_index)
        raise RuntimeError(f"Unhandled coffee automation state: {self.state}")

    def _measured_joint_action(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self._joint_ids].detach().clone()

    def _begin_operator_cycle(self, recorded_step_index: int) -> None:
        cycle = self.current_cycle
        if cycle is None:
            self.state = SeedAutomationState.TASK_COMPLETE
            self._markers["task_complete_step"] = int(recorded_step_index)
            print("[CoffeeSeed] Two cans complete. Press N to validate and save.")
            return
        self._operator_actions = []
        self._operator_previous_action = None
        self._operator_start_step = None
        self._operator_gripper_input_baseline = None
        self._stable_steps = 0
        self._markers[f"cycle_{cycle.order}_start_step"] = int(recorded_step_index)
        self.state = SeedAutomationState.WAITING_FOR_OPERATOR_TACT
        print(
            f"[CoffeeSeed] Cycle {cycle.order + 1}/{len(COFFEE_CAN_CYCLES)} ready at HOME. "
            f"Press the {cycle.arm} A3 tact once to enable that arm immediately."
        )

    def _mask_operator_action(self, action: torch.Tensor) -> torch.Tensor:
        assert self._nominal_action is not None
        assert self._operator_previous_action is not None
        cycle = self.current_cycle
        assert cycle is not None
        result = action.clone()
        inactive = "left" if cycle.arm == "right" else "right"
        result[:, self._arm_indices[inactive]] = self._nominal_action[
            :, self._arm_indices[inactive]
        ]
        result[:, self._head_indices] = self._nominal_action[:, self._head_indices]
        result[:, self._lift_index] = self._nominal_action[:, self._lift_index]
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0

        gripper_index = self._gripper_index[cycle.arm]
        if self._operator_gripper_input_baseline is None:
            self._operator_gripper_input_baseline = float(
                action[0, gripper_index].item()
            )
        gripper_name = FFW_SG2_ACTION_JOINT_NAMES[gripper_index]
        lower, upper = FFW_SG2_JOINT_POSITION_LIMITS[gripper_name]
        result[:, gripper_index] = torch.clamp(
            self._nominal_action[:, gripper_index]
            + action[:, gripper_index]
            - self._operator_gripper_input_baseline,
            min=lower,
            max=upper,
        )

        arm_joint_indices = [
            index
            for index in self._arm_indices[cycle.arm]
            if index != gripper_index
        ]
        previous = self._operator_previous_action.to(
            device=result.device, dtype=result.dtype
        )
        arm_delta = self.cfg.operator_arm_max_delta_rad
        result[:, arm_joint_indices] = torch.maximum(
            torch.minimum(
                result[:, arm_joint_indices],
                previous[:, arm_joint_indices] + arm_delta,
            ),
            previous[:, arm_joint_indices] - arm_delta,
        )
        gripper_delta = self.cfg.operator_gripper_max_delta_rad
        result[:, gripper_index] = torch.clamp(
            result[:, gripper_index],
            min=float(previous[0, gripper_index].item()) - gripper_delta,
            max=float(previous[0, gripper_index].item()) + gripper_delta,
        )
        self._operator_previous_action = result[:, :19].detach().clone()
        return result

    def _record_operator_frame(self, action: torch.Tensor) -> None:
        self._operator_actions.append(action[:, :19].detach().clone())

    def _infer_grasp_step(self) -> int:
        cycle = self.current_cycle
        assert cycle is not None
        index = self._gripper_index[cycle.arm]
        values = torch.stack(self._operator_actions)[:, 0, index]
        prefix = values[: min(10, len(values))]
        open_value = torch.median(prefix)
        lower, upper = torch.min(values), torch.max(values)
        closed_value = (
            upper
            if torch.abs(upper - open_value) >= torch.abs(lower - open_value)
            else lower
        )
        excursion = closed_value - open_value
        if torch.abs(excursion) < 0.1:
            raise ValueError("active gripper has no clear close excursion")
        threshold = open_value + 0.9 * excursion
        closed = values >= threshold if excursion > 0 else values <= threshold
        stable = 3
        for step in range(len(values) - stable + 1):
            if bool(torch.all(closed[step : step + stable]).item()):
                return step
        raise ValueError("active gripper never reached a stable closed plateau")

    def _accept_g_or_fail(self, recorded_step_index: int) -> None:
        cycle = self.current_cycle
        assert cycle is not None
        if len(self._operator_actions) < self.cfg.minimum_operator_frames:
            self._fail("G pressed before enough pickup/extract frames were recorded")
            return
        try:
            grasp_step = self._infer_grasp_step()
        except ValueError as exc:
            self._fail(str(exc))
            return
        assert self._operator_start_step is not None
        self._markers[f"cycle_{cycle.order}_grasp_step"] = (
            self._operator_start_step + grasp_step
        )
        self._markers[f"cycle_{cycle.order}_g_step"] = int(recorded_step_index)
        self._return_start_action = self._measured_joint_action()
        self._closed_gripper_value = float(
            self._operator_actions[-1][0, self._gripper_index[cycle.arm]].item()
        )
        self._held_action = self._nominal_action.detach().clone()
        self._held_action[
            :, self._gripper_index[cycle.arm]
        ] = self._closed_gripper_value
        self._carry_to_home_actions = list(self._operator_actions[grasp_step:])
        self._approach_reverse = list(reversed(self._operator_actions[:grasp_step]))
        self._enter(SeedAutomationState.RETURNING_HOME)
        print(
            f"[CoffeeSeed] G accepted for {cycle.can_name}; "
            "body-carry gate bypassed. Returning to HOME and waiting for "
            "a stable hold."
        )

    def _held_can_guard_passes(self) -> bool:
        cycle = self.current_cycle
        assert cycle is not None
        can_position = self.env.scene[cycle.can_name].data.root_pos_w[0]
        distance = float(
            torch.linalg.vector_norm(
                self._active_eef_position(cycle.arm) - can_position
            ).item()
        )
        if distance <= self.cfg.held_can_distance_abort_m:
            return True
        self._fail(
            f"held can separated from {cycle.arm} EEF "
            f"({distance:.3f} m > {self.cfg.held_can_distance_abort_m:.3f} m)"
        )
        return False

    def _active_eef_position(self, side: str) -> torch.Tensor:
        sensor = self.env.scene["left_eef" if side == "left" else "right_eef"]
        return sensor.data.target_pos_w[0, 0, :]

    def _enter(self, state: SeedAutomationState) -> None:
        self.state = state
        self._state_steps = 0
        self._stable_steps = 0
        self._replay_index = 0

    def _safe_hold(self, action: torch.Tensor) -> torch.Tensor:
        result = action.clone()
        if self._safe_hold_action is None:
            source = self._held_action
            if source is None:
                source = self._measured_joint_action()
            self._safe_hold_action = source.detach().clone()
        result[:, :19] = self._safe_hold_action.to(
            device=result.device, dtype=result.dtype
        )
        result[:, self._head_indices] = self._nominal_action[:, self._head_indices]
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0
        return result

    def _apply_return_home(
        self,
        action: torch.Tensor,
        recorded_step_index: int,
    ) -> torch.Tensor:
        """Blend to the fixed episode HOME, then require a measured stable hold."""

        assert self._return_start_action is not None
        assert self._held_action is not None
        result = action.clone()
        self._state_steps += 1
        blend_steps = max(
            1, round(self.cfg.return_home_blend_seconds * self.cfg.control_hz)
        )
        alpha = min(1.0, self._state_steps / blend_steps)
        start = self._return_start_action.to(device=result.device, dtype=result.dtype)
        target = self._held_action.to(device=result.device, dtype=result.dtype)
        result[:, :19] = start + alpha * (target - start)
        result[:, self._head_indices] = self._nominal_action[:, self._head_indices]
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0

        if self._state_steps <= blend_steps:
            self._carry_to_home_actions.append(result[:, :19].detach().clone())

        measured = self._measured_joint_action()[0]
        home = self._held_action[0]
        settled = bool(
            torch.all(
                torch.abs(
                    measured[self._home_settle_indices]
                    - home[self._home_settle_indices]
                )
                <= self.cfg.home_joint_tolerance_rad
            ).item()
        )
        self._stable_steps = self._stable_steps + 1 if settled else 0
        if self._stable_steps >= self.cfg.home_stable_steps:
            cycle = self.current_cycle
            assert cycle is not None
            self._markers[f"cycle_{cycle.order}_home_stable_step"] = int(
                recorded_step_index
            )
            self._carry_reverse = list(reversed(self._carry_to_home_actions))
            self._begin_axis_motion(destination=True)
            print(
                f"[CoffeeSeed] HOME stable for {cycle.can_name}; starting "
                f"{'lift' if self.task_id == '000001' else 'base'} motion."
            )
        elif self._state_steps > round(
            self.cfg.return_home_timeout_seconds * self.cfg.control_hz
        ):
            self._fail("returning_home timed out before the fixed pose settled")
        return result

    def _automatic_joint_action(self, action: torch.Tensor) -> torch.Tensor:
        assert self._held_action is not None
        result = action.clone()
        result[:, :19] = self._held_action
        result[:, self._head_indices] = self._nominal_action[:, self._head_indices]
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0
        return result

    def _current_axis_scalar(self) -> float:
        if self.task_id == "000001":
            return float(self.robot.data.joint_pos[0, self._joint_ids[self._lift_index]].item())
        root = self.robot.data.root_pos_w[0]
        home = self.env._coffee_transport_episode_home_root_pose_w[0]
        return float(root[1] - home[1])

    def _begin_axis_motion(self, *, destination: bool) -> None:
        self._enter(
            SeedAutomationState.AXIS_TO_DESTINATION
            if destination
            else SeedAutomationState.AXIS_TO_SOURCE
        )
        self._axis_start_value = self._current_axis_scalar()
        if destination:
            if self.task_id == "000001":
                self._axis_source_value = 0.0
                self._axis_destination_value = LIFT_TRAVEL_COMMAND_M
            else:
                self._axis_source_value = self._axis_start_value
                self._axis_destination_value = CABINET_RIGHT_DELTA_Y_M

    def _apply_axis_motion(
        self,
        action: torch.Tensor,
        recorded_step_index: int,
    ) -> torch.Tensor:
        destination = self.state is SeedAutomationState.AXIS_TO_DESTINATION
        result = self._automatic_joint_action(action)
        self._state_steps += 1
        if self.task_id == "000001":
            target = (
                self._axis_destination_value
                if destination
                else self._axis_source_value
            )
            # The lift action is an absolute position target. Keep the
            # observation as the measured position, but label every motion
            # frame with the endpoint the controller is pursuing.
            result[:, self._lift_index] = target
            measured = self._current_axis_scalar()
            reached = abs(measured - target) <= self.cfg.lift_tolerance_m
            timeout = self.cfg.lift_timeout_seconds
        else:
            target_offset = CABINET_RIGHT_DELTA_Y_M if destination else 0.0
            velocity, reached = self._base_command(target_offset)
            result[:, 19:22] = velocity
            timeout = self.cfg.base_timeout_seconds

        self._held_action = result[:, :19].detach().clone()
        self._stable_steps = self._stable_steps + 1 if reached else 0
        if self._stable_steps >= self.cfg.axis_stable_steps:
            if destination:
                self._enter(SeedAutomationState.INVERSE_CARRY_CLOSED)
            else:
                cycle = self.current_cycle
                assert cycle is not None
                self._markers[f"cycle_{cycle.order}_complete_step"] = int(
                    recorded_step_index
                )
                self.cycle_index += 1
                self._begin_operator_cycle(recorded_step_index + 1)
            return result
        if self._state_steps > round(timeout * self.cfg.control_hz):
            self._fail(f"{self.state.value} timed out")
        return result

    def _base_command(self, target_offset_y_m: float) -> tuple[torch.Tensor, bool]:
        root_pose = self.robot.data.root_pose_w[0]
        home = self.env._coffee_transport_episode_home_root_pose_w[0]
        target_xy = home[:2].clone()
        target_xy[1] += target_offset_y_m
        world_error = target_xy - root_pose[:2]
        w, x, y, z = root_pose[3:7]
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        hw, hx, hy, hz = home[3:7]
        target_yaw = torch.atan2(
            2.0 * (hw * hz + hx * hy),
            1.0 - 2.0 * (hy * hy + hz * hz),
        )
        yaw_error = torch.atan2(
            torch.sin(target_yaw - yaw),
            torch.cos(target_yaw - yaw),
        )
        cos_yaw, sin_yaw = torch.cos(yaw), torch.sin(yaw)
        body_error = torch.stack(
            (
                cos_yaw * world_error[0] + sin_yaw * world_error[1],
                -sin_yaw * world_error[0] + cos_yaw * world_error[1],
            )
        )
        body_velocity = self.cfg.base_linear_gain * body_error
        speed = torch.linalg.vector_norm(body_velocity)
        if speed > self.cfg.base_linear_max_mps:
            body_velocity *= self.cfg.base_linear_max_mps / speed
        angular = torch.clamp(
            self.cfg.base_angular_gain * yaw_error,
            -self.cfg.base_angular_max_radps,
            self.cfg.base_angular_max_radps,
        )
        reached = bool(
            (
                torch.linalg.vector_norm(world_error)
                <= self.cfg.base_position_tolerance_m
            ).item()
            and (torch.abs(yaw_error) <= self.cfg.base_yaw_tolerance_rad).item()
        )
        if reached:
            return root_pose.new_zeros(3), True
        return torch.cat((body_velocity, angular.reshape(1))), False

    def _apply_replay(
        self,
        action: torch.Tensor,
        *,
        closed: bool,
        recorded_step_index: int,
    ) -> torch.Tensor:
        frames = self._carry_reverse if closed else self._approach_reverse
        if self._replay_index >= len(frames):
            if closed:
                self._enter(SeedAutomationState.RELEASE)
                return self.apply(action, recorded_step_index=recorded_step_index)
            return self._apply_post_release_home(action, recorded_step_index)

        result = action.clone()
        result[:, :19] = frames[self._replay_index].to(
            device=result.device, dtype=result.dtype
        )
        cycle = self.current_cycle
        assert cycle is not None
        gripper_index = self._gripper_index[cycle.arm]
        result[:, gripper_index] = (
            self._closed_gripper_value
            if closed
            else self._nominal_action[:, gripper_index]
        )
        result[:, self._head_indices] = self._nominal_action[:, self._head_indices]
        result[:, self._lift_index] = (
            self._axis_destination_value if self.task_id == "000001" else 0.0
        )
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0
        self._held_action = result[:, :19].detach().clone()
        self._replay_index += 1
        return result

    def _apply_post_release_home(
        self,
        action: torch.Tensor,
        recorded_step_index: int,
    ) -> torch.Tensor:
        """End inverse approach at exact destination-side HOME and settle it."""

        assert self._nominal_action is not None
        result = action.clone()
        target = self._nominal_action.to(device=result.device, dtype=result.dtype).clone()
        if self.task_id == "000001":
            target[:, self._lift_index] = self._axis_destination_value
        result[:, :19] = target
        if result.shape[1] == 22:
            result[:, 19:22] = 0.0
        self._held_action = target.detach().clone()

        self._state_steps += 1
        measured = self._measured_joint_action()[0]
        settled = bool(
            torch.all(
                torch.abs(
                    measured[self._home_settle_indices]
                    - target[0, self._home_settle_indices]
                )
                <= self.cfg.home_joint_tolerance_rad
            ).item()
        )
        self._stable_steps = self._stable_steps + 1 if settled else 0
        if self._stable_steps >= self.cfg.post_release_home_stable_steps:
            self._enter(SeedAutomationState.VALIDATE_PLACEMENT)
            return self.apply(action, recorded_step_index=recorded_step_index)
        if self._state_steps > round(
            self.cfg.post_release_home_timeout_seconds * self.cfg.control_hz
        ):
            self._fail("post-release HOME timed out before the fixed pose settled")
        return result

    def _apply_release(
        self,
        action: torch.Tensor,
        recorded_step_index: int,
    ) -> torch.Tensor:
        result = self._automatic_joint_action(action)
        cycle = self.current_cycle
        assert cycle is not None
        gripper_index = self._gripper_index[cycle.arm]
        result[:, gripper_index] = self._nominal_action[:, gripper_index]
        self._held_action = result[:, :19].detach().clone()
        self._state_steps += 1
        if self._state_steps >= round(
            self.cfg.release_dwell_seconds * self.cfg.control_hz
        ):
            self._markers[f"cycle_{cycle.order}_release_step"] = int(
                recorded_step_index
            )
            self._enter(SeedAutomationState.INVERSE_APPROACH_OPEN)
        return result

    def _apply_placement_gate(
        self,
        action: torch.Tensor,
        recorded_step_index: int,
    ) -> torch.Tensor:
        result = self._automatic_joint_action(action)
        cycle = self.current_cycle
        assert cycle is not None
        result[:, self._gripper_index[cycle.arm]] = self._nominal_action[
            :, self._gripper_index[cycle.arm]
        ]
        self._held_action = result[:, :19].detach().clone()
        decision = self._placement_decision()
        self._stable_steps = self._stable_steps + 1 if decision.passed else 0
        self._state_steps += 1
        placement_passed = (
            self._stable_steps >= DEFAULT_PLACEMENT_QUALITY.stable_steps
        )
        return_wait_elapsed = self._state_steps >= round(
            self.cfg.placement_return_wait_seconds * self.cfg.control_hz
        )
        if placement_passed or return_wait_elapsed:
            self._markers[f"cycle_{cycle.order}_placement_check_step"] = int(
                recorded_step_index
            )
            if placement_passed:
                self._markers[f"cycle_{cycle.order}_placed_step"] = int(
                    recorded_step_index
                )
            else:
                print(
                    f"[CoffeeSeed] {cycle.can_name} placement is not yet valid "
                    f"({','.join(decision.failures)}). Returning the task axis; "
                    "N will run the strict final placement check."
                )
            self._begin_axis_motion(destination=False)
        return result

    def _placement_decision(self, cycle=None):
        cycle = self.current_cycle if cycle is None else cycle
        assert cycle is not None
        square = next(
            item for item in SOURCE_SPAWN_SQUARES if item.name == cycle.can_name
        ).translated(
            dz_m=SHELF_LEVEL_DELTA_Z_M if self.task_id == "000001" else 0.0,
            dy_m=CABINET_RIGHT_DELTA_Y_M if self.task_id == "000002" else 0.0,
        )
        asset = self.env.scene[cycle.can_name]
        pose = asset.data.root_pose_w[0]
        origin = self.env.scene.env_origins[0]
        local_position = pose[:3] - origin
        quaternion = tuple(float(value) for value in pose[3:7])
        eef_distance = float(
            torch.linalg.vector_norm(
                self._active_eef_position(cycle.arm) - pose[:3]
            ).item()
        )
        expected_z = (
            SOURCE_UPPER_SUPPORT_Z_M
            + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
            + (SHELF_LEVEL_DELTA_Z_M if self.task_id == "000001" else 0.0)
        )
        return evaluate_placement(
            center_x_m=float(local_position[0]),
            center_y_m=float(local_position[1]),
            origin_z_m=float(local_position[2]),
            expected_origin_z_m=expected_z,
            upright_cosine=upright_cosine_from_wxyz(quaternion),
            can_linear_speed_mps=float(
                torch.linalg.vector_norm(asset.data.root_lin_vel_w[0]).item()
            ),
            can_angular_speed_radps=float(
                torch.linalg.vector_norm(asset.data.root_ang_vel_w[0]).item()
            ),
            active_eef_distance_m=eef_distance,
            center_region_xyxy_m=(
                square.x_min_m,
                square.x_max_m,
                square.y_min_m,
                square.y_max_m,
            ),
        )

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason
        if self._safe_hold_action is None:
            source = self._held_action
            if source is None:
                source = self._measured_joint_action()
            self._safe_hold_action = source.detach().clone()
        self.state = SeedAutomationState.FAILED
        print(f"[CoffeeSeed] Automation stopped: {reason}. Press R to reset.")
