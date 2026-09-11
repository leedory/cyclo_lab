"""Coffee-transport seed-recording controller for Task000001/000002."""

from __future__ import annotations

from typing import Any, Callable


def add_cli_arguments(parser: Any) -> None:
    parser.add_argument(
        "--coffee_transport_seed_automation",
        action="store_true",
        help=(
            "Task000001/000002 nominal seed mode: the operator performs each "
            "approach, body grasp, and extraction, then G automatically returns "
            "HOME, places, and returns."
        ),
    )


def is_requested(args: Any) -> bool:
    return bool(args.coffee_transport_seed_automation)


class CoffeeTransportRecordingFactory:
    """Validated construction data for a nominal coffee seed session."""

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id

    def configure_env_cfg(self, env_cfg: Any) -> None:
        """Coffee nominal seed collection uses the environment's existing config."""

    def create(
        self,
        env: Any,
        teleop: Any,
        *,
        control_hz: float,
        wheel_speed_norm: Callable[[Any], float],
    ) -> "CoffeeTransportRecordingController":
        del wheel_speed_norm
        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport import (
            seed_automation,
        )
        from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES

        automation = seed_automation.CoffeeTransportSeedAutomation(
            env,
            task_id=self.task_id,
            cfg=seed_automation.CoffeeSeedAutomationCfg(control_hz=control_hz),
        )
        return CoffeeTransportRecordingController(
            self,
            env,
            teleop,
            automation=automation,
            action_joint_names=FFW_SG2_ACTION_JOINT_NAMES,
        )


def build_recording_factory(args: Any) -> CoffeeTransportRecordingFactory | None:
    if not is_requested(args):
        return None
    if args.robot_type != "FFW_SG2":
        raise ValueError("Coffee seed automation requires --robot_type FFW_SG2.")
    task_name = args.task or ""
    matching_ids = [
        task_id for task_id in ("000001", "000002") if f"Task{task_id}" in task_name
    ]
    if len(matching_ids) != 1 or "-Random-" in task_name:
        raise ValueError(
            "Coffee seed automation requires the nominal Task000001 or "
            "Task000002 environment, not a Random environment."
        )
    return CoffeeTransportRecordingFactory(matching_ids[0])


class CoffeeTransportRecordingController:
    """Own Coffee seed save gates, callbacks, holds, and automation lifecycle."""

    def __init__(
        self,
        factory: CoffeeTransportRecordingFactory,
        env: Any,
        teleop: Any,
        *,
        automation: Any,
        action_joint_names: tuple[str, ...],
    ) -> None:
        self.factory = factory
        self.env = env
        self.teleop = teleop
        self.automation = automation
        self.action_joint_names = action_joint_names
        self.reset_joint_hold_target = None
        self.operator_tact_generation = None
        self._is_recording: Callable[[], bool] = lambda: False

    def register_callbacks(
        self,
        *,
        is_recording: Callable[[], bool],
        recorded_step: Callable[[], int],
    ) -> None:
        del recorded_step
        self._is_recording = is_recording
        self.teleop.add_callback("G", self._request_place_cycle)

    def _request_place_cycle(self) -> None:
        if not self._is_recording():
            print("[CoffeeSeed] G ignored: start coffee seed recording with B first.")
            return
        cycle = self.automation.current_cycle
        if self.automation.operator_enabled and cycle is not None:
            self.teleop.set_arm_enabled(cycle.arm, False)
        self.automation.request_g()

    def _home_joint_target(self):
        robot = self.env.scene["robot"]
        joint_ids = [
            robot.joint_names.index(name) for name in self.action_joint_names
        ]
        return robot.data.default_joint_pos[:, joint_ids].detach().clone()

    def validate_save(self) -> bool:
        if self.automation.failed:
            print(
                "[CoffeeSeed] Save refused after fail-closed stop: "
                f"{self.automation.failure_reason}. Press R to discard."
            )
            return False
        if not self.automation.task_complete:
            print(
                "[CoffeeSeed] Save refused: both G-driven placement cycles "
                "must complete before N."
            )
            return False
        placement_failures = self.automation.final_placement_failures()
        if placement_failures:
            print(
                "[CoffeeSeed] Save refused by final placement gate: "
                f"{placement_failures}. Press R to discard."
            )
            return False
        return True

    def dataset_metadata(self) -> dict[str, Any]:
        task_id = self.factory.task_id
        return {
            "coffee_seed_mode": (
                "two_operator_pickups_tact_immediate_g_automatic_place_v4"
                if task_id == "000001"
                else "two_operator_pickups_tact_immediate_g_automatic_place_v3"
            ),
            "coffee_task_id": task_id,
            "coffee_can_order": "coffee_can_right,coffee_can_left",
            "coffee_arm_order": "right,left",
            "coffee_operator_scope": "approach_grasp_extract_only",
            "coffee_g_scope": (
                "return_home_axis_move_inverse_replay_release_axis_return"
            ),
            "coffee_head_policy": "fixed_at_episode_nominal",
            "coffee_seed_randomization": "disabled_nominal_only",
            "coffee_task_axis_action_contract": (
                "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1"
                if task_id == "000001"
                else "base_body_velocity_feedback_v1"
            ),
        }

    def episode_metadata(self) -> dict[str, Any]:
        return self.automation.metadata()

    def advance_seed(self) -> None:
        """Coffee automation advances its own can cycles within one episode."""

    def after_initial_reset(self) -> None:
        self.reset_joint_hold_target = self._home_joint_target()
        self.teleop.set_all_arms_idle()

    def on_recording_started(self) -> None:
        self.teleop.set_all_arms_idle()
        self.automation.begin_episode(recorded_step_index=0)
        self.reset_joint_hold_target = None
        self.operator_tact_generation = None

    def on_environment_reset(self) -> None:
        self.automation.reset()
        self.reset_joint_hold_target = self._home_joint_target()
        self.operator_tact_generation = None
        self.teleop.set_all_arms_idle()
        self.teleop.clear_command_cache()

    def record_step_metadata(self, recorder_manager: Any, device: str) -> None:
        del recorder_manager, device

    def apply_actions(self, actions: Any, *, recording: bool, step_index: int):
        if not recording:
            if self.reset_joint_hold_target is not None:
                actions[:, :19] = self.reset_joint_hold_target.to(
                    device=actions.device, dtype=actions.dtype
                )
                if actions.shape[1] == 22:
                    actions[:, 19:22] = 0.0
            return actions

        actions = self.automation.apply(actions, recorded_step_index=step_index)
        if not self.automation.awaiting_operator_tact:
            return actions

        cycle = self.automation.current_cycle
        if cycle is None:
            raise RuntimeError("Coffee tact wait has no active can cycle.")
        current_generation = self.teleop.arm_tact_generation(cycle.arm)
        if self.operator_tact_generation is None:
            self.operator_tact_generation = current_generation
        elif current_generation > self.operator_tact_generation:
            if self.automation.enable_operator_cycle(recorded_step_index=step_index):
                self.teleop.begin_control_activation()
                self.operator_tact_generation = None
        return actions

    def before_shutdown(self) -> None:
        self.teleop.set_all_arms_idle()
