"""Task000525 seed-recording hooks for the generic demonstration recorder.

This module is intentionally safe to load before Isaac Sim starts.  Runtime
dependencies are imported only when the factory configures or creates a
controller.
"""

from __future__ import annotations

import json
from typing import Any, Callable


TASK_TOKEN = "Task000525"
REGION_TO_SIDE = {"A": "left", "B": "left", "C": "right", "D": "right"}
PHASE_NAMES = (
    "grasp_step",
    "lift_step",
    "navigate_step",
    "place_step",
    "release_step",
)


def add_cli_arguments(parser: Any) -> None:
    """Add the legacy-compatible Task000525 recording arguments."""

    parser.add_argument(
        "--task525_seed_regions",
        type=str,
        help="Comma-separated same-arm Task525 seed schedule: A,B or C,D.",
    )
    parser.add_argument(
        "--task525_phase_markers",
        action="store_true",
        help=(
            "Enable Task525 phase markers. In Dijkstra mode G starts the "
            "automatic carry-home and navigation transition; F/H are optional "
            "annotations. Manual mode also uses M and P."
        ),
    )
    parser.add_argument(
        "--task525_base_mode",
        choices=("manual", "dijkstra"),
        default="manual",
        help=(
            "Task525 base source. 'manual' preserves foot-pad/keyboard control; "
            "'dijkstra' plans and executes the simulated route after G."
        ),
    )
    parser.add_argument(
        "--task525_dijkstra_linear_max",
        type=float,
        default=0.10,
        help="Task525 online Dijkstra maximum linear body speed in m/s.",
    )
    parser.add_argument(
        "--task525_dijkstra_angular_max",
        type=float,
        default=0.25,
        help="Task525 online Dijkstra maximum yaw speed in rad/s.",
    )
    parser.add_argument(
        "--task525_dijkstra_timeout_s",
        type=float,
        default=90.0,
        help="Task525 online Dijkstra fail-closed timeout in seconds.",
    )


def is_requested(args: Any) -> bool:
    """Return whether any Task525 recording mode was explicitly requested."""

    return bool(args.task525_phase_markers) or args.task525_base_mode == "dijkstra"


class Task000525RecordingFactory:
    """Validated construction data for one Task000525 recording session."""

    def __init__(
        self,
        *,
        seed_regions: tuple[str, ...],
        active_side: str,
        base_mode: str,
        linear_max: float,
        angular_max: float,
        timeout_s: float,
    ) -> None:
        self.seed_regions = seed_regions
        self.active_side = active_side
        self.base_mode = base_mode
        self.linear_max = linear_max
        self.angular_max = angular_max
        self.timeout_s = timeout_s

    @property
    def automatic(self) -> bool:
        return self.base_mode == "dijkstra"

    def configure_env_cfg(self, env_cfg: Any) -> None:
        """Select the first fixed-region seed profile before environment creation."""

        from isaaclab.managers import EventTermCfg as EventTerm

        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.profiles import (
            TASK000525_SEED_PROFILES,
        )
        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.reset_events import (
            randomize_coffee_can_center_regions,
        )

        first_region = self.seed_regions[0]
        env_cfg.randomization = TASK000525_SEED_PROFILES[first_region]
        coffee_positions = env_cfg.randomization.coffee_positions
        env_cfg.events.randomize_task000525_coffee_positions = EventTerm(
            func=randomize_coffee_can_center_regions,
            mode="reset",
            params={
                "layout_key": coffee_positions.layout_key,
                "target_region": None,
                "sample_positions": coffee_positions.sample_positions,
                "shuffle_distractors": coffee_positions.shuffle_distractors,
            },
        )

    def create(
        self,
        env: Any,
        teleop: Any,
        *,
        control_hz: float,
        wheel_speed_norm: Callable[[Any], float],
    ) -> "Task000525RecordingController":
        """Create the runtime controller after Isaac Sim and the env are ready."""

        from cyclo_lab.robot_specs.ffw.sg2 import (
            FFW_SG2_ACTION_JOINT_NAMES,
            FFW_SG2_JOINT_POSITION_LIMITS,
            FFW_SG2_LIFT_POSITION_UPPER,
        )

        navigator = None
        if self.automatic:
            import omni.kit.app

            extension_manager = omni.kit.app.get_app().get_extension_manager()
            if not extension_manager.is_extension_enabled(
                "isaacsim.replicator.mobility_gen"
            ):
                extension_manager.set_extension_enabled_immediate(
                    "isaacsim.replicator.mobility_gen", True
                )

            from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.online_dijkstra import (
                Task525OnlineDijkstraCfg,
                Task525OnlineDijkstraNavigator,
            )

            navigator = Task525OnlineDijkstraNavigator(
                env,
                Task525OnlineDijkstraCfg(
                    linear_max=self.linear_max,
                    angular_max=self.angular_max,
                    max_navigation_seconds=self.timeout_s,
                ),
                control_hz=control_hz,
            )

        return Task000525RecordingController(
            self,
            env,
            teleop,
            wheel_speed_norm=wheel_speed_norm,
            action_joint_names=FFW_SG2_ACTION_JOINT_NAMES,
            joint_position_limits=FFW_SG2_JOINT_POSITION_LIMITS,
            lift_position_upper=FFW_SG2_LIFT_POSITION_UPPER,
            navigator=navigator,
        )


def build_recording_factory(args: Any) -> Task000525RecordingFactory | None:
    """Return the Task000525 factory selected by the legacy CLI.

    Task-name recognition deliberately lives only here.  The generic recorder
    sees an optional controller and never branches on a Task000525 name.
    """

    if not args.task525_phase_markers:
        if args.task525_base_mode == "dijkstra":
            raise ValueError(
                "--task525_base_mode=dijkstra requires --task525_phase_markers."
            )
        return None
    if args.robot_type != "FFW_SG2" or TASK_TOKEN not in (args.task or ""):
        raise ValueError(
            "--task525_phase_markers is only valid for the FFW_SG2 "
            "Task000525 environment."
        )
    if not args.task525_seed_regions:
        raise ValueError(
            "Task525 seed collection requires --task525_seed_regions A,B or C,D"
        )

    seed_regions = tuple(
        value.strip().upper()
        for value in args.task525_seed_regions.split(",")
        if value.strip()
    )
    unknown = [region for region in seed_regions if region not in REGION_TO_SIDE]
    if unknown:
        raise ValueError(f"Unknown Task525 seed regions: {unknown}")
    if seed_regions not in (("A", "B"), ("C", "D")):
        raise ValueError(
            "Task525 seed schedule must be exactly A,B (left) or C,D (right)"
        )
    sides = {REGION_TO_SIDE[region] for region in seed_regions}
    if len(sides) != 1:
        raise ValueError("One Task525 recorder session must use one manipulation arm")
    if args.num_demos != len(seed_regions):
        raise ValueError(
            f"Task525 {args.task525_seed_regions} collection requires "
            f"--num_demos {len(seed_regions)}"
        )

    return Task000525RecordingFactory(
        seed_regions=seed_regions,
        active_side=sides.pop(),
        base_mode=args.task525_base_mode,
        linear_max=args.task525_dijkstra_linear_max,
        angular_max=args.task525_dijkstra_angular_max,
        timeout_s=args.task525_dijkstra_timeout_s,
    )


class Task000525RecordingController:
    """Own Task000525 phase, navigation, hold, and metadata lifecycle."""

    def __init__(
        self,
        factory: Task000525RecordingFactory,
        env: Any,
        teleop: Any,
        *,
        wheel_speed_norm: Callable[[Any], float],
        action_joint_names: tuple[str, ...],
        joint_position_limits: dict[str, tuple[float, float]],
        lift_position_upper: float,
        navigator: Any = None,
    ) -> None:
        self.factory = factory
        self.env = env
        self.teleop = teleop
        self.wheel_speed_norm = wheel_speed_norm
        self.action_joint_names = action_joint_names
        self.lift_position_upper = lift_position_upper
        self.navigator = navigator

        self.gripper_action_index = {
            "left": action_joint_names.index("gripper_l_joint1"),
            "right": action_joint_names.index("gripper_r_joint1"),
        }
        self.head_pitch_action_index = action_joint_names.index("head_joint1")
        self.head_yaw_action_index = action_joint_names.index("head_joint2")
        self.lift_action_index = action_joint_names.index("lift_joint")
        self.head_pitch_down_max_rad = joint_position_limits["head_joint1"][1]

        self.seed_region_index = 0
        self.markers = dict.fromkeys(PHASE_NAMES)
        self.should_start_auto_navigation = False
        self.auto_navigation_failure_reported = False
        self.home_arm_action = None
        self.place_activation_generation = None
        self.reset_joint_hold_target = None
        self.head_hold_target = None
        self.reset_lift_hold_target = None
        self._is_recording: Callable[[], bool] = lambda: False
        self._recorded_step: Callable[[], int] = lambda: 0

        env._task525_target_region = self.current_region()

    @property
    def automatic(self) -> bool:
        return self.factory.automatic

    def current_region(self) -> str:
        return self.factory.seed_regions[self.seed_region_index]

    def current_side(self) -> str:
        return self.factory.active_side

    def current_phase(self) -> int:
        if self.markers["release_step"] is not None:
            return 4
        if self.markers["place_step"] is not None:
            return 3
        if self.markers["navigate_step"] is not None:
            return 2
        if self.markers["lift_step"] is not None:
            return 1
        return 0

    def register_callbacks(
        self,
        *,
        is_recording: Callable[[], bool],
        recorded_step: Callable[[], int],
    ) -> None:
        self._is_recording = is_recording
        self._recorded_step = recorded_step
        self.teleop.add_callback("F", self._mark_grasp)
        self.teleop.add_callback("G", self._mark_lift)
        self.teleop.add_callback("M", self._mark_navigation)
        self.teleop.add_callback("P", self._mark_place)
        self.teleop.add_callback("H", self._mark_release)

    def _mark_grasp(self) -> None:
        step = self._recorded_step()
        if not self._is_recording():
            print("[Task525] F ignored: start a Task525 marked recording with B first.")
            return
        if self.markers["grasp_step"] is not None:
            print("[Task525] F ignored: stable grasp has already been marked.")
            return
        if step < 2:
            print(
                "[Task525] F ignored: record the approach and gripper close "
                "before marking a stable grasp."
            )
            return
        self.markers["grasp_step"] = step
        self.markers["lift_step"] = step
        print(
            f"[Task525] grasp_step={step}, lift_step={step}. Remove the can and "
            f"return the {self.current_side()} arm to the carry/home pose."
        )

    def _mark_lift(self) -> None:
        step = self._recorded_step()
        if not self._is_recording():
            print("[Task525] G ignored: start a Task525 marked recording with B first.")
            return
        if self.automatic:
            if self.navigator is not None and self.navigator.active:
                print(
                    "[Task525] G ignored: automatic carrying-home/Dijkstra "
                    "navigation is already running."
                )
                return
            if self.navigator is not None and self.navigator.status == self.navigator.FAILED:
                print(
                    "[Task525] G ignored: automatic navigation failed. Press R "
                    "to reset before retrying."
                )
                return
            if self.navigator is not None and self.navigator.completed:
                print("[Task525] G ignored: automatic navigation already completed.")
                return
            if self.navigator is not None and self.navigator.awaiting_place_activation:
                print(
                    f"[Task525] G ignored: base arrived; toggle the "
                    f"{self.current_side()} A3 tact to enable place."
                )
                return
            if self.markers["grasp_step"] is None:
                self.markers["grasp_step"] = step
            self.markers["lift_step"] = step
            self.should_start_auto_navigation = True
            print(
                f"[Task525] G accepted: preserving the closed {self.current_side()} "
                "gripper, returning both arms home, then starting Dijkstra navigation."
            )
            return
        if self.markers["grasp_step"] is None:
            print("[Task525] G ignored: press F after the stable grasp first.")
            return
        if self.markers["navigate_step"] is not None:
            print("[Task525] G ignored: navigation has already been marked.")
            return
        print("[Task525] G accepted: carry/home is ready; press M to begin manual navigation.")

    def _mark_navigation(self) -> None:
        if not self._is_recording():
            print("[Task525] M ignored: start a Task525 marked recording with B first.")
            return
        if self.automatic:
            print("[Task525] M is not needed in Dijkstra mode; press G after pick/carry.")
            return
        if self.markers["lift_step"] is None:
            print("[Task525] M ignored: press F after grasp and G after carry/home first.")
            return
        self.markers["navigate_step"] = self._recorded_step()
        self.markers["place_step"] = None
        print(
            f"[Task525] navigate_step={self.markers['navigate_step']}. Move the base; "
            "A3 arm mappings remain unchanged."
        )

    def _mark_place(self) -> None:
        if not self._is_recording():
            print("[Task525] P ignored: start a Task525 marked recording with B first.")
            return
        if self.markers["navigate_step"] is None:
            print("[Task525] P ignored: mark navigation start with M first.")
            return
        if self.automatic:
            if self.navigator is None or not self.navigator.completed:
                print("[Task525] P ignored: wait for automatic Dijkstra arrival.")
                return
            self.markers["place_step"] = self._recorded_step()
            print(
                f"[Task525] place_step={self.markers['place_step']}. Begin "
                f"{self.current_side()}-arm place."
            )
            return
        wheel_speed = self.wheel_speed_norm(self.env)
        if wheel_speed > 0.1:
            print(
                "[Task525] P ignored because the base is still moving "
                f"({wheel_speed:.3f} rad/s > 0.100 rad/s)."
            )
            return
        self.markers["place_step"] = self._recorded_step()
        print(
            f"[Task525] place_step={self.markers['place_step']}. Keep the base "
            "stopped and perform place."
        )

    def _mark_release(self) -> None:
        if not self._is_recording():
            print("[Task525] H ignored: start a Task525 marked recording with B first.")
            return
        if self.markers["place_step"] is None:
            print("[Task525] H ignored: wait for place before marking release.")
            return
        if self.markers["release_step"] is not None:
            print("[Task525] H ignored: release has already been marked.")
            return
        self.markers["release_step"] = self._recorded_step()
        print(
            f"[Task525] release_step={self.markers['release_step']}. Return the "
            f"{self.current_side()} arm home, then press N to save."
        )

    def validate_save(self) -> bool:
        if self.automatic and (self.navigator is None or not self.navigator.completed):
            status = "unavailable" if self.navigator is None else self.navigator.status
            reason = (
                ""
                if self.navigator is None
                else f" ({self.navigator.failure_reason})"
            )
            print(
                "[Task525] Save refused: automatic Dijkstra navigation has not "
                f"completed (status={status}){reason}."
            )
            return False
        required = (
            ("grasp_step", "lift_step", "navigate_step", "place_step")
            if self.automatic
            else PHASE_NAMES
        )
        missing = [name for name in required if self.markers[name] is None]
        if missing:
            instruction = (
                f"Use G, wait for navigation, toggle the {self.current_side()} "
                "tact, perform place/release, then press N."
                if self.automatic
                else "Use F, G, M, P, H, return home, then press N."
            )
            print(f"[Task525] Save refused: missing phase markers {missing}. {instruction}")
            return False
        return True

    def dataset_metadata(self) -> dict[str, Any]:
        auto = self.automatic
        return {
            "task525_seed_mode": (
                "continuous_pick_online_dijkstra_base_place"
                if auto
                else "continuous_pick_manual_base_place"
            ),
            "task525_phase_semantics": (
                "0=grasp,1=clear_cabinet_and_return_carry_home,2=navigation,"
                "3=place_and_release,4=return_home_without_object"
            ),
            "task525_base_command_source": (
                "online_dijkstra_fixed_yaw_holonomic"
                if auto
                else "external_cmd_vel_foot_pad_or_keyboard"
            ),
            "task525_online_dijkstra": auto,
            "task525_a3_mapping": "original_dual_arm_left_to_left_right_to_right",
            "task525_collection_arm_usage": "both_leaders_enabled",
            "task525_seed_regions": ",".join(self.factory.seed_regions),
            "task525_manipulation_side": self.current_side(),
            "task525_region_side_policy": json.dumps(REGION_TO_SIDE),
        }

    def episode_metadata(self) -> dict[str, Any]:
        arrangement = getattr(self.env, "_task525_arrangements", {}).get(0, {})
        return {
            **self.markers,
            "target_side": self.current_side(),
            "task525_arrangement_version": 1,
            "task525_target_region": self.current_region(),
            "task525_manipulation_side": self.current_side(),
            "task525_region_to_object": json.dumps(
                arrangement.get("region_to_object", {}), sort_keys=True
            ),
        }

    def advance_seed(self) -> None:
        if self.seed_region_index + 1 >= len(self.factory.seed_regions):
            return
        self.seed_region_index += 1
        self.env._task525_target_region = self.current_region()
        print(
            f"[Task525] Next seed: region {self.current_region()} with the "
            f"{self.current_side()} arm."
        )

    def _save_pose_target(self):
        import torch

        from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.home_pose import (
            TASK000525_SAVE_POSE_3_JOINT_POSITIONS,
        )

        values = [
            TASK000525_SAVE_POSE_3_JOINT_POSITIONS[name]
            for name in self.action_joint_names
        ]
        return (
            torch.tensor(values, dtype=torch.float32, device=self.env.device)
            .unsqueeze(0)
            .expand(self.env.num_envs, -1)
            .clone()
        )

    def _reset_head_target(self):
        import torch

        save_pose = self._save_pose_target()
        return torch.stack(
            (
                save_pose[:, self.head_pitch_action_index],
                save_pose[:, self.head_yaw_action_index],
            ),
            dim=1,
        )

    def _reset_lift_target(self):
        import torch

        return torch.full(
            (self.env.num_envs,),
            self.lift_position_upper,
            dtype=torch.float32,
            device=self.env.device,
        )

    def after_initial_reset(self) -> None:
        if self.automatic:
            self.head_hold_target = self._reset_head_target()
            self.reset_lift_hold_target = self._reset_lift_target()
            self.reset_joint_hold_target = self._save_pose_target()

    def on_recording_started(self) -> None:
        self.markers.update(dict.fromkeys(PHASE_NAMES))
        self.should_start_auto_navigation = False
        self.auto_navigation_failure_reported = False
        self.place_activation_generation = None
        if self.navigator is not None:
            self.navigator.reset()
            self.home_arm_action = self._save_pose_target()
        self.reset_joint_hold_target = None
        if self.automatic:
            self.teleop.begin_control_activation()
            print(
                f"[Task525] B activation: region {self.current_region()}, using "
                "absolute A3 commands; Task525 holds head and lift until G."
            )
        else:
            self.reset_lift_hold_target = None
        if self.automatic:
            print(
                "[Task525] Recording is armed. After a stable grasp, lift clear "
                "and press G to start automatic navigation."
            )

    def on_environment_reset(self) -> None:
        self.head_hold_target = self._reset_head_target()
        self.reset_lift_hold_target = self._reset_lift_target()
        self.reset_joint_hold_target = self._save_pose_target()
        self.teleop.clear_command_cache()
        self.markers.update(dict.fromkeys(PHASE_NAMES))
        self.should_start_auto_navigation = False
        self.auto_navigation_failure_reported = False
        self.home_arm_action = None
        self.place_activation_generation = None
        if self.navigator is not None:
            self.navigator.reset()
        print(
            "[Task525] Reset complete: head and lift are held at their start "
            "targets; G turns the head down and later lowers the lift by 30 cm."
        )

    def record_step_metadata(self, recorder_manager: Any, device: str) -> None:
        import torch

        recorder_manager.add_to_episodes(
            "obs/task525_demo_phase",
            torch.tensor([[self.current_phase()]], dtype=torch.int64, device=device),
        )

    def apply_actions(self, actions: Any, *, recording: bool, step_index: int):
        """Apply the Task525-owned holds and automatic navigation transition."""

        if not self.automatic:
            return actions
        if not recording and self.reset_joint_hold_target is not None:
            actions[:, :19] = self.reset_joint_hold_target.to(
                device=actions.device, dtype=actions.dtype
            )
        if self.head_hold_target is not None:
            head_hold = self.head_hold_target.to(
                device=actions.device, dtype=actions.dtype
            )
            actions[:, self.head_pitch_action_index] = head_hold[:, 0]
            actions[:, self.head_yaw_action_index] = head_hold[:, 1]
        if self.reset_lift_hold_target is not None:
            actions[:, self.lift_action_index] = self.reset_lift_hold_target.to(
                device=actions.device, dtype=actions.dtype
            )

        if self.should_start_auto_navigation:
            self.should_start_auto_navigation = False
            if self.navigator is None:
                raise RuntimeError("Task525 Dijkstra navigator was not initialized.")
            if self.home_arm_action is None:
                raise RuntimeError(
                    "Task525 automatic home target is unavailable; press B first."
                )
            home_action = self.home_arm_action.to(
                device=actions.device, dtype=actions.dtype
            ).clone()
            measured = self.teleop.get_measured_joint_hold_action()
            gripper_index = self.gripper_action_index[self.current_side()]
            home_action[:, gripper_index] = measured[:, gripper_index]
            home_action[:, self.head_pitch_action_index] = self.head_pitch_down_max_rad
            home_action[:, self.head_yaw_action_index] = 0.0
            import torch

            self.head_hold_target = torch.stack(
                (
                    home_action[:, self.head_pitch_action_index],
                    home_action[:, self.head_yaw_action_index],
                ),
                dim=1,
            ).detach().clone()
            if self.navigator.start(actions, home_arm_action=home_action):
                self.markers["place_step"] = None
                print(
                    "[Task525 AutoNav] Dijkstra path ready: "
                    f"{len(self.navigator.path.points)} waypoints, "
                    f"{float(self.navigator.path.get_path_length()):.3f} m. "
                    "Returning the arm home before base motion."
                )
            else:
                self.auto_navigation_failure_reported = True
                print(
                    "[Task525 AutoNav] Failed before motion. "
                    f"{self.navigator.failure_reason} Press R to reset."
                )

        if self.navigator is None:
            raise RuntimeError("Task525 Dijkstra navigator was not initialized.")
        status_before = self.navigator.status
        actions = self.navigator.apply(actions)
        if (
            status_before == self.navigator.RETURNING_HOME
            and self.navigator.status == self.navigator.NAVIGATING
        ):
            self.markers["navigate_step"] = step_index
            print(
                "[Task525 AutoNav] Carrying-home pose settled. Starting Dijkstra "
                "base motion; manual base commands remain ignored."
            )
        if (
            status_before != self.navigator.WAITING_FOR_PLACE_ACTIVATION
            and self.navigator.awaiting_place_activation
        ):
            if self.navigator.frozen_arm_action is None:
                raise RuntimeError("Task525 lift target is unavailable at place handoff.")
            self.reset_lift_hold_target = self.navigator.frozen_arm_action[
                :, self.lift_action_index
            ].detach().clone()
            self.place_activation_generation = self.teleop.arm_tact_generation(
                self.current_side()
            )
            print(
                "[Task525 AutoNav] Arrived, wheel motion settled, and lift lowered. "
                f"Press the {self.current_side()} A3 tact once to enable place."
            )
        if self.navigator.awaiting_place_activation:
            if self.place_activation_generation is None:
                raise RuntimeError(
                    "Task525 place handoff is missing its active-arm tact snapshot."
                )
            if (
                self.teleop.arm_tact_generation(self.current_side())
                > self.place_activation_generation
                and self.navigator.enable_place_control()
            ):
                self.teleop.begin_control_activation()
                self.markers["place_step"] = step_index
                print(
                    f"[Task525 AutoNav] {self.current_side().title()} A3 tact "
                    "received. Place is enabled; base remains held at zero."
                )
        if (
            self.navigator.status == self.navigator.FAILED
            and not self.auto_navigation_failure_reported
        ):
            self.auto_navigation_failure_reported = True
            print(
                "[Task525 AutoNav] Stopped without reaching the goal. "
                f"{self.navigator.failure_reason} Press R to reset."
            )
        return actions

    def before_shutdown(self) -> None:
        """Task525 has no extra teleoperation shutdown transition."""
