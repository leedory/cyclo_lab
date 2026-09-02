"""Task000525 mobile 22D, HDF5, and locomanipulation SDG contracts."""

from pathlib import Path
import importlib.util
import math
import unittest


REPO = Path(__file__).resolve().parents[3]
TASK_ROOT = (
    REPO
    / "source"
    / "cyclo_lab"
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
)
SG2_ROOT = (
    REPO / "source" / "cyclo_lab" / "cyclo_lab" / "robot_specs" / "ffw" / "sg2"
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class Task000525MobileContractTest(unittest.TestCase):
    def test_mobile_contract_is_canonical_22d(self):
        source = (SG2_ROOT / "mobile_contract.py").read_text(encoding="utf-8")
        self.assertIn('("linear_x", "linear_y", "angular_z")', source)
        self.assertIn('("rad",) * 18 + ("m",)', source)
        self.assertIn('("m/s", "m/s", "rad/s")', source)
        self.assertIn('"ffw_sg2_rev1_mobile_22d_v1"', source)
        self.assertIn(
            '"ffw_sg2_task525_locomanipulation_sdg_eef22_v1"', source
        )

    def test_task_observes_command_and_measured_mobile_values_separately(self):
        source = (TASK_ROOT / "env_cfg.py").read_text(encoding="utf-8")
        home_pose = (TASK_ROOT / "home_pose.py").read_text(encoding="utf-8")
        for name in (
            "base_velocity_body",
            "robot_root_pose_world",
            "target_object_pose_world",
            "left_eef_pose_world",
            "right_eef_pose_world",
        ):
            self.assertIn(name, source)
        self.assertIn("ContinuousShowroomActionsCfg", source)
        self.assertIn("self.actions.base_action.linear_deadband = 0.01", source)
        self.assertIn("TASK000525_SAVE_POSE_3_JOINT_POSITIONS", source)
        self.assertIn('"arm_l_joint1": 0.0005', home_pose)
        self.assertIn('"arm_l_joint7": 0.7391', home_pose)
        self.assertIn('"arm_r_joint2": -0.6040', home_pose)
        self.assertIn('"arm_r_joint7": -0.7391', home_pose)
        self.assertIn('"head_joint1": 0.2', home_pose)
        self.assertIn('"head_joint2": 0.0', home_pose)
        self.assertIn("exactly mirror", home_pose)
        self.assertIn("self.scene.robot.init_state.joint_pos.update", source)
        self.assertIn('self.events.set_robot_joint_pose.params["joint_positions"]', source)

    def test_recorder_uses_environment_contract_not_fixed_19d(self):
        source = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "recorder"
            / "record_demos.py"
        ).read_text(encoding="utf-8")
        self.assertIn("hdf5_contract_metadata(env_cfg.actions)", source)
        self.assertNotIn('"ffw_sg2_rev1_fixed_base_19d"', source)
        self.assertIn('"previous_step_action"', source)

    def test_continuous_demo_recorder_has_required_phase_markers(self):
        recorder = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "recorder"
            / "record_demos.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"grasp_step": None', recorder)
        self.assertIn('"lift_step": None', recorder)
        self.assertIn('"navigate_step": None', recorder)
        self.assertIn('"place_step": None', recorder)
        self.assertIn('"release_step": None', recorder)
        self.assertIn('teleop_interface.add_callback("F", mark_grasp_step)', recorder)
        self.assertIn('teleop_interface.add_callback("G", mark_lift_step)', recorder)
        self.assertIn('teleop_interface.add_callback("H", mark_task525_release)', recorder)
        self.assertIn('teleop_interface.add_callback("M", mark_task525_navigation)', recorder)
        self.assertIn('teleop_interface.add_callback("P", mark_task525_place)', recorder)
        self.assertIn('"obs/task525_demo_phase"', recorder)
        self.assertIn('choices=("manual", "dijkstra")', recorder)
        self.assertIn('G accepted: preserving the closed right gripper', recorder)
        self.assertIn('home_arm_action[:, TASK525_RIGHT_GRIPPER_ACTION_INDEX]', recorder)
        self.assertIn("TASK525_HEAD_PITCH_DOWN_MAX_RAD", recorder)
        self.assertIn("home_arm_action[:, TASK525_HEAD_PITCH_ACTION_INDEX]", recorder)
        self.assertIn("home_arm_action[:, TASK525_HEAD_YAW_ACTION_INDEX] = 0.0", recorder)
        self.assertIn('get_measured_joint_hold_action', recorder)
        self.assertIn('Press the right A3 tact once to enable right-arm place', recorder)
        self.assertIn('right_arm_tact_generation()', recorder)
        self.assertNotIn(
            'teleop_interface.trajectory_command_generation("right_arm")\n                                    > task525_place_activation_generation',
            recorder,
        )
        self.assertIn('Arrived, wheel motion settled, and lift lowered', recorder)
        self.assertIn('Press R to reset before retrying', recorder)
        self.assertIn("TASK525_LIFT_ACTION_INDEX", recorder)
        self.assertIn("task525_reset_lift_hold_target", recorder)
        self.assertIn("def task525_reset_lift_target", recorder)
        self.assertIn("def task525_save_pose_target", recorder)
        self.assertIn("task525_reset_joint_hold_target", recorder)
        self.assertIn("task525_home_arm_action = task525_save_pose_target()", recorder)
        self.assertIn("FFW_SG2_LIFT_POSITION_UPPER", recorder)
        self.assertIn("teleop_interface.clear_command_cache()", recorder)
        self.assertIn("using absolute A3 joint commands", recorder)
        self.assertNotIn("TASK525_RELATIVE_ARM_JOINT_NAMES", recorder)
        self.assertNotIn("relative_joint_names=", recorder)

    def test_recorder_keeps_original_a3_mapping_and_supports_online_base_commands(self):
        recorder = (
            REPO / "scripts" / "sim2real" / "imitation_learning" / "recorder" / "record_demos.py"
        ).read_text(encoding="utf-8")
        bridge = (
            REPO / "source" / "cyclo_lab" / "cyclo_lab" / "runtime" / "bridges" / "sg2" / "topic_bridge.py"
        ).read_text(encoding="utf-8")
        self.assertIn('"online_dijkstra_fixed_yaw_holonomic"', recorder)
        self.assertIn('"task525_a3_mapping": "original_dual_arm_left_to_left_right_to_right"', recorder)
        self.assertIn('actions = task525_auto_navigation.apply(actions)', recorder)
        self.assertIn(
            'TRAJECTORY_COMMAND_GROUPS = ("left_arm", "right_arm", "head", "lift")',
            bridge,
        )
        self.assertIn(
            '("right_arm", "head", "lift")',
            recorder,
        )
        self.assertIn('FFW_SG2_ACTION_TOPICS["mobile"]', bridge)
        self.assertIn('def trajectory_command_generation', bridge)
        self.assertIn('FFW_SG2_RIGHT_ARM_ENABLE_TOPIC', bridge)
        self.assertIn('def right_arm_tact_generation', bridge)
        self.assertIn('int(getattr(msg, "data", -1)) != 2', bridge)
        self.assertIn('def get_measured_joint_hold_action', bridge)
        sdk = (
            REPO / "source" / "cyclo_lab" / "cyclo_lab" / "runtime" / "sdk" / "ffw_sg2.py"
        ).read_text(encoding="utf-8")
        self.assertIn("None so external /cmd_vel passes through", sdk)
        self.assertIn("action[:, -3:] = action.new_tensor(keyboard_command)", sdk)

    def test_online_dijkstra_uses_inflated_static_map_and_freezes_only_during_navigation(self):
        source = (TASK_ROOT / "online_dijkstra.py").read_text(encoding="utf-8")
        self.assertIn("SHOWROOM_STATIC_OBSTACLE_AABBS", source)
        self.assertIn("STATIC_MAP_PREFILL_BUFFER_M + UPSTREAM_FINAL_BUFFER_M", source)
        self.assertIn("plan_path(", source)
        self.assertIn("off_path_replan_distance", source)
        self.assertIn("result[:, :19] = self.frozen_arm_action", source)
        self.assertIn("result[:, 19:22] = result.new_tensor(self.step())", source)
        self.assertIn('RETURNING_HOME = "returning_home"', source)
        self.assertIn('WAITING_FOR_PLACE_ACTIVATION = "waiting_for_place_activation"', source)
        self.assertIn('LOWERING_LIFT = "lowering_lift"', source)
        self.assertIn("def enable_place_control", source)
        self.assertIn("return_home_blend_seconds: float = 2.0", source)
        self.assertIn("home_joint_tolerance_rad: float = 0.08", source)
        self.assertIn("_home_settle_action_indices", source)
        self.assertIn("def _return_home_arm_action", source)
        self.assertIn("def _begin_navigation", source)
        self.assertIn("navigation-entry replan failed", source)
        self.assertIn("absolute 19D Task525", source)
        self.assertIn("lift_lower_distance_m: float = 0.30", source)
        self.assertIn("def _begin_lift_lowering", source)
        self.assertIn("initial_yaw_counterclockwise: bool = True", source)
        self.assertIn("_initial_yaw_alignment_pending", source)
        self.assertIn("% math.tau", source)
        launcher = (
            REPO / "scripts" / "sim2real" / "imitation_learning" / "record_task000525_mobile_demo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--task525_base_mode dijkstra", launcher)
        manual_launcher = (
            REPO / "scripts" / "sim2real" / "imitation_learning" / "record_task000525_manual_base_demo.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("--task525_base_mode manual", manual_launcher)
        self.assertIn("--keyboard_mobile", manual_launcher)

    def test_mimic_adapter_supports_22d_without_owning_navigation(self):
        common = (
            REPO
            / "source"
            / "cyclo_lab"
            / "cyclo_lab"
            / "manager_based"
            / "manipulation"
            / "common"
            / "ffw_sg2_mimic_env.py"
        ).read_text(encoding="utf-8")
        converter = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "mimic"
            / "action_data_converter.py"
        ).read_text(encoding="utf-8")
        self.assertIn("MOBILE_ACTION_DIM = 22", common)
        self.assertIn("upper_body_action.new_zeros(3)", common)
        self.assertIn("joint_actions[:, 19:22]", converter)
        self.assertIn("ik_actions[:, 19:22]", converter)

    def test_task525_generator_is_local_repeat_free_and_preserves_release(self):
        upstream = (
            REPO
            / "third_party"
            / "IsaacLab"
            / "scripts"
            / "imitation_learning"
            / "locomanipulation_sdg"
            / "generate_data.py"
        ).read_text(encoding="utf-8")
        local = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "tasks"
            / "task_000525"
            / "generate_trajectories.py"
        ).read_text(encoding="utf-8")
        launcher = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "run_task000525_trajectory_generation.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("[Task525]", upstream)
        self.assertIn('active_side in ("legacy", "left")', local)
        self.assertIn('recorded_base_reference=place_step is not None', local)
        self.assertIn("infer_dropoff_replay_step", local)
        self.assertIn("recording_step = dropoff_replay_step", local)
        self.assertNotIn("manipulation_step_repeat", local)
        self.assertNotIn("manipulation_repeat_index", local)
        self.assertIn("build_navigation_scene(", local)
        self.assertIn("plan_navigation_path(", local)
        self.assertIn('"navigation_entry"', local)
        self.assertIn("TASK000525_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M = 0.005", local)
        self.assertIn('"carry_root_xy_max_displacement_m"', local)
        self.assertIn("possible arm-cabinet contact", local)
        self.assertIn('"task525_generation_quality_gate_v2"', local)
        self.assertIn("--active_side right", launcher)
        self.assertIn("--navigation_mode fixed_yaw_holonomic", launcher)
        self.assertIn("--approach_distance 0.0", launcher)

    def test_converter_preserves_canonical_order_and_measured_base_state(self):
        source = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "data_converter"
            / "isaaclab2lerobot.py"
        ).read_text(encoding="utf-8")
        self.assertIn("FFW_SG2_MOBILE_ACTION_NAMES", source)
        self.assertIn('demo_group["obs/base_velocity_body"]', source)
        self.assertNotIn("actions[:, 17:19]", source)
        self.assertNotIn("actions[:, 16:17]", source)
        self.assertIn('"eef" in normalized_contract_id', source)
        self.assertIn('normalized_representation in {"ik", "eef"}', source)

    def test_sdg_hdf_converter_uses_causal_joint_and_base_alignment(self):
        source = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "data_converter"
            / "locomanipulation_sdg_to_joint22.py"
        ).read_text(encoding="utf-8")
        self.assertIn("joint_targets[1:]", source)
        self.assertIn("raw_sdg_actions[:-1, 19:22]", source)
        self.assertIn('POLICY_CONTRACT_ID = "ffw_sg2_rev1_mobile_22d_v1"', source)

    def test_sdg_adapter_is_local_and_registered(self):
        adapter = (TASK_ROOT / "locomanipulation_sdg_env.py").read_text(encoding="utf-8")
        registration = (
            TASK_ROOT.parents[1] / "__init__.py"
        ).read_text(encoding="utf-8")
        self.assertIn("class Task000525LocomanipulationSDGEnv", adapter)
        self.assertIn("posture_bias_sides=(\"r\",)", adapter)
        mimic_cfg = (
            REPO
            / "source"
            / "cyclo_lab"
            / "cyclo_lab"
            / "manager_based"
            / "manipulation"
            / "common"
            / "ffw_sg2_mimic_action_cfg.py"
        ).read_text(encoding="utf-8")
        self.assertIn("null_projector", mimic_cfg)
        self.assertIn("base_velocity_target", adapter)
        self.assertIn(
            "Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0",
            registration,
        )
        self.assertIn(
            "Cyclo-Real-Showroom-Task000525-Trajectory-Generation-FFW-SG2-v0",
            registration,
        )

    def test_candidate_docking_parent_reconstructs_candidate_goal(self):
        contract = load_module(
            "task525_sdg_contract",
            TASK_ROOT / "locomanipulation_sdg_contract.py",
        )
        source_x, source_y = contract.SOURCE_FIXTURE_POSE_WXYZ[:2]
        base_x, base_y, base_yaw = (-1.47138, 0.775837960613148, math.pi)
        relative_x = base_x - source_x
        relative_y = base_y - source_y

        destination = contract.DESTINATION_DOCKING_PARENT_POSE_WXYZ
        destination_yaw = math.pi
        goal_x = destination[0] + math.cos(destination_yaw) * relative_x - math.sin(
            destination_yaw
        ) * relative_y
        goal_y = destination[1] + math.sin(destination_yaw) * relative_x + math.cos(
            destination_yaw
        ) * relative_y
        goal_yaw = contract.wrap_to_pi(destination_yaw + base_yaw)
        expected = contract.CANDIDATE_BASE_GOAL_XYYAW
        self.assertAlmostEqual(goal_x, expected[0], places=6)
        self.assertAlmostEqual(goal_y, expected[1], places=6)
        self.assertAlmostEqual(goal_yaw, expected[2], places=6)
        total_buffer = (
            contract.STATIC_MAP_PREFILL_BUFFER_M
            + contract.UPSTREAM_FINAL_BUFFER_M
        )
        self.assertAlmostEqual(total_buffer, 0.465, places=6)

    def test_smoke_script_checks_hdf5_timing_and_replay(self):
        source = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "validate_task000525_mobile.py"
        ).read_text(encoding="utf-8")
        self.assertIn("last_actions[1:], actions[:-1]", source)
        self.assertIn("HDF5DatasetFileHandler", source)
        self.assertIn("replay_dataset(", source)
        self.assertIn("body_delta(initial, final)", source)

    def test_visual_replay_holds_finished_padded_episode_state(self):
        source = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "tasks"
            / "task_000525"
            / "replay_visual_policy_staging.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "post_step[min(step - 1, len(post_step) - 1)]",
            source,
        )
        self.assertIn("shorter padded env is inactive", source)


if __name__ == "__main__":
    unittest.main()
