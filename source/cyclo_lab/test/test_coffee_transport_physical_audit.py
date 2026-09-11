"""Pure HDF5 regression tests for the Coffee physical trajectory audit."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[3]
SCRIPT_DIR = REPO / "scripts/sim2real/imitation_learning/tasks/coffee_transport"
SCRIPT = SCRIPT_DIR / "audit_and_select_physical_trajectories.py"
GENERATION_METHODS = {
    "000001": "coffee_two_can_endpoint_lift_targets_pending_validation_v13",
    "000002": "coffee_two_can_production_realized_initial_commanded_joint_hold_v12",
}
INITIAL_NORMALIZATION_CONTRACTS = {
    "000001": "coffee_realized_pose_endpoint_lift_targets_v13",
    "000002": "coffee_production_realized_pose_commanded_joint_hold_v12",
}
GENERATION_PROFILES = {
    "000001": "coffee_physical_trajectory_generation_endpoint_lift_targets_v13",
    "000002": "coffee_physical_trajectory_generation_commanded_joint_hold_v12",
}
PRODUCTION_PROMOTIONS = {
    "000001": "coffee_endpoint_lift_targets_v13_reviewed",
    "000002": "coffee_commanded_joint_hold_v12_promoted_20260908",
}
COLLISION_PROFILE = "cylinder_r33p5_h90_v1"
COLLISION_ASSET_SHA256 = (
    "fe340f411e93fd3fd54bc06b1b4ed1e6dec24214179e69ae5dd6ce1dd50d4a82"
)


def production_attrs(task_id: str) -> dict:
    attrs = {
        "generation_method": GENERATION_METHODS[task_id],
        "initial_normalization_contract_id": INITIAL_NORMALIZATION_CONTRACTS[task_id],
        "generation_profile_id": GENERATION_PROFILES[task_id],
        "production_promotion_id": PRODUCTION_PROMOTIONS[task_id],
        "production_approved": True,
        "task_axis_action_contract": (
            "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1"
            if task_id == "000001"
            else "base_body_velocity_feedback_v1"
        ),
        "coffee_can_collision_profile_id": COLLISION_PROFILE,
        "coffee_can_collision_asset_sha256": COLLISION_ASSET_SHA256,
        "coffee_can_collision_12gon_applied": False,
    }
    if task_id == "000001":
        attrs["production_validation_status"] = (
            "reviewed_endpoint_lift_targets_v13"
        )
    return attrs




def load_module():
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location("coffee_physical_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def root_pose(x: float, y: float, yaw: float = math.pi) -> np.ndarray:
    return np.asarray(
        [x, y, 0.0, math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)],
        dtype=np.float32,
    )


def add_episode(data: h5py.Group, task_id: str, *, seed: int = 101) -> h5py.Group:
    frames = 120
    action_width = 19 if task_id == "000001" else 22
    home_x, home_y = -1.47138, 0.80011813
    route_spawn = 0.0 if task_id == "000001" else 0.40
    spawn = root_pose(home_x, home_y + route_spawn)
    home = root_pose(home_x, home_y)
    demo = data.create_group("demo_0")

    actions = np.zeros((frames, action_width), dtype=np.float32)
    joint = np.zeros((frames, 19), dtype=np.float32)
    joint_target = np.zeros((frames, 19), dtype=np.float32)
    if task_id == "000001":
        lift = np.zeros(frames, dtype=np.float32)
        lift[:10] = np.linspace(-0.10, 0.0, 10)
        lift[35:46] = -0.28
        lift[85:96] = -0.28
        joint[:, 18] = lift
        lift_target = np.zeros(frames, dtype=np.float32)
        lift_target[30:55] = -0.28
        lift_target[80:105] = -0.28
        actions[:, 18] = lift_target
        joint_target[:, 18] = lift_target
    robot = np.repeat(home[None], frames, axis=0)
    if task_id == "000002":
        robot[:10, 1] = np.linspace(home_y + route_spawn, home_y, 10)
        robot[35:46, 1] = home_y + 0.815072039386852
        robot[85:96, 1] = home_y + 0.815072039386852
        robot[0] = spawn

    demo.create_dataset("actions", data=actions)
    demo.create_dataset("processed_actions", data=actions)
    obs = demo.create_group("obs")
    obs.create_dataset("actions", data=actions)
    obs.create_dataset("joint_pos", data=joint)
    obs.create_dataset("joint_pos_target", data=joint_target)
    obs.create_dataset("left_eef_pose", data=np.zeros((frames, 7), dtype=np.float32))
    obs.create_dataset("right_eef_pose", data=np.zeros((frames, 7), dtype=np.float32))
    image = np.broadcast_to(
        np.arange(672, dtype=np.uint16)[None, :, None] % 256,
        (376, 672, 3),
    ).astype(np.uint8)
    camera = obs.create_dataset(
        "cam_head",
        shape=(frames, 376, 672, 3),
        dtype=np.uint8,
        compression="gzip",
    )
    for index in range(frames):
        camera[index] = (image.astype(np.uint16) + index % 3).astype(np.uint8)

    states = demo.create_group("states")
    articulation = states.create_group("articulation/robot")
    articulation.create_dataset("joint_position", data=np.zeros((frames, 31), dtype=np.float32))
    articulation.create_dataset("joint_velocity", data=np.zeros((frames, 31), dtype=np.float32))
    articulation.create_dataset("root_pose", data=robot)
    articulation.create_dataset("root_velocity", data=np.zeros((frames, 6), dtype=np.float32))

    initial = demo.create_group("initial_state")
    initial_robot = initial.create_group("articulation/robot")
    initial_robot.create_dataset("root_pose", data=spawn[None])
    initial_robot.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))
    initial_robot.create_dataset("joint_position", data=np.zeros((1, 31), dtype=np.float32))
    initial_robot.create_dataset("joint_velocity", data=np.zeros((1, 31), dtype=np.float32))

    initial_xy = {
        "coffee_can_right": (-2.17, 0.99),
        "coffee_can_left": (-2.17, 0.58),
    }
    onset = {"coffee_can_right": 20, "coffee_can_left": 70}
    for can, (x, y) in initial_xy.items():
        initial_can = initial.create_group(f"rigid_object/{can}")
        start_pose = np.asarray([x, y, 1.350128932, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        initial_can.create_dataset("root_pose", data=start_pose[None])
        initial_can.create_dataset("root_velocity", data=np.zeros((1, 6), dtype=np.float32))
        destination_y = y if task_id == "000001" else y + 0.815072039386852
        destination_z = 1.062119288 if task_id == "000001" else 1.350128932
        final_pose = np.asarray([x, destination_y, destination_z, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        poses = np.repeat(start_pose[None], frames, axis=0)
        poses[onset[can] :] = final_pose
        rigid = states.create_group(f"rigid_object/{can}")
        rigid.create_dataset("root_pose", data=poses)
        rigid.create_dataset("root_velocity", data=np.zeros((frames, 6), dtype=np.float32))

        prefix = f"quality_{can}_grasp_"
        diagnostics = {
            "close_onset_step": 10.0,
            "close_frame_count": 15.0,
            "open_gripper_value": 0.0,
            "closed_gripper_value": 1.1,
            "diagnostic_eef_can_distance_m": 0.03,
            "diagnostic_can_local_eef_x_m": 0.01,
            "diagnostic_can_local_eef_y_m": 0.01,
            "diagnostic_can_local_eef_z_m": 0.00,
            "diagnostic_source_can_local_eef_x_m": -0.01,
            "diagnostic_source_can_local_eef_y_m": 0.00,
            "diagnostic_source_can_local_eef_z_m": 0.01,
            "diagnostic_can_local_error_x_m": 0.02,
            "diagnostic_can_local_error_y_m": 0.01,
            "diagnostic_can_local_error_z_m": -0.01,
            "diagnostic_orientation_error_rad": 0.1,
        }
        for key, value in diagnostics.items():
            demo.attrs[prefix + key] = value
        carry = {
            "eef_object_distance_m": 0.03,
            "source_eef_object_distance_m": 0.025,
            "tcp_z_relative_to_can_m": 0.0,
            "can_displacement_m": 0.25,
            "home_joint_max_error": 0.04,
            "can_linear_speed_mps": 5.0,
            "can_angular_speed_radps": 50.0,
        }
        for key, value in carry.items():
            demo.attrs[f"quality_{can}_carry_{key}"] = value
        placement = {
            "center_x_m": x,
            "center_y_m": destination_y,
            "origin_z_m": destination_z,
            "expected_origin_z_m": destination_z,
            "upright_cosine": 1.0,
            "can_linear_speed_mps": 0.0,
            "can_angular_speed_radps": 0.0,
            "active_eef_distance_m": float(np.linalg.norm(home[:3] - final_pose[:3])),
        }
        for key, value in placement.items():
            demo.attrs[f"quality_{can}_placement_{key}"] = value

    demo.attrs.update(
        {
            "success": True,
            "failure_reason": "",
            "task_id": task_id,
            "success_criterion_id": "coffee_two_can_carry_and_placement_quality_gate_v3",
            "coffee_phase_contract_version": 7 if task_id == "000001" else 6,
            "generation_seed": seed,
            "generation_attempt": 1,
            "grasp_correction_can_x_m": -0.020,
            "grasp_correction_can_z_m": 0.030,
            "grasp_correction_blend_steps": 30,
            "num_samples": frames,
            "robot_spawn_x_m": float(spawn[0]),
            "robot_spawn_y_m": float(spawn[1]),
            "robot_spawn_yaw_rad": math.pi,
            "robot_home_x_m": home_x,
            "robot_home_y_m": home_y,
            "robot_home_yaw_rad": math.pi,
            "lift_spawn_m": -0.10 if task_id == "000001" else 0.0,
            "base_route_spawn_offset_m": route_spawn,
        }
    )
    demo.attrs.update(production_attrs(task_id))
    for can, (x, y) in initial_xy.items():
        demo.attrs[f"{can}_x_m"] = x
        demo.attrs[f"{can}_y_m"] = y
        demo.attrs[f"{can}_yaw_rad"] = 0.0
    return demo


def create_file(path: Path, task_id: str) -> None:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        data.attrs.update(
            {
                "schema_version": "cyclo_lab_hdf5_v1",
                "dataset_origin": "coffee_transport_physical_trajectory_generation",
                    "success_criterion_id": "coffee_two_can_carry_and_placement_quality_gate_v3",
                "task_id": task_id,
                "observation_semantics": "pre_step",
                "scene_state_semantics": "post_step",
                "control_hz": 15.0,
                "camera_hz": 15.0,
                "total": 120,
                "action_semantics": "pre_step_dual_eef_pose_wxyz_grippers_head_lift_plus_optional_body_velocity",
                "action_names": json.dumps(list((
                    "left_eef_x", "left_eef_y", "left_eef_z",
                    "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz",
                    "gripper_l_joint1",
                    "right_eef_x", "right_eef_y", "right_eef_z",
                    "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz",
                    "gripper_r_joint1", "head_joint1", "head_joint2", "lift_joint",
                ) + (("linear_x", "linear_y", "angular_z") if task_id == "000002" else ()))),
            }
        )
        data.attrs.update(production_attrs(task_id))
        add_episode(data, task_id)


class CoffeePhysicalAuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_working_vertical_and_lateral_contracts_pass(self):
        for task_id in ("000001", "000002"):
            with self.subTest(task_id=task_id), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "physical.hdf5"
                create_file(path, task_id)
                report = self.module.audit_inputs([path], task_id, 1, False)
                self.assertTrue(report["passed"], report["errors"])
                self.assertEqual(report["accepted_episode_count"], 1)

    def test_allow_rejected_selects_requested_count_from_larger_pool(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "physical_first.hdf5"
            second = Path(directory) / "physical_second.hdf5"
            create_file(first, "000001")
            create_file(second, "000001")
            with h5py.File(second, "r+") as handle:
                handle["data/demo_0"].attrs["generation_seed"] = 102

            strict = self.module.audit_inputs([first, second], "000001", 1, False)
            self.assertFalse(strict["passed"])
            pooled = self.module.audit_inputs([first, second], "000001", 1, True)
            self.assertTrue(pooled["passed"], pooled["errors"])
            self.assertEqual(pooled["accepted_episode_count"], 2)
            self.assertEqual(pooled["selected_episode_count"], 1)
            self.assertEqual(
                pooled["selection_rule"],
                "first N accepted episodes in input order, then numeric demo order",
            )
            self.assertEqual(pooled["selected_records"][0]["source_file"], str(first.resolve()))

    def test_grasp_xyz_and_carry_speed_are_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                demo = handle["data/demo_0"]
                demo.attrs["quality_coffee_can_right_grasp_diagnostic_can_local_error_x_m"] = 100.0
                demo.attrs["quality_coffee_can_right_carry_can_linear_speed_mps"] = 100.0
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertTrue(report["passed"], report["errors"])
            self.assertFalse(report["grasp_diagnostics_are_acceptance_gate"])

    def test_wrong_campaign_correction_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                handle["data/demo_0"].attrs["grasp_correction_can_x_m"] = -0.01
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertFalse(report["passed"])
            self.assertTrue(any("grasp_correction_can_x_m" in reason for reason in report["records"][0]["reasons"]))


    def test_missing_v13_production_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                del handle["data"].attrs["production_promotion_id"]
                del handle["data/demo_0"].attrs["production_promotion_id"]
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("production_promotion_id" in error for error in report["errors"])
            )
            self.assertTrue(
                any(
                    "production_promotion_id" in reason
                    for reason in report["records"][0]["reasons"]
                )
            )

    def test_intermediate_task000001_lift_targets_are_rejected(self):
        for path in ("actions", "processed_actions", "obs/joint_pos_target"):
            with self.subTest(path=path), tempfile.TemporaryDirectory() as directory:
                dataset = Path(directory) / "physical.hdf5"
                create_file(dataset, "000001")
                with h5py.File(dataset, "r+") as handle:
                    frame = 1 if path == "obs/joint_pos_target" else 0
                    handle[f"data/demo_0/{path}"][frame, 18] = -0.10
                report = self.module.audit_inputs([dataset], "000001", 1, False)
                self.assertFalse(report["passed"])
                reasons = report["records"][0]["reasons"]
                self.assertTrue(
                    any(
                        path in reason and "intermediate absolute target" in reason
                        for reason in reasons
                    ),
                    reasons,
                )


    def test_placement_checkpoint_uses_state_t_and_eef_t_plus_one(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                handle["data/demo_0"].attrs[
                    "quality_coffee_can_right_placement_can_angular_speed_radps"
                ] = 0.1
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertFalse(report["passed"])
            self.assertTrue(
                any(
                    "cannot be aligned" in reason
                    for reason in report["records"][0]["reasons"]
                )
            )

    def test_fixed_base_root_drift_is_reported_but_not_an_invented_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                handle[
                    "data/demo_0/states/articulation/robot/root_pose"
                ][-5:, 0] += 0.04
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertTrue(report["passed"], report["errors"])
            route = report["records"][0]["task_route"]
            self.assertTrue(route["fixed_base_root_drift_reference_exceeded"])
            self.assertFalse(route["fixed_base_root_stationarity_is_acceptance_gate"])

            curated = self.module.audit_inputs(
                [path],
                "000001",
                1,
                True,
                require_fixed_base_stationarity=True,
            )
            self.assertFalse(curated["passed"])
            self.assertEqual(curated["accepted_episode_count"], 0)
            self.assertEqual(curated["rejected_episode_count"], 1)
            self.assertTrue(curated["require_fixed_base_stationarity"])
            self.assertEqual(
                curated["task_diagnostics_summary"][
                    "fixed_base_root_policy_excluded_count"
                ],
                1,
            )
            self.assertTrue(
                any(
                    "17D arms+grippers+lift policy excludes base state/action" in reason
                    for reason in curated["records"][0]["reasons"]
                ),
                curated["records"][0]["reasons"],
            )

    def test_final_placement_is_recomputed_from_raw_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                handle["data/demo_0/states/rigid_object/coffee_can_left/root_pose"][-1, 0] = -3.0
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertFalse(report["passed"])
            reasons = report["records"][0]["reasons"]
            self.assertTrue(any("final center" in reason for reason in reasons), reasons)

    def test_head_motion_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                handle["data/demo_0/obs/joint_pos"][-1, 16] = 0.02
            report = self.module.audit_inputs([path], "000001", 1, False)
            self.assertFalse(report["passed"])
            self.assertTrue(any("head changes" in reason for reason in report["records"][0]["reasons"]))

    def test_previous_episode_frame_leak_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "physical.hdf5"
            create_file(path, "000001")
            with h5py.File(path, "r+") as handle:
                data = handle["data"]
                data.copy(data["demo_0"], "demo_1")
                data["demo_1"].attrs["generation_seed"] = 102
                data.attrs["total"] = 240
                camera0 = data["demo_1/obs/cam_head"]
                camera0[0] = data["demo_0/obs/cam_head"][-1]
            report = self.module.audit_inputs([path], "000001", 2, False)
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("previous episode" in reason for reason in report["records"][1]["reasons"]),
                report["records"][1]["reasons"],
            )


    def test_selected_hdf_is_contiguous_and_stamped(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "physical.hdf5"
            output = Path(directory) / "selected.hdf5"
            create_file(source, "000001")
            report = self.module.audit_inputs([source], "000001", 1, False)
            evidence = {
                "process_started_utc": "2026-09-07T17:46:36Z",
                "generation_log_path": "/evidence/generation.log",
                "generation_log_sha256": "b" * 64,
                "isaac_kit_log_name": "kit_20260908_024636.log",
                "environment_cfg": "Task000001TrajectoryGenerationEnvCfg",
                "initial_seed": 101,
                "logged_environment_steps_s": {
                    "physics": 1.0 / 120.0,
                    "rendering": 1.0 / 15.0,
                    "environment": 1.0 / 15.0,
                },
            }
            report["runtime_rate_attestation"] = {
                "schema": "cyclo.coffee_transport.runtime_rate_attestation.v1",
                "attestation_id": "coffee_transport_final450_v12_runtime_rates_20260908",
                "review_status": "reviewed",
                "reviewed_utc": "2026-09-08T09:30:00Z",
                "manifest_path": "/reviewed/rates.json",
                "manifest_sha256": "a" * 64,
                "rate_config_commit": "f106306bb450f6fb38cb3abf1f35f528b9c1b994",
                "rate_contract": {
                    "physics_hz": 120.0,
                    "control_hz": 15.0,
                    "camera_hz": 15.0,
                    "decimation": 8,
                    "render_interval": 8,
                    "swerve_controller_hz": 30.0,
                    "swerve_update_interval_physics_steps": 4,
                },
                "inputs": [{
                    "source_file": str(source.resolve()),
                    "hdf5_sha256": "c" * 64,
                    "actual_process_start_evidence": evidence,
                }],
            }
            self.module.write_selected_hdf5(report, output)
            with h5py.File(output, "r") as handle:
                self.assertEqual(list(handle["data"]), ["demo_0"])
                self.assertEqual(handle["data"].attrs["physical_audit_selected_count"], 1)
                self.assertEqual(
                    handle["data/demo_0"].attrs["physical_audit_contract_id"],
                    self.module.AUDIT_CONTRACT_ID,
                )
                self.assertEqual(
                    handle["data"].attrs["physical_audit_selection_rule"],
                    report["selection_rule"],
                )
                self.assertEqual(
                    handle["data/demo_0"].attrs["physical_audit_source_file"],
                    str(source.resolve()),
                )
                self.assertEqual(
                    handle["data/demo_0"].attrs["physical_audit_source_demo"],
                    "demo_0",
                )
                self.assertEqual(handle["data"].attrs["physics_hz"], 120.0)
                self.assertEqual(handle["data"].attrs["decimation"], 8)
                self.assertEqual(
                    handle["data"].attrs["rate_config_commit"],
                    "f106306bb450f6fb38cb3abf1f35f528b9c1b994",
                )
                self.assertEqual(
                    handle["data/demo_0"].attrs[
                        "runtime_rate_attestation_source_process_started_utc"
                    ],
                    "2026-09-07T17:46:36Z",
                )


if __name__ == "__main__":
    unittest.main()
