"""Contract tests for Coffee Transport causal ACT staging."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
import sys

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
COMMON_PATH = ROOT / "scripts/sim2real/imitation_learning/tasks/coffee_transport/policy_staging_common.py"


def load_common():
    sys.path.insert(0, str(COMMON_PATH.parent))
    spec = importlib.util.spec_from_file_location("coffee_policy_staging_common", COMMON_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class TestCoffeePolicyStagingContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.common = load_common()

    def make_group(self, task_id: str):
        handle = h5py.File(tempfile.NamedTemporaryFile(suffix=".hdf5").name, "w")
        group = handle.create_group("data/demo_0")
        frames = 4
        width = 19 if task_id == "000001" else 22
        group.create_dataset("actions", data=np.zeros((frames, width), dtype=np.float32))
        observation = group.create_group("obs")
        joint = np.arange(frames * 19, dtype=np.float32).reshape(frames, 19)
        target = joint + 1000.0
        if task_id == "000001":
            # Row 0 is the unused reset target. ACT uses target[t + 1].
            target[:, 18] = np.asarray([-0.10, 0.0, -0.28, 0.0], np.float32)
        observation.create_dataset("joint_pos", data=joint)
        observation.create_dataset("joint_pos_target", data=target)
        initial = group.create_group("initial_state/articulation/robot")
        initial.create_dataset("root_pose", data=np.asarray([[0, 0, 0, 1, 0, 0, 0]], np.float32))
        initial.create_dataset("root_velocity", data=np.zeros((1, 6), np.float32))
        states = group.create_group("states/articulation/robot")
        pose = np.tile(np.asarray([0, 0, 0, 1, 0, 0, 0], np.float32), (frames, 1))
        states.create_dataset("root_pose", data=pose)
        velocity = np.zeros((frames, 6), np.float32)
        velocity[:, 1] = np.asarray([1, 2, 3, 4], np.float32)
        states.create_dataset("root_velocity", data=velocity)
        return handle, group, joint

    def test_task000001_is_causal_arm_gripper_lift_17d(self):
        handle, group, joint = self.make_group("000001")
        with handle:
            state, action = self.common.derive_policy_arrays(group, "000001")
        self.assertEqual(state.shape, (3, 17))
        self.assertTrue(np.array_equal(state[:, :16], joint[:-1, :16]))
        self.assertTrue(np.array_equal(state[:, 16], joint[:-1, 18]))
        self.assertTrue(np.array_equal(action[:, :16], joint[1:, :16] + 1000.0))
        self.assertTrue(
            np.array_equal(
                action[:, 16],
                np.asarray([0.0, -0.28, 0.0], dtype=np.float32),
            )
        )
        self.assertNotIn("head_joint1", self.common.TASK_POLICY_NAMES["000001"])

    def test_task000001_rejects_intermediate_lift_action_label(self):
        handle, group, _joint = self.make_group("000001")
        group["obs/joint_pos_target"][1, 18] = -0.10
        with handle, self.assertRaisesRegex(
            self.common.CoffeePolicyDataError,
            "absolute endpoints 0.0 or -0.28 m",
        ):
            self.common.derive_policy_arrays(group, "000001")

    def test_task000002_is_causal_arm_gripper_mobile_19d(self):
        handle, group, joint = self.make_group("000002")
        source_action = np.asarray(group["actions"])
        source_action[:, 19:22] = np.asarray(
            [[0.01, 0.1, -0.001], [0.02, 0.2, -0.002],
             [0.03, 0.3, -0.003], [0.04, 0.4, -0.004]],
            dtype=np.float32,
        )
        del group["actions"]
        group.create_dataset("actions", data=source_action)
        with handle:
            state, action = self.common.derive_policy_arrays(group, "000002")
        self.assertEqual(state.shape, (3, 19))
        # pre-step t=0 uses initial twist 0; t=1/2 use post-step 0/1.
        self.assertTrue(np.array_equal(state[:, 16], np.zeros(3)))
        self.assertTrue(np.array_equal(state[:, 17], np.asarray([0.0, 1.0, 2.0])))
        self.assertTrue(np.array_equal(state[:, 18], np.zeros(3)))
        self.assertTrue(np.allclose(action[:, 16:19], source_action[:-1, 19:22]))
        self.assertEqual(
            self.common.TASK_POLICY_NAMES["000002"][-3:],
            ("linear_x", "linear_y", "angular_z"),
        )

    def test_body_lateral_velocity_respects_root_yaw(self):
        handle, group, _joint = self.make_group("000002")
        # +90 degree yaw: world -X is body +Y.
        q = np.sqrt(0.5)
        group["initial_state/articulation/robot/root_pose"][0, 3:] = [q, 0, 0, q]
        group["states/articulation/robot/root_pose"][:, 3:] = [q, 0, 0, q]
        group["initial_state/articulation/robot/root_velocity"][0, :3] = [-2, 0, 0]
        with handle:
            twist = self.common._pre_step_body_twist(group, 4)
        self.assertAlmostEqual(float(twist[0, 0]), 0.0, places=5)
        self.assertAlmostEqual(float(twist[0, 1]), 2.0, places=5)
        self.assertAlmostEqual(float(twist[0, 2]), 0.0, places=5)

    def test_staging_requires_reviewed_task_specific_source_metadata(self):
        for task_id, task_metadata in self.common.REVIEWED_SOURCE_METADATA.items():
            metadata = {
                **self.common.REVIEWED_COMMON_SOURCE_METADATA,
                **task_metadata,
            }
            with self.subTest(task_id=task_id):
                self.common.validate_reviewed_source_metadata(metadata, task_id)

    def test_staging_rejects_superseded_task000001_v12_source(self):
        metadata = {
            **self.common.REVIEWED_COMMON_SOURCE_METADATA,
            **self.common.REVIEWED_SOURCE_METADATA["000001"],
            "generation_profile_id": (
                "coffee_physical_trajectory_generation_commanded_joint_hold_v12"
            ),
            "production_promotion_id": (
                "coffee_commanded_joint_hold_v12_promoted_20260908"
            ),
        }
        with self.assertRaisesRegex(
            self.common.CoffeePolicyDataError,
            "not an approved task 000001 physical dataset",
        ):
            self.common.validate_reviewed_source_metadata(metadata, "000001")

    def test_allcams_contract_has_canonical_shapes_and_no_rotation(self):
        self.assertEqual(
            self.common.CAMERA_MAP,
            {
                "cam_head": "cam_left_head",
                "cam_wrist_left": "cam_left_wrist",
                "cam_wrist_right": "cam_right_wrist",
            },
        )
        for camera, shape in self.common.CAMERA_SHAPES.items():
            frame = np.zeros((*shape, 4), dtype=np.uint8)
            canonical = self.common.canonicalize_camera_frame(camera, frame)
            self.assertEqual(canonical.shape, (*shape, 3))
            self.assertEqual(self.common.CAMERA_ROTATION_DEG[camera], 0)


if __name__ == "__main__":
    unittest.main()
