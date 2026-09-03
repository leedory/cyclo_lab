"""Regression tests for canonical Task525 joint22 policy staging."""

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np


REPO = Path(__file__).resolve().parents[3]
COMMON = (
    REPO
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000525"
    / "policy_staging_common.py"
)


def load_common():
    spec = spec_from_file_location("task_000525_policy_staging_common", COMMON)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTask525PolicyStagingContract(unittest.TestCase):
    def make_source(self, module, path: Path):
        source = h5py.File(path, "w")
        data = source.create_group("data")
        names = list(module.CANONICAL_STATE_ACTION_NAMES)
        data.attrs["target_object_name"] = module.POLICY_TARGET_OBJECT_NAME
        data.attrs["robot_contract_id"] = module.POLICY_CONTRACT_ID
        data.attrs["action_semantics"] = module.POLICY_ACTION_SEMANTICS
        data.attrs["observation_state_names"] = json.dumps(names)
        data.attrs["action_names"] = json.dumps(names)
        data.attrs["control_hz"] = 15.0
        demo = data.create_group("demo_0")
        actions = np.arange(6 * 22, dtype=np.float32).reshape(6, 22)
        joint = np.arange(6 * 19, dtype=np.float32).reshape(6, 19) / 10.0
        base = np.arange(6 * 3, dtype=np.float32).reshape(6, 3) / 100.0
        demo.create_dataset("actions", data=actions)
        observations = demo.create_group("obs")
        observations.create_dataset("joint_pos", data=joint)
        observations.create_dataset("base_velocity_body", data=base)
        observations.create_dataset(
            "task525_demo_phase",
            data=np.asarray([[0], [0], [1], [1], [2], [2]], dtype=np.int64),
        )
        return source, data, demo, actions, joint, base

    def test_canonical_source_uses_direct_actions_without_dropping_last_row(self):
        module = load_common()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "canonical.hdf5"
            source, data, demo, actions, joint, base = self.make_source(module, path)
            with source:
                contract = module.validate_source(data)
                state, action, tasks = module.derive_policy_arrays(demo)

            self.assertEqual(contract["source_contract_id"], module.POLICY_CONTRACT_ID)
            self.assertEqual(contract["state_names"], list(module.CANONICAL_STATE_ACTION_NAMES))
            self.assertEqual(contract["action_names"], list(module.CANONICAL_STATE_ACTION_NAMES))
            np.testing.assert_array_equal(action, actions)
            np.testing.assert_array_equal(state, np.concatenate((joint, base), axis=-1))
            np.testing.assert_array_equal(tasks, [0, 0, 1, 1, 2, 2])
            self.assertEqual(module.phase_bounds(tasks, "pick"), (0, 4))

    def test_current_dual_eef_generator_derives_causal_joint22_and_trims_once(self):
        module = load_common()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "generator.hdf5"
            source, data, demo, source_actions, joint, base = self.make_source(
                module, path
            )
            data.attrs["robot_contract_id"] = module.SDG_CONTRACT_ID
            data.attrs["action_semantics"] = module.SDG_ACTION_SEMANTICS
            data.attrs["action_names"] = json.dumps(list(module.SDG_ACTION_NAMES))
            del data.attrs["target_object_name"]
            demo.attrs["target_object_name"] = module.POLICY_TARGET_OBJECT_NAME
            observations = demo["obs"]
            phase = np.asarray(observations["task525_demo_phase"])
            del observations["task525_demo_phase"]
            task_output = demo.create_group("locomanipulation_sdg_output_data")
            task_output.create_dataset("task", data=phase)
            joint_target = (
                np.arange(6 * 19, dtype=np.float32).reshape(6, 19) / 7.0
            )
            observations.create_dataset("joint_pos_target", data=joint_target)

            with source:
                contract = module.validate_source(data)
                state, action, tasks = module.derive_policy_arrays(demo)

            expected_action = np.concatenate(
                (joint_target[1:], source_actions[:-1, 19:22]), axis=-1
            )
            expected_state = np.concatenate((joint[:-1], base[:-1]), axis=-1)
            self.assertEqual(contract["source_format"], module.SDG_SOURCE_FORMAT)
            self.assertEqual(contract["source_action_names"], list(module.SDG_ACTION_NAMES))
            self.assertEqual(contract["action_names"], list(module.CANONICAL_STATE_ACTION_NAMES))
            np.testing.assert_array_equal(action, expected_action)
            np.testing.assert_array_equal(state, expected_state)
            np.testing.assert_array_equal(tasks, [0, 0, 1, 1, 2])
            self.assertEqual(module.phase_bounds(tasks, "pick"), (0, 4))

    def test_noncanonical_contract_and_semantics_are_rejected(self):
        module = load_common()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.hdf5"
            source, data, _demo, _actions, _joint, _base = self.make_source(module, path)
            with source:
                data.attrs["robot_contract_id"] = "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
                with self.assertRaisesRegex(module.Task525PolicyDataError, "robot_contract_id"):
                    module.validate_source(data)
                data.attrs["robot_contract_id"] = module.POLICY_CONTRACT_ID
                data.attrs["action_semantics"] = "derived_legacy_action"
                with self.assertRaisesRegex(module.Task525PolicyDataError, "action_semantics"):
                    module.validate_source(data)
                data.attrs["robot_contract_id"] = "ffw_sg2_task525_hybrid22_v0"
                data.attrs["action_semantics"] = module.SDG_ACTION_SEMANTICS
                with self.assertRaisesRegex(module.Task525PolicyDataError, "active exact"):
                    module.validate_source(data)

    def test_camera_subset_is_returned_in_canonical_order(self):
        module = load_common()
        camera_map = module.select_camera_map(
            ["cam_wrist_right", "cam_head", "cam_wrist_left"]
        )
        self.assertEqual(list(camera_map), list(module.CANONICAL_CAMERA_MAP))

    def test_pick_rejects_nonmonotonic_zero_one_phases(self):
        module = load_common()
        tasks = np.asarray([0, 1, 0, 1, 2, 2], dtype=np.int64)
        with self.assertRaisesRegex(
            module.Task525PolicyDataError, "monotonic 0 then 1"
        ):
            module.phase_bounds(tasks, "pick")


if __name__ == "__main__":
    unittest.main()
