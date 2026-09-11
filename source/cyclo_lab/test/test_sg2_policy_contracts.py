"""Regression tests for the two deployable SG2 learned-policy contracts."""

import unittest

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000458.policy import (
    POLICY_CONTRACT as TASK458_POLICY,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.policy import (
    POLICY_CONTRACT as TASK525_POLICY,
)
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_ACTION_JOINT_NAMES,
    FFW_SG2_MOBILE_ACTION_NAMES,
)


EXPECTED_CAMERAS = {
    "cam_left_head": {"width": 672, "height": 376},
    "cam_left_wrist": {"width": 480, "height": 640},
    "cam_right_wrist": {"width": 480, "height": 640},
}


class TestSG2PolicyContracts(unittest.TestCase):
    def test_task458_is_stationary_19d(self):
        self.assertEqual(TASK458_POLICY["policy_hz"], 15)
        self.assertEqual(TASK458_POLICY["state_names"], FFW_SG2_ACTION_JOINT_NAMES)
        self.assertEqual(TASK458_POLICY["action_names"], FFW_SG2_ACTION_JOINT_NAMES)
        self.assertEqual(len(TASK458_POLICY["action_names"]), 19)
        self.assertNotIn("mobile", TASK458_POLICY["action_components"])

    def test_task525_is_mobile_22d(self):
        self.assertEqual(TASK525_POLICY["policy_hz"], 15)
        self.assertEqual(TASK525_POLICY["state_names"], FFW_SG2_MOBILE_ACTION_NAMES)
        self.assertEqual(TASK525_POLICY["action_names"], FFW_SG2_MOBILE_ACTION_NAMES)
        self.assertEqual(len(TASK525_POLICY["action_names"]), 22)

    def test_both_tasks_use_the_canonical_cameras_and_safe_envs(self):
        for policy in (TASK458_POLICY, TASK525_POLICY):
            self.assertEqual(policy["cameras"], EXPECTED_CAMERAS)
            self.assertEqual(policy["simulation"]["default_reset"], "deterministic")
            self.assertNotIn("Locomanipulation", policy["simulation"]["environment"])
            self.assertNotIn("Mimic", policy["simulation"]["environment"])


if __name__ == "__main__":
    unittest.main()
