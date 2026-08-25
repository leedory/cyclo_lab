"""Unit tests for the evidence-scoped SG2 joint response profile."""

from __future__ import annotations

import unittest

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
    SG2_MEASURED_RESPONSE_GROUPS,
    SG2_MEASURED_RESPONSE_VARIATION_SCALE_BOUNDS,
    SG2_MEASURED_RESPONSE_VARIATION_STD_FRACTION,
    SG2_MEASURED_TARGET_OFFSETS_RAD,
)


class TestSG2JointResponseProfile(unittest.TestCase):
    def test_profile_covers_only_measured_arm_and_gripper_joints(self) -> None:
        expected = {
            *FFW_SG2_LEFT_ARM_JOINT_NAMES,
            *FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
            *FFW_SG2_RIGHT_ARM_JOINT_NAMES,
            *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
        }
        covered = [name for group in SG2_MEASURED_RESPONSE_GROUPS for name in group.joint_names]

        self.assertEqual(set(covered), expected)
        self.assertEqual(len(covered), len(set(covered)))
        self.assertEqual(set(SG2_MEASURED_TARGET_OFFSETS_RAD), expected)

    def test_nominal_profile_matches_selected_values(self) -> None:
        values = {
            group.name: (group.delay_seconds, group.filter_time_constant_seconds)
            for group in SG2_MEASURED_RESPONSE_GROUPS
        }

        self.assertEqual(values["arm joints 1 to 6"], (0.085, 0.070))
        self.assertEqual(values["arm joint 7"], (0.070, 0.035))
        self.assertEqual(values["gripper"], (0.010, 0.050))
        self.assertEqual(SG2_MEASURED_RESPONSE_VARIATION_STD_FRACTION, 0.05)
        self.assertEqual(SG2_MEASURED_RESPONSE_VARIATION_SCALE_BOUNDS, (0.85, 1.15))


if __name__ == "__main__":
    unittest.main()
