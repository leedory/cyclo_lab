"""Tests for Coffee Transport recorder timing alignment."""

from pathlib import Path
import runpy
import unittest

import numpy as np


TIMING_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "coffee_transport"
    / "trajectory_timing.py"
)


class CoffeeTransportTrajectoryTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene_pose_at_pre_step = staticmethod(
            runpy.run_path(str(TIMING_PATH))["scene_pose_at_pre_step"]
        )

    def test_step_zero_uses_initial_state(self):
        initial = np.full(7, -1.0)
        post_step = np.arange(21).reshape(3, 7)
        np.testing.assert_array_equal(
            self.scene_pose_at_pre_step(initial, post_step, 0), initial
        )

    def test_later_step_uses_previous_post_step_state(self):
        initial = np.full(7, -1.0)
        post_step = np.arange(21).reshape(3, 7)
        np.testing.assert_array_equal(
            self.scene_pose_at_pre_step(initial, post_step, 1), post_step[0]
        )
        np.testing.assert_array_equal(
            self.scene_pose_at_pre_step(initial, post_step, 3), post_step[2]
        )

    def test_invalid_step_is_rejected_instead_of_clamped(self):
        initial = np.zeros(7)
        post_step = np.zeros((2, 7))
        with self.assertRaisesRegex(IndexError, "non-negative"):
            self.scene_pose_at_pre_step(initial, post_step, -1)
        with self.assertRaisesRegex(IndexError, "only 2 states"):
            self.scene_pose_at_pre_step(initial, post_step, 3)


if __name__ == "__main__":
    unittest.main()
