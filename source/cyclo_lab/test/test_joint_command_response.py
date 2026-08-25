"""CPU unit tests for the delayed first-order joint command response."""

from __future__ import annotations

import math
import unittest

import torch

from cyclo_lab.utils.joint_command_response import CommandResponseGroup, JointCommandResponse


class TestJointCommandResponse(unittest.TestCase):
    def test_zero_delay_and_filter_are_identity(self) -> None:
        response = JointCommandResponse(
            num_envs=1,
            num_joints=1,
            physics_dt=0.01,
            groups=(CommandResponseGroup("joint", (0,), 0.0, 0.0),),
        )
        response.reset(torch.zeros(1, 1))

        target = torch.tensor([[0.75]])
        torch.testing.assert_close(response.update(target), target)

    def test_delay_is_quantized_to_physics_steps(self) -> None:
        response = JointCommandResponse(
            num_envs=1,
            num_joints=1,
            physics_dt=0.01,
            groups=(CommandResponseGroup("joint", (0,), 0.085, 0.0),),
        )
        response.reset(torch.zeros(1, 1))

        step = torch.ones(1, 1)
        self.assertAlmostEqual(response.realized_delay_seconds["joint"].item(), 0.08, places=6)
        for _ in range(8):
            self.assertAlmostEqual(response.update(step).item(), 0.0, places=6)
        self.assertAlmostEqual(response.update(step).item(), 1.0, places=6)

    def test_filter_uses_exact_discrete_first_order_update(self) -> None:
        time_constant = 0.07
        response = JointCommandResponse(
            num_envs=1,
            num_joints=1,
            physics_dt=0.01,
            groups=(CommandResponseGroup("joint", (0,), 0.0, time_constant),),
        )
        response.reset(torch.zeros(1, 1))

        expected = 1.0 - math.exp(-0.01 / time_constant)
        self.assertAlmostEqual(response.update(torch.ones(1, 1)).item(), expected, places=6)

    def test_one_gaussian_scale_changes_delay_and_filter_together(self) -> None:
        torch.manual_seed(7)
        response = JointCommandResponse(
            num_envs=32,
            num_joints=2,
            physics_dt=0.01,
            groups=(
                CommandResponseGroup(
                    "arm",
                    (0, 1),
                    0.08,
                    0.04,
                    variation_std_fraction=0.05,
                    variation_scale_bounds=(0.85, 1.15),
                ),
            ),
        )
        response.reset(torch.zeros(32, 2))

        scales = response.response_scales["arm"]
        self.assertGreater(torch.std(scales).item(), 0.0)
        self.assertGreaterEqual(torch.min(scales).item(), 0.85)
        self.assertLessEqual(torch.max(scales).item(), 1.15)
        torch.testing.assert_close(response.delay_seconds["arm"] / 0.08, scales)
        torch.testing.assert_close(response.filter_time_constants["arm"] / 0.04, scales)

    def test_offset_is_applied_after_dynamic_response(self) -> None:
        response = JointCommandResponse(
            num_envs=1,
            num_joints=1,
            physics_dt=0.01,
            groups=(CommandResponseGroup("joint", (0,), 0.0, 0.0),),
            target_offsets=torch.tensor([0.2]),
        )
        response.reset(torch.zeros(1, 1))

        self.assertAlmostEqual(response.update(torch.tensor([[0.5]])).item(), 0.7, places=6)


if __name__ == "__main__":
    unittest.main()
