"""Pure tests for coffee body-carry and placement quality gates."""

from pathlib import Path
import math
import runpy
import unittest


QUALITY_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "coffee_transport"
    / "quality_contract.py"
)


class CoffeeTransportQualityContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.contract = runpy.run_path(str(QUALITY_PATH))

    def valid_body_carry(self, **overrides):
        values = {
            "eef_object_distance_m": 0.080,
            "source_eef_object_distance_m": 0.075,
            "tcp_z_relative_to_can_m": 0.020,
            "can_displacement_m": 0.300,
            "home_joint_max_error": 0.050,
            "can_linear_speed_mps": 0.010,
            "can_angular_speed_radps": 0.100,
        }
        values.update(overrides)
        return self.contract["evaluate_body_carry"](**values)

    def valid_placement(self, **overrides):
        values = {
            "center_x_m": 0.0,
            "center_y_m": 0.0,
            "origin_z_m": 1.0,
            "expected_origin_z_m": 1.0,
            "upright_cosine": 1.0,
            "can_linear_speed_mps": 0.010,
            "can_angular_speed_radps": 0.100,
            "active_eef_distance_m": 0.150,
            "center_region_xyxy_m": (-0.06, 0.06, -0.06, 0.06),
        }
        values.update(overrides)
        return self.contract["evaluate_placement"](**values)

    def test_body_grasp_passes_but_measured_task525_rim_grasps_fail(self):
        self.assertTrue(self.valid_body_carry().passed)
        for rim_tcp_z_m in (0.06960, 0.07639, 0.07684, 0.07899):
            with self.subTest(rim_tcp_z_m=rim_tcp_z_m):
                decision = self.valid_body_carry(
                    tcp_z_relative_to_can_m=rim_tcp_z_m
                )
                self.assertFalse(decision.passed)
                self.assertIn(
                    "tcp_outside_body_grasp_band",
                    decision.failures,
                )

    def test_body_carry_rejects_drop_motion_and_missing_home(self):
        cases = (
            ({"eef_object_distance_m": 0.120}, "eef_too_far_from_can"),
            ({"can_displacement_m": 0.050}, "can_not_clear_of_source"),
            ({"home_joint_max_error": 0.200}, "carry_home_not_reached"),
            (
                {"can_linear_speed_mps": 0.060},
                "can_linear_motion_not_stable",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.valid_body_carry(**overrides).failures)

    def test_generation_carry_treats_speed_as_diagnostic(self):
        decision = self.valid_body_carry(
            can_linear_speed_mps=0.25,
            can_angular_speed_radps=6.0,
            require_stability=False,
        )
        self.assertTrue(decision.passed)

    def test_body_grasp_lower_bound_is_three_centimeters_below_can_origin(self):
        self.assertTrue(
            self.valid_body_carry(tcp_z_relative_to_can_m=-0.030).passed
        )
        decision = self.valid_body_carry(tcp_z_relative_to_can_m=-0.0301)
        self.assertIn("tcp_outside_body_grasp_band", decision.failures)

    def test_upright_cosine_uses_can_local_z(self):
        upright = self.contract["upright_cosine_from_wxyz"]
        self.assertAlmostEqual(upright((1.0, 0.0, 0.0, 0.0)), 1.0)
        ten_degrees = math.radians(10.0)
        self.assertAlmostEqual(
            upright((math.cos(ten_degrees / 2.0), math.sin(ten_degrees / 2.0), 0.0, 0.0)),
            math.cos(ten_degrees),
        )

    def test_placement_requires_named_square_support_upright_settle_and_release(self):
        self.assertTrue(self.valid_placement().passed)
        cases = (
            (
                {"center_x_m": 0.067},
                "can_center_outside_named_destination_region",
            ),
            ({"origin_z_m": 1.011}, "can_not_on_destination_support"),
            (
                {"upright_cosine": math.cos(math.radians(11.0))},
                "can_not_upright",
            ),
            (
                {"can_angular_speed_radps": 0.251},
                "can_angular_motion_not_settled",
            ),
            (
                {"active_eef_distance_m": 0.099},
                "active_gripper_not_released",
            ),
        )
        for overrides, expected in cases:
            with self.subTest(expected=expected):
                self.assertIn(expected, self.valid_placement(**overrides).failures)

    def test_placement_center_uses_the_task_six_millimeter_tolerance(self):
        self.assertTrue(self.valid_placement(center_x_m=0.066).passed)
        self.assertIn(
            "can_center_outside_named_destination_region",
            self.valid_placement(center_x_m=0.0661).failures,
        )

    def test_stability_windows_are_explicit(self):
        self.assertEqual(
            self.contract["DEFAULT_BODY_CARRY_QUALITY"].stable_steps,
            5,
        )
        self.assertEqual(
            self.contract["DEFAULT_PLACEMENT_QUALITY"].stable_steps,
            8,
        )


if __name__ == "__main__":
    unittest.main()
