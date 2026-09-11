"""Pure tests for measured coffee-grasp replay helpers."""

from pathlib import Path
import runpy
import unittest


GRASP_TRACKING_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "coffee_transport"
    / "grasp_tracking.py"
)


class CoffeeTransportGraspTrackingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = runpy.run_path(str(GRASP_TRACKING_PATH))

    def test_close_onset_ignores_early_open_gripper_settling(self):
        values = (
            [0.0, 0.01, 0.03, 0.05, 0.07]
            + [0.07] * 15
            + [0.08, 0.12, 0.24, 0.48, 0.72, 0.96, 1.09]
            + [1.10] * 12
        )
        segment = self.module["infer_close_segment"](
            values,
            operator_start_step=0,
            grasp_step=27,
            g_step=len(values),
        )
        self.assertEqual(segment.onset_step, 20)
        self.assertAlmostEqual(segment.open_value, 0.07)
        self.assertAlmostEqual(segment.closed_value, 1.10)

    def test_close_onset_allows_short_gaps_inside_close_ramp(self):
        values = [0.06] * 30
        values[17] = 0.07
        values[18] = 0.07
        values[20] = 0.10
        values[21] = 0.10
        values[24:30] = [0.22, 0.46, 0.70, 0.94, 1.10, 1.10]
        segment = self.module["infer_close_segment"](
            values,
            operator_start_step=0,
            grasp_step=28,
            g_step=30,
        )
        self.assertEqual(segment.onset_step, 17)

    def test_close_segment_rejects_missing_excursion(self):
        with self.assertRaisesRegex(ValueError, "no clear close excursion"):
            self.module["infer_close_segment"](
                [0.05] * 20,
                operator_start_step=0,
                grasp_step=10,
                g_step=20,
            )

    def test_translation_alignment_is_per_axis_inclusive_at_one_centimeter(self):
        passes = self.module["translation_alignment_passes"]
        self.assertTrue(passes((0.010, -0.010, 0.009)))
        self.assertFalse(passes((0.0101, 0.0, 0.0)))
        self.assertFalse(passes((0.0, -0.0101, 0.0)))

    def test_approved_timeouts_and_stability_are_explicit(self):
        cfg = self.module["DEFAULT_GRASP_TRACKING"]
        self.assertEqual(cfg.position_tolerance_m, 0.010)
        self.assertEqual(cfg.stable_steps, 5)
        self.assertGreater(cfg.preclose_timeout_steps, 0)
        self.assertGreater(cfg.closed_alignment_timeout_steps, 0)


if __name__ == "__main__":
    unittest.main()
