"""Regression tests for the SG2 Showroom multi-rate contract."""

import importlib.util
import math
import unittest
from pathlib import Path


PACKAGE_ROOT = Path(__file__).parents[1] / "cyclo_lab"


def _load_module(name: str, relative_path: str):
    spec = importlib.util.spec_from_file_location(name, PACKAGE_ROOT / relative_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {relative_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controller_rate = _load_module(
    "sg2_controller_rate", "robot_specs/ffw/mobile_base/controller_rate.py"
)
showroom_rates = _load_module(
    "sg2_showroom_rates",
    "manager_based/manipulation/showroom/config/ffw_sg2/platform/rates.py",
)


class TestSg2ShowroomRateContract(unittest.TestCase):
    def test_showroom_defaults_keep_control_and_render_at_15_hz(self):
        physics_hz = showroom_rates.SG2_SHOWROOM_PHYSICS_HZ
        control_hz = showroom_rates.SG2_SHOWROOM_CONTROL_HZ
        camera_hz = showroom_rates.SG2_SHOWROOM_CAMERA_HZ

        self.assertEqual(physics_hz, 120.0)
        self.assertEqual(control_hz, 15.0)
        self.assertEqual(camera_hz, 15.0)
        self.assertEqual(physics_hz / control_hz, 8.0)
        self.assertEqual(physics_hz / camera_hz, 8.0)

    def test_swerve_updates_every_four_physics_ticks(self):
        scheduler = controller_rate.ControllerRateScheduler(
            physics_dt=1.0 / 120.0, controller_hz=30.0
        )

        updates = [tick for tick in range(12) if scheduler.update_due()]
        self.assertEqual(updates, [0, 4, 8])
        self.assertEqual(scheduler.physics_ticks_per_update, 4)
        self.assertTrue(math.isclose(scheduler.controller_dt, 1.0 / 30.0))

        scheduler.reset()
        self.assertTrue(scheduler.update_due())

    def test_non_integer_rate_ratio_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "integer multiple"):
            controller_rate.ControllerRateScheduler(
                physics_dt=1.0 / 120.0, controller_hz=100.0
            )


if __name__ == "__main__":
    unittest.main()
