"""Regression tests for the SG2 raw wrist-camera orientation contract."""

from __future__ import annotations

import math
import unittest

from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    FFW_SG2_D405_CAMERA_UPRIGHT_ROT,
    FFW_SG2_WRIST_CAMERA_HEIGHT,
    FFW_SG2_WRIST_CAMERA_WIDTH,
)


class TestSG2WristCameraOrientation(unittest.TestCase):
    def test_upright_raw_frame_is_portrait(self) -> None:
        self.assertEqual((FFW_SG2_WRIST_CAMERA_WIDTH, FFW_SG2_WRIST_CAMERA_HEIGHT), (480, 640))

    def test_upright_rotation_aligns_optical_axis_without_image_roll(self) -> None:
        """The wrist camera uses only +90 degrees about link Y (w, x, y, z)."""

        expected = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)
        for actual_component, expected_component in zip(
            FFW_SG2_D405_CAMERA_UPRIGHT_ROT, expected
        ):
            self.assertAlmostEqual(actual_component, expected_component, places=7)

        norm = math.sqrt(sum(component**2 for component in FFW_SG2_D405_CAMERA_UPRIGHT_ROT))
        self.assertAlmostEqual(norm, 1.0, places=7)


if __name__ == "__main__":
    unittest.main()
