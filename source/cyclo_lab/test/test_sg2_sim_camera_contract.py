"""Focused SG2 simulation camera-construction contracts."""

import unittest

from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    FFW_SG2_HEAD_CAMERA_HEIGHT,
    FFW_SG2_HEAD_CAMERA_WIDTH,
    FFW_SG2_WRIST_CAMERA_HEIGHT,
    FFW_SG2_WRIST_CAMERA_WIDTH,
)
from cyclo_lab.robot_specs.ffw.sg2 import load_sg2_camera_profile


class TestSG2SimulationCameraContract(unittest.TestCase):
    def test_shared_policy_camera_shapes(self):
        self.assertEqual(
            (FFW_SG2_HEAD_CAMERA_HEIGHT, FFW_SG2_HEAD_CAMERA_WIDTH), (376, 672)
        )
        self.assertEqual(
            (FFW_SG2_WRIST_CAMERA_HEIGHT, FFW_SG2_WRIST_CAMERA_WIDTH), (640, 480)
        )

    def test_1050_profile_uses_canonical_head_raster(self):
        profile_head = load_sg2_camera_profile("1050").camera("head")
        self.assertEqual((profile_head.height, profile_head.width), (376, 672))
        self.assertEqual(profile_head.intrinsic_matrix[0], 365.1824645996094)


if __name__ == "__main__":
    unittest.main()
