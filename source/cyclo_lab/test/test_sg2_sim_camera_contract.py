"""Focused SG2 simulation camera-construction contracts."""

from pathlib import Path
import unittest

from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    FFW_SG2_HEAD_CAMERA_HEIGHT,
    FFW_SG2_HEAD_CAMERA_WIDTH,
    FFW_SG2_WRIST_CAMERA_HEIGHT,
    FFW_SG2_WRIST_CAMERA_WIDTH,
)
from cyclo_lab.robot_specs.ffw.sg2 import load_sg2_camera_profile


ROBOT_CFG = (
    Path(__file__).parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "platform"
    / "robot_cfg.py"
)


class TestSG2SimulationCameraContract(unittest.TestCase):
    def test_shared_policy_camera_shapes(self):
        self.assertEqual(
            (FFW_SG2_HEAD_CAMERA_HEIGHT, FFW_SG2_HEAD_CAMERA_WIDTH), (376, 672)
        )
        self.assertEqual(
            (FFW_SG2_WRIST_CAMERA_HEIGHT, FFW_SG2_WRIST_CAMERA_WIDTH), (640, 480)
        )

    def test_showroom_uses_canonical_rasters_without_relabeling_old_head_k(self):
        source = ROBOT_CFG.read_text(encoding="utf-8")
        profile_head = load_sg2_camera_profile("1050").camera("head")
        self.assertEqual((profile_head.height, profile_head.width), (480, 640))
        self.assertEqual(profile_head.intrinsic_matrix[0], 489.7808024)
        profile_block = source.split(
            "def apply_sg2_showroom_camera_profile", 1
        )[1].split("def enable_sg2_showroom_operator_cameras", 1)[0]
        self.assertIn("width=FFW_SG2_HEAD_CAMERA_WIDTH", profile_block)
        self.assertIn("height=FFW_SG2_HEAD_CAMERA_HEIGHT", profile_block)
        self.assertIn(
            "== (FFW_SG2_HEAD_CAMERA_HEIGHT, FFW_SG2_HEAD_CAMERA_WIDTH)",
            profile_block,
        )
        self.assertIn("else None", profile_block)
        self.assertIn("width=wrist_left.height", profile_block)
        self.assertIn("height=wrist_left.width", profile_block)

    def test_ui_session_attaches_only_center_overhead_camera(self):
        source = ROBOT_CFG.read_text(encoding="utf-8")
        operator_block, ui_block = source.split(
            "def enable_sg2_showroom_operator_cameras", 1
        )[1].split("def enable_sg2_showroom_ui_session_camera", 1)
        self.assertIn("cam_overhead_left", operator_block)
        self.assertIn("cam_overhead_center", operator_block)
        self.assertIn("cam_overhead_right", operator_block)
        self.assertIn("cam_overhead_center", ui_block)
        self.assertNotIn("cam_overhead_left", ui_block)
        self.assertNotIn("cam_overhead_right", ui_block)
        self.assertIn(
            'make_ffw_sg2_overhead_camera_cfg(\n        "center", update_period=0.0\n    )',
            ui_block,
        )


if __name__ == "__main__":
    unittest.main()
