"""Focused contracts for the FFW F2 asset and Continuous template."""

from pathlib import Path
import unittest

from cyclo_lab.assets.sensors.ffw_f2_cameras import FFW_F2_CAMERA_HEIGHT, FFW_F2_CAMERA_WIDTH
from cyclo_lab.robot_specs.ffw.f2 import (
    DEFAULT_F2_CAMERA_PROFILE,
    F2_SWERVE_WHEEL_RADIUS,
    FFW_F2_ACTION_JOINT_NAMES,
    FFW_F2_JOINT_POSITION_LIMITS,
    FFW_F2_MOBILE_ACTION_NAMES,
    classify_f2_visual_material,
    load_f2_camera_profile,
)


class TestF2Support(unittest.TestCase):
    def test_public_action_contract(self) -> None:
        self.assertEqual(len(FFW_F2_ACTION_JOINT_NAMES), 19)
        self.assertEqual(len(FFW_F2_MOBILE_ACTION_NAMES), 22)
        self.assertEqual(FFW_F2_ACTION_JOINT_NAMES[0], "arm_l_joint1")
        self.assertEqual(FFW_F2_ACTION_JOINT_NAMES[-1], "lift_joint")

    def test_canonical_visual_material_mapping(self) -> None:
        expected_materials = {
            "/Robot/camera_head_link/visuals/mesh": "aluminum",
            "/Robot/arm_l_link4/visuals/mesh": "dark",
            "/Robot/gripper_r_hx2_d1_r2/visuals/mesh": "dark",
            "/Robot/body_arm_assy/visuals/mesh": "white",
            "/Robot/end_effector_l/visuals/mesh": "red",
        }
        for prim_path, material in expected_materials.items():
            with self.subTest(prim_path=prim_path):
                self.assertEqual(classify_f2_visual_material(prim_path), material)

    def test_canonical_geometry_and_limits(self) -> None:
        self.assertEqual(F2_SWERVE_WHEEL_RADIUS, 0.0825)
        self.assertEqual(FFW_F2_JOINT_POSITION_LIMITS["arm_l_joint7"], (-1.5804, 1.8201))
        self.assertEqual(FFW_F2_JOINT_POSITION_LIMITS["arm_r_joint7"], (-1.8201, 1.5804))

    def test_9004_camera_profile(self) -> None:
        profile = load_f2_camera_profile()
        self.assertEqual(DEFAULT_F2_CAMERA_PROFILE, "9004")
        self.assertEqual(profile.profile_id, "f2-9004-2026-09-03")
        self.assertEqual(profile.robot_model, "ffw_f2")
        self.assertEqual((FFW_F2_CAMERA_HEIGHT, FFW_F2_CAMERA_WIDTH), (480, 640))
        self.assertEqual(set(profile.cameras), {"head", "wrist_left", "wrist_right"})
        self.assertEqual(len({camera.serial for camera in profile.cameras.values()}), 3)
        self.assertEqual(len(profile.source_sha256), 64)

    def test_generated_usd_is_present(self) -> None:
        usd_path = Path(__file__).parents[1] / "data/robots/FFW/FFW_F2.usd"
        self.assertTrue(usd_path.is_file())
        self.assertGreater(usd_path.stat().st_size, 1_000_000)


if __name__ == "__main__":
    unittest.main()
