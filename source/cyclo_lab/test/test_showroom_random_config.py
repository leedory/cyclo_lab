"""Static contracts for deterministic and randomized Continuous showroom presets."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "cyclo_lab" / "manager_based" / "manipulation" / "showroom" / "config" / "ffw_sg2"


class ShowroomArchitectureTest(unittest.TestCase):
    def test_normal_and_random_are_continuous_presets(self):
        registration = (ROOT / "__init__.py").read_text()
        self.assertEqual(registration.count("gym.register("), 6)
        self.assertIn("platform.env_cfg:ContinuousShowroomEnvCfg", registration)
        self.assertIn("platform.env_cfg:ContinuousRandomShowroomEnvCfg", registration)
        self.assertEqual(
            registration.count("continuous_env:ContinuousManagerBasedEnv"), 2
        )

    def test_default_reset_has_no_randomizer(self):
        source = (ROOT / "platform" / "env_cfg.py").read_text()
        block = source.split("class DeterministicResetEventsCfg", 1)[1].split(
            "@configclass", 1
        )[0]
        self.assertIn("randomize_robot_root_pose = None", block)
        self.assertIn("randomize_selected_objects = None", block)
        self.assertNotIn("randomize_root_pose_in_xy_box", block)
        self.assertNotIn("randomize_root_poses_in_xy_box", block)

    def test_random_profile_keeps_22d_mobile_action(self):
        env_source = (ROOT / "platform" / "env_cfg.py").read_text()
        action_source = (ROOT / "platform" / "action_cfg.py").read_text()
        profile = (ROOT / "randomization" / "profiles.py").read_text()
        self.assertIn("class ContinuousRandomShowroomEnvCfg(ContinuousShowroomEnvCfg)", env_source)
        self.assertIn("class ContinuousShowroomActionsCfg(FFWSG2MobileActionsCfg)", action_source)
        self.assertIn('DIGITAL_TWIN_SELECTED_OBJECTS = ("peanut_mix_bag_02",)', profile)
        self.assertIn("depth_x_max_m=0.030", profile)
        self.assertIn("x_max_m=0.010", profile)

    def test_continuous_uses_the_common_profile_builder(self):
        env_source = (ROOT / "platform" / "env_cfg.py").read_text()
        builder = (ROOT / "randomization" / "event_cfg.py").read_text()
        self.assertIn("self.apply_randomization_profile(self.randomization)", env_source)
        self.assertNotIn("class ContinuousRandomResetEventsCfg", env_source)
        self.assertIn("if robot.enabled", builder)
        self.assertIn("if objects.enabled", builder)
        for parameter in (
            '"x_max": robot.depth_x_max_m',
            '"y_max": robot.lateral_y_max_m',
            '"yaw_max": robot.yaw_max_rad',
            '"object_names": objects.object_names',
            '"x_max": objects.x_max_m',
            '"y_max": objects.y_max_m',
            '"yaw_max": objects.yaw_max_rad',
        ):
            self.assertIn(parameter, builder)


if __name__ == "__main__":
    unittest.main()
