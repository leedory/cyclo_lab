"""Static contracts for deterministic and randomized Continuous showroom presets."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "cyclo_lab" / "manager_based" / "manipulation" / "showroom" / "config" / "ffw_sg2"


class ShowroomArchitectureTest(unittest.TestCase):
    def test_normal_and_random_are_continuous_presets(self):
        module = ast.parse((ROOT / "__init__.py").read_text())
        expected_ids = {
            "Cyclo-Real-Showroom-FFW-SG2-v0",
            "Cyclo-Real-Showroom-Random-FFW-SG2-v0",
        }
        registrations = {}
        for statement in module.body:
            call = statement.value if isinstance(statement, ast.Expr) else None
            if not isinstance(call, ast.Call):
                continue
            if not (
                isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name)
                and call.func.value.id == "gym"
                and call.func.attr == "register"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in call.keywords}
            task_id = ast.literal_eval(keywords["id"])
            if task_id in expected_ids:
                registrations[task_id] = ast.literal_eval(keywords["entry_point"])

        expected_entry_point = "cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv"
        self.assertEqual(
            registrations["Cyclo-Real-Showroom-FFW-SG2-v0"], expected_entry_point
        )
        self.assertEqual(
            registrations["Cyclo-Real-Showroom-Random-FFW-SG2-v0"],
            expected_entry_point,
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
