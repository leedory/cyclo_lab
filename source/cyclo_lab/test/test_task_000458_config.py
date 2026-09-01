"""Static contracts for the Task000458 episodic presets."""

from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1] / "cyclo_lab" / "manager_based" / "manipulation" / "showroom" / "config" / "ffw_sg2"
TASK = ROOT / "tasks" / "task_000458"


class Task000458ConfigTest(unittest.TestCase):
    def test_registration_surface_uses_explicit_presets(self):
        source = (ROOT / "__init__.py").read_text()
        for task_id in (
            "Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
            "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
            "Cyclo-Real-Showroom-Task000458-Mimic-Seed-FFW-SG2-v0",
            "Cyclo-Real-Showroom-Task000458-Mimic-Generate-FFW-SG2-v0",
        ):
            self.assertIn(task_id, source)
        self.assertNotIn("Pick-Peanut", source)

    def test_task_inherits_common_episdodic_shell(self):
        source = (TASK / "env_cfg.py").read_text()
        common = (ROOT / "tasks" / "common.py").read_text()
        self.assertIn("class Task000458EnvCfg(EpisodicShowroomTaskEnvCfg)", source)
        self.assertIn("class EpisodicShowroomTaskEnvCfg(ManagerBasedRLEnvCfg)", common)
        actions = (ROOT / "platform" / "action_cfg.py").read_text()
        self.assertIn("class EpisodicShowroomActionsCfg(FFWSG2JointPositionActionsCfg)", actions)

    def test_profiles_are_explicit(self):
        source = (TASK / "env_cfg.py").read_text()
        profiles = (TASK / "profiles.py").read_text()
        self.assertIn(
            "randomization: ShowroomRandomizationCfg = TASK000458_RECORD_DETERMINISTIC",
            source,
        )
        self.assertIn("TASK000458_RECORD_RANDOM", source)
        self.assertIn("object_names=(TASK_000458_SPEC.target_object,)", profiles)

    def test_record_random_profile_wires_robot_and_object_events(self):
        source = (TASK / "env_cfg.py").read_text()
        common = (ROOT / "tasks" / "common.py").read_text()
        builder = (ROOT / "randomization" / "event_cfg.py").read_text()
        self.assertIn("apply_randomization_profile(self.randomization)", common)
        self.assertIn("configure_profiled_reset_events(", common)
        self.assertIn("randomize_robot_root_pose", builder)
        self.assertIn("randomize_selected_objects", builder)
        self.assertNotIn("EventTerm", source)
        self.assertNotIn(".params.update(", source)

    def test_target_has_one_task_owned_source(self):
        spec = (TASK / "spec.py").read_text()
        profiles = (TASK / "profiles.py").read_text()
        mimic = (TASK / "mimic_cfg.py").read_text()
        self.assertIn("TASK_000458_TARGET_OBJECT", spec)
        self.assertIn("TASK_000458_SPEC.target_object", profiles)
        self.assertNotIn("TARGET_OBJECT", mimic)
        self.assertIn("self.target_object", mimic)

if __name__ == "__main__":
    unittest.main()
