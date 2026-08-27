"""Focused source contracts for Task000458 Mimic."""

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
ROOT = REPO / "source" / "cyclo_lab" / "cyclo_lab" / "manager_based" / "manipulation"
TASK = ROOT / "showroom" / "config" / "ffw_sg2" / "tasks" / "task_000458"


class Task000458MimicTest(unittest.TestCase):
    def test_task_runtime_uses_generic_sg2_mimic_base(self):
        source = (TASK / "mimic_env.py").read_text()
        self.assertIn("class Task000458MimicEnv(FFWSG2MimicEnv)", source)
        self.assertNotIn("FFWSG2PickPlaceMimicEnv", source)

    def test_cfg_has_no_pick_place_dependency(self):
        source = (TASK / "mimic_cfg.py").read_text()
        compact = "".join(source.split())
        self.assertIn("configure_ffw_sg2_mimic_ik_actions(self.actions)", compact)
        self.assertNotIn("PickPlaceEnvCfg", source)
        self.assertIn("generation_interpolate_from_last_target_pose=False", compact)
        self.assertEqual(compact.count("subtask_term_offset_range=(0,0)"), 3)

    def test_pick_place_and_task_are_siblings(self):
        generic = (ROOT / "common" / "ffw_sg2_mimic_env.py").read_text()
        pick_place = (ROOT / "pick_place" / "config" / "ffw_sg2" / "pick_place_mimic_env.py").read_text()
        self.assertIn("class FFWSG2MimicEnv(ManagerBasedRLMimicEnv)", generic)
        self.assertIn("class FFWSG2PickPlaceMimicEnv(FFWSG2MimicEnv)", pick_place)


if __name__ == "__main__":
    unittest.main()
