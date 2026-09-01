"""Task000458 success, action order, and converter contracts."""

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
TASK = REPO / "source" / "cyclo_lab" / "cyclo_lab" / "manager_based" / "manipulation" / "showroom" / "config" / "ffw_sg2" / "tasks" / "task_000458"


class Task000458DataContractTest(unittest.TestCase):
    def test_takeout_success_is_task_local(self):
        source = (TASK / "takeout_terms.py").read_text()
        compact = "".join(source.split())
        self.assertIn("task_success_value=held_current&target_outside_shelf&neighbors_static", compact)
        self.assertNotIn("trajectory_collision_free", source)

    def test_seed_remains_operator_accepted(self):
        recorder = (REPO / "scripts" / "sim2real" / "imitation_learning" / "recorder" / "record_demos.py").read_text()
        self.assertIn("Saving operator-accepted demo.", recorder)
        self.assertNotIn("Save rejected by success metric", recorder)

    def test_action_order_conversion_is_canonical(self):
        converter = (REPO / "scripts" / "sim2real" / "imitation_learning" / "mimic" / "action_data_converter.py").read_text()
        self.assertIn("head_action = joint_actions[:, 16:18]", converter)
        self.assertIn("lift_action = joint_actions[:, 18:19]", converter)
        self.assertNotIn("joint_targets[:, 18:19]", converter)
        self.assertNotIn("joint_targets[:, 16:18]", converter)

    def test_mimic_tail_is_head_then_lift(self):
        task_mimic = (TASK / "mimic_env.py").read_text()
        task_compact = "".join(task_mimic.split())
        self.assertIn("\"head_joint1\",\"head_joint2\",\"lift_joint\",", task_compact)

        common = (REPO / "source" / "cyclo_lab" / "cyclo_lab" / "manager_based" / "manipulation" / "common" / "ffw_sg2_mimic_env.py").read_text()
        self.assertIn("                head_action,           # 17-18: head", common)
        self.assertIn("                lift_action            # 19: lift", common)
        self.assertNotIn("                lift_action,           # 17: lift", common)

    def test_mimic_scripts_use_the_resolved_environment(self):
        mimic_dir = (
            REPO
            / "scripts"
            / "sim2real"
            / "imitation_learning"
            / "mimic"
        )
        annotate = (mimic_dir / "annotate_demos.py").read_text()
        generate = (mimic_dir / "generate_dataset.py").read_text()
        self.assertIn("gym.make(env_name, cfg=env_cfg)", annotate)
        self.assertNotIn("gym.make(args_cli.task, cfg=env_cfg)", annotate)
        self.assertIn(
            'env_name.replace("-Mimic-Seed-", "-Mimic-Generate-")',
            generate,
        )
        self.assertIn("generation_env_name in gym.registry", generate)


if __name__ == "__main__":
    unittest.main()
