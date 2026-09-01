"""Contracts for Task000525 visual replay yaw provenance and guards."""

from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
TASK_SCRIPTS = (
    REPO / "scripts" / "sim2real" / "imitation_learning" / "tasks" / "task_000525"
)


class Task000525VisualEventWiringTest(unittest.TestCase):
    def test_replay_requires_and_applies_task_visual_yaw(self):
        source = (TASK_SCRIPTS / "replay_visual_policy_staging.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Task000525RandomizationCfg", source)
        self.assertIn("profile.coffee_positions.enabled", source)
        self.assertIn("profile.coffee_visual_yaw.enabled", source)
        self.assertIn("randomize_coffee_can_visual_yaw(", source)
        self.assertIn('"coffee_can_visual_yaw"', source)

    def test_replay_guards_all_rigid_roots(self):
        source = (TASK_SCRIPTS / "replay_visual_policy_staging.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("def protected_root_pose_snapshot", source)
        self.assertIn("verify_protected_root_poses", source)
        self.assertIn("if error != 0.0", source)
        self.assertIn('"protected_pose_max_abs_error"', source)

    def test_audit_rejects_missing_or_changed_visual_evidence(self):
        source = (TASK_SCRIPTS / "audit_policy_staging.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("validate_visual_randomization(record)", source)
        self.assertIn("coffee visual-yaw samples do not cover exactly", source)
        self.assertIn("inconsistent visual yaw rad/deg pair", source)
        self.assertIn("visual randomization changed protected rigid roots", source)


if __name__ == "__main__":
    unittest.main()
