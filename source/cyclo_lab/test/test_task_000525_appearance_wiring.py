"""Static profile wiring for Task000525 visual-only can yaw."""

from pathlib import Path
import unittest


TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
)


class Task000525AppearanceWiringTest(unittest.TestCase):
    def test_profiles_select_visual_yaw_explicitly(self):
        source = (TASK_DIR / "profiles.py").read_text(encoding="utf-8")
        deterministic = source.split("TASK000525_DETERMINISTIC =", 1)[1].split(
            "def make_task000525_seed_profile", 1
        )[0]
        physical = source.split(
            "TASK000525_PHYSICAL_TRAJECTORY_GENERATION =", 1
        )[1].split("TASK000525_VISUAL_REPLAY_AUGMENTATION =", 1)[0]
        visual = source.split(
            "TASK000525_VISUAL_REPLAY_AUGMENTATION =", 1
        )[1]
        self.assertNotIn("enabled=True", deterministic)
        self.assertNotIn("coffee_visual_yaw=", physical)
        self.assertIn(
            "coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True)",
            visual,
        )

    def test_visual_profile_selects_only_non_target_appearance_shuffle(self):
        source = (TASK_DIR / "profiles.py").read_text(encoding="utf-8")
        physical = source.split(
            "TASK000525_PHYSICAL_TRAJECTORY_GENERATION =", 1
        )[1].split("TASK000525_VISUAL_REPLAY_AUGMENTATION =", 1)[0]
        visual = source.split(
            "TASK000525_VISUAL_REPLAY_AUGMENTATION =", 1
        )[1]
        self.assertNotIn("coffee_distractor_appearance=", physical)
        self.assertIn(
            "coffee_distractor_appearance=CoffeeDistractorAppearanceRandomizationCfg(",
            visual,
        )
        self.assertIn("if object_name != TASK000525_TARGET_OBJECT", source)
        self.assertIn(
            "Task525 distractor appearance must never include the orange target object.",
            source,
        )

    def test_env_builds_event_from_profile_values(self):
        source = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("coffee_visual_yaw = self.randomization.coffee_visual_yaw", source)
        self.assertIn("func=randomize_coffee_can_visual_yaw", source)
        self.assertIn('"object_names": coffee_visual_yaw.object_names', source)
        self.assertIn('"yaw_range_rad": coffee_visual_yaw.yaw_range_rad', source)


if __name__ == "__main__":
    unittest.main()
