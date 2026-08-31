"""Contracts for Task000525 Random visual-only coffee yaw wiring."""

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


class Task000525VisualEventWiringTest(unittest.TestCase):
    def test_random_preset_enables_visual_yaw_event(self):
        source = (TASK_DIR / "env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("randomize_coffee_visual_yaw: bool = True", source)
        self.assertIn("randomize_task000525_coffee_visual_yaw", source)
        self.assertIn("func=randomize_coffee_can_visual_yaw", source)
        self.assertIn('"object_names": TASK000525_CAN_NAMES', source)

    def test_physical_reset_still_samples_xy_only(self):
        source = (TASK_DIR / "reset_events.py").read_text(encoding="utf-8")
        self.assertIn("torch.rand((len(env_ids), 2)", source)
        for component in range(3, 7):
            self.assertNotIn(f"root_pose[:, {component}", source)


if __name__ == "__main__":
    unittest.main()
