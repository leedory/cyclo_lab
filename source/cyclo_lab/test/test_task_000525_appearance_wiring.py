"""Static config wiring for Task000525 visual-only yaw randomization."""

from pathlib import Path
import unittest


ENV_CFG_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
    / "env_cfg.py"
)


class Task000525AppearanceWiringTest(unittest.TestCase):
    def test_deterministic_preset_keeps_visual_yaw_fixed(self):
        source = ENV_CFG_PATH.read_text(encoding="utf-8")
        base_class = source.split("class Task000525EnvCfg", 1)[1].split(
            "class Task000525RandomEnvCfg", 1
        )[0]
        self.assertIn("randomize_coffee_visual_yaw: bool = False", base_class)

    def test_random_preset_enables_visual_yaw_event(self):
        source = ENV_CFG_PATH.read_text(encoding="utf-8")
        random_class = source.split("class Task000525RandomEnvCfg", 1)[1]
        self.assertIn("randomize_coffee_visual_yaw: bool = True", random_class)
        self.assertIn("func=randomize_coffee_can_visual_yaw", source)
        self.assertIn('params={"object_names": TASK000525_CAN_NAMES}', source)


if __name__ == "__main__":
    unittest.main()
