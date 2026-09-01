"""Regression checks for optional bridge activation hooks in generic bringup."""

from pathlib import Path
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
BRINGUP_PATH = REPOSITORY_ROOT / "scripts" / "sim2real" / "bringup.py"


class TestBringupBridgeActivation(unittest.TestCase):
    def test_activation_hook_is_optional_for_non_sg2_bridges(self):
        source = BRINGUP_PATH.read_text(encoding="utf-8")

        self.assertIn(
            'control_activation = getattr(bridge, "begin_control_activation", None)',
            source,
        )
        self.assertIn("if callable(control_activation):", source)
        self.assertIn("control_activation()", source)
        self.assertNotIn("bridge.begin_control_activation()", source)


if __name__ == "__main__":
    unittest.main()
