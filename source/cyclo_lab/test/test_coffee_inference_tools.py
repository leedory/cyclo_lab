"""Static tests for the checkpoint-independent Coffee inference tools."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "scripts/sim2real/imitation_learning/tasks/coffee_transport"


def load_control():
    spec = importlib.util.spec_from_file_location(
        "coffee_control_inference", TOOLS / "control_inference.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class CoffeeInferenceToolsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.control = load_control()

    def args(self, command: str, **overrides):
        values = {
            "command": command,
            "task_id": None,
            "checkpoint": None,
            "publish_actions": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_task000001_load_is_scoped_to_model_root(self):
        request = self.control.build_request(
            self.args(
                "load",
                task_id="000001",
                checkpoint="/workspace/model/lerobot/task_000001/checkpoints/last/pretrained_model",
                publish_actions=True,
            )
        )
        self.assertEqual(request["command"], 0)
        self.assertEqual(request["robot_type"], "ffw_sg2_rev1")
        self.assertTrue(request["publish_to_robot"])
        self.assertIn("lower cabinet shelf", request["task_instruction"])

    def test_task000002_instruction_and_lifecycle_commands(self):
        request = self.control.build_request(
            self.args(
                "load",
                task_id="000002",
                checkpoint="/workspace/model/lerobot/task_000002",
            )
        )
        self.assertIn("cabinet on the right", request["task_instruction"])
        start = self.control.build_request(
            self.args("start", publish_actions=True)
        )
        self.assertEqual(start["command"], 1)
        self.assertTrue(start["publish_to_robot"])

    def test_load_rejects_paths_outside_policy_root(self):
        with self.assertRaisesRegex(ValueError, "below /workspace/model/lerobot"):
            self.control.build_request(
                self.args(
                    "load", task_id="000001", checkpoint="/tmp/checkpoint"
                )
            )

    def test_nonload_rejects_stale_checkpoint_arguments(self):
        with self.assertRaisesRegex(ValueError, "valid only for load"):
            self.control.build_request(
                self.args("start", task_id="000001", checkpoint="/workspace/model/lerobot/x")
            )

if __name__ == "__main__":
    unittest.main()
