"""Pure-Python contracts for Task000525 appearance-only can yaw."""

import ast
from pathlib import Path
import unittest


APPEARANCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
    / "appearance_events.py"
)


class Task000525AppearanceTest(unittest.TestCase):
    def test_event_interface_is_stable(self):
        module = ast.parse(APPEARANCE_PATH.read_text(encoding="utf-8"))
        function = next(
            node
            for node in module.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "randomize_coffee_can_visual_yaw"
        )
        self.assertEqual(
            [argument.arg for argument in function.args.args],
            ["env", "env_ids", "object_names", "yaw_range_rad"],
        )

    def test_event_authors_only_the_visual_mesh_transform(self):
        source = APPEARANCE_PATH.read_text(encoding="utf-8")
        self.assertIn('/Visual/SharedMesh"', source)
        self.assertIn('op.GetOpName() == "xformOp:rotateZ"', source)
        self.assertIn("AddRotateZOp", source)
        for physical_mutator in (
            "write_root_pose_to_sim",
            "write_root_velocity_to_sim",
            "set_world_pose",
            "physics:approximation",
        ):
            self.assertNotIn(physical_mutator, source)

    def test_yaw_range_is_explicitly_bounded(self):
        source = APPEARANCE_PATH.read_text(encoding="utf-8")
        self.assertIn("-math.pi <= yaw_min <= yaw_max <= math.pi", source)


if __name__ == "__main__":
    unittest.main()
