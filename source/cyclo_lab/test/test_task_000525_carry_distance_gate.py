"""Pure/static contracts for Task000525's source-relative carry distance gate."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
ENV_PATH = (
    REPO
    / "source"
    / "cyclo_lab"
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
    / "tasks"
    / "task_000525"
    / "locomanipulation_sdg_env.py"
)
LIMIT_FUNCTION = "task000525_carry_eef_object_distance_limit_m"
CONSTANT_NAMES = {
    "TASK000525_CARRY_EEF_OBJECT_DISTANCE_MIN_M",
    "TASK000525_CARRY_EEF_OBJECT_DISTANCE_TOLERANCE_M",
}


def load_limit_contract():
    tree = ast.parse(ENV_PATH.read_text(encoding="utf-8"))
    selected = [ast.Import(names=[ast.alias(name="math")])]
    selected.extend(
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id in CONSTANT_NAMES
            for target in node.targets
        )
    )
    selected.extend(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == LIMIT_FUNCTION
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(selected, [])), str(ENV_PATH), "exec"), namespace)
    return namespace


class Task000525CarryDistanceGateTest(unittest.TestCase):
    def test_limit_uses_source_rim_grasp_plus_explicit_tolerance(self):
        contract = load_limit_contract()
        limit = contract[LIMIT_FUNCTION]
        self.assertAlmostEqual(
            contract["TASK000525_CARRY_EEF_OBJECT_DISTANCE_MIN_M"], 0.080
        )
        self.assertAlmostEqual(
            contract["TASK000525_CARRY_EEF_OBJECT_DISTANCE_TOLERANCE_M"], 0.025
        )
        self.assertAlmostEqual(limit(0.0875), 0.1125)
        self.assertAlmostEqual(limit(0.0400), 0.0800)

    def test_measured_near_source_attempts_pass_but_gross_drops_do_not(self):
        limit = load_limit_contract()[LIMIT_FUNCTION]
        for source_distance, current_distance in (
            (0.0875, 0.091),
            (0.0800, 0.093),
            (0.0780, 0.091),
        ):
            with self.subTest(source_distance=source_distance):
                self.assertLessEqual(current_distance, limit(source_distance))
                self.assertGreater(0.200, limit(source_distance))

    def test_invalid_source_distance_is_rejected(self):
        limit = load_limit_contract()[LIMIT_FUNCTION]
        for source_distance in (-0.001, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                limit(source_distance)

    def test_checkpoint_reads_active_source_geometry_and_records_contract(self):
        source = ENV_PATH.read_text(encoding="utf-8")
        for token in (
            '"obs/left_eef_pose_world"',
            '"obs/right_eef_pose_world"',
            '"obs/target_object_pose_world"',
            "navigate_step",
            '"carry_source_eef_object_distance_m"',
            '"carry_eef_object_distance_tolerance_m"',
            '"carry_eef_object_distance_limit_m"',
            "if eef_object_distance > eef_object_distance_limit:",
        ):
            self.assertIn(token, source)
        self.assertIn("if object_displacement < 0.200:", source)
        self.assertIn("if home_joint_max_error > 0.150:", source)
        self.assertNotIn("if eef_object_distance > 0.080:", source)


if __name__ == "__main__":
    unittest.main()
