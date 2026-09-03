"""Focused static regressions for Task525 pick-only trajectory generation."""

import ast
from pathlib import Path
import unittest


REPO = Path(__file__).resolve().parents[3]
GENERATOR = (
    REPO
    / "scripts"
    / "sim2real"
    / "imitation_learning"
    / "tasks"
    / "task_000525"
    / "generate_trajectories.py"
)


def function(tree: ast.AST, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def named_calls(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        candidate
        for candidate in ast.walk(node)
        if isinstance(candidate, ast.Call)
        and isinstance(candidate.func, ast.Name)
        and candidate.func.id == name
    ]


def branch_contains(branch: list[ast.stmt], target: ast.AST) -> bool:
    return any(target in ast.walk(statement) for statement in branch)


class Task000525PickScopeGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source = GENERATOR.read_text(encoding="utf-8")
        cls.tree = ast.parse(cls.source)
        cls.replay = function(cls.tree, "replay")

    def test_explicit_equal_grasp_and_lift_boundary_is_supported(self):
        resolver = function(self.tree, "resolve_grasp_boundary_step")
        explicit_boundary = next(
            node
            for node in resolver.body
            if isinstance(node, ast.If)
            and ast.unparse(node.test)
            == "grasp_step is not None and 0 <= grasp_step <= lift_step"
        )
        self.assertEqual(
            ast.unparse(explicit_boundary.body[0]),
            "return (grasp_step, 'explicit grasp_step')",
        )

    def test_pick_scope_does_not_plan_or_run_navigation_control(self):
        parents: dict[ast.AST, ast.AST] = {}
        for parent in ast.walk(self.replay):
            for child in ast.iter_child_nodes(parent):
                parents[child] = parent

        planner_calls = named_calls(self.replay, "plan_navigation_path")
        self.assertEqual(len(planner_calls), 2)
        for call in planner_calls:
            ancestor = parents.get(call)
            while ancestor is not None:
                if (
                    isinstance(ancestor, ast.If)
                    and ast.unparse(ancestor.test) == "acceptance_scope == 'all'"
                ):
                    break
                ancestor = parents.get(ancestor)
            self.assertIsNotNone(
                ancestor,
                "every route-plan call must be guarded by acceptance_scope == 'all'",
            )

        build_call = named_calls(self.replay, "build_navigation_scene")[0]
        placement_keyword = next(
            keyword
            for keyword in build_call.keywords
            if keyword.arg == "randomize_placement"
        )
        self.assertEqual(
            ast.unparse(placement_keyword.value),
            "randomize_placement and acceptance_scope == 'all'",
        )

        sentinel_call = named_calls(self.replay, "populate_pick_navigate_sentinel")[0]
        controller_call = named_calls(self.replay, "handle_navigate_state")[0]
        dispatches = [
            node
            for node in ast.walk(self.replay)
            if isinstance(node, ast.If)
            and ast.unparse(node.test) == "acceptance_scope == 'pick'"
            and branch_contains(node.body, sentinel_call)
            and branch_contains(node.orelse, controller_call)
        ]
        self.assertEqual(len(dispatches), 1)

    def test_pick_sentinel_is_exactly_two_post_step_zero_base_rows(self):
        sentinel = function(self.tree, "populate_pick_navigate_sentinel")
        sentinel_source = ast.get_source_segment(self.source, sentinel)
        self.assertIsNotNone(sentinel_source)
        self.assertIn("current_base_pose.new_zeros(3)", sentinel_source)
        self.assertIn("LocomanipulationSDGDataGenerationState.NAVIGATE", sentinel_source)
        for forbidden in (
            "plan_navigation_path",
            "handle_navigate_state",
            "compute_navigation_velocity",
            "compute_fixed_yaw_holonomic_velocity",
        ):
            self.assertNotIn(forbidden, sentinel_source)

        constants = {
            target.id: node.value.value
            for node in self.tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "TASK000525_PICK_NAVIGATE_SENTINEL_ROWS"
            and isinstance(node.value, ast.Constant)
            for target in node.targets
        }
        self.assertEqual(constants["TASK000525_PICK_NAVIGATE_SENTINEL_ROWS"], 2)

        step_call = next(
            node
            for node in ast.walk(self.replay)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "env"
            and node.func.attr == "step"
        )
        increment = next(
            node
            for node in ast.walk(self.replay)
            if isinstance(node, ast.AugAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "pick_navigate_rows_recorded"
        )
        self.assertLess(step_call.lineno, increment.lineno)

    def test_pick_phase_handlers_are_monotonic_zero_one_two(self):
        enum_node = next(
            node
            for node in self.tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LocomanipulationSDGDataGenerationState"
        )
        values = {
            node.targets[0].id: node.value.value
            for node in enum_node.body
            if isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.Constant)
        }
        self.assertEqual(
            [values[name] for name in ("GRASP_OBJECT", "LIFT_OBJECT", "NAVIGATE")],
            [0, 1, 2],
        )

        expected_states = {
            "handle_grasp_state": {"GRASP_OBJECT", "LIFT_OBJECT"},
            "handle_lift_state": {"LIFT_OBJECT", "NAVIGATE"},
            "populate_pick_navigate_sentinel": {"NAVIGATE"},
        }
        for handler_name, expected in expected_states.items():
            handler = function(self.tree, handler_name)
            referenced = {
                node.attr
                for node in ast.walk(handler)
                if isinstance(node, ast.Attribute)
                and isinstance(node.value, ast.Name)
                and node.value.id == "LocomanipulationSDGDataGenerationState"
            }
            self.assertEqual(referenced, expected)


if __name__ == "__main__":
    unittest.main()
