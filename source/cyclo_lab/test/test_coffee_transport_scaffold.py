"""Static contracts for the shared temporary coffee-transport package."""

from pathlib import Path
import runpy
import unittest


SOURCE_ROOT = Path(__file__).resolve().parents[1]
FFW_SG2_ROOT = (
    SOURCE_ROOT
    / "cyclo_lab"
    / "manager_based"
    / "manipulation"
    / "showroom"
    / "config"
    / "ffw_sg2"
)
TASK_ROOT = FFW_SG2_ROOT / "tasks" / "coffee_transport"


class CoffeeTransportScaffoldTest(unittest.TestCase):
    def test_shared_package_owns_common_and_task_specific_files(self):
        for name in (
            "README.md",
            "coffee_transport_common.py",
            "generation_contract.py",
            "task_000001/spec.py",
            "task_000001/env_cfg.py",
            "task_000002/spec.py",
            "task_000002/env_cfg.py",
        ):
            self.assertTrue((TASK_ROOT / name).is_file(), name)

    def test_can_order_arm_assignment_and_phase_order_are_fixed(self):
        contract = runpy.run_path(str(TASK_ROOT / "generation_contract.py"))
        cycles = contract["COFFEE_CAN_CYCLES"]
        self.assertEqual(
            [(cycle.can_name, cycle.arm) for cycle in cycles],
            [
                ("coffee_can_right", "right"),
                ("coffee_can_center", "right"),
                ("coffee_can_left", "left"),
            ],
        )
        phases = contract["COFFEE_TRANSPORT_PHASES"]
        self.assertEqual(len(phases), 16)
        self.assertEqual(phases[0].key, "normalize_start")
        expected_suffixes = ("grasp_", "clear_", "transport_", "place_", "recover_")
        for cycle_index, cycle in enumerate(cycles):
            cycle_phases = phases[1 + 5 * cycle_index : 6 + 5 * cycle_index]
            self.assertEqual(
                tuple(phase.key.startswith(prefix) for phase, prefix in zip(cycle_phases, expected_suffixes)),
                (True, True, True, True, True),
            )
            self.assertEqual(cycle_phases[0].can_name, cycle.can_name)
            self.assertEqual(cycle_phases[0].arm, cycle.arm)

    def test_registration_uses_nested_task_modules(self):
        registration = (FFW_SG2_ROOT / "__init__.py").read_text(encoding="utf-8")
        self.assertIn('for _task_id in ("000001", "000002")', registration)
        self.assertIn(".tasks.coffee_transport.task_{_task_id}.env_cfg", registration)


if __name__ == "__main__":
    unittest.main()
