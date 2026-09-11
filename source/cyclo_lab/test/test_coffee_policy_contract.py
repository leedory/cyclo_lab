"""Regression tests for Coffee Transport learned-policy contracts."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest


ARM_AND_GRIPPER_NAMES = (
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
)
JOINT19_NAMES = (*ARM_AND_GRIPPER_NAMES, "head_joint1", "head_joint2", "lift_joint")
ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = (
    ROOT
    / "cyclo_lab/manager_based/manipulation/showroom/config/ffw_sg2/tasks/coffee_transport"
)


def load_contract(task_id: str):
    robot_spec = types.ModuleType("cyclo_lab.robot_specs.ffw.sg2")
    robot_spec.FFW_SG2_ACTION_JOINT_NAMES = JOINT19_NAMES
    previous = sys.modules.get("cyclo_lab.robot_specs.ffw.sg2")
    sys.modules["cyclo_lab.robot_specs.ffw.sg2"] = robot_spec
    try:
        spec = importlib.util.spec_from_file_location(
            f"coffee_task_{task_id}_policy", TASK_ROOT / f"task_{task_id}/policy.py"
        )
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module.POLICY_CONTRACT
    finally:
        if previous is None:
            sys.modules.pop("cyclo_lab.robot_specs.ffw.sg2", None)
        else:
            sys.modules["cyclo_lab.robot_specs.ffw.sg2"] = previous


TASK_000001_POLICY = load_contract("000001")
TASK_000002_POLICY = load_contract("000002")


class CoffeePolicyContractTest(unittest.TestCase):
    def test_task_000001_is_head_only_arm_gripper_lift_17d(self):
        contract = TASK_000001_POLICY
        self.assertEqual(
            contract["state_names"], (*ARM_AND_GRIPPER_NAMES, "lift_joint")
        )
        self.assertEqual(contract["action_names"], contract["state_names"])
        self.assertEqual(
            contract["action_components"], ("arm_left", "arm_right", "lift")
        )
        self.assertEqual(
            contract["inactive_actions"],
            {"head": "hold_current", "mobile": "hold_current"},
        )
        self.assertEqual(set(contract["cameras"]), {"cam_left_head"})

    def test_task_000002_is_head_only_arm_gripper_mobile_19d(self):
        contract = TASK_000002_POLICY
        self.assertEqual(
            contract["state_names"],
            (*ARM_AND_GRIPPER_NAMES, "linear_x", "linear_y", "angular_z"),
        )
        self.assertEqual(contract["action_names"], contract["state_names"])
        self.assertEqual(
            contract["action_components"],
            ("arm_left", "arm_right", "mobile"),
        )
        self.assertEqual(
            contract["inactive_actions"],
            {"head": "hold_current", "lift": "hold_current"},
        )
        self.assertEqual(set(contract["cameras"]), {"cam_left_head"})

    def test_simulation_ids_match_task_and_profile(self):
        for task_id, contract in (
            ("000001", TASK_000001_POLICY),
            ("000002", TASK_000002_POLICY),
        ):
            with self.subTest(task_id=task_id):
                simulation = contract["simulation"]
                self.assertEqual(
                    simulation["environment"],
                    f"Cyclo-Real-Showroom-Task{task_id}-FFW-SG2-v0",
                )
                self.assertEqual(
                    simulation["randomized_environment"],
                    f"Cyclo-Real-Showroom-Task{task_id}-Random-FFW-SG2-v0",
                )
                self.assertEqual(simulation["default_reset"], "deterministic")


if __name__ == "__main__":
    unittest.main()
