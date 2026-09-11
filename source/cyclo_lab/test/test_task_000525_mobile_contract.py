"""Behavioral contracts for Task000525 mobile recording and validation tools."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_BASE_ACTION_NAMES,
    FFW_SG2_BASE_ACTION_UNITS,
    FFW_SG2_MOBILE_ACTION_DIM,
    FFW_SG2_MOBILE_ACTION_NAMES,
    hdf5_contract_metadata,
)


REPO = Path(__file__).resolve().parents[3]
TASK_ROOT = (
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
)
COFFEE_TASK_ROOT = TASK_ROOT.parent / "coffee_transport"
def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_parser(registry) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--robot_type", default="FFW_SG2")
    parser.add_argument("--num_demos", type=int, default=2)
    registry.add_cli_arguments(parser)
    return parser


class Task000525MobileContractTest(unittest.TestCase):
    def test_mobile_metadata_exposes_the_full_three_axis_base(self):
        actions_cfg = SimpleNamespace(
            base_action=object(),
            arm_l_action=object(),
            arm_r_action=object(),
        )

        metadata = hdf5_contract_metadata(actions_cfg)

        self.assertEqual(FFW_SG2_BASE_ACTION_NAMES, ("linear_x", "linear_y", "angular_z"))
        self.assertEqual(FFW_SG2_BASE_ACTION_UNITS, ("m/s", "m/s", "rad/s"))
        self.assertEqual(FFW_SG2_MOBILE_ACTION_DIM, 22)
        self.assertEqual(tuple(metadata["action_names"]), FFW_SG2_MOBILE_ACTION_NAMES)
        self.assertEqual(metadata["robot_contract_id"], "ffw_sg2_rev1_mobile_22d_v1")
        self.assertEqual(metadata["base_action_frame"], "robot_body")
        self.assertEqual(
            metadata["observation_state_paths"],
            ["obs/joint_pos", "obs/base_velocity_body"],
        )

    def test_registry_builds_the_manual_task525_controller(self):
        registry = load_module("task_recording_registry", TASK_ROOT.parent / "recording.py")
        recording = load_module("task525_recording", TASK_ROOT / "recording.py")
        args = make_parser(registry).parse_args(
            [
                "--task",
                "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
                "--task525_phase_markers",
                "--task525_seed_regions",
                "a,b",
            ]
        )
        factory = registry.build_recording_factory(args)

        self.assertEqual(factory.seed_regions, ("A", "B"))
        self.assertEqual(factory.active_side, "left")
        self.assertFalse(factory.automatic)

        class FakeTeleop:
            def __init__(self):
                self.callbacks = {}

            def add_callback(self, key, callback):
                self.callbacks[key] = callback

        teleop = FakeTeleop()
        controller = recording.Task000525RecordingController(
            factory,
            SimpleNamespace(),
            teleop,
            wheel_speed_norm=lambda _env: 0.0,
            action_joint_names=(
                "gripper_l_joint1",
                "gripper_r_joint1",
                "head_joint1",
                "head_joint2",
                "lift_joint",
            ),
            joint_position_limits={"head_joint1": (-0.2, 1.2)},
            lift_position_upper=0.0,
        )
        step = {"value": 2}
        controller.register_callbacks(
            is_recording=lambda: True,
            recorded_step=lambda: step["value"],
        )

        self.assertEqual(set(teleop.callbacks), {"F", "G", "M", "P", "H"})
        for key, value in (("F", 2), ("G", 2), ("M", 3), ("P", 4), ("H", 5)):
            step["value"] = value
            teleop.callbacks[key]()

        self.assertEqual(
            controller.markers,
            {
                "grasp_step": 2,
                "lift_step": 2,
                "navigate_step": 3,
                "place_step": 4,
                "release_step": 5,
            },
        )
        self.assertTrue(controller.validate_save())
        self.assertEqual(
            controller.dataset_metadata()["task525_base_command_source"],
            "external_cmd_vel_foot_pad_or_keyboard",
        )

    def test_registry_builds_the_coffee_controller_and_rejects_mixed_modes(self):
        registry = load_module(
            "coffee_task_recording_registry", TASK_ROOT.parent / "recording.py"
        )
        coffee_recording = load_module(
            "coffee_transport_recording", COFFEE_TASK_ROOT / "recording.py"
        )
        parser = make_parser(registry)
        args = parser.parse_args(
            [
                "--task",
                "Cyclo-Real-Showroom-Task000001-FFW-SG2-v0",
                "--coffee_transport_seed_automation",
            ]
        )
        factory = registry.build_recording_factory(args)
        self.assertEqual(factory.task_id, "000001")
        self.assertIsNone(factory.configure_env_cfg(SimpleNamespace()))

        exclusive = parser.parse_args(
            [
                "--task",
                "Cyclo-Real-Showroom-Task000001-FFW-SG2-v0",
                "--coffee_transport_seed_automation",
                "--task525_phase_markers",
                "--task525_seed_regions",
                "A,B",
            ]
        )
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            registry.build_recording_factory(exclusive)

        automation = Mock(
            failed=False,
            failure_reason="",
            task_complete=False,
            operator_enabled=True,
            awaiting_operator_tact=True,
            current_cycle=SimpleNamespace(arm="right"),
        )
        automation.apply.side_effect = lambda actions, **_kwargs: actions
        automation.final_placement_failures.return_value = {}
        automation.metadata.return_value = {"coffee_completed_cycles": 2}
        automation.enable_operator_cycle.return_value = True
        teleop = Mock()
        callbacks = {}
        teleop.add_callback.side_effect = callbacks.__setitem__
        teleop.arm_tact_generation.return_value = 1
        controller = coffee_recording.CoffeeTransportRecordingController(
            factory,
            SimpleNamespace(),
            teleop,
            automation=automation,
            action_joint_names=(),
        )
        controller.register_callbacks(
            is_recording=lambda: True,
            recorded_step=lambda: 0,
        )
        controller.on_recording_started()
        callbacks["G"]()
        automation.begin_episode.assert_called_once_with(recorded_step_index=0)
        automation.request_g.assert_called_once_with()
        teleop.set_arm_enabled.assert_called_once_with("right", False)
        self.assertFalse(controller.validate_save())
        automation.task_complete = True
        self.assertTrue(controller.validate_save())

        actions = object()
        self.assertIs(controller.apply_actions(actions, recording=True, step_index=7), actions)
        teleop.arm_tact_generation.return_value = 2
        self.assertIs(controller.apply_actions(actions, recording=True, step_index=8), actions)
        automation.enable_operator_cycle.assert_called_once_with(recorded_step_index=8)
        teleop.begin_control_activation.assert_called_once_with()
        self.assertEqual(controller.episode_metadata(), {"coffee_completed_cycles": 2})
        self.assertEqual(controller.dataset_metadata()["coffee_task_id"], "000001")

    def test_automatic_mode_requires_phase_markers(self):
        registry = load_module("task_recording_registry_invalid", TASK_ROOT.parent / "recording.py")
        args = make_parser(registry).parse_args(
            [
                "--task",
                "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
                "--task525_base_mode",
                "dijkstra",
            ]
        )

        with self.assertRaisesRegex(ValueError, "requires --task525_phase_markers"):
            registry.build_recording_factory(args)

if __name__ == "__main__":
    unittest.main()
