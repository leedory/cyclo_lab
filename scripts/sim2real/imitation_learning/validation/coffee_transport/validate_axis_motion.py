"""Validate one Isaac lift/base round trip for coffee automation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task_id", choices=("000001", "000002"), required=True)
parser.add_argument("--report_path", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

try:
    import gymnasium as gym
    import torch

    from isaaclab.managers import DatasetExportMode
    from isaaclab_tasks.utils import parse_env_cfg

    import cyclo_lab  # noqa: F401
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.coffee_transport_common import (
        CABINET_RIGHT_DELTA_Y_M,
        LIFT_TRAVEL_COMMAND_M,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.seed_automation import (
        CoffeeTransportSeedAutomation,
        SeedAutomationState,
    )

    task_name = f"Cyclo-Real-Showroom-Task{args.task_id}-FFW-SG2-v0"
    cfg = parse_env_cfg(task_name, device=args.device, num_envs=1)
    cfg.init_action_cfg("record")
    for camera_name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(cfg.scene, camera_name, None)
        setattr(cfg.observations.policy, camera_name, None)
    cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env = gym.make(task_name, cfg=cfg).unwrapped
    env.reset()

    controller = CoffeeTransportSeedAutomation(env, task_id=args.task_id)
    controller.begin_episode()
    home = env._coffee_transport_episode_home_root_pose_w[0].detach().clone()
    root_samples = []

    def yaw_from_wxyz(quaternion):
        w, x, y, z = quaternion
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def run_axis(state):
        controller._held_action = controller._measured_joint_action()
        controller._begin_axis_motion(
            destination=state is SeedAutomationState.AXIS_TO_DESTINATION
        )
        initial_state = state
        limit = round(20.0 * controller.cfg.control_hz)
        lift_commands = []
        for step in range(limit):
            action = controller._measured_joint_action()
            if args.task_id == "000002":
                action = torch.cat((action, torch.zeros((1, 3), device=env.device)), dim=1)
            output = controller._apply_axis_motion(action, recorded_step_index=step)
            if args.task_id == "000001":
                lift_commands.append(float(output[0, controller._lift_index].item()))
            env.step(output)
            root_samples.append(env.scene["robot"].data.root_pose_w[0].detach().clone())
            if controller.failed:
                raise AssertionError(controller.failure_reason)
            if controller.state is not initial_state:
                return step + 1, lift_commands
        raise AssertionError(f"{initial_state.value} did not complete")

    destination_steps, destination_lift_commands = run_axis(
        SeedAutomationState.AXIS_TO_DESTINATION
    )
    destination_axis = controller._current_axis_scalar()
    source_steps, source_lift_commands = run_axis(
        SeedAutomationState.AXIS_TO_SOURCE
    )
    source_axis = controller._current_axis_scalar()

    roots = torch.stack(root_samples)
    yaw_errors = torch.atan2(
        torch.sin(torch.stack([yaw_from_wxyz(row[3:7]) for row in roots]) - yaw_from_wxyz(home[3:7])),
        torch.cos(torch.stack([yaw_from_wxyz(row[3:7]) for row in roots]) - yaw_from_wxyz(home[3:7])),
    )
    x_deviation = torch.max(torch.abs(roots[:, 0] - home[0]))
    report = {
        "task_id": args.task_id,
        "destination_steps": destination_steps,
        "source_steps": source_steps,
        "destination_axis_m": round(destination_axis, 6),
        "source_axis_m": round(source_axis, 6),
        "max_root_x_deviation_m": round(float(x_deviation), 6),
        "max_root_yaw_error_deg": round(float(torch.max(torch.abs(yaw_errors))) * 180.0 / math.pi, 6),
    }
    destination_target = LIFT_TRAVEL_COMMAND_M if args.task_id == "000001" else CABINET_RIGHT_DELTA_Y_M
    assert abs(destination_axis - destination_target) <= 0.012
    assert abs(source_axis) <= 0.012
    if args.task_id == "000001":
        assert destination_lift_commands
        assert source_lift_commands
        assert all(
            abs(value - LIFT_TRAVEL_COMMAND_M) <= 1.0e-6
            for value in destination_lift_commands
        )
        assert all(abs(value) <= 1.0e-6 for value in source_lift_commands)
        report["destination_lift_action_unique_m"] = sorted(
            set(round(value, 7) for value in destination_lift_commands)
        )
        report["source_lift_action_unique_m"] = sorted(
            set(round(value, 7) for value in source_lift_commands)
        )
    else:
        assert float(x_deviation) <= 0.02
        assert float(torch.max(torch.abs(yaw_errors))) <= math.radians(1.0)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COFFEE_AXIS_ROUND_TRIP_OK " + json.dumps(report, sort_keys=True), flush=True)
    env.close()
except BaseException as exc:
    args.report_path.with_suffix(".fail.txt").write_text(
        f"{type(exc).__name__}: {exc}\n", encoding="utf-8"
    )
    print(f"COFFEE_AXIS_ROUND_TRIP_FAILED: {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    app.close()
