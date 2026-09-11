"""Run a single-environment Isaac smoke check for coffee seed automation."""

from __future__ import annotations

import argparse
import json
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
        COFFEE_CAN_NAMES,
        COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
        SHELF_LEVEL_DELTA_Z_M,
        SOURCE_SPAWN_SQUARES,
        SOURCE_UPPER_SUPPORT_Z_M,
        coffee_transport_success,
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
    nominal_can_errors = {}
    for square in SOURCE_SPAWN_SQUARES:
        position = env.scene[square.name].data.root_pos_w[0]
        expected = torch.tensor(
            square.default_position_m,
            dtype=position.dtype,
            device=position.device,
        ) + env.scene.env_origins[0]
        nominal_can_errors[square.name] = float(
            torch.max(torch.abs(position - expected)).item()
        )
    assert max(nominal_can_errors.values()) <= 0.006

    destination = "down" if args.task_id == "000001" else "right"
    initial_transport_success = bool(
        coffee_transport_success(env, destination=destination)[0].item()
    )
    assert not initial_transport_success

    controller = CoffeeTransportSeedAutomation(env, task_id=args.task_id)
    configured_home = env.scene["robot"].data.default_joint_pos[
        :, controller._joint_ids
    ].detach().clone()
    initial_spawn_error = float(
        torch.max(
            torch.abs(controller._measured_joint_action() - configured_home)
        ).item()
    )
    initial_lift_position = float(
        controller._measured_joint_action()[0, controller._lift_index].item()
    )
    assert initial_spawn_error <= 0.03
    assert abs(initial_lift_position) <= 0.006
    controller.begin_episode()
    assert controller.state is SeedAutomationState.WAITING_FOR_OPERATOR_TACT
    assert controller.enable_operator_cycle(recorded_step_index=0)
    assert controller.state is SeedAutomationState.OPERATOR_PICK_AND_EXTRACT
    action = controller._measured_joint_action()
    if args.task_id == "000002":
        action = torch.cat((action, torch.zeros((1, 3), device=env.device)), dim=1)
    output = controller.apply(action, recorded_step_index=1)
    assert output.shape == (1, 19 if args.task_id == "000001" else 22)
    env.step(output)

    home_action = controller._nominal_action.detach().clone()
    probe_action = home_action.clone()
    probe_action[:, controller._action_index["arm_r_joint1"]] += 0.25
    for _ in range(20):
        probe = probe_action
        if args.task_id == "000002":
            probe = torch.cat((probe, torch.zeros((1, 3), device=env.device)), dim=1)
        env.step(probe)
    return_start_error = float(
        torch.max(torch.abs(controller._measured_joint_action() - home_action)).item()
    )
    assert return_start_error > 0.10, f"probe did not move arm: {return_start_error:.6f}"

    controller._return_start_action = controller._measured_joint_action()
    controller._held_action = home_action
    controller._carry_to_home_actions = []
    controller._enter(SeedAutomationState.RETURNING_HOME)
    for return_step in range(330):
        return_action = controller._measured_joint_action()
        if args.task_id == "000002":
            return_action = torch.cat(
                (return_action, torch.zeros((1, 3), device=env.device)), dim=1
            )
        output = controller.apply(return_action, recorded_step_index=return_step + 1)
        env.step(output)
        if controller.state is SeedAutomationState.AXIS_TO_DESTINATION:
            break
    else:
        raise AssertionError(controller.failure_reason or "HOME did not settle")

    measured = controller._measured_joint_action()[0]
    final_home_error = float(
        torch.max(
            torch.abs(
                measured[controller._home_settle_indices]
                - home_action[0, controller._home_settle_indices]
            )
        ).item()
    )
    held_arm_error = float(
        torch.max(
            torch.abs(
                output[0, controller._home_settle_indices]
                - home_action[0, controller._home_settle_indices]
            )
        ).item()
    )
    assert final_home_error <= controller.cfg.home_joint_tolerance_rad, (
        f"HOME error {final_home_error:.6f} exceeds "
        f"{controller.cfg.home_joint_tolerance_rad:.6f}"
    )
    assert held_arm_error <= 1.0e-6, (
        f"command is not fixed HOME: {held_arm_error:.9f}"
    )

    lowered_action = home_action.clone()
    lowered_action[:, controller._lift_index] = -0.20
    for _ in range(30):
        lower_command = lowered_action
        if args.task_id == "000002":
            lower_command = torch.cat(
                (lower_command, torch.zeros((1, 3), device=env.device)), dim=1
            )
        env.step(lower_command)
    lift_before_reset = float(
        controller._measured_joint_action()[0, controller._lift_index].item()
    )
    assert lift_before_reset <= -0.15

    env.reset()
    for _ in range(5):
        reset_command = home_action
        if args.task_id == "000002":
            reset_command = torch.cat(
                (reset_command, torch.zeros((1, 3), device=env.device)), dim=1
            )
        env.step(reset_command)
    lift_after_reset = float(
        controller._measured_joint_action()[0, controller._lift_index].item()
    )
    reset_home_error = float(
        torch.max(
            torch.abs(controller._measured_joint_action() - configured_home)
        ).item()
    )
    assert abs(lift_after_reset) <= 0.006
    assert reset_home_error <= 0.03

    for square in SOURCE_SPAWN_SQUARES:
        asset = env.scene[square.name]
        pose = asset.data.root_state_w[:, :7].clone()
        pose[:, 0] = (
            (square.x_min_m + square.x_max_m) / 2.0 + env.scene.env_origins[:, 0]
        )
        pose[:, 1] = (
            (square.y_min_m + square.y_max_m) / 2.0 + env.scene.env_origins[:, 1]
        )
        pose[:, 2] = (
            SOURCE_UPPER_SUPPORT_Z_M
            + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
            + env.scene.env_origins[:, 2]
        )
        if destination == "down":
            pose[:, 2] += SHELF_LEVEL_DELTA_Z_M
        else:
            pose[:, 1] += CABINET_RIGHT_DELTA_Y_M
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()
    translated_transport_success = bool(
        coffee_transport_success(env, destination=destination)[0].item()
    )
    assert translated_transport_success

    scene_keys = set(env.scene.keys())
    report = {
        "task_id": args.task_id,
        "state": controller.state.value,
        "action_dim": int(output.shape[1]),
        "scene_has_two_cans": (
            all(name in scene_keys for name in COFFEE_CAN_NAMES)
            and "coffee_can_center" not in scene_keys
        ),
        "scene_has_both_eefs": all(name in scene_keys for name in ("left_eef", "right_eef")),
        "home_cached": hasattr(env, "_coffee_transport_episode_home_root_pose_w"),
        "initial_spawn_error_rad": round(initial_spawn_error, 6),
        "initial_lift_position_m": round(initial_lift_position, 6),
        "nominal_can_max_errors_m": {
            name: round(error, 6) for name, error in nominal_can_errors.items()
        },
        "initial_transport_success": initial_transport_success,
        "translated_transport_success": translated_transport_success,
        "lift_before_reset_m": round(lift_before_reset, 6),
        "lift_after_reset_m": round(lift_after_reset, 6),
        "reset_home_error_rad": round(reset_home_error, 6),
        "return_home_steps": return_step + 1,
        "return_start_error_rad": round(return_start_error, 6),
        "final_home_error_rad": round(final_home_error, 6),
        "held_arm_error_rad": round(held_arm_error, 6),
    }
    assert all((
        report["scene_has_two_cans"],
        report["scene_has_both_eefs"],
        report["home_cached"],
    ))
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("COFFEE_SEED_ENV_SMOKE_OK " + json.dumps(report, sort_keys=True), flush=True)
    env.close()
except BaseException as exc:
    args.report_path.with_suffix(".fail.txt").write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )
    print(f"COFFEE_SEED_ENV_SMOKE_FAILED: {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    app.close()
