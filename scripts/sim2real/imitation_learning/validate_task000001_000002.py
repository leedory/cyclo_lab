"""Headless reset and success smoke test for the temporary coffee transport tasks."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report_path", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401  Register Cyclo Lab showroom tasks.
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.coffee_transport_common import (
    CABINET_RIGHT_DELTA_Y_M,
    COFFEE_CAN_NAMES,
    COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
    LIFT_TRAVEL_COMMAND_M,
    ROBOT_SPAWN_XY_RANDOM_M,
    ROBOT_SPAWN_YAW_RANDOM_RAD,
    SHELF_LEVEL_DELTA_Z_M,
    SOURCE_SPAWN_SQUARES,
    SOURCE_UPPER_SUPPORT_Z_M,
)

TASKS = {
    "000001": ("Cyclo-Real-Showroom-Task000001-FFW-SG2-v0", "down", 19),
    "000002": ("Cyclo-Real-Showroom-Task000002-FFW-SG2-v0", "right", 22),
}


def disable_cameras(env_cfg) -> None:
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, name, None)
        setattr(env_cfg.observations.policy, name, None)


def assert_source_regions(env) -> None:
    for square in SOURCE_SPAWN_SQUARES:
        position = env.scene[square.name].data.root_pos_w[0]
        if not square.x_min_m <= float(position[0]) <= square.x_max_m:
            raise AssertionError(f"{square.name} X is outside its source square: {position.tolist()}")
        if not square.y_min_m <= float(position[1]) <= square.y_max_m:
            raise AssertionError(f"{square.name} Y is outside its source square: {position.tolist()}")


def assert_robot_spawn(env, task_id: str) -> float:
    robot = env.scene["robot"]
    episode_home = env._coffee_transport_episode_home_root_pose_w[0]
    default_home = robot.data.default_root_state[0, :7]
    if torch.any(torch.abs(episode_home[:2] - default_home[:2]) > ROBOT_SPAWN_XY_RANDOM_M + 1e-5):
        raise AssertionError(f"Task{task_id} HOME XY randomization exceeds 2 cm")
    dot = torch.sum(episode_home[3:7] * default_home[3:7]).abs().clamp(max=1.0)
    if float(2.0 * torch.acos(dot)) > ROBOT_SPAWN_YAW_RANDOM_RAD + 1e-5:
        raise AssertionError(f"Task{task_id} HOME yaw randomization exceeds 1 degree")
    root_pose = robot.data.root_state_w[0, :7]
    route_y_m = float(root_pose[1] - episode_home[1])
    if task_id == "000001":
        if not torch.allclose(root_pose, episode_home, atol=1e-5):
            raise AssertionError("Task000001 must spawn directly in its episode HOME region")
        return 0.0
    if not -1e-5 <= route_y_m <= CABINET_RIGHT_DELTA_Y_M + 1e-5:
        raise AssertionError(f"Task000002 route offset is out of range: {route_y_m}")
    return route_y_m


def place_in_destination(env, destination: str) -> None:
    for square in SOURCE_SPAWN_SQUARES:
        asset = env.scene[square.name]
        pose = asset.data.root_state_w[:, :7].clone()
        pose[:, 0] = (square.x_min_m + square.x_max_m) / 2.0
        pose[:, 1] = (square.y_min_m + square.y_max_m) / 2.0
        pose[:, 2] = SOURCE_UPPER_SUPPORT_Z_M + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
        if destination == "down":
            pose[:, 2] += SHELF_LEVEL_DELTA_Z_M
        else:
            pose[:, 1] += CABINET_RIGHT_DELTA_Y_M
        asset.write_root_pose_to_sim(pose)
        asset.write_root_velocity_to_sim(torch.zeros((1, 6), device=env.device))
    env.scene.write_data_to_sim()
    env.sim.forward()

def main() -> None:
    report: dict[str, dict[str, object]] = {}
    for task_id, (task_name, destination, expected_action_dim) in TASKS.items():
        cfg = parse_env_cfg(task_name, device=args.device, num_envs=1)
        disable_cameras(cfg)
        cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
        env = gym.make(task_name, cfg=cfg).unwrapped
        samples = []
        for _ in range(5):
            env.reset()
            assert_source_regions(env)
            samples.append({
                name: [round(float(value), 5) for value in env.scene[name].data.root_pos_w[0, :3]]
                for name in COFFEE_CAN_NAMES
            })
        if env.action_manager.total_action_dim != expected_action_dim:
            raise AssertionError(f"Task{task_id} action dimension is {env.action_manager.total_action_dim}")
        route_offset_m = assert_robot_spawn(env, task_id)
        if task_id == "000001":
            lift_id = env.scene["robot"].joint_names.index("lift_joint")
            reset_value = float(env.scene["robot"].data.joint_pos[0, lift_id])
            if not LIFT_TRAVEL_COMMAND_M - 1e-5 <= reset_value <= 1e-5:
                raise AssertionError(f"Task000001 lift reset is out of range: {reset_value}")
        else:
            reset_value = route_offset_m
        env.termination_manager.compute()
        initial_success = bool(env.termination_manager.get_term("success")[0])
        if initial_success:
            raise AssertionError(f"Task{task_id} incorrectly succeeds before transport")
        place_in_destination(env, destination)
        env.termination_manager.compute()
        final_success = bool(env.termination_manager.get_term("success")[0])
        if not final_success:
            raise AssertionError(f"Task{task_id} did not succeed in its translated destination region")
        report[task_id] = {
            "task": task_name,
            "action_dim": expected_action_dim,
            "static_jelly_bag_count": sum(name.startswith("jelly_bag") for name in env.scene.keys()),
            "initial_success": initial_success,
            "translated_destination_success": final_success,
            "last_reset_axis_value_m": round(reset_value, 5),
            "last_reset_cans_world_m": samples[-1],
        }
        env.close()
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
