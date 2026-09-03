"""Smoke-test the Task000525 locomanipulation_sdg adapter in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report_path", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import torch

from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401
from cyclo_lab.robot_specs.ffw.sg2 import hdf5_contract_metadata

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.arrangement import (
    TASK000525_TARGET_OBJECT,
    manipulation_side_for_region,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.layout import (
    TASK000525_SELECTED_LAYOUT_KEY,
    selected_sampling_regions,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.reset_events import (
    randomize_coffee_can_center_regions,
)

TASK = "Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0"
SAFE_TEST_SPAWN = (1.0, 0.7, 0.0)
SAFE_TEST_ROT_WXYZ = (0.0, 0.0, 0.0, 1.0)


def disable_policy_cameras(env_cfg) -> None:
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, name, None)
        setattr(env_cfg.observations.policy, name, None)


def main() -> None:
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    disable_policy_cameras(env_cfg)
    env_cfg.scene.robot.init_state.pos = SAFE_TEST_SPAWN
    env_cfg.scene.robot.init_state.rot = SAFE_TEST_ROT_WXYZ
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env_cfg.recorders.record_pre_step_locomanipulation_sdg_output_data = None

    env = gym.make(TASK, cfg=env_cfg).unwrapped
    observation, _ = env.reset()
    policy = observation["policy"]
    joint_pos = policy["joint_pos"][0]
    # build_action_vector normally receives this through load_input_data().
    # This standalone smoke has no source EpisodeData, so use the measured
    # reset joint19 as its explicit passive/left-arm hold command.
    env._source_joint_action = joint_pos.detach().clone()
    base_command = torch.tensor((0.20, 0.0, 0.0), device=env.device)
    left_target_world = policy["left_eef_pose_world"][0].clone()
    right_target_world = policy["right_eef_pose_world"][0].clone()

    def build_action():
        return env.build_action_vector(
            left_target_world,
            right_target_world,
            joint_pos[7:8],
            joint_pos[15:16],
            base_command,
        )

    action = build_action()
    if tuple(action.shape) != (1, 22):
        raise AssertionError(f"SDG adapter action shape is {tuple(action.shape)}")
    if not torch.allclose(action[0, 19:22], base_command):
        raise AssertionError("SDG adapter changed the body velocity command")

    initial_root = env.get_base().get_pose()[0].detach().cpu().numpy()
    for _ in range(40):
        action = build_action()
        env.step(action)
    final_root = env.get_base().get_pose()[0].detach().cpu().numpy()
    world_delta = final_root[:3] - initial_root[:3]
    w, x, y, z = initial_root[3:7]
    initial_yaw = math.atan2(
        2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)
    )
    body_forward = math.cos(initial_yaw) * world_delta[0] + math.sin(
        initial_yaw
    ) * world_delta[1]
    if body_forward <= 0.15:
        raise AssertionError(
            f"SDG +linear_x did not move forward: body delta={body_forward}"
        )

    occupancy_map = env.get_start_fixture().get_occupancy_map()
    occupied_cells = int(np.count_nonzero(occupancy_map.occupied_mask()))
    if occupied_cells <= 0:
        raise AssertionError("Task525 static occupancy map has no occupied cells")

    contract = hdf5_contract_metadata(env.cfg.actions)
    if contract["robot_contract_id"] != (
        "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
    ):
        raise AssertionError(f"wrong SDG HDF5 contract: {contract}")

    arrangement_report: dict[str, dict[str, object]] = {}
    env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
    for region_key in ("A", "B", "C", "D"):
        expected_side = manipulation_side_for_region(region_key)
        env.set_task525_episode_context(
            target_region=region_key,
            manipulation_side=expected_side,
            source_demo="runtime_smoke",
        )
        randomize_coffee_can_center_regions(
            env,
            env_ids,
            layout_key=TASK000525_SELECTED_LAYOUT_KEY,
            target_region=region_key,
            sample_positions=False,
            shuffle_distractors=True,
        )
        env.scene.write_data_to_sim()
        env.sim.forward()
        resolved = env._task525_arrangements[0]
        if resolved["region_to_object"][region_key] != TASK000525_TARGET_OBJECT:
            raise AssertionError(f"orange target was not placed in region {region_key}")
        target_region = next(
            region
            for region in selected_sampling_regions()
            if region.region_key == region_key
        )
        orange_xy = env.scene[TASK000525_TARGET_OBJECT].data.root_pos_w[0, :2]
        expected_xy = torch.tensor(
            target_region.default_position_m[:2],
            device=env.device,
            dtype=orange_xy.dtype,
        )
        if not torch.allclose(orange_xy, expected_xy, atol=1e-6, rtol=0.0):
            raise AssertionError(
                f"region {region_key} orange center mismatch: "
                f"{orange_xy.tolist()} vs {expected_xy.tolist()}"
            )
        arrangement_report[region_key] = {
            "manipulation_side": expected_side,
            "region_to_object": resolved["region_to_object"],
        }

    report = {
        "task": TASK,
        "hdf5_contract": contract,
        "action_shape": list(action.shape),
        "base_command_body": action[0, 19:22].detach().cpu().tolist(),
        "world_translation_after_40_steps_m": world_delta.tolist(),
        "body_forward_after_40_steps_m": float(body_forward),
        "start_fixture_pose_wxyz": env.get_start_fixture().get_pose()[0]
        .detach()
        .cpu()
        .tolist(),
        "end_fixture_pose_wxyz": env.get_end_fixture().get_pose()[0]
        .detach()
        .cpu()
        .tolist(),
        "occupancy_map": {
            "shape": list(occupancy_map.data.shape),
            "resolution_m": float(occupancy_map.resolution),
            "origin_xyyaw": list(occupancy_map.origin),
            "occupied_cells": occupied_cells,
        },
        "passed": True,
        "arrangements": arrangement_report,
    }
    if args.report_path is not None:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
        args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
