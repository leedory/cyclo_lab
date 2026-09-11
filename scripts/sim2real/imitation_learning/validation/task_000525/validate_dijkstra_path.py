"""Validate the Task525 Dijkstra path with zero approach offset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--report_path", type=Path, required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import numpy as np
import omni.kit.app
import torch

_extension_manager = omni.kit.app.get_app().get_extension_manager()
if not _extension_manager.is_extension_enabled("isaacsim.replicator.mobility_gen"):
    _extension_manager.set_extension_enabled_immediate("isaacsim.replicator.mobility_gen", True)

from isaaclab.managers import DatasetExportMode
from isaaclab_tasks.utils import parse_env_cfg
from isaaclab_mimic.locomanipulation_sdg.occupancy_map_utils import Point2d
from isaaclab_mimic.locomanipulation_sdg.path_utils import plan_path
from isaaclab_mimic.locomanipulation_sdg.scene_utils import RelativePose
from isaaclab_mimic.locomanipulation_sdg.transform_utils import transform_inv, transform_mul

import cyclo_lab  # noqa: F401
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.locomanipulation_sdg_contract import (
    CANDIDATE_BASE_GOAL_XYYAW,
)


TASK = "Cyclo-Real-Showroom-Task000525-Locomanipulation-SDG-FFW-SG2-v0"


def disable_policy_cameras(env_cfg) -> None:
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, name, None)
        setattr(env_cfg.observations.policy, name, None)


def main() -> None:
    # Upstream plan_path currently converts poses to NumPy, so planning runs on CPU.
    env_cfg = parse_env_cfg(TASK, device="cpu", num_envs=1)
    disable_policy_cameras(env_cfg)
    env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    env_cfg.recorders.record_pre_step_locomanipulation_sdg_output_data = None
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    env.reset()

    initial_base_pose = env.get_base().get_pose()
    source_fixture_pose = env.get_start_fixture().get_pose()
    base_goal = RelativePose(
        relative_pose=transform_mul(transform_inv(source_fixture_pose), initial_base_pose),
        parent=env.get_end_fixture(),
    )
    # approach_distance=0 means the Dijkstra endpoint is the final candidate;
    # it does not create the old -0.5 m local-X point inside an obstacle.
    planning_map = env.get_start_fixture().get_occupancy_map().buffered_meters(0.15)
    path = plan_path(start=env.get_base(), end=base_goal, occupancy_map=planning_map)

    start_xy = initial_base_pose[0, :2].detach().cpu().numpy()
    goal_xy = base_goal.get_pose_2d()[0, :2].detach().cpu().numpy()
    expected_goal_xy = np.asarray(CANDIDATE_BASE_GOAL_XYYAW[:2])
    old_approach_xy = goal_xy + np.asarray((-0.5, 0.0))
    goal_free = planning_map.check_world_point_in_freespace(Point2d(*goal_xy))
    old_approach_free = planning_map.check_world_point_in_freespace(Point2d(*old_approach_xy))
    endpoint_error = float(np.linalg.norm(path[-1].numpy() - goal_xy))
    if not goal_free:
        raise AssertionError(f"Task525 candidate goal is occupied: {goal_xy.tolist()}")
    if old_approach_free:
        raise AssertionError(f"Expected old approach to be blocked: {old_approach_xy.tolist()}")
    if endpoint_error > planning_map.resolution * 1.5:
        raise AssertionError(f"Dijkstra endpoint error is {endpoint_error} m")

    segment_lengths = torch.linalg.vector_norm(path[1:] - path[:-1], dim=1)
    report = {
        "task": TASK,
        "planner": "isaacsim.replicator.mobility_gen.generate_paths (Dijkstra tree)",
        "approach_distance_m": 0.0,
        "start_xy": start_xy.tolist(),
        "goal_xy": goal_xy.tolist(),
        "expected_candidate_goal_xy": expected_goal_xy.tolist(),
        "goal_is_free": bool(goal_free),
        "old_minus_0_5m_approach_xy": old_approach_xy.tolist(),
        "old_approach_is_free": bool(old_approach_free),
        "path_waypoint_count": int(path.shape[0]),
        "path_length_m": float(segment_lengths.sum()),
        "endpoint_error_m": endpoint_error,
        "path_xy": path.tolist(),
        "passed": True,
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
