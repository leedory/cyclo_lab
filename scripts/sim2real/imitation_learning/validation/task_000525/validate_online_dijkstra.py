"""Record and replay one Task525 online-Dijkstra carrying segment in Isaac Lab."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument("--mode", choices=("record", "validate", "replay"), required=True)
parser.add_argument("--pre_pick_steps", type=int, default=8)
parser.add_argument("--place_steps", type=int, default=12)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym
import h5py
import numpy as np
import omni.kit.app
import torch

extension_manager = omni.kit.app.get_app().get_extension_manager()
if not extension_manager.is_extension_enabled("isaacsim.replicator.mobility_gen"):
    extension_manager.set_extension_enabled_immediate("isaacsim.replicator.mobility_gen", True)

from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.online_dijkstra import (
    Task525OnlineDijkstraCfg,
    Task525OnlineDijkstraNavigator,
)


TASK = "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0"
DATASET_STEM = "task000525_online_dijkstra_smoke"


def disable_policy_cameras(env_cfg) -> None:
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, name, None)
        setattr(env_cfg.observations.policy, name, None)


def make_cfg(*, record: bool, output_dir: Path | None = None):
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    disable_policy_cameras(env_cfg)
    env_cfg.episode_length_s = 600.0
    if record:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = str(output_dir)
        env_cfg.recorders.dataset_filename = DATASET_STEM
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    return env_cfg


def yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def root_pose(env) -> np.ndarray:
    robot = env.scene["robot"]
    return torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1)[0].detach().cpu().numpy()


def hold_action(observation) -> torch.Tensor:
    action = torch.zeros((1, 22), dtype=torch.float32, device=args.device)
    action[0, :19] = observation["policy"]["joint_pos"][0]
    return action


def close_recording_handler(env) -> None:
    handler = env.recorder_manager._dataset_file_handler
    if handler is not None:
        handler.close()
        env.recorder_manager._dataset_file_handler = None


def validate_recorded_hdf5(
    dataset_path: Path,
    expected_goal_xyyaw: tuple[float, float, float],
    *,
    expected_lift_target_m: float | None = None,
) -> dict:
    with h5py.File(dataset_path, "r") as hdf:
        data = hdf["data"]
        names = sorted(data.keys())
        if len(names) != 1:
            raise AssertionError(f"expected exactly one recorded auto-nav demo, got {names}")
        demo = data[names[0]]
        actions = np.asarray(demo["actions"])
        root = np.asarray(demo["obs/robot_root_pose_world"])
        if actions.ndim != 2 or actions.shape[1] != 22:
            raise AssertionError(f"expected [T,22] actions, got {actions.shape}")
        nonzero_base = np.linalg.norm(actions[:, 19:22], axis=1) > 1e-6
        if not np.any(nonzero_base):
            raise AssertionError("online navigation wrote no non-zero base3 command to HDF5")
        if np.any(np.abs(actions[: args.pre_pick_steps, 19:22]) > 1e-7):
            raise AssertionError("base command was non-zero before the simulated G transition")
        lift_target_error = None
        if expected_lift_target_m is not None:
            # The final synthetic place hold can use a slightly lagged measured
            # lift state, so verify that the commanded autonomous target itself
            # was present in the recorded 19D action prefix.
            lift_target_error = float(np.min(np.abs(actions[:, 18] - expected_lift_target_m)))
            if lift_target_error > 0.01:
                raise AssertionError(
                    "recorded lift action never reached the autonomous target: "
                    f"error={lift_target_error:.4f} m"
                )
        goal_xy = np.asarray(expected_goal_xyyaw[:2])
        final_xy = root[-1, :2]
        final_yaw = yaw_from_wxyz(root[-1, 3:7])
        position_error = float(np.linalg.norm(final_xy - goal_xy))
        yaw_error = abs(wrap_to_pi(final_yaw - expected_goal_xyyaw[2]))
        if position_error > 0.08 or yaw_error > 0.10:
            raise AssertionError(
                f"recorded final pose misses goal: position={position_error:.3f} m, yaw={yaw_error:.3f} rad"
            )
        return {
            "demo": names[0],
            "frames": int(actions.shape[0]),
            "nonzero_base_frames": int(nonzero_base.sum()),
            "base_command_peak_abs": np.max(np.abs(actions[:, 19:22]), axis=0).tolist(),
            "recorded_final_position_error_m": position_error,
            "recorded_final_yaw_error_rad": yaw_error,
            "recorded_lift_target_error_m": lift_target_error,
        }


def record(output_dir: Path) -> dict:
    env_cfg = make_cfg(record=True, output_dir=output_dir)
    env = gym.make(TASK, cfg=env_cfg).unwrapped
    try:
        observation, _ = env.reset()
        for _ in range(args.pre_pick_steps):
            observation, *_ = env.step(hold_action(observation))
        initial_lift_position = float(observation["policy"]["joint_pos"][0, 18].item())

        navigator = Task525OnlineDijkstraNavigator(
            env,
            Task525OnlineDijkstraCfg(),
            control_hz=float(env_cfg.control_hz),
        )
        if not navigator.start(hold_action(observation)):
            raise RuntimeError(navigator.failure_reason)
        planned_path = navigator.path.points.detach().cpu().tolist()
        while navigator.active:
            action = navigator.apply(hold_action(observation))
            observation, *_ = env.step(action)
        if navigator.awaiting_place_activation:
            # The recorder waits for a fresh right-leader tact message here.
            # This simulator-only test has no A3 leader, so it explicitly
            # performs that state-machine handoff before recording place hold.
            if not navigator.enable_place_control():
                raise RuntimeError("failed to release Task525 place control after arrival")
        else:
            raise AssertionError(
                "online navigation must wait for an explicit place activation after lift lowering"
            )
        if not navigator.completed:
            pose = navigator.last_pose_2d
            raise RuntimeError(
                "online navigation failed: "
                f"{navigator.failure_reason}; steps={navigator.navigation_steps}; "
                f"last_pose={None if pose is None else pose.tolist()}; "
                f"last_command={navigator.last_command}; replans={navigator.replan_count}"
            )

        for _ in range(args.place_steps):
            observation, *_ = env.step(hold_action(observation))
        final_pose = root_pose(env)
        env.reset()  # finish/export the one recorder episode before closing the HDF5 handler
        close_recording_handler(env)
        report = validate_recorded_hdf5(
            output_dir / f"{DATASET_STEM}.hdf5",
            navigator.cfg.goal_xyyaw,
            expected_lift_target_m=navigator.lift_lower_target_position,
        )
        report.update(
            {
                "planned_path_xy": planned_path,
                "controller_status": navigator.status,
                "return_home_steps": navigator.return_home_steps,
                "lift_lower_start_position_m": navigator.lift_lower_start_position,
                "lift_lower_target_position_m": navigator.lift_lower_target_position,
                "lift_lower_steps": navigator.lift_lower_steps,
                "lift_lower_requested_distance_m": float(
                    initial_lift_position - (navigator.lift_lower_target_position or 0.0)
                ),
                "controller_replans": navigator.replan_count,
                "live_final_root_pose_wxyz": final_pose.tolist(),
            }
        )
        return report
    finally:
        env.close()


def replay(dataset_path: Path) -> dict:
    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    names = sorted(handler.get_episode_names())
    if len(names) != 1:
        raise AssertionError(f"expected one demo, got {names}")
    episode = handler.load_episode(names[0], args.device)
    handler.close()

    env = gym.make(TASK, cfg=make_cfg(record=False)).unwrapped
    try:
        env.reset_to(
            state=episode.get_initial_state(),
            env_ids=torch.tensor([0], dtype=torch.long, device=args.device),
            is_relative=True,
        )
        for action in episode.data["actions"]:
            env.step(action.reshape(1, -1))
        final_pose = root_pose(env)
        expected = episode.data["states"]["articulation"]["robot"]["root_pose"][-1].detach().cpu().numpy()
        position_error = float(np.linalg.norm(final_pose[:3] - expected[:3]))
        yaw_error = abs(wrap_to_pi(yaw_from_wxyz(final_pose[3:7]) - yaw_from_wxyz(expected[3:7])))
        if position_error > 0.03 or yaw_error > 0.03:
            raise AssertionError(f"replay mismatch: position={position_error:.4f}, yaw={yaw_error:.4f}")
        return {
            "demo": names[0],
            "position_error_m": position_error,
            "yaw_error_rad": yaw_error,
        }
    finally:
        env.close()


def main() -> None:
    output_dir = args.output_dir.resolve()
    dataset_path = output_dir / f"{DATASET_STEM}.hdf5"
    report_path = output_dir / "online_dijkstra_validation_report.json"
    if args.mode == "record":
        output_dir.mkdir(parents=True, exist_ok=False)
        report = {"task": TASK, "record": record(output_dir), "replay": "pending", "passed": False}
    elif args.mode == "validate":
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        report = {
            "task": TASK,
            "record": validate_recorded_hdf5(
                dataset_path, Task525OnlineDijkstraCfg().goal_xyyaw
            ),
            "replay": "pending",
            "passed": False,
        }
    else:
        if not dataset_path.is_file() or not report_path.is_file():
            raise FileNotFoundError("run --mode record before --mode replay")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["replay"] = replay(dataset_path)
        report["passed"] = True
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
