"""Record and replay a deterministic Task000525 mobile-direction smoke dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--mode",
    choices=("record", "replay"),
    required=True,
    help="Run recording and replay in separate Isaac processes.",
)
parser.add_argument("--settle_steps", type=int, default=20)
parser.add_argument("--command_steps", type=int, default=40)
parser.add_argument("--stop_steps", type=int, default=10)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import json
import math

import gymnasium as gym
import h5py
import numpy as np
import torch

from isaaclab.managers import DatasetExportMode
from isaaclab.utils.datasets import HDF5DatasetFileHandler
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_MOBILE_ACTION_DIM,
    FFW_SG2_MOBILE_ACTION_NAMES,
    hdf5_contract_metadata,
)


TASK = "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0"
DATASET_STEM = "task000525_mobile_direction_smoke"
SAFE_TEST_SPAWN = (1.0, 0.7, 0.0)
SAFE_TEST_ROT_WXYZ = (0.0, 0.0, 0.0, 1.0)
COMMANDS = {
    "positive_vx": (0.20, 0.0, 0.0),
    "positive_vy": (0.0, 0.20, 0.0),
    "positive_wz": (0.0, 0.0, 0.30),
}


def yaw_from_wxyz(quaternion) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def body_delta(initial_pose: np.ndarray, final_pose: np.ndarray) -> np.ndarray:
    yaw = yaw_from_wxyz(initial_pose[3:7])
    world = final_pose[:2] - initial_pose[:2]
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    body_xy = np.asarray(
        (cosine * world[0] + sine * world[1], -sine * world[0] + cosine * world[1])
    )
    yaw_delta = wrap_to_pi(
        yaw_from_wxyz(final_pose[3:7]) - yaw_from_wxyz(initial_pose[3:7])
    )
    return np.asarray((body_xy[0], body_xy[1], yaw_delta), dtype=np.float64)


def disable_policy_cameras(env_cfg) -> None:
    for name in ("cam_head", "cam_wrist_left", "cam_wrist_right"):
        setattr(env_cfg.scene, name, None)
        setattr(env_cfg.observations.policy, name, None)


def make_cfg(*, record: bool, output_dir: Path | None = None):
    env_cfg = parse_env_cfg(TASK, device=args.device, num_envs=1)
    disable_policy_cameras(env_cfg)
    env_cfg.scene.robot.init_state.pos = SAFE_TEST_SPAWN
    env_cfg.scene.robot.init_state.rot = SAFE_TEST_ROT_WXYZ
    env_cfg.episode_length_s = 600.0
    if record:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_ALL
        env_cfg.recorders.dataset_export_dir_path = str(output_dir)
        env_cfg.recorders.dataset_filename = DATASET_STEM
    else:
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_NONE
    return env_cfg


def root_pose(env) -> np.ndarray:
    robot = env.scene["robot"]
    return torch.cat((robot.data.root_pos_w, robot.data.root_quat_w), dim=-1)[
        0
    ].detach().cpu().numpy()


def hold_action(observation, base_command=(0.0, 0.0, 0.0)) -> torch.Tensor:
    action = torch.zeros(
        (1, FFW_SG2_MOBILE_ACTION_DIM), dtype=torch.float32, device=args.device
    )
    action[0, :19] = observation["policy"]["joint_pos"][0]
    action[0, 19:22] = torch.tensor(
        base_command, dtype=torch.float32, device=args.device
    )
    return action


def step_segment(env, command: tuple[float, float, float]) -> dict:
    observation, _ = env.reset()
    initial = root_pose(env)
    measured = []
    actions = []

    for _ in range(args.settle_steps):
        action = hold_action(observation)
        observation, *_ = env.step(action)
        measured.append(
            observation["policy"]["base_velocity_body"][0].detach().cpu().numpy()
        )
        actions.append(action[0].detach().cpu().numpy())

    for _ in range(args.command_steps):
        action = hold_action(observation, command)
        observation, *_ = env.step(action)
        measured.append(
            observation["policy"]["base_velocity_body"][0].detach().cpu().numpy()
        )
        actions.append(action[0].detach().cpu().numpy())

    for _ in range(args.stop_steps):
        action = hold_action(observation)
        observation, *_ = env.step(action)
        measured.append(
            observation["policy"]["base_velocity_body"][0].detach().cpu().numpy()
        )
        actions.append(action[0].detach().cpu().numpy())

    final = root_pose(env)
    return {
        "initial_root_pose_wxyz": initial,
        "final_root_pose_wxyz": final,
        "body_delta_xyyaw": body_delta(initial, final),
        "measured_base_velocity_body": np.asarray(measured),
        "submitted_actions": np.asarray(actions),
    }


def json_attr(value):
    if isinstance(value, (tuple, list, dict)):
        return json.dumps(value)
    return value


def attach_and_validate_hdf5_metadata(dataset_path: Path, env_cfg) -> dict:
    checks = {}
    metadata = {
        "schema_version": "cyclo_lab_hdf5_v1",
        "task_env_name": TASK,
        "dataset_origin": "task000525_mobile_direction_smoke",
        "control_hz": float(env_cfg.control_hz),
        "observation_semantics": "pre_step",
        "obs_last_action_semantics": "previous_step_action",
        "scene_state_semantics": "post_step",
        **hdf5_contract_metadata(env_cfg.actions),
    }
    with h5py.File(dataset_path, "r+") as hdf:
        data = hdf["data"]
        for key, value in metadata.items():
            data.attrs[key] = json_attr(value)

        demo_names = sorted(data.keys())
        checks["episode_count"] = len(demo_names)
        checks["episodes"] = {}
        if len(demo_names) != len(COMMANDS):
            raise AssertionError(f"expected 3 recorded demos, got {demo_names}")

        for demo_name, (label, command) in zip(demo_names, COMMANDS.items()):
            demo = data[demo_name]
            actions = np.asarray(demo["actions"])
            last_actions = np.asarray(demo["obs/actions"])
            measured = np.asarray(demo["obs/base_velocity_body"])
            root_world = np.asarray(demo["obs/robot_root_pose_world"])
            if actions.shape[1] != 22 or last_actions.shape != actions.shape:
                raise AssertionError(
                    f"{demo_name}: bad action shapes {actions.shape}, {last_actions.shape}"
                )
            command_rows = actions[
                args.settle_steps : args.settle_steps + args.command_steps, 19:22
            ]
            if not np.allclose(command_rows, np.asarray(command), atol=1e-7):
                raise AssertionError(f"{demo_name}: HDF5 base command values changed")
            if not np.allclose(last_actions[1:], actions[:-1], atol=1e-6):
                raise AssertionError(f"{demo_name}: obs/actions is not the previous action")
            if measured.shape != (actions.shape[0], 3):
                raise AssertionError(f"{demo_name}: measured base state shape is {measured.shape}")
            if root_world.shape != (actions.shape[0], 7):
                raise AssertionError(f"{demo_name}: root pose shape is {root_world.shape}")

            checks["episodes"][demo_name] = {
                "label": label,
                "frames": int(actions.shape[0]),
                "command": list(command),
                "measured_peak_abs": np.max(np.abs(measured), axis=0).tolist(),
            }
    return checks


def validate_direction(label: str, result: dict) -> dict:
    delta = result["body_delta_xyyaw"]
    measured = result["measured_base_velocity_body"]
    axis = {"positive_vx": 0, "positive_vy": 1, "positive_wz": 2}[label]
    minimum = 0.05 if axis < 2 else 0.10
    if delta[axis] <= minimum:
        raise AssertionError(f"{label}: positive command produced delta {delta.tolist()}")
    if np.max(measured[:, axis]) <= 0.03:
        raise AssertionError(f"{label}: measured body velocity never became positive")
    if axis < 2:
        cross_axis = 1 - axis
        if abs(delta[cross_axis]) > max(0.08, 0.6 * abs(delta[axis])):
            raise AssertionError(f"{label}: excessive cross-axis drift {delta.tolist()}")
    return {
        "body_delta_xyyaw": delta.tolist(),
        "measured_peak": np.max(measured, axis=0).tolist(),
        "measured_min": np.min(measured, axis=0).tolist(),
    }


def replay_dataset(dataset_path: Path) -> dict:
    print(f"[SMOKE] loading replay dataset: {dataset_path}", flush=True)
    handler = HDF5DatasetFileHandler()
    handler.open(str(dataset_path))
    demo_names = sorted(handler.get_episode_names())
    episodes = [(name, handler.load_episode(name, args.device)) for name in demo_names]
    handler.close()

    replay_env = gym.make(TASK, cfg=make_cfg(record=False)).unwrapped
    print("[SMOKE] replay environment ready", flush=True)
    replay_report = {}
    try:
        for (demo_name, episode), (label, _command) in zip(episodes, COMMANDS.items()):
            print(f"[SMOKE] replaying {demo_name} ({label})", flush=True)
            replay_env.reset_to(
                state=episode.get_initial_state(),
                env_ids=torch.tensor([0], dtype=torch.long, device=args.device),
                is_relative=True,
            )
            actions = episode.data["actions"]
            for action in actions:
                replay_env.step(action.reshape(1, -1))
            final_pose = root_pose(replay_env)
            expected = episode.data["states"]["articulation"]["robot"][
                "root_pose"
            ][-1].detach().cpu().numpy()
            position_error = float(np.linalg.norm(final_pose[:3] - expected[:3]))
            yaw_error = abs(
                wrap_to_pi(
                    yaw_from_wxyz(final_pose[3:7]) - yaw_from_wxyz(expected[3:7])
                )
            )
            if position_error > 0.03 or yaw_error > 0.03:
                raise AssertionError(
                    f"{demo_name}: replay mismatch position={position_error}, yaw={yaw_error}"
                )
            replay_report[demo_name] = {
                "label": label,
                "position_error_m": position_error,
                "yaw_error_rad": yaw_error,
            }
    finally:
        print("[SMOKE] replay steps complete; SimulationApp owns cleanup", flush=True)
        replay_env.close()
    return replay_report


def main() -> None:
    output_dir = args.output_dir.resolve()
    dataset_path = output_dir / f"{DATASET_STEM}.hdf5"
    report_path = output_dir / "validation_report.json"

    if args.mode == "record":
        output_dir.mkdir(parents=True, exist_ok=False)
        print("[SMOKE] creating Task525 recording environment", flush=True)
        env_cfg = make_cfg(record=True, output_dir=output_dir)
        env = gym.make(TASK, cfg=env_cfg).unwrapped
        print("[SMOKE] recording environment ready", flush=True)
        original_results = {}
        for label, command in COMMANDS.items():
            print(f"[SMOKE] recording {label}: {command}", flush=True)
            original_results[label] = step_segment(env, command)
        print("[SMOKE] exporting final episode", flush=True)
        env.reset()
        handler = env.recorder_manager._dataset_file_handler
        handler.close()
        env.recorder_manager._dataset_file_handler = None
        print("[SMOKE] HDF5 recorder closed", flush=True)

        direction_report = {
            label: validate_direction(label, result)
            for label, result in original_results.items()
        }
        hdf5_report = attach_and_validate_hdf5_metadata(dataset_path, env_cfg)
        report = {
            "task": TASK,
            "dataset": str(dataset_path),
            "test_spawn_position": list(SAFE_TEST_SPAWN),
            "test_spawn_quaternion_wxyz": list(SAFE_TEST_ROT_WXYZ),
            "action_names": list(FFW_SG2_MOBILE_ACTION_NAMES),
            "directions": direction_report,
            "hdf5": hdf5_report,
            "replay": "pending separate --mode replay process",
            "passed": False,
        }
        env.close()
    else:
        if not dataset_path.is_file():
            raise FileNotFoundError(dataset_path)
        print("[SMOKE] starting separate replay process", flush=True)
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["replay"] = replay_dataset(dataset_path)
        report["passed"] = True

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
