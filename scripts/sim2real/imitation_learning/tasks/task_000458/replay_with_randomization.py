#!/usr/bin/env python3
"""Replay Task000458 HDF5 actions under a reset randomization profile.

The source initial state and raw actions are replayed through physics. Policy
observations and RGB videos are captured immediately before each source action.
The neutral staging output can be reviewed, merged, then converted to LeRobot.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import re
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="Registered Isaac Lab environment ID.")
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument(
    "--randomization-profile",
    required=True,
    help="Import path in module:attribute form.",
)
parser.add_argument("--repeats", type=int, default=1)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=20260826)
parser.add_argument("--select-episodes", nargs="*", default=None)
parser.add_argument("--limit-episodes", type=int)
parser.add_argument(
    "--camera-map",
    action="append",
    default=[],
    metavar="SENSOR=OUTPUT",
    help="Camera sensor to output name mapping. May be repeated.",
)
parser.add_argument(
    "--camera-refresh-updates",
    type=int,
    default=4,
    help="Forced render/sensor updates after reset and randomization before frame zero.",
)
parser.add_argument(
    "--allow-pose-randomization",
    action="store_true",
    help="Allow a profile that moves the target or robot root.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = True
app_launcher = AppLauncher(vars(args))
simulation_app = app_launcher.app


import cv2
import gymnasium as gym
import h5py
import numpy as np
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization import (
    appearance_events,
    presence_events,
    task_pose_events,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
    ShowroomGenerationRandomizationCfg,
    validate_randomization_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.replay_state import (
    prepare_sg2_position_replay_state,
    restore_sg2_replay_root_pose,
)


SCHEMA = "cyclo.isaac_action_replay_staging.v1"
ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"


class ReplayError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_name_list(value: Any, label: str) -> list[str]:
    value = jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ReplayError(f"{label} must be a JSON name list") from error
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ReplayError(f"{label} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ReplayError(f"{label} contains duplicate names")
    return value


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def load_profile(path: str) -> ShowroomGenerationRandomizationCfg:
    if ":" not in path:
        raise ReplayError("--randomization-profile must use module:attribute form")
    module_name, attribute_name = path.rsplit(":", 1)
    module = importlib.import_module(module_name)
    try:
        profile = getattr(module, attribute_name)
    except AttributeError as error:
        raise ReplayError(f"randomization profile is missing: {path}") from error
    if not isinstance(profile, ShowroomGenerationRandomizationCfg):
        raise ReplayError(
            "action replay requires ShowroomGenerationRandomizationCfg, got "
            f"{type(profile)!r}"
        )
    validate_randomization_cfg(profile)
    if not args.allow_pose_randomization and (profile.target_pose.enabled or profile.robot_root.enabled):
        raise ReplayError(
            "profile moves the target or robot root; pass --allow-pose-randomization "
            "only when trajectory-context changes are intentional"
        )
    return profile


def parse_camera_map(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ReplayError(f"invalid --camera-map value: {value!r}")
        sensor, output = (item.strip() for item in value.split("=", 1))
        if not sensor or not output or sensor in result or output in result.values():
            raise ReplayError(f"invalid or duplicate camera mapping: {value!r}")
        result[sensor] = output
    if not result:
        raise ReplayError("at least one --camera-map SENSOR=OUTPUT is required")
    return result


def collect_dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths: list[str] = []
    for name, value in group.items():
        path = f"{prefix}/{name}" if prefix else name
        if isinstance(value, h5py.Dataset):
            paths.append(path)
        elif isinstance(value, h5py.Group):
            paths.extend(collect_dataset_paths(value, path))
    return paths


def nested_assign(root: dict, path: str, value: torch.Tensor) -> None:
    parts = path.split("/")
    cursor = root
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_initial_state(source: h5py.File, names: Sequence[str], device: str) -> dict:
    paths = collect_dataset_paths(source[f"data/{names[0]}/initial_state"])
    state: dict = {}
    for path in paths:
        rows = []
        expected_shape = None
        for name in names:
            dataset = source[f"data/{name}/initial_state/{path}"]
            value = np.asarray(dataset)
            if value.ndim < 2 or value.shape[0] != 1:
                raise ReplayError(f"{dataset.name} must have a singleton environment axis")
            row = value[0]
            if expected_shape is None:
                expected_shape = row.shape
            elif row.shape != expected_shape:
                raise ReplayError(f"initial-state shape mismatch at {dataset.name}")
            rows.append(row)
        nested_assign(state, path, torch.as_tensor(np.stack(rows), device=device))
    return state


def normalized_timestamps(group: h5py.Group, length: int, fps: float) -> tuple[np.ndarray, str]:
    for path in ("obs/timestamp_s", "obs/timestamp", "timestamp"):
        if path in group:
            values = np.asarray(group[path], dtype=np.float64).reshape(-1)
            if values.shape != (length,) or not np.isfinite(values).all():
                raise ReplayError(f"invalid timestamps at {group.name}/{path}")
            values = values - values[0]
            if length > 1 and np.any(np.diff(values) <= 0.0):
                raise ReplayError(f"timestamps are not strictly increasing at {group.name}/{path}")
            return values, path
    return np.arange(length, dtype=np.float64) / fps, f"synthesized_{fps:g}hz"


def source_contract(source: h5py.File, names: Sequence[str]) -> dict[str, Any]:
    data = source["data"]
    action_names = parse_name_list(data.attrs.get("action_names"), "data/action_names")
    state_names = parse_name_list(
        data.attrs.get("observation_state_names"), "data/observation_state_names"
    )
    semantics = str(jsonable(data.attrs.get("action_semantics", "")))
    if semantics != ACTION_SEMANTICS:
        raise ReplayError(
            f"source action_semantics={semantics!r}; required {ACTION_SEMANTICS!r}"
        )
    fps = float(data.attrs.get("control_hz", 0.0))
    if not np.isfinite(fps) or fps <= 0.0:
        raise ReplayError("data/control_hz must be positive")
    lengths = {}
    for name in names:
        group = data[name]
        actions = group["actions"]
        if actions.ndim != 2 or actions.shape[1] != len(action_names):
            raise ReplayError(f"invalid action shape at {actions.name}: {actions.shape}")
        if not bool(group.attrs.get("success", False)):
            raise ReplayError(f"source episode is not marked successful: {name}")
        lengths[name] = int(actions.shape[0])
    return {
        "action_names": action_names,
        "state_names": state_names,
        "fps": fps,
        "lengths": lengths,
        "task_instruction": str(jsonable(data.attrs.get("task_instruction", args.task))),
        "target_object_name": str(jsonable(data.attrs.get("target_object_name", ""))),
        "source_attrs": {str(key): jsonable(data.attrs[key]) for key in data.attrs},
    }


def disable_configured_events(env_cfg: Any) -> None:
    if env_cfg.events is not None:
        for name in tuple(vars(env_cfg.events)):
            setattr(env_cfg.events, name, None)


def apply_profile(env: Any, env_ids: torch.Tensor, profile: ShowroomGenerationRandomizationCfg) -> None:
    if profile.target_pose.enabled:
        task_pose_events.refresh_shelf_support_collider(env, env_ids)
        target = profile.target_pose
        task_pose_events.randomize_target_pose(
            env, env_ids, target.lateral_y_max_m, target.yaw_max_rad,
            SceneEntityCfg(env.cfg.target_object),
        )
    if profile.robot_root.enabled:
        robot = profile.robot_root
        task_pose_events.randomize_robot_root(
            env, env_ids, robot.depth_x_max_m, robot.lateral_y_max_m,
            robot.yaw_max_rad, SceneEntityCfg("robot"),
        )
    if profile.presence.enabled:
        presence = profile.presence
        presence_events.randomize_non_target_presence(
            env, env_ids, presence.object_names, presence.disappearance_probability
        )
    if profile.lighting.enabled:
        if hasattr(env, "_task458_dome_sample"):
            delattr(env, "_task458_dome_sample")
        lighting = profile.lighting
        appearance_events.randomize_dome_and_weak_keys(
            env, env_ids, lighting.dome_intensity_range, lighting.dome_rgb_range,
            lighting.weak_key_intensity_range,
        )
    if profile.shelf.enabled:
        shelf = profile.shelf
        appearance_events.randomize_shelf_texture_scale(
            env, env_ids, shelf.brightness_range, shelf.channel_tint_max
        )
    if profile.wall.enabled:
        wall = profile.wall
        appearance_events.randomize_wall_solid_rgb(
            env, env_ids, wall.mode, wall.rgb_range, wall.near_white_range
        )
    if profile.camera.enabled:
        camera = profile.camera
        appearance_events.randomize_policy_cameras(
            env, env_ids, camera.camera_names, camera.coupled_focal_scale_range,
            camera.local_roll_max_rad, camera.local_pitch_max_rad,
            camera.local_yaw_max_rad,
        )


def profile_snapshot(env: Any, env_ids: torch.Tensor, profile: ShowroomGenerationRandomizationCfg) -> list[dict]:
    snapshots = []
    for env_id in env_ids.detach().cpu().tolist():
        item: dict[str, Any] = {}
        if profile.presence.enabled:
            absent = [
                name for name in profile.presence.object_names
                if not bool(env._task458_non_target_presence[name][env_id])
            ]
            item["presence"] = {
                "absent_objects": absent,
                "present_count": len(profile.presence.object_names) - len(absent),
                "total_non_targets": len(profile.presence.object_names),
            }
        if profile.lighting.enabled:
            item["lighting"] = {
                "dome": env._task458_dome_sample,
                "weak_key": env._task458_weak_key_state[env_id],
            }
        if profile.shelf.enabled:
            item["shelf_rgb_scale"] = env._task458_shelf_rgb_scale[env_id]
        if profile.wall.enabled:
            item["wall_rgb"] = env._task458_wall_rgb[env_id]
        if profile.camera.enabled:
            item["cameras"] = {
                name: {
                    "coupled_focal_scale": env._task458_camera_focal_scale[name][env_id],
                    "local_rpy_rad": env._task458_camera_local_rpy[name][env_id],
                }
                for name in profile.camera.camera_names
            }
        snapshots.append(jsonable(item))
    return snapshots


def refresh_camera_buffers(
    env: Any,
    camera_names: Sequence[str],
    env_ids: torch.Tensor,
    updates: int,
) -> None:
    if updates < 1:
        raise ReplayError("--camera-refresh-updates must be at least 1")
    env.scene.write_data_to_sim()
    for _ in range(updates):
        for camera_name in camera_names:
            env.scene.sensors[camera_name].reset(env_ids)
        env.sim.render()
        for camera_name in camera_names:
            _ = env.scene.sensors[camera_name].data.output["rgb"]


def open_video_writers(
    output_dir: Path,
    output_indices: Sequence[int],
    camera_map: Mapping[str, str],
    env: Any,
    fps: float,
) -> dict[tuple[int, str], tuple[cv2.VideoWriter, Path]]:
    writers = {}
    for local_index, output_index in enumerate(output_indices):
        for sensor_name, output_name in camera_map.items():
            rgb = env.scene.sensors[sensor_name].data.output["rgb"][local_index]
            height, width = (int(value) for value in rgb.shape[:2])
            path = output_dir / "videos" / f"episode_{output_index:06d}" / f"{output_name}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
            )
            if not writer.isOpened():
                raise ReplayError(f"OpenCV could not open video writer: {path}")
            writers[(local_index, sensor_name)] = (writer, path)
    return writers


def capture_frame(
    env: Any,
    active: Sequence[bool],
    camera_map: Mapping[str, str],
    writers: Mapping[tuple[int, str], tuple[cv2.VideoWriter, Path]],
    state_rows: list[list[np.ndarray]],
) -> None:
    observations = env.observation_manager.compute(update_history=False)["policy"]
    joint_pos = observations["joint_pos"].detach().cpu().numpy().astype(np.float32)
    for local_index, is_active in enumerate(active):
        if not is_active:
            continue
        state_rows[local_index].append(joint_pos[local_index].copy())
        for sensor_name in camera_map:
            rgb = env.scene.sensors[sensor_name].data.output["rgb"][local_index]
            frame = rgb[..., :3].detach().cpu().numpy().astype(np.uint8)
            writers[(local_index, sensor_name)][0].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))


def gpu_sample(label: str) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        used, total, utilization = [int(item.strip()) for item in output.split(",")]
        return {
            "label": label, "used_mib": used, "total_mib": total,
            "utilization_percent": utilization,
        }
    except Exception as error:
        return {"label": label, "error": f"{type(error).__name__}: {error}"}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(jsonable(manifest), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    if args.repeats < 1 or args.num_envs < 1:
        raise ReplayError("--repeats and --num-envs must be positive")
    input_path = args.input_file.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise ReplayError(f"refusing existing output directory: {output_dir}")
    if not input_path.is_file():
        raise ReplayError(f"input HDF5 does not exist: {input_path}")
    output_dir.mkdir(parents=True)
    (output_dir / "policy_arrays").mkdir()

    profile = load_profile(args.randomization_profile)
    camera_map = parse_camera_map(args.camera_map)
    started = time.monotonic()
    input_digest = sha256(input_path)

    with h5py.File(input_path, "r") as source:
        if "data" not in source:
            raise ReplayError("input HDF5 has no data group")
        available = sorted(source["data"].keys(), key=natural_key)
        names = list(args.select_episodes or available)
        missing = [name for name in names if name not in available]
        if missing:
            raise ReplayError(f"selected episodes are missing: {missing}")
        if args.limit_episodes is not None:
            names = names[: args.limit_episodes]
        if not names:
            raise ReplayError("no source episodes were selected")
        contract = source_contract(source, names)

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        env_cfg.init_action_cfg("record")
        env_cfg.recorders = None
        env_cfg.terminations = None
        disable_configured_events(env_cfg)
        env_cfg.rerender_on_reset = True
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        if env.action_space.shape[-1] != len(contract["action_names"]):
            raise ReplayError(
                f"environment action dimension {env.action_space.shape[-1]} does not match "
                f"source {len(contract['action_names'])}"
            )
        missing_sensors = [name for name in camera_map if name not in env.scene.sensors]
        if missing_sensors:
            raise ReplayError(f"environment cameras are missing: {missing_sensors}")

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": utc_now(),
            "status": "replayed_unreviewed",
            "training_ready": False,
            "source_hdf": str(input_path),
            "source_hdf_sha256": input_digest,
            "source_episode_count": len(names),
            "source_episodes": names,
            "source_attrs": contract["source_attrs"],
            "task": args.task,
            "task_instruction": contract["task_instruction"],
            "target_object_name": contract["target_object_name"],
            "randomization_profile": args.randomization_profile,
            "randomization_profile_values": jsonable(profile),
            "repeats": args.repeats,
            "seed": args.seed,
            "fps": contract["fps"],
            "camera_map": camera_map,
            "camera_refresh_updates": args.camera_refresh_updates,
            "action_semantics": ACTION_SEMANTICS,
            "observation_semantics": "replay_pre_step",
            "render_contract": "source initial_state + source actions -> Isaac physics -> policy RGB",
            "episodes": [],
            "batches": [],
            "gpu_samples": [gpu_sample("started")],
        }
        write_manifest(output_dir / "manifest.json", manifest)

        try:
            for repeat_index in range(args.repeats):
                for batch_index, start in enumerate(range(0, len(names), args.num_envs)):
                    batch_names = names[start : start + args.num_envs]
                    local_count = len(batch_names)
                    padded_names = batch_names + [batch_names[-1]] * (args.num_envs - local_count)
                    output_indices = [
                        repeat_index * len(names) + start + index for index in range(local_count)
                    ]
                    batch_seed = args.seed + repeat_index * 1_000_003 + batch_index * 1_009
                    torch.manual_seed(batch_seed)
                    np.random.seed(batch_seed % (2**32 - 1))

                    initial_state = prepare_sg2_position_replay_state(
                        load_initial_state(source, padded_names, env.device)
                    )
                    all_env_ids = torch.arange(args.num_envs, dtype=torch.long, device=env.device)
                    env.reset_to(initial_state, all_env_ids, seed=batch_seed, is_relative=True)
                    expected_target = initial_state["rigid_object"][env.cfg.target_object]["root_pose"].clone()
                    expected_robot = initial_state["articulation"]["robot"]["root_pose"].clone()
                    apply_profile(env, all_env_ids, profile)
                    env.sim.forward()
                    refresh_camera_buffers(
                        env,
                        tuple(camera_map),
                        all_env_ids,
                        args.camera_refresh_updates,
                    )

                    actual_target = env.scene[env.cfg.target_object].data.root_pose_w[all_env_ids].clone()
                    actual_target[:, :3] -= env.scene.env_origins[all_env_ids]
                    actual_robot = env.scene["robot"].data.root_pose_w[all_env_ids].clone()
                    actual_robot[:, :3] -= env.scene.env_origins[all_env_ids]
                    target_pose_error = float(torch.max(torch.abs(actual_target - expected_target)).item())
                    robot_pose_error = float(torch.max(torch.abs(actual_robot - expected_robot)).item())
                    if not args.allow_pose_randomization and max(target_pose_error, robot_pose_error) > 1.0e-5:
                        raise ReplayError(
                            "appearance profile changed a protected initial pose: "
                            f"target={target_pose_error:.3e}, robot={robot_pose_error:.3e}"
                        )

                    snapshots = profile_snapshot(env, all_env_ids, profile)
                    writers = open_video_writers(
                        output_dir, output_indices, camera_map, env, contract["fps"]
                    )
                    state_rows: list[list[np.ndarray]] = [[] for _ in range(local_count)]
                    action_rows = [
                        np.asarray(source[f"data/{name}/actions"], dtype=np.float32)
                        for name in padded_names
                    ]
                    timestamps = [
                        normalized_timestamps(
                            source[f"data/{name}"], len(action_rows[index]), contract["fps"]
                        )
                        for index, name in enumerate(padded_names)
                    ]
                    maximum_length = max(len(value) for value in action_rows)
                    batch_started = time.monotonic()
                    try:
                        for step_index in range(maximum_length):
                            active = [
                                step_index < len(action_rows[index]) for index in range(local_count)
                            ]
                            capture_frame(env, active, camera_map, writers, state_rows)
                            action_batch = np.stack(
                                [value[min(step_index, len(value) - 1)] for value in action_rows]
                            )
                            env.step(torch.as_tensor(action_batch, device=env.device))
                            restore_sg2_replay_root_pose(
                                env, expected_robot, all_env_ids
                            )
                            refresh_camera_buffers(
                                env, tuple(camera_map), all_env_ids, 1
                            )
                    finally:
                        for writer, _path in writers.values():
                            writer.release()

                    for local_index, (source_name, output_index) in enumerate(
                        zip(batch_names, output_indices)
                    ):
                        length = len(action_rows[local_index])
                        replay_state = np.stack(state_rows[local_index]).astype(np.float32)
                        if replay_state.shape != (length, len(contract["state_names"])):
                            raise ReplayError(
                                f"replay state shape mismatch for {source_name}: {replay_state.shape}"
                            )
                        array_path = output_dir / "policy_arrays" / f"episode_{output_index:06d}.npz"
                        np.savez_compressed(
                            array_path,
                            observation_state=replay_state,
                            action=action_rows[local_index],
                            timestamp_s=timestamps[local_index][0],
                        )
                        video_paths = {
                            output_name: str(
                                writers[(local_index, sensor_name)][1].relative_to(output_dir)
                            )
                            for sensor_name, output_name in camera_map.items()
                        }
                        manifest["episodes"].append(
                            {
                                "episode_index": output_index,
                                "source_episode": source_name,
                                "source_episode_ordinal": names.index(source_name),
                                "repeat_index": repeat_index,
                                "random_seed": batch_seed,
                                "length": length,
                                "arrays": str(array_path.relative_to(output_dir)),
                                "array_sha256": sha256(array_path),
                                "videos": video_paths,
                                "state_names": contract["state_names"],
                                "action_names": contract["action_names"],
                                "timestamp_source": timestamps[local_index][1],
                                "randomization": snapshots[local_index],
                                "protected_pose_max_abs_error": {
                                    "target": target_pose_error,
                                    "robot_root": robot_pose_error,
                                },
                            }
                        )
                    manifest["batches"].append(
                        {
                            "repeat_index": repeat_index,
                            "batch_index": batch_index,
                            "seed": batch_seed,
                            "source_episodes": batch_names,
                            "output_indices": output_indices,
                            "elapsed_s": time.monotonic() - batch_started,
                            "gpu": gpu_sample(f"repeat_{repeat_index}_batch_{batch_index}"),
                        }
                    )
                    write_manifest(output_dir / "manifest.json", manifest)
                    print(
                        f"REPLAY_BATCH repeat={repeat_index} batch={batch_index} "
                        f"episodes={local_count} total={len(manifest['episodes'])}",
                        flush=True,
                    )
        finally:
            env.close()

    manifest["episode_count"] = len(manifest["episodes"])
    manifest["total_frames"] = sum(item["length"] for item in manifest["episodes"])
    manifest["elapsed_s"] = time.monotonic() - started
    manifest["gpu_samples"].append(gpu_sample("finished"))
    expected_count = len(names) * args.repeats
    if manifest["episode_count"] != expected_count:
        raise ReplayError(f"episode count mismatch: {manifest['episode_count']} != {expected_count}")
    write_manifest(output_dir / "manifest.json", manifest)
    print(
        "ISAAC_ACTION_REPLAY="
        + json.dumps(
            {
                "staging": str(output_dir),
                "episodes": manifest["episode_count"],
                "frames": manifest["total_frames"],
                "elapsed_s": manifest["elapsed_s"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        simulation_app.close()
        raise
    else:
        simulation_app.close()
