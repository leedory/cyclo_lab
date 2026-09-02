#!/usr/bin/env python3
"""Replay Task525 joint22 actions with visual-only randomization.

Every episode is reset from the source initial state and replayed from frame
zero.  Only the selected pick or navigation phase is captured.  This is
important for mobile navigation: starting directly at task 2 would lose the
carried-can state produced by tasks 0 and 1.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from typing import Any, Mapping, Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--randomization-profile", required=True)
parser.add_argument("--policy", choices=("pick", "mobile_ccw", "all"), required=True)
parser.add_argument("--repeats", type=int, required=True)
parser.add_argument("--num-envs", type=int, default=8)
parser.add_argument("--seed", type=int, default=20260901)
parser.add_argument("--limit-episodes", type=int)
parser.add_argument("--expected-source-episodes", type=int)
parser.add_argument(
    "--resume",
    action="store_true",
    help=(
        "Resume an interrupted output from its atomically committed manifest. "
        "Uncommitted episode files at or after that boundary are removed."
    ),
)
parser.add_argument(
    "--resume-from-episode",
    type=int,
    help=(
        "With --resume, roll the committed manifest back to this batch boundary "
        "before regenerating the suffix."
    ),
)
parser.add_argument(
    "--source-hybrid-action-replay",
    action="store_true",
    help=(
        "Replay the source SDG hybrid22 command through the matching SDG environment, "
        "while exporting the derived causal joint22 ACT label."
    ),
)
parser.add_argument(
    "--source-state-replay",
    action="store_true",
    help=(
        "Render the exact recorded scene state at every policy frame. Use this when "
        "contact-sensitive action replay cannot preserve the carried object."
    ),
)
parser.add_argument("--camera-refresh-updates", type=int, default=4)
parser.add_argument("--max-root-position-error-m", type=float, default=0.050)
parser.add_argument("--max-target-position-error-m", type=float, default=0.050)
parser.add_argument("--max-root-angle-error-rad", type=float, default=0.175)
parser.add_argument("--max-target-angle-error-rad", type=float, default=0.175)
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
from isaaclab_tasks.utils import parse_env_cfg

import cyclo_lab  # noqa: F401

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization import (
    appearance_events,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.appearance_events import (
    randomize_coffee_can_visual_yaw,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.profiles import (
    Task000525RandomizationCfg,
    validate_task000525_randomization_cfg,
)
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.replay_state import (
    prepare_sg2_position_replay_state,
)

from policy_staging_common import (
    CANONICAL_CAMERA_SHAPES,
    CAMERA_ROTATION_DEG,
    POLICY_ACTION_SEMANTICS,
    POLICY_CONTRACT_ID,
    POLICY_INSTRUCTIONS,
    STAGING_ACTION_SEMANTICS,
    STAGING_SCHEMA,
    Task525PolicyDataError,
    canonicalize_camera_frame,
    derive_policy_arrays,
    jsonable,
    phase_bounds,
    selected_episode_names,
    sha256,
    validate_source,
)


CAMERA_MAP = {
    "cam_head": "cam_left_head",
    "cam_wrist_left": "cam_left_wrist",
    "cam_wrist_right": "cam_right_wrist",
}


class ReplayError(RuntimeError):
    """Raised when replay cannot preserve the Task525 policy contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return jsonable(value)


def collect_dataset_paths(group: h5py.Group, prefix: str = "") -> list[str]:
    paths = []
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
                raise ReplayError(f"{dataset.name} must have singleton environment axis")
            row = value[0]
            if expected_shape is None:
                expected_shape = row.shape
            elif row.shape != expected_shape:
                raise ReplayError(f"initial-state shape mismatch at {dataset.name}")
            rows.append(row)
        nested_assign(state, path, torch.as_tensor(np.stack(rows), device=device))
    return state


def preload_source_states(
    source: h5py.File, names: Sequence[str], maximum_step: int
) -> list[dict[str, tuple[np.ndarray, np.ndarray]]]:
    """Load compact scene states needed for exact pre-step frame rendering."""

    paths = collect_dataset_paths(source[f"data/{names[0]}/initial_state"])
    result = []
    for name in names:
        episode = {}
        for path in paths:
            initial = np.asarray(source[f"data/{name}/initial_state/{path}"])[0]
            states_path = f"data/{name}/states/{path}"
            if states_path not in source:
                raise ReplayError(f"missing recorded scene state: {states_path}")
            # obs[t] is pre-step. states[t-1] is therefore its matching scene
            # state for t>0; initial_state is used for t=0.
            post_step = np.asarray(source[states_path][0 : max(0, maximum_step - 1)])
            episode[path] = (initial, post_step)
        result.append(episode)
    return result


def source_state_at_step(
    trajectories: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]],
    step: int,
    device: str,
) -> dict:
    state: dict = {}
    for path in trajectories[0]:
        rows = []
        for trajectory in trajectories:
            initial, post_step = trajectory[path]
            if step == 0:
                rows.append(initial)
            elif len(post_step) == 0:
                raise ReplayError(f"recorded trajectory has no state for step {step}")
            else:
                # Batches may contain episodes with different lengths. Once a
                # shorter padded env is inactive, hold its final recorded state
                # while longer envs finish; active frames are never clamped.
                rows.append(post_step[min(step - 1, len(post_step) - 1)])
        nested_assign(
            state,
            path,
            torch.as_tensor(np.stack(rows), device=device),
        )
    return state


def preload_pose_references(
    source: h5py.File, names: Sequence[str], maximum_step: int
) -> list[dict[str, np.ndarray]]:
    return [
        {
            "robot": np.asarray(
                source[f"data/{name}/obs/robot_root_pose_world"][:maximum_step],
                dtype=np.float64,
            ),
            "target": np.asarray(
                source[f"data/{name}/obs/target_object_pose_world"][:maximum_step],
                dtype=np.float64,
            ),
        }
        for name in names
    ]


def disable_configured_events(env_cfg: Any) -> None:
    if env_cfg.events is not None:
        for name in tuple(vars(env_cfg.events)):
            setattr(env_cfg.events, name, None)


def load_profile(path: str) -> Task000525RandomizationCfg:
    if ":" not in path:
        raise ReplayError("--randomization-profile must be module:attribute")
    module_name, attribute_name = path.rsplit(":", 1)
    profile = getattr(importlib.import_module(module_name), attribute_name)
    if not isinstance(profile, Task000525RandomizationCfg):
        raise ReplayError(f"unexpected Task525 randomization profile: {type(profile)!r}")
    validate_task000525_randomization_cfg(profile)
    forbidden = {
        "target_pose": profile.target_pose.enabled,
        "robot_root": profile.robot_root.enabled,
        "presence": profile.presence.enabled,
        "shelf": profile.shelf.enabled,
        "coffee_positions": profile.coffee_positions.enabled,
    }
    enabled_forbidden = [name for name, enabled in forbidden.items() if enabled]
    if enabled_forbidden:
        raise ReplayError(
            f"Task525 visual replay forbids physical/scene-content axes: {enabled_forbidden}"
        )
    if not (
        profile.lighting.enabled
        and profile.wall.enabled
        and profile.camera.enabled
        and profile.coffee_visual_yaw.enabled
    ):
        raise ReplayError(
            "Task525 visual replay requires lighting, wall, camera, and "
            "coffee visual-yaw axes"
        )
    return profile


def apply_profile(
    env: Any,
    env_ids: torch.Tensor,
    profile: Task000525RandomizationCfg,
) -> None:
    if hasattr(env, "_task458_dome_sample"):
        delattr(env, "_task458_dome_sample")
    lighting = profile.lighting
    appearance_events.randomize_dome_and_weak_keys(
        env,
        env_ids,
        lighting.dome_intensity_range,
        lighting.dome_rgb_range,
        lighting.weak_key_intensity_range,
    )
    wall = profile.wall
    appearance_events.randomize_wall_solid_rgb(
        env, env_ids, wall.mode, wall.rgb_range, wall.near_white_range
    )
    camera = profile.camera
    appearance_events.randomize_policy_cameras(
        env,
        env_ids,
        camera.camera_names,
        camera.coupled_focal_scale_range,
        camera.local_roll_max_rad,
        camera.local_pitch_max_rad,
        camera.local_yaw_max_rad,
    )
    coffee_yaw = profile.coffee_visual_yaw
    randomize_coffee_can_visual_yaw(
        env,
        env_ids,
        coffee_yaw.object_names,
        coffee_yaw.yaw_range_rad,
    )


def profile_snapshot(
    env: Any,
    env_ids: torch.Tensor,
    profile: Task000525RandomizationCfg,
) -> list[dict]:
    result = []
    for env_id in env_ids.detach().cpu().tolist():
        result.append(
            json_value(
                {
                    "lighting": {
                        "dome": env._task458_dome_sample,
                        "weak_key": env._task458_weak_key_state[env_id],
                    },
                    "wall_rgb": env._task458_wall_rgb[env_id],
                    "cameras": {
                        name: {
                            "coupled_focal_scale": env._task458_camera_focal_scale[name][env_id],
                            "local_rpy_rad": env._task458_camera_local_rpy[name][env_id],
                        }
                        for name in profile.camera.camera_names
                    },
                    "coffee_can_visual_yaw": {
                        name: env._task000525_coffee_visual_yaw[env_id][name]
                        for name in profile.coffee_visual_yaw.object_names
                    },
                }
            )
        )
    return result


def protected_root_pose_snapshot(
    env: Any,
    env_ids: torch.Tensor,
    object_names: Sequence[str],
) -> dict[str, torch.Tensor]:
    """Capture rigid roots around appearance-only mutation."""

    assets = {"robot": env.scene["robot"]}
    assets.update({name: env.scene[name] for name in object_names})
    return {
        name: asset.data.root_pose_w[env_ids].detach().clone()
        for name, asset in assets.items()
    }


def verify_protected_root_poses(
    before: Mapping[str, torch.Tensor],
    after: Mapping[str, torch.Tensor],
) -> list[dict[str, float]]:
    """Require visual randomization to leave every protected root bit-identical."""

    if before.keys() != after.keys():
        raise ReplayError("protected root asset set changed during visual randomization")
    env_count = next(iter(before.values())).shape[0]
    result = [dict() for _ in range(env_count)]
    for name in before:
        difference = torch.abs(after[name] - before[name])
        for env_id in range(env_count):
            error = float(torch.max(difference[env_id]).item())
            result[env_id][name] = error
            if error != 0.0:
                raise ReplayError(
                    f"visual randomization changed protected root {name} "
                    f"in env {env_id}: max_abs_error={error}"
                )
    return result


def refresh_camera_buffers(env: Any, env_ids: torch.Tensor, updates: int) -> None:
    if updates < 1:
        raise ReplayError("camera refresh updates must be positive")
    env.scene.write_data_to_sim()
    for _ in range(updates):
        for camera_name in CAMERA_MAP:
            env.scene.sensors[camera_name].reset(env_ids)
        env.sim.render()
        for camera_name in CAMERA_MAP:
            _ = env.scene.sensors[camera_name].data.output["rgb"]


def open_video_writers(
    output: Path, output_indices: Sequence[int], env: Any, fps: int
) -> dict[tuple[int, str], tuple[cv2.VideoWriter, Path]]:
    writers = {}
    for local_index, output_index in enumerate(output_indices):
        for source_camera, output_camera in CAMERA_MAP.items():
            sample = env.scene.sensors[source_camera].data.output["rgb"][local_index]
            frame = canonicalize_camera_frame(
                source_camera, sample.detach().cpu().numpy()
            )
            height, width = frame.shape[:2]
            path = output / "videos" / f"episode_{output_index:06d}" / f"{output_camera}.mp4"
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
            )
            if not writer.isOpened():
                raise ReplayError(f"could not open video writer: {path}")
            writers[(local_index, source_camera)] = (writer, path)
    return writers


def pose_errors(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(actual[:3] - expected[:3]))
    dot = float(np.clip(abs(np.dot(actual[3:7], expected[3:7])), 0.0, 1.0))
    angle = float(2.0 * np.arccos(dot))
    return position, angle


def capture_frame(
    env: Any,
    pose_references: Sequence[Mapping[str, np.ndarray]],
    source_step: int,
    active: Sequence[bool],
    writers: Mapping[tuple[int, str], tuple[cv2.VideoWriter, Path]],
    state_rows: list[list[np.ndarray]],
    errors: list[dict[str, float]],
) -> None:
    observations = env.observation_manager.compute(update_history=False)["policy"]
    joint = observations["joint_pos"].detach().cpu().numpy().astype(np.float32)
    base = observations["base_velocity_body"].detach().cpu().numpy().astype(np.float32)
    origins = env.scene.env_origins.detach().cpu().numpy()
    robot = env.scene["robot"].data.root_pose_w.detach().cpu().numpy().copy()
    target = env.scene[env.cfg.target_object].data.root_pose_w.detach().cpu().numpy().copy()
    robot[:, :3] -= origins
    target[:, :3] -= origins
    camera_frames = {
        source_camera: env.scene.sensors[source_camera]
        .data.output["rgb"][: len(active)]
        .detach()
        .cpu()
        .numpy()
        for source_camera in CAMERA_MAP
    }

    for local_index, is_active in enumerate(active):
        if not is_active:
            continue
        state_rows[local_index].append(np.concatenate((joint[local_index], base[local_index])))
        expected_robot = pose_references[local_index]["robot"][source_step]
        expected_target = pose_references[local_index]["target"][source_step]
        root_position, root_angle = pose_errors(robot[local_index], expected_robot)
        target_position, target_angle = pose_errors(target[local_index], expected_target)
        errors[local_index]["root_position_m"] = max(
            errors[local_index]["root_position_m"], root_position
        )
        errors[local_index]["root_angle_rad"] = max(
            errors[local_index]["root_angle_rad"], root_angle
        )
        errors[local_index]["target_position_m"] = max(
            errors[local_index]["target_position_m"], target_position
        )
        errors[local_index]["target_angle_rad"] = max(
            errors[local_index]["target_angle_rad"], target_angle
        )
        for source_camera in CAMERA_MAP:
            frame = canonicalize_camera_frame(
                source_camera, camera_frames[source_camera][local_index]
            )
            writers[(local_index, source_camera)][0].write(
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            )


def check_trajectory_error(name: str, values: Mapping[str, float]) -> None:
    limits = {
        "root_position_m": args.max_root_position_error_m,
        "target_position_m": args.max_target_position_error_m,
        "root_angle_rad": args.max_root_angle_error_rad,
        "target_angle_rad": args.max_target_angle_error_rad,
    }
    failed = {
        key: {"actual": float(values[key]), "limit": float(limit)}
        for key, limit in limits.items()
        if float(values[key]) > float(limit)
    }
    if failed:
        raise ReplayError(f"{name}: replay trajectory diverged: {failed}")


def gpu_sample(label: str) -> dict[str, Any]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        ).strip().splitlines()[0]
        used, total, utilization = [int(item.strip()) for item in output.split(",")]
        return {
            "label": label,
            "used_mib": used,
            "total_mib": total,
            "utilization_percent": utilization,
        }
    except Exception as error:
        return {"label": label, "error": f"{type(error).__name__}: {error}"}


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_value(manifest), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_resume_manifest(
    output: Path,
    expected: Mapping[str, Any],
    resume_from_episode: int | None,
) -> tuple[dict[str, Any], int, list[str]]:
    path = output / "manifest.json"
    if not path.is_file():
        raise ReplayError(f"--resume output has no manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    compatibility_keys = (
        "schema",
        "source_hdf",
        "source_hdf_sha256",
        "source_episode_count",
        "source_episodes",
        "policy_robot_contract_id",
        "policy_action_semantics",
        "policy",
        "task",
        "task_instruction",
        "randomization_profile",
        "randomization_profile_values",
        "repeats",
        "seed",
        "fps",
        "camera_map",
        "camera_shapes_h_w",
        "camera_rotation_deg",
        "camera_refresh_updates",
        "action_semantics",
        "replay_action_source",
    )
    mismatches = {
        key: (manifest.get(key), expected.get(key))
        for key in compatibility_keys
        if manifest.get(key) != expected.get(key)
    }
    if mismatches:
        raise ReplayError(f"--resume manifest is incompatible: {mismatches}")
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ReplayError("--resume manifest has no episode records")
    indices = [int(record["episode_index"]) for record in records]
    if indices != list(range(len(records))):
        raise ReplayError("--resume episode indices are not contiguous from zero")
    original_completed = len(records)
    completed = original_completed
    if resume_from_episode is not None:
        if not 0 <= resume_from_episode <= original_completed:
            raise ReplayError(
                f"--resume-from-episode {resume_from_episode} is outside "
                f"[0, {original_completed}]"
            )
        for batch in manifest.get("batches", []):
            output_indices = [int(value) for value in batch.get("output_indices", [])]
            if not output_indices:
                continue
            before = output_indices[-1] < resume_from_episode
            after = output_indices[0] >= resume_from_episode
            if not before and not after:
                raise ReplayError(
                    "--resume-from-episode falls inside an existing batch: "
                    f"{resume_from_episode} in {output_indices}"
                )
        completed = resume_from_episode
        manifest["episodes"] = records[:completed]
        manifest["batches"] = [
            batch
            for batch in manifest.get("batches", [])
            if not batch.get("output_indices")
            or int(batch["output_indices"][-1]) < completed
        ]
    removed: list[str] = []
    arrays = output / "policy_arrays"
    for path in arrays.glob("episode_*.npz"):
        try:
            index = int(path.stem.rsplit("_", 1)[-1])
        except ValueError as error:
            raise ReplayError(f"unexpected policy array name: {path}") from error
        if index >= completed:
            path.unlink()
            removed.append(str(path.relative_to(output)))
    videos = output / "videos"
    if videos.is_dir():
        for path in videos.iterdir():
            if not path.is_dir() or not path.name.startswith("episode_"):
                raise ReplayError(f"unexpected video entry: {path}")
            try:
                index = int(path.name.rsplit("_", 1)[-1])
            except ValueError as error:
                raise ReplayError(f"unexpected video directory name: {path}") from error
            if index >= completed:
                shutil.rmtree(path)
                removed.append(str(path.relative_to(output)))
    manifest.pop("episode_count", None)
    manifest.pop("total_frames", None)
    manifest.pop("elapsed_s", None)
    manifest.setdefault("resume_events", []).append(
        {
            "resumed_utc": utc_now(),
            "committed_episode_count_before_rollback": original_completed,
            "committed_episode_count": completed,
            "removed_uncommitted_outputs": removed,
        }
    )
    manifest.setdefault("gpu_samples", []).append(gpu_sample("resumed"))
    write_manifest(path=output / "manifest.json", manifest=manifest)
    return manifest, completed, removed


def main() -> None:
    if args.repeats < 1 or args.num_envs < 1:
        raise ReplayError("--repeats and --num-envs must be positive")
    if args.resume_from_episode is not None and not args.resume:
        raise ReplayError("--resume-from-episode requires --resume")
    if args.source_state_replay and args.source_hybrid_action_replay:
        raise ReplayError("choose either source state replay or source hybrid action replay")
    input_path = args.input_file.resolve()
    output = args.output_dir.resolve()
    if output.exists() and not args.resume:
        raise ReplayError(f"refusing existing output: {output}")
    if not input_path.is_file():
        raise ReplayError(f"missing source HDF5: {input_path}")
    output.mkdir(parents=True, exist_ok=args.resume)
    (output / "policy_arrays").mkdir(exist_ok=args.resume)
    profile = load_profile(args.randomization_profile)
    started = time.monotonic()

    with h5py.File(input_path, "r") as source:
        if "data" not in source:
            raise ReplayError("source HDF5 has no /data")
        data = source["data"]
        contract = validate_source(data)
        names, selection_evidence = selected_episode_names(data, args.policy)
        if args.expected_source_episodes is not None and len(names) != args.expected_source_episodes:
            raise ReplayError(
                f"selected {len(names)} source episodes, expected {args.expected_source_episodes}"
            )
        if args.limit_episodes is not None:
            names = names[: args.limit_episodes]
        if not names:
            raise ReplayError("no source episodes selected")

        env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
        if not args.source_hybrid_action_replay:
            env_cfg.init_action_cfg("record")
        env_cfg.recorders = None
        env_cfg.terminations = None
        disable_configured_events(env_cfg)
        env_cfg.rerender_on_reset = True
        env = gym.make(args.task, cfg=env_cfg).unwrapped
        env.reset()
        if env.action_space.shape[-1] != 22:
            raise ReplayError(f"Task525 replay environment action width is {env.action_space.shape[-1]}")
        missing_cameras = [name for name in CAMERA_MAP if name not in env.scene.sensors]
        if missing_cameras:
            raise ReplayError(f"missing policy cameras: {missing_cameras}")

        fresh_manifest: dict[str, Any] = {
            "schema": STAGING_SCHEMA,
            "created_utc": utc_now(),
            "source_hdf": str(input_path),
            "source_hdf_sha256": sha256(input_path),
            "source_episode_count": len(names),
            "source_episodes": names,
            "source_attrs": {str(key): jsonable(data.attrs[key]) for key in data.attrs},
            "source_robot_contract_id": contract["source_contract_id"],
            "policy_robot_contract_id": POLICY_CONTRACT_ID,
            "policy_action_semantics": POLICY_ACTION_SEMANTICS,
            "policy": args.policy,
            "selection_rule": (
                "tasks 0 and 1 from frame zero through immediately before task 2"
                if args.policy == "pick"
                else (
                    "all task IDs 0 through 4 over the complete causal episode"
                    if args.policy == "all"
                    else "task 2 and first non-zero angular_z command > 0"
                )
            ),
            "selection_evidence": selection_evidence,
            "task": args.task,
            "task_instruction": POLICY_INSTRUCTIONS[args.policy],
            "target_object_name": "coffee_can_green",
            "randomization_profile": args.randomization_profile,
            "randomization_profile_values": json_value(profile),
            "repeats": args.repeats,
            "seed": args.seed,
            "fps": contract["fps"],
            "camera_map": CAMERA_MAP,
            "camera_shapes_h_w": CANONICAL_CAMERA_SHAPES,
            "camera_rotation_deg": CAMERA_ROTATION_DEG,
            "camera_refresh_updates": args.camera_refresh_updates,
            "action_semantics": STAGING_ACTION_SEMANTICS,
            "observation_semantics": "replay pre_step phase crop",
            "render_contract": (
                "source initial state + "
                + (
                    "exact recorded pre-step scene states + derived causal joint22 ACT labels"
                    if args.source_state_replay
                    else (
                        "source SDG hybrid22 replay actions + derived causal joint22 ACT labels"
                        if args.source_hybrid_action_replay
                        else "derived causal joint22 replay actions and ACT labels"
                    )
                )
                + " + visual-only profile; "
                + (
                    "selected phase rendered directly from exact source states"
                    if args.source_state_replay
                    else "full prefix replayed before selected phase capture"
                )
            ),
            "replay_action_source": (
                "recorded_scene_state"
                if args.source_state_replay
                else (
                    "source_sdg_hybrid22"
                    if args.source_hybrid_action_replay
                    else "derived_policy_joint22"
                )
            ),
            "trajectory_error_limits": {
                "root_position_m": args.max_root_position_error_m,
                "target_position_m": args.max_target_position_error_m,
                "root_angle_rad": args.max_root_angle_error_rad,
                "target_angle_rad": args.max_target_angle_error_rad,
            },
            "episodes": [],
            "batches": [],
            "gpu_samples": [gpu_sample("started")],
        }
        if args.resume:
            manifest, completed_count, removed_outputs = load_resume_manifest(
                output, fresh_manifest, args.resume_from_episode
            )
            print(
                f"TASK525_RESUME policy={args.policy} committed={completed_count} "
                f"removed_uncommitted={len(removed_outputs)}",
                flush=True,
            )
        else:
            manifest = fresh_manifest
            completed_count = 0
            write_manifest(output / "manifest.json", manifest)

        try:
            for repeat_index in range(args.repeats):
                for batch_index, start_index in enumerate(range(0, len(names), args.num_envs)):
                    batch_names = names[start_index : start_index + args.num_envs]
                    local_count = len(batch_names)
                    padded_names = batch_names + [batch_names[-1]] * (args.num_envs - local_count)
                    output_indices = [
                        repeat_index * len(names) + start_index + index
                        for index in range(local_count)
                    ]
                    if output_indices[-1] < completed_count:
                        continue
                    if output_indices[0] < completed_count:
                        raise ReplayError(
                            "resume boundary falls inside a batch: "
                            f"completed={completed_count}, batch={output_indices}"
                        )
                    seed = args.seed + repeat_index * 1_000_003 + batch_index * 1_009
                    torch.manual_seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    all_env_ids = torch.arange(args.num_envs, device=env.device, dtype=torch.long)
                    initial_state = prepare_sg2_position_replay_state(
                        load_initial_state(source, padded_names, env.device)
                    )
                    env.reset_to(initial_state, all_env_ids, seed=seed, is_relative=True)
                    protected_before = protected_root_pose_snapshot(
                        env,
                        all_env_ids,
                        profile.coffee_visual_yaw.object_names,
                    )
                    apply_profile(env, all_env_ids, profile)
                    protected_after = protected_root_pose_snapshot(
                        env,
                        all_env_ids,
                        profile.coffee_visual_yaw.object_names,
                    )
                    protected_pose_errors = verify_protected_root_poses(
                        protected_before, protected_after
                    )
                    env.sim.forward()
                    refresh_camera_buffers(env, all_env_ids, args.camera_refresh_updates)
                    snapshots = profile_snapshot(env, all_env_ids, profile)

                    policy_rows = []
                    replay_rows = []
                    bounds = []
                    for name in padded_names:
                        _state, action, tasks = derive_policy_arrays(data[name])
                        policy_rows.append(action)
                        replay_rows.append(
                            np.asarray(data[name]["actions"], dtype=np.float32)
                            if args.source_hybrid_action_replay
                            else action
                        )
                        bounds.append(phase_bounds(tasks, args.policy))
                    maximum_step = max(end for _start, end in bounds[:local_count])
                    minimum_step = (
                        min(start for start, _end in bounds[:local_count])
                        if args.source_state_replay
                        else 0
                    )
                    pose_references = preload_pose_references(
                        source, padded_names, maximum_step
                    )
                    source_trajectories = (
                        preload_source_states(source, padded_names, maximum_step)
                        if args.source_state_replay
                        else None
                    )
                    writers = open_video_writers(output, output_indices, env, contract["fps"])
                    state_rows: list[list[np.ndarray]] = [[] for _ in range(local_count)]
                    errors = [
                        {
                            "root_position_m": 0.0,
                            "root_angle_rad": 0.0,
                            "target_position_m": 0.0,
                            "target_angle_rad": 0.0,
                        }
                        for _ in range(local_count)
                    ]
                    batch_started = time.monotonic()
                    try:
                        for step_index in range(minimum_step, maximum_step):
                            if source_trajectories is not None:
                                env.scene.reset_to(
                                    source_state_at_step(
                                        source_trajectories, step_index, env.device
                                    ),
                                    all_env_ids,
                                    is_relative=True,
                                )
                                env.sim.forward()
                                refresh_camera_buffers(env, all_env_ids, 1)
                            active = [
                                bounds[index][0] <= step_index < bounds[index][1]
                                for index in range(local_count)
                            ]
                            if any(active):
                                capture_frame(
                                    env,
                                    pose_references,
                                    step_index,
                                    active,
                                    writers,
                                    state_rows,
                                    errors,
                                )
                            if source_trajectories is None:
                                action_batch = np.stack(
                                    [
                                        row[min(step_index, len(row) - 1)]
                                        for row in replay_rows
                                    ]
                                )
                                env.step(torch.as_tensor(action_batch, device=env.device))
                                refresh_camera_buffers(env, all_env_ids, 1)
                    finally:
                        for writer, _path in writers.values():
                            writer.release()

                    for local_index, (name, output_index) in enumerate(
                        zip(batch_names, output_indices)
                    ):
                        segment_start, segment_end = bounds[local_index]
                        action = policy_rows[local_index][segment_start:segment_end]
                        state = np.asarray(state_rows[local_index], dtype=np.float32)
                        if state.shape != action.shape or state.shape[1] != 22:
                            raise ReplayError(
                                f"{name}: state/action mismatch {state.shape} vs {action.shape}"
                            )
                        check_trajectory_error(name, errors[local_index])
                        array_path = output / "policy_arrays" / f"episode_{output_index:06d}.npz"
                        np.savez_compressed(
                            array_path,
                            observation_state=state,
                            action=action,
                            timestamp_s=np.arange(len(action), dtype=np.float64) / contract["fps"],
                        )
                        videos = {
                            output_camera: str(
                                writers[(local_index, source_camera)][1].relative_to(output)
                            )
                            for source_camera, output_camera in CAMERA_MAP.items()
                        }
                        manifest["episodes"].append(
                            {
                                "episode_index": output_index,
                                "source_episode": name,
                                "source_episode_ordinal": int(name.rsplit("_", 1)[-1]),
                                "repeat_index": repeat_index,
                                "random_seed": seed,
                                "length": len(action),
                                "source_segment_start": segment_start,
                                "source_segment_end": segment_end,
                                "selection_evidence": selection_evidence[name],
                                "arrays": str(array_path.relative_to(output)),
                                "array_sha256": sha256(array_path),
                                "videos": videos,
                                "state_names": contract["state_names"],
                                "action_names": contract["action_names"],
                                "timestamp_source": f"phase_crop_synthesized_{contract['fps']}hz",
                                "randomization": snapshots[local_index],
                                "trajectory_replay_max_error": errors[local_index],
                                "protected_pose_max_abs_error": protected_pose_errors[
                                    local_index
                                ],
                            }
                        )
                    manifest["batches"].append(
                        {
                            "repeat_index": repeat_index,
                            "batch_index": batch_index,
                            "seed": seed,
                            "source_episodes": batch_names,
                            "output_indices": output_indices,
                            "elapsed_s": time.monotonic() - batch_started,
                            "gpu": gpu_sample(f"repeat_{repeat_index}_batch_{batch_index}"),
                        }
                    )
                    write_manifest(output / "manifest.json", manifest)
                    print(
                        f"TASK525_REPLAY policy={args.policy} repeat={repeat_index} "
                        f"batch={batch_index} total={len(manifest['episodes'])}",
                        flush=True,
                    )
        finally:
            env.close()

    manifest["episode_count"] = len(manifest["episodes"])
    manifest["total_frames"] = sum(int(item["length"]) for item in manifest["episodes"])
    manifest["elapsed_s"] = time.monotonic() - started
    manifest["gpu_samples"].append(gpu_sample("finished"))
    expected = len(names) * args.repeats
    if manifest["episode_count"] != expected:
        raise ReplayError(f"episode count {manifest['episode_count']} != {expected}")
    write_manifest(output / "manifest.json", manifest)
    print(
        "TASK525_VISUAL_REPLAY="
        + json.dumps(
            {
                "staging": str(output),
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
