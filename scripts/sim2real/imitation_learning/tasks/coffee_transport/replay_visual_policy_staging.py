#!/usr/bin/env python3
"""Render Coffee trajectories from exact scene state with visual-only variation."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import json
from pathlib import Path
import shutil
import time
from typing import Any, Mapping, Sequence

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-id", choices=("000001", "000002"), required=True)
parser.add_argument("--task", required=True)
parser.add_argument("--input-file", type=Path, required=True)
parser.add_argument("--output-dir", type=Path, required=True)
parser.add_argument("--randomization-profile", required=True)
parser.add_argument("--repeats", type=int, default=8)
parser.add_argument("--num-envs", type=int, default=5)
parser.add_argument("--seed", type=int, default=20260907)
parser.add_argument("--expected-source-episodes", type=int)
parser.add_argument("--limit-episodes", type=int)
parser.add_argument("--camera-refresh-updates", type=int, default=4)
parser.add_argument(
    "--cameras",
    nargs="+",
    choices=("cam_head", "cam_wrist_left", "cam_wrist_right"),
    default=["cam_head", "cam_wrist_left", "cam_wrist_right"],
    help="policy cameras to render and encode into staging",
)
parser.add_argument("--per-state-render-updates", type=int, default=1)
parser.add_argument("--resume", action="store_true")
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
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization import appearance_events
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.replay_state import prepare_sg2_position_replay_state
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.reset_events import randomize_coffee_can_visual_yaw
from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.randomization_cfg import validate_coffee_transport_randomization_cfg

from ffmpeg_frame_writer import FrameWriterGroup
from frame0_metrics import Frame0Collector, check_frame0_metrics
from policy_staging_common import (
    CAMERA_MAP,
    POLICY_CAMERA_NAMES,
    CAMERA_ROTATION_DEG,
    CAMERA_SHAPES,
    STAGING_ACTION_SEMANTICS,
    STAGING_SCHEMA,
    TASK_INSTRUCTIONS,
    CoffeePolicyDataError,
    canonicalize_camera_frame,
    derive_policy_arrays,
    jsonable,
    selected_names,
    sha256,
    validate_source,
)
from visual_replay_completion import (
    VisualReplayCompletionGuard,
    finalize_manifest,
    validate_resume_checkpoint,
)


class ReplayError(RuntimeError):
    pass


RENDERER_WARMUP_CONTRACT = (
    "initial_N_forced_updates_use_last_frame0_then_forced_updates_per_state_v2"
)


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


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(json_value(manifest), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


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
    cursor = root
    parts = path.split("/")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def load_initial_state(source: h5py.File, names: Sequence[str], device: str) -> dict:
    paths = collect_dataset_paths(source[f"data/{names[0]}/initial_state"])
    state: dict = {}
    for path in paths:
        rows = [np.asarray(source[f"data/{name}/initial_state/{path}"])[0] for name in names]
        nested_assign(state, path, torch.as_tensor(np.stack(rows), device=device))
    return state


def preload_states(source: h5py.File, names: Sequence[str], maximum_step: int) -> list[dict]:
    paths = collect_dataset_paths(source[f"data/{names[0]}/initial_state"])
    trajectories = []
    for name in names:
        episode = {}
        for path in paths:
            states_path = f"data/{name}/states/{path}"
            if states_path not in source:
                raise ReplayError(f"missing recorded scene state: {states_path}")
            episode[path] = (
                np.asarray(source[f"data/{name}/initial_state/{path}"])[0],
                np.asarray(source[states_path][0 : max(0, maximum_step - 1)]),
            )
        trajectories.append(episode)
    return trajectories


def state_at(trajectories: Sequence[Mapping[str, tuple[np.ndarray, np.ndarray]]], step: int, device: str) -> dict:
    state: dict = {}
    for path in trajectories[0]:
        rows = []
        for trajectory in trajectories:
            initial, post = trajectory[path]
            rows.append(initial if step == 0 else post[min(step - 1, len(post) - 1)])
        nested_assign(state, path, torch.as_tensor(np.stack(rows), device=device))
    return state


def disable_events(env_cfg: Any) -> None:
    if env_cfg.events is not None:
        for name in tuple(vars(env_cfg.events)):
            setattr(env_cfg.events, name, None)


def disable_recorders(env_cfg: Any) -> None:
    """Visual replay writes staging directly and must not open recorder HDF5."""

    env_cfg.recorders = None


def load_profile(path: str) -> Any:
    if ":" not in path:
        raise ReplayError("--randomization-profile must be module:attribute")
    module_name, attribute_name = path.rsplit(":", 1)
    profile = getattr(importlib.import_module(module_name), attribute_name)
    validate_coffee_transport_randomization_cfg(
        profile,
        coffee_positions=profile.coffee_positions,
        coffee_visual_yaw=profile.coffee_visual_yaw,
        episode_home=profile.episode_home,
    )
    forbidden = {
        "target_pose": profile.target_pose.enabled,
        "robot_root": profile.robot_root.enabled,
        "presence": profile.presence.enabled,
        "shelf": profile.shelf.enabled,
        "coffee_positions": profile.coffee_positions.enabled,
        "episode_home": profile.episode_home.enabled,
    }
    enabled = [name for name, value in forbidden.items() if value]
    if enabled:
        raise ReplayError(f"visual replay forbids physical axes: {enabled}")
    if not (profile.lighting.enabled and profile.wall.enabled and profile.camera.enabled and profile.coffee_visual_yaw.enabled):
        raise ReplayError("visual replay requires lighting, wall, camera, and coffee visual yaw")
    # CameraRandomizationCfg.camera_names is the set that receives focal/rpy
    # jitter, and the profile owns that choice. It must be a non-empty subset of
    # the policy cameras; it is deliberately NOT required to be all of them,
    # because every policy camera is rendered and stored regardless of whether
    # its pose is jittered.
    jittered = tuple(profile.camera.camera_names)
    unknown = [name for name in jittered if name not in POLICY_CAMERA_NAMES]
    if unknown:
        raise ReplayError(f"profile jitters cameras outside the policy set: {unknown}")
    if not jittered:
        raise ReplayError("camera randomization is enabled but jitters no camera")
    return profile


def root_snapshot(env: Any, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        name: env.scene[name].data.root_pose_w[env_ids].detach().clone()
        for name in ("robot", "coffee_can_right", "coffee_can_left")
    }


def apply_profile(env: Any, env_ids: torch.Tensor, profile: Any) -> tuple[list[dict], list[dict]]:
    before = root_snapshot(env, env_ids)
    if hasattr(env, "_task458_dome_sample"):
        delattr(env, "_task458_dome_sample")
    appearance_events.randomize_dome_and_weak_keys(
        env, env_ids, profile.lighting.dome_intensity_range,
        profile.lighting.dome_rgb_range, profile.lighting.weak_key_intensity_range,
    )
    appearance_events.randomize_wall_solid_rgb(
        env, env_ids, profile.wall.mode, profile.wall.rgb_range, profile.wall.near_white_range,
    )
    camera = profile.camera
    # Which cameras get focal/rpy jitter is a randomization-profile decision:
    # CameraRandomizationCfg.camera_names owns it. The set of cameras that are
    # rendered and stored is a separate, script-level choice (--cameras).
    appearance_events.randomize_policy_cameras(
        env, env_ids, tuple(camera.camera_names), camera.coupled_focal_scale_range,
        camera.local_roll_max_rad, camera.local_pitch_max_rad, camera.local_yaw_max_rad,
    )
    randomize_coffee_can_visual_yaw(
        env, env_ids, profile.coffee_visual_yaw.object_names,
        profile.coffee_visual_yaw.yaw_range_rad,
    )
    after = root_snapshot(env, env_ids)
    errors = [dict() for _ in range(len(env_ids))]
    for name in before:
        difference = torch.max(torch.abs(after[name] - before[name]), dim=1).values
        for row, value in enumerate(difference.detach().cpu().tolist()):
            errors[row][name] = float(value)
            if value != 0.0:
                raise ReplayError(f"visual randomization changed {name} rigid root: {value}")
    snapshots = []
    for env_id in env_ids.detach().cpu().tolist():
        snapshots.append(json_value({
            "lighting": {"dome": env._task458_dome_sample, "weak_key": env._task458_weak_key_state[env_id]},
            "background_wall_rgb": env._task458_wall_rgb[env_id],
            "camera": {
                name: {
                    "coupled_focal_scale": env._task458_camera_focal_scale[name][env_id],
                    "local_rpy_rad": env._task458_camera_local_rpy[name][env_id],
                }
                for name in profile.camera.camera_names
            },
            "coffee_can_texture_yaw": env._coffee_transport_coffee_visual_yaw[env_id],
        }))
    return snapshots, errors


def read_camera_frames(
    env: Any, camera_names: Sequence[str], count: int
) -> dict[str, np.ndarray]:
    """Slice RGB on the GPU, then make one host copy per camera.

    ``sim.render()`` rasterizes every render product in the scene, so reading
    three cameras costs no extra GPU work over reading one - only the readback.
    Slicing to 3 channels and making contiguous on the device keeps the copy to
    exactly the bytes the encoder consumes (raw rgb24), and removes the CPU-side
    RGBA slice / ascontiguousarray / cvtColor passes the cv2 writer needed.
    """

    result: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        rgb = env.scene.sensors[camera_name].data.output["rgb"][:count, ..., :3]
        result[camera_name] = rgb.contiguous().to("cpu", non_blocking=False).numpy()
    return result


def warm_up_cameras(
    env: Any, env_ids: torch.Tensor, updates: int, camera_names: Sequence[str]
) -> tuple[dict[str, np.ndarray], list[dict[str, list]]]:
    """Converge RTX history at the exact episode-initial state.

    Force each same-state sensor update and use the final buffer as frame zero.
    This adds no simulation/action step, preserving causal policy alignment.
    Diagnostics are tracked on cam_head, which is the camera the frame-0
    renderer gate has a reviewed baseline for.
    """

    env.scene.write_data_to_sim()
    diagnostics = [
        {"channel_mean_per_render": [], "pixel_mae_from_previous_render": []}
        for _ in range(len(env_ids))
    ]
    previous = None
    for _ in range(updates):
        for camera_name in camera_names:
            env.scene.sensors[camera_name].reset(env_ids)
        env.sim.render()
        for camera_name in camera_names:
            _ = env.scene.sensors[camera_name].data.output["rgb"]
        values = (
            env.scene.sensors["cam_head"].data.output["rgb"].detach().to(dtype=torch.float32)
        )
        means = values.mean(dim=(1, 2)).cpu().tolist()
        deltas = (
            torch.mean(torch.abs(values - previous), dim=(1, 2, 3)).cpu().tolist()
            if previous is not None
            else None
        )
        for local in range(len(env_ids)):
            diagnostics[local]["channel_mean_per_render"].append(means[local])
            if deltas is not None:
                diagnostics[local]["pixel_mae_from_previous_render"].append(
                    float(deltas[local])
                )
        previous = values.clone()
    return read_camera_frames(env, camera_names, len(env_ids)), diagnostics


def render_cameras(
    env: Any, env_ids: torch.Tensor, updates: int, camera_names: Sequence[str]
) -> dict[str, np.ndarray]:
    """Force and capture one exact replay state without advancing physics."""

    env.scene.write_data_to_sim()
    for _ in range(updates):
        for camera_name in camera_names:
            env.scene.sensors[camera_name].reset(env_ids)
        env.sim.render()
        for camera_name in camera_names:
            _ = env.scene.sensors[camera_name].data.output["rgb"]
    return read_camera_frames(env, camera_names, len(env_ids))


def open_writers(
    output: Path, output_indices: Sequence[int], camera_map: Mapping[str, str], fps: int
) -> FrameWriterGroup:
    """Open one ffmpeg encoder per (episode, camera).

    Encoding runs in separate processes, so it overlaps with the next render
    instead of blocking it, and the pinned H.264 settings make the staging
    videos eligible for the LeRobot v3.0 stream-copy concat fast path
    (no re-encode during conversion). See lib/ffmpeg_frame_writer.py.
    """

    shapes = {name: CAMERA_SHAPES[name] for name in camera_map}
    return FrameWriterGroup(output, output_indices, camera_map, shapes, fps)


def resume_manifest(
    output: Path, fresh: dict, batch_size: int, expected_outputs: int
) -> tuple[dict, int]:
    path = output / "manifest.json"
    if not args.resume:
        if output.exists():
            raise ReplayError(f"refusing existing output: {output}")
        output.mkdir(parents=True)
        (output / "policy_arrays").mkdir()
        write_manifest(path, fresh)
        return fresh, 0
    if not path.is_file():
        raise ReplayError(f"resume output has no manifest: {path}")
    current = json.loads(path.read_text(encoding="utf-8"))
    for key in (
        "schema", "source_hdf_sha256", "source_episodes", "task_id",
        "randomization_profile", "randomization_profile_values", "repeats",
        "num_envs", "seed", "camera_map", "camera_refresh_updates",
        "per_state_render_updates",
        "renderer_warmup_contract",
        "action_semantics",
    ):
        if current.get(key) != fresh.get(key):
            raise ReplayError(f"resume mismatch for {key}")
    try:
        completed = validate_resume_checkpoint(
            output,
            current,
            batch_size=batch_size,
            expected_outputs=expected_outputs,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ReplayError(str(error)) from error
    for path in (output / "policy_arrays").glob("episode_*.npz"):
        if int(path.stem.rsplit("_", 1)[-1]) >= completed:
            path.unlink()
    for path in (output / "videos").glob("episode_*") if (output / "videos").is_dir() else ():
        if int(path.name.rsplit("_", 1)[-1]) >= completed:
            shutil.rmtree(path)
    return current, completed


def main() -> None:
    if (
        args.repeats < 1 or args.num_envs < 1
        or args.camera_refresh_updates < 1 or args.per_state_render_updates < 1
    ):
        raise ReplayError("repeats, num-envs, and render update counts must be positive")
    source_path = args.input_file.resolve()
    output = args.output_dir.resolve()
    profile = load_profile(args.randomization_profile)
    cameras = tuple(dict.fromkeys(args.cameras))
    unknown = [name for name in cameras if name not in CAMERA_MAP]
    if unknown:
        raise ReplayError(f"unknown policy cameras: {unknown}")
    if "cam_head" not in cameras:
        raise ReplayError("cam_head is required: the frame-0 renderer gate baselines it")
    camera_map = {name: CAMERA_MAP[name] for name in cameras}
    env_cfg = parse_env_cfg(args.task, device=args.device, num_envs=args.num_envs)
    disable_events(env_cfg)
    disable_recorders(env_cfg)
    env = gym.make(args.task, cfg=env_cfg).unwrapped
    started = time.monotonic()
    completion_guard = VisualReplayCompletionGuard()
    try:
        with h5py.File(source_path, "r") as source:
            data = source["data"]
            contract = validate_source(data, args.task_id)
            names = selected_names(contract["episode_names"], args.limit_episodes)
            if args.expected_source_episodes is not None and len(names) != args.expected_source_episodes:
                raise ReplayError(f"selected {len(names)} sources, expected {args.expected_source_episodes}")
            if len(names) % args.num_envs:
                raise ReplayError("source episode count must be divisible by --num-envs for resumable batches")
            fresh = {
                "schema": STAGING_SCHEMA, "created_utc": utc_now(),
                "source_hdf": str(source_path), "source_hdf_sha256": sha256(source_path),
                "source_episode_count": len(names), "source_episodes": names,
                "source_attrs": {str(key): jsonable(data.attrs[key]) for key in data.attrs},
                "physical_runtime_rate_provenance": contract[
                    "physical_runtime_rate_provenance"
                ],
                "task_id": args.task_id, "task": args.task,
                "task_instruction": TASK_INSTRUCTIONS[args.task_id],
                "randomization_profile": args.randomization_profile,
                "randomization_profile_values": json_value(profile),
                "repeats": args.repeats, "num_envs": args.num_envs,
                "seed": args.seed, "fps": contract["fps"],
                "cameras": list(cameras),
                "camera_map": {name: CAMERA_MAP[name] for name in cameras},
                "jittered_cameras": list(profile.camera.camera_names),
                "per_state_render_updates": args.per_state_render_updates,
                "camera_shapes_h_w": {key: list(value) for key, value in CAMERA_SHAPES.items()},
                "camera_rotation_deg": CAMERA_ROTATION_DEG,
                "camera_refresh_updates": args.camera_refresh_updates,
                "renderer_warmup_contract": RENDERER_WARMUP_CONTRACT,
                "action_semantics": STAGING_ACTION_SEMANTICS,
                "observation_semantics": "exact source pre_step policy arrays",
                "render_contract": (
                    "exact recorded pre-step scene state + visual-only profile + canonical AllCams RGB; "
                    "frame0 is the final unchanged-state renderer warmup buffer"
                ),
                "replay_action_source": "recorded_scene_state",
                "episodes": [], "batches": [],
            }
            expected_outputs = len(names) * args.repeats
            manifest, completed = resume_manifest(
                output, fresh, args.num_envs, expected_outputs
            )
            completion_guard.arm(output / "manifest.json", expected_outputs)
            previous_head_last_hash: str | None = None

            for repeat in range(args.repeats):
                for start in range(0, len(names), args.num_envs):
                    batch_names = names[start : start + args.num_envs]
                    output_indices = [repeat * len(names) + start + local for local in range(args.num_envs)]
                    if output_indices[-1] < completed:
                        continue
                    if output_indices[0] < completed:
                        raise ReplayError("resume boundary falls inside a batch")
                    seed = args.seed + repeat * 1_000_003 + (start // args.num_envs) * 1_009
                    torch.manual_seed(seed)
                    np.random.seed(seed % (2**32 - 1))
                    env_ids = torch.arange(args.num_envs, dtype=torch.long, device=env.device)
                    initial = prepare_sg2_position_replay_state(load_initial_state(source, batch_names, env.device))
                    env.reset_to(initial, env_ids, seed=seed, is_relative=True)
                    snapshots, pose_errors = apply_profile(env, env_ids, profile)
                    env.sim.forward()
                    initial_frames, warmup_diagnostics = warm_up_cameras(
                        env, env_ids, args.camera_refresh_updates, cameras
                    )
                    arrays = [derive_policy_arrays(data[name], args.task_id) for name in batch_names]
                    lengths = [len(action) for _state, action in arrays]
                    trajectories = preload_states(source, batch_names, max(lengths))
                    writers = open_writers(
                        output, output_indices, camera_map, contract["fps"]
                    )
                    frame0 = Frame0Collector()
                    try:
                        for step in range(max(lengths)):
                            if step == 0:
                                frames = initial_frames
                            else:
                                env.scene.reset_to(
                                    state_at(trajectories, step, env.device),
                                    env_ids,
                                    is_relative=True,
                                )
                                env.sim.forward()
                                frames = render_cameras(
                                    env, env_ids, args.per_state_render_updates, cameras
                                )
                            for local, length in enumerate(lengths):
                                if step >= length:
                                    continue
                                for source_camera in cameras:
                                    rgb = canonicalize_camera_frame(
                                        source_camera, frames[source_camera][local]
                                    )
                                    writers.write(local, source_camera, rgb)
                                    frame0.observe(local, source_camera, rgb)
                    except Exception:
                        writers.abort()
                        raise
                    else:
                        writers.close()
                    # Renderer frame-0 sanity gate. This used to live in
                    # audit_and_select_physical_trajectories.py against
                    # obs/cam_head in the generation HDF5; once generation stops
                    # storing pixels, this is the only place that sees them.
                    for local in range(len(batch_names)):
                        gate_reasons = check_frame0_metrics(
                            "cam_head",
                            frame0.metrics(local, "cam_head"),
                            previous_last_hash=previous_head_last_hash,
                        )
                        if gate_reasons:
                            raise ReplayError(
                                f"renderer frame-0 gate failed for output "
                                f"{output_indices[local]}: {'; '.join(gate_reasons)}"
                            )
                        previous_head_last_hash = frame0.metrics(
                            local, "cam_head"
                        )["last_frame_hash"]
                    records = []
                    for local, (name, output_index) in enumerate(zip(batch_names, output_indices)):
                        state, action = arrays[local]
                        array_path = output / "policy_arrays" / f"episode_{output_index:06d}.npz"
                        np.savez_compressed(
                            array_path, observation_state=state, action=action,
                            timestamp_s=np.arange(len(action), dtype=np.float64) / contract["fps"],
                        )
                        records.append({
                            "episode_index": output_index, "source_episode": name,
                            "source_episode_ordinal": int(name.rsplit("_", 1)[-1]),
                            "repeat_index": repeat, "random_seed": seed,
                            "random_stream_index": local,
                            "length": len(action), "source_segment_start": 0,
                            "source_segment_end": len(action),
                            "arrays": str(array_path.relative_to(output)), "array_sha256": sha256(array_path),
                            "videos": {
                                camera_map[source_camera]: str(
                                    writers.paths[(local, source_camera)].relative_to(output)
                                )
                                for source_camera in cameras
                            },
                            "renderer_frame0": {
                                source_camera: frame0.metrics(local, source_camera)
                                for source_camera in cameras
                            },
                            "state_names": contract["state_names"], "action_names": contract["action_names"],
                            "timestamp_source": f"source_pre_step_{contract['fps']}hz",
                            "randomization": snapshots[local],
                            "renderer_warmup": warmup_diagnostics[local],
                            "protected_pose_max_abs_error": pose_errors[local],
                        })
                    manifest["episodes"].extend(records)
                    manifest["batches"].append({
                        "repeat_index": repeat, "source_episodes": batch_names,
                        "output_indices": output_indices, "seed": seed,
                    })
                    write_manifest(output / "manifest.json", manifest)
                    print(f"COFFEE_VISUAL task={args.task_id} repeat={repeat} outputs={output_indices}", flush=True)
            finalize_manifest(
                manifest,
                expected_outputs=expected_outputs,
                elapsed_s=time.monotonic() - started,
            )
            write_manifest(output / "manifest.json", manifest)
            completion_guard.mark_complete()
    finally:
        env.close()
        simulation_app.close()
    print("COFFEE_VISUAL_STAGING=" + json.dumps({
        "staging": str(output), "episodes": manifest["episode_count"], "frames": manifest["total_frames"]
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
