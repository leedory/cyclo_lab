#!/usr/bin/env python3
"""Export recorded Isaac Lab HDF5 episodes to neutral replay staging.

This is the non-randomized branch of the replay pipeline: it preserves the RGB
frames already stored in HDF5 and packages them with policy arrays for review,
optional merging, and eventual LeRobot v3 conversion.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping, Sequence

import cv2
import h5py
import numpy as np


SCHEMA = "cyclo.isaac_action_replay_staging.v1"
ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"


class ExportError(RuntimeError):
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
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_names(value: Any, label: str) -> list[str]:
    value = jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ExportError(f"{label} must be a JSON list") from error
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise ExportError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise ExportError(f"{label} contains duplicate names")
    return value


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def parse_camera_map(values: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ExportError(f"invalid --camera-map value: {value!r}")
        source, output = (item.strip() for item in value.split("=", 1))
        if not source or not output or source in result or output in result.values():
            raise ExportError(f"invalid or duplicate camera mapping: {value!r}")
        result[source] = output
    if not result:
        raise ExportError("at least one --camera-map SOURCE=OUTPUT is required")
    return result


def timestamps(group: h5py.Group, length: int, fps: float) -> tuple[np.ndarray, str]:
    for key in ("obs/timestamp_s", "obs/timestamp", "timestamp"):
        if key in group:
            values = np.asarray(group[key], dtype=np.float64).reshape(-1)
            if values.shape != (length,) or not np.isfinite(values).all():
                raise ExportError(f"invalid timestamps at {group.name}/{key}")
            values -= values[0]
            if length > 1 and np.any(np.diff(values) <= 0.0):
                raise ExportError(f"timestamps are not strictly increasing at {group.name}/{key}")
            return values, key
    return np.arange(length, dtype=np.float64) / fps, f"synthesized_{fps:g}hz"


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_video(path: Path, frames: h5py.Dataset, fps: float) -> None:
    if frames.ndim != 4 or frames.shape[-1] not in (3, 4):
        raise ExportError(f"RGB dataset must be [N,H,W,3/4], got {frames.name} {frames.shape}")
    height, width = (int(value) for value in frames.shape[1:3])
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise ExportError(f"OpenCV could not open video writer: {path}")
    try:
        for index in range(len(frames)):
            rgb = np.asarray(frames[index, ..., :3], dtype=np.uint8)
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def export(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    source_path = args.input_file.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise ExportError(f"refusing existing output directory: {output}")
    if not source_path.is_file():
        raise ExportError(f"input HDF5 does not exist: {source_path}")
    camera_map = parse_camera_map(args.camera_map)
    output.mkdir(parents=True)
    (output / "policy_arrays").mkdir()

    with h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise ExportError("input HDF5 is missing /data")
        data = source["data"]
        semantics = str(jsonable(data.attrs.get("action_semantics", "")))
        if semantics != ACTION_SEMANTICS:
            raise ExportError(
                f"source action_semantics={semantics!r}; required {ACTION_SEMANTICS!r}"
            )
        state_names = parse_names(data.attrs.get("observation_state_names"), "observation_state_names")
        action_names = parse_names(data.attrs.get("action_names"), "action_names")
        fps = float(data.attrs.get("control_hz", 0.0))
        if not fps.is_integer() or fps <= 0:
            raise ExportError(f"positive integer control_hz is required, got {fps}")
        names = sorted(data.keys(), key=natural_key)
        if args.select_episodes:
            missing = [name for name in args.select_episodes if name not in data]
            if missing:
                raise ExportError(f"selected episodes do not exist: {missing}")
            names = list(args.select_episodes)
        if args.limit_episodes is not None:
            names = names[: args.limit_episodes]
        if not names:
            raise ExportError("no episodes selected")

        manifest: dict[str, Any] = {
            "schema": SCHEMA,
            "created_utc": utc_now(),
            "source_hdf": str(source_path),
            "source_hdf_sha256": sha256(source_path),
            "source_episode_count": len(names),
            "source_episodes": names,
            "source_attrs": {str(key): jsonable(data.attrs[key]) for key in data.attrs},
            "task": str(jsonable(data.attrs.get("task_env_name", ""))),
            "task_instruction": str(jsonable(data.attrs.get("task_instruction", ""))),
            "target_object_name": str(jsonable(data.attrs.get("target_object_name", ""))),
            "randomization_profile": "native_hdf5_recording",
            "randomization_profile_values": {},
            "repeats": 1,
            "seed": None,
            "fps": fps,
            "camera_map": camera_map,
            "action_semantics": ACTION_SEMANTICS,
            "observation_semantics": str(
                jsonable(data.attrs.get("observation_semantics", "pre_step"))
            ),
            "render_contract": "native RGB/state/action observations stored in source HDF5",
            "episodes": [],
            "batches": [],
            "gpu_samples": [],
        }
        write_json(output / "manifest.json", manifest)

        for episode_index, name in enumerate(names):
            group = data[name]
            action = np.asarray(group["actions"], dtype=np.float32)
            state = np.asarray(group["obs/joint_pos"], dtype=np.float32)
            length = len(action)
            if action.shape != (length, len(action_names)):
                raise ExportError(f"bad action shape at {group.name}: {action.shape}")
            if state.shape != (length, len(state_names)):
                raise ExportError(f"bad state shape at {group.name}: {state.shape}")
            if not np.isfinite(action).all() or not np.isfinite(state).all():
                raise ExportError(f"non-finite policy values at {group.name}")
            episode_timestamps, timestamp_source = timestamps(group, length, fps)
            array_path = output / "policy_arrays" / f"episode_{episode_index:06d}.npz"
            np.savez_compressed(
                array_path,
                observation_state=state,
                action=action,
                timestamp_s=episode_timestamps,
            )
            videos: dict[str, str] = {}
            for source_camera, output_camera in camera_map.items():
                key = f"obs/{source_camera}"
                if key not in group:
                    raise ExportError(f"missing camera dataset: {group.name}/{key}")
                camera_frames = group[key]
                if len(camera_frames) != length:
                    raise ExportError(
                        f"camera length mismatch at {camera_frames.name}: {len(camera_frames)} != {length}"
                    )
                video_path = (
                    output / "videos" / f"episode_{episode_index:06d}" / f"{output_camera}.mp4"
                )
                write_video(video_path, camera_frames, fps)
                videos[output_camera] = str(video_path.relative_to(output))
            manifest["episodes"].append(
                {
                    "episode_index": episode_index,
                    "source_episode": name,
                    "source_episode_ordinal": episode_index,
                    "repeat_index": 0,
                    "random_seed": None,
                    "length": length,
                    "arrays": str(array_path.relative_to(output)),
                    "array_sha256": sha256(array_path),
                    "videos": videos,
                    "state_names": state_names,
                    "action_names": action_names,
                    "timestamp_source": timestamp_source,
                    "randomization": {"native_hdf5_recording": True},
                    "protected_pose_max_abs_error": {"target": 0.0, "robot_root": 0.0},
                }
            )
            manifest["batches"].append(
                {"episode_index": episode_index, "source_episode": name}
            )
            write_json(output / "manifest.json", manifest)
            print(f"NATIVE_EXPORT episode={episode_index} source={name} frames={length}", flush=True)

    manifest["episode_count"] = len(manifest["episodes"])
    manifest["total_frames"] = sum(int(item["length"]) for item in manifest["episodes"])
    manifest["elapsed_s"] = time.monotonic() - started
    write_json(output / "manifest.json", manifest)
    return {
        "staging": str(output),
        "episodes": manifest["episode_count"],
        "frames": manifest["total_frames"],
        "elapsed_s": manifest["elapsed_s"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-map", action="append", default=[], metavar="SOURCE=OUTPUT")
    parser.add_argument("--select-episodes", nargs="*")
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args()


def main() -> int:
    result = export(parse_args())
    print("ISAAC_HDF5_NATIVE_EXPORT=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
