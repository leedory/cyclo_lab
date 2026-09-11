#!/usr/bin/env python3
"""Export Coffee Transport native AllCams RGB and causal policy arrays."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import h5py
import numpy as np

from ffmpeg_frame_writer import FrameWriterGroup
from policy_staging_common import (
    CAMERA_MAP,
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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_camera_videos(
    output: Path,
    episode_index: int,
    group: h5py.Group,
    length: int,
    fps: int,
) -> dict[str, str]:
    """Encode every policy camera without changing pixels or trajectory rows."""

    datasets: dict[str, h5py.Dataset] = {}
    for source_camera in CAMERA_MAP:
        dataset_path = f"obs/{source_camera}"
        if dataset_path not in group:
            raise CoffeePolicyDataError(f"{group.name}: missing {dataset_path}")
        frames = group[dataset_path]
        if frames.shape[0] < length:
            raise CoffeePolicyDataError(
                f"{group.name}/{dataset_path}: {frames.shape[0]} frames < {length}"
            )
        datasets[source_camera] = frames

    writers = FrameWriterGroup(
        output, [episode_index], CAMERA_MAP, CAMERA_SHAPES, fps
    )
    try:
        # Bounded blocks avoid per-frame HDF5 traversal while keeping peak host
        # memory stable across long episodes.
        for start in range(0, length, 64):
            stop = min(start + 64, length)
            blocks = {
                camera: np.asarray(frames[start:stop])
                for camera, frames in datasets.items()
            }
            for offset in range(stop - start):
                for source_camera in CAMERA_MAP:
                    rgb = canonicalize_camera_frame(
                        source_camera, blocks[source_camera][offset]
                    )
                    writers.write(0, source_camera, rgb)
    except Exception:
        writers.abort()
        raise
    else:
        writers.close()
    return {
        output_camera: str(
            writers.paths[(0, source_camera)].relative_to(output)
        )
        for source_camera, output_camera in CAMERA_MAP.items()
    }


def export(args: argparse.Namespace) -> dict:
    source_path = args.input_file.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise CoffeePolicyDataError(f"refusing existing output: {output}")
    output.mkdir(parents=True)
    (output / "policy_arrays").mkdir()
    started = time.monotonic()
    with h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise CoffeePolicyDataError("source has no /data")
        data = source["data"]
        contract = validate_source(data, args.task_id)
        names = selected_names(contract["episode_names"], args.limit_episodes)
        if args.expected_episodes is not None and len(names) != args.expected_episodes:
            raise CoffeePolicyDataError(
                f"selected {len(names)} episodes, expected {args.expected_episodes}"
            )
        manifest = {
            "schema": STAGING_SCHEMA,
            "created_utc": utc_now(),
            "source_hdf": str(source_path),
            "source_hdf_sha256": sha256(source_path),
            "source_episode_count": len(names),
            "source_episodes": names,
            "source_attrs": {str(key): jsonable(data.attrs[key]) for key in data.attrs},
            "physical_runtime_rate_provenance": contract[
                "physical_runtime_rate_provenance"
            ],
            "task_id": args.task_id,
            "task": f"Cyclo-Real-Showroom-Task{args.task_id}-FFW-SG2-v0",
            "task_instruction": TASK_INSTRUCTIONS[args.task_id],
            "randomization_profile": "native_physical_trajectory",
            "randomization_profile_values": {},
            "repeats": 1,
            "seed": None,
            "fps": contract["fps"],
            "camera_map": CAMERA_MAP,
            "camera_shapes_h_w": {key: list(value) for key, value in CAMERA_SHAPES.items()},
            "camera_rotation_deg": CAMERA_ROTATION_DEG,
            "action_semantics": STAGING_ACTION_SEMANTICS,
            "observation_semantics": "source pre_step; causal scene velocity at t",
            "render_contract": "native canonical AllCams RGB; no rotation or resizing",
            "episodes": [],
            "batches": [],
        }
        write_json(output / "manifest.json", manifest)
        for episode_index, name in enumerate(names):
            group = data[name]
            state, action = derive_policy_arrays(group, args.task_id)
            length = len(action)
            array_path = output / "policy_arrays" / f"episode_{episode_index:06d}.npz"
            np.savez_compressed(
                array_path,
                observation_state=state,
                action=action,
                timestamp_s=np.arange(length, dtype=np.float64) / contract["fps"],
            )
            videos = write_camera_videos(
                output, episode_index, group, length, contract["fps"]
            )
            record = {
                "episode_index": episode_index,
                "source_episode": name,
                "source_episode_ordinal": int(name.rsplit("_", 1)[-1]),
                "repeat_index": 0,
                "random_seed": None,
                "length": length,
                "source_segment_start": 0,
                "source_segment_end": length,
                "arrays": str(array_path.relative_to(output)),
                "array_sha256": sha256(array_path),
                "videos": videos,
                "state_names": contract["state_names"],
                "action_names": contract["action_names"],
                "timestamp_source": f"source_pre_step_{contract['fps']}hz",
                "randomization": {"native_physical_trajectory": True},
                "protected_pose_max_abs_error": {},
            }
            manifest["episodes"].append(record)
            manifest["batches"].append({"episode_index": episode_index, "source_episode": name})
            write_json(output / "manifest.json", manifest)
            print(f"COFFEE_NATIVE task={args.task_id} episode={episode_index} source={name} frames={length}", flush=True)
    manifest["episode_count"] = len(manifest["episodes"])
    manifest["total_frames"] = sum(int(item["length"]) for item in manifest["episodes"])
    manifest["elapsed_s"] = time.monotonic() - started
    write_json(output / "manifest.json", manifest)
    return {"staging": str(output), "episodes": manifest["episode_count"], "frames": manifest["total_frames"]}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=("000001", "000002"), required=True)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument("--limit-episodes", type=int)
    return parser.parse_args()


if __name__ == "__main__":
    result = export(parse_args())
    print("COFFEE_NATIVE_STAGING=" + json.dumps(result, sort_keys=True), flush=True)
