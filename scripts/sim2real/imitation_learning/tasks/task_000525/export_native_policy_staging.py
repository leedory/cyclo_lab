#!/usr/bin/env python3
"""Export phase-cropped native Task525 RGB and causal joint22 arrays."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
import json
import multiprocessing as mp
from pathlib import Path
import time

import cv2
import h5py
import numpy as np

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
    crop_policy_episode,
    jsonable,
    selected_episode_names,
    sha256,
    validate_source,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_video(
    path: Path,
    frames: h5py.Dataset,
    source_camera: str,
    start: int,
    end: int,
    fps: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = CANONICAL_CAMERA_SHAPES[source_camera]
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    if not writer.isOpened():
        raise Task525PolicyDataError(f"could not open video writer: {path}")
    try:
        for index in range(start, end):
            rgb = canonicalize_camera_frame(source_camera, frames[index])
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def write_video_from_hdf(
    source_path: str,
    group_name: str,
    source_camera: str,
    output_path: str,
    start: int,
    end: int,
    fps: int,
) -> None:
    """Encode one camera in an isolated process with its own HDF5 handle."""
    with h5py.File(source_path, "r") as source:
        frames = source["data"][group_name][f"obs/{source_camera}"]
        write_video(Path(output_path), frames, source_camera, start, end, fps)


def export(args: argparse.Namespace) -> dict:
    source_path = args.input_file.resolve()
    output = args.output_dir.resolve()
    if output.exists():
        raise Task525PolicyDataError(f"refusing existing output: {output}")
    if not source_path.is_file():
        raise Task525PolicyDataError(f"missing source HDF5: {source_path}")
    started = time.monotonic()
    output.mkdir(parents=True)
    (output / "policy_arrays").mkdir()
    camera_map = {
        "cam_head": "cam_left_head",
        "cam_wrist_left": "cam_left_wrist",
        "cam_wrist_right": "cam_right_wrist",
    }

    executor_context = (
        ProcessPoolExecutor(
            max_workers=args.camera_workers,
            mp_context=mp.get_context("spawn"),
        )
        if args.camera_workers > 1
        else nullcontext(None)
    )
    with executor_context as executor, h5py.File(source_path, "r") as source:
        if "data" not in source:
            raise Task525PolicyDataError("source HDF5 has no /data")
        data = source["data"]
        contract = validate_source(data)
        names, selection_evidence = selected_episode_names(data, args.policy)
        if args.expected_episodes is not None and len(names) != args.expected_episodes:
            raise Task525PolicyDataError(
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
            "task": "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
            "task_instruction": POLICY_INSTRUCTIONS[args.policy],
            "target_object_name": "coffee_can_green",
            "randomization_profile": "native_hdf5_recording",
            "randomization_profile_values": {},
            "repeats": 1,
            "seed": None,
            "fps": contract["fps"],
            "camera_map": camera_map,
            "camera_shapes_h_w": CANONICAL_CAMERA_SHAPES,
            "camera_rotation_deg": CAMERA_ROTATION_DEG,
            "action_semantics": STAGING_ACTION_SEMANTICS,
            "observation_semantics": "source pre_step phase crop",
            "render_contract": "native canonical RGB; no rotation or resizing",
            "episodes": [],
            "batches": [],
            "gpu_samples": [],
        }
        write_json(output / "manifest.json", manifest)

        pending_episodes = []
        for episode_index, name in enumerate(names):
            group = data[name]
            state, action, tasks, (start, end) = crop_policy_episode(group, args.policy)
            length = end - start
            array_path = output / "policy_arrays" / f"episode_{episode_index:06d}.npz"
            np.savez_compressed(
                array_path,
                observation_state=state,
                action=action,
                timestamp_s=np.arange(length, dtype=np.float64) / contract["fps"],
            )
            videos = {}
            video_jobs = []
            for source_camera, output_camera in camera_map.items():
                dataset_path = f"obs/{source_camera}"
                if dataset_path not in group:
                    raise Task525PolicyDataError(f"{group.name}: missing {dataset_path}")
                video_path = (
                    output / "videos" / f"episode_{episode_index:06d}" / f"{output_camera}.mp4"
                )
                if executor is None:
                    write_video(
                        video_path,
                        group[dataset_path],
                        source_camera,
                        start,
                        end,
                        contract["fps"],
                    )
                else:
                    video_jobs.append(
                        executor.submit(
                            write_video_from_hdf,
                            str(source_path),
                            name,
                            source_camera,
                            str(video_path),
                            start,
                            end,
                            contract["fps"],
                        )
                    )
                videos[output_camera] = str(video_path.relative_to(output))
            record = {
                "episode_index": episode_index,
                "source_episode": name,
                "source_episode_ordinal": int(name.rsplit("_", 1)[-1]),
                "repeat_index": 0,
                "random_seed": None,
                "length": length,
                "source_segment_start": start,
                "source_segment_end": end,
                "source_task_ids": sorted(int(value) for value in np.unique(tasks)),
                "selection_evidence": selection_evidence[name],
                "arrays": str(array_path.relative_to(output)),
                "array_sha256": sha256(array_path),
                "videos": videos,
                "state_names": contract["state_names"],
                "action_names": contract["action_names"],
                "timestamp_source": f"phase_crop_synthesized_{contract['fps']}hz",
                "randomization": {"native_hdf5_recording": True},
                "protected_pose_max_abs_error": {},
            }
            pending_episodes.append((episode_index, name, length, record, video_jobs))

        # Queue every episode-camera pair before waiting so that workers can
        # encode several episodes concurrently. Each worker still owns an
        # independent read-only HDF5 handle, preserving the original data
        # isolation contract.
        for episode_index, name, length, record, video_jobs in pending_episodes:
            for job in video_jobs:
                job.result()
            manifest["episodes"].append(record)
            manifest["batches"].append(
                {"episode_index": episode_index, "source_episode": name}
            )
            write_json(output / "manifest.json", manifest)
            print(
                f"NATIVE_TASK525 policy={args.policy} episode={episode_index} "
                f"source={name} frames={length}",
                flush=True,
            )

    manifest["episode_count"] = len(manifest["episodes"])
    manifest["total_frames"] = sum(int(item["length"]) for item in manifest["episodes"])
    manifest["elapsed_s"] = time.monotonic() - started
    write_json(output / "manifest.json", manifest)
    return {
        "staging": str(output),
        "episodes": manifest["episode_count"],
        "frames": manifest["total_frames"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy", choices=tuple(POLICY_INSTRUCTIONS), required=True)
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument(
        "--camera-workers",
        type=int,
        default=3,
        choices=(1, 2, 3, 6, 9, 12),
        help=(
            "Independent HDF5/video processes scheduled across all "
            "episode-camera jobs."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    print("TASK525_NATIVE_STAGING=" + json.dumps(export(parse_args()), sort_keys=True))
