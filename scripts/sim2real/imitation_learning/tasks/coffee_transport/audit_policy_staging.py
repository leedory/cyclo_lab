#!/usr/bin/env python3
"""Fail-closed audit for native plus visual Coffee policy staging."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from policy_staging_common import (
    CAMERA_MAP,
    CAMERA_SHAPES,
    STAGING_ACTION_SEMANTICS,
    STAGING_SCHEMA,
    TASK_POLICY_NAMES,
    sha256,
)
from runtime_rate_attestation import (
    RuntimeRateAttestationError,
    validate_physical_runtime_rate_provenance,
)


class AuditError(RuntimeError):
    pass


def load_manifest(root: Path, count: int, task_id: str) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise AuditError(f"missing manifest: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "schema": STAGING_SCHEMA,
        "task_id": task_id,
        "episode_count": count,
        "camera_map": CAMERA_MAP,
        "camera_shapes_h_w": {key: list(shape) for key, shape in CAMERA_SHAPES.items()},
        "action_semantics": STAGING_ACTION_SEMANTICS,
    }
    mismatch = {key: (value.get(key), expected_value) for key, expected_value in expected.items() if value.get(key) != expected_value}
    if mismatch:
        raise AuditError(f"manifest mismatch at {path}: {mismatch}")
    records = value.get("episodes")
    if not isinstance(records, list) or len(records) != count:
        raise AuditError(f"episode records mismatch at {path}")
    if [int(record["episode_index"]) for record in records] != list(range(count)):
        raise AuditError(f"episode indices are not contiguous at {path}")
    try:
        validate_physical_runtime_rate_provenance(
            value.get("physical_runtime_rate_provenance", {})
        )
    except RuntimeRateAttestationError as error:
        raise AuditError(f"runtime-rate provenance mismatch at {path}: {error}") from error
    return value


def load_arrays(root: Path, record: dict, names: tuple[str, ...], fps: int):
    path = root / record["arrays"]
    if not path.is_file() or sha256(path) != record["array_sha256"]:
        raise AuditError(f"missing or hash-mismatched arrays: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["observation_state"], dtype=np.float32)
        action = np.asarray(archive["action"], dtype=np.float32)
        timestamp = np.asarray(archive["timestamp_s"], dtype=np.float64)
    length = int(record["length"])
    expected_shape = (length, len(names))
    if state.shape != expected_shape or action.shape != expected_shape:
        raise AuditError(
            f"bad policy arrays at {path}: {state.shape}/{action.shape} != {expected_shape}"
        )
    if record.get("state_names") != list(names) or record.get("action_names") != list(names):
        raise AuditError(f"named policy contract mismatch at {path}")
    expected_time = np.arange(length, dtype=np.float64) / fps
    if timestamp.shape != (length,) or not np.array_equal(timestamp, expected_time):
        raise AuditError(f"timestamp mismatch at {path}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise AuditError(f"non-finite policy arrays at {path}")
    return state, action


def verify_video(path: Path, frames: int, fps: int, source_camera: str) -> None:
    if not path.is_file():
        raise AuditError(f"missing video: {path}")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise AuditError(f"could not open video: {path}")
        actual = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            float(capture.get(cv2.CAP_PROP_FPS)),
        )
    finally:
        capture.release()
    expected = (*CAMERA_SHAPES[source_camera], frames, float(fps))
    if actual != expected:
        raise AuditError(f"video contract mismatch {path}: {actual} != {expected}")


def audit(args: argparse.Namespace) -> dict[str, Any]:
    native_root, augmented_root = args.native.resolve(), args.augmented.resolve()
    native = load_manifest(native_root, args.expected_sources, args.task_id)
    augmented = load_manifest(
        augmented_root, args.expected_sources * args.augmented_repeats, args.task_id
    )
    if native["source_hdf_sha256"] != augmented["source_hdf_sha256"]:
        raise AuditError("native and augmented source hashes differ")
    if (
        native["physical_runtime_rate_provenance"]
        != augmented["physical_runtime_rate_provenance"]
    ):
        raise AuditError("native and augmented physical runtime-rate provenance differs")
    if int(native["fps"]) != 15 or int(augmented["fps"]) != 15:
        raise AuditError("Coffee policy fps must be 15")
    names = TASK_POLICY_NAMES[args.task_id]
    policy_dimension = len(names)
    native_arrays = {}
    total_frames = 0
    for record in native["episodes"]:
        source_name = str(record["source_episode"])
        native_arrays[source_name] = load_arrays(native_root, record, names, 15)
        if set(record.get("videos", {})) != set(CAMERA_MAP.values()):
            raise AuditError(f"native AllCams record is incomplete for {source_name}")
        for source_camera, output_camera in CAMERA_MAP.items():
            verify_video(
                native_root / record["videos"][output_camera],
                int(record["length"]),
                15,
                source_camera,
            )
        total_frames += int(record["length"])
    repeats: dict[str, set[int]] = defaultdict(set)
    visual_samples = {
        "dome_intensity": [],
        "wall_rgb": [],
        "can_yaw": [],
        **{
            f"{camera}_{field}": []
            for camera in CAMERA_MAP
            for field in ("focal", "rpy")
        },
    }
    max_protected_error = 0.0
    for record in augmented["episodes"]:
        source_name = str(record["source_episode"])
        if source_name not in native_arrays:
            raise AuditError(f"augmented source missing native peer: {source_name}")
        repeat = int(record["repeat_index"])
        if repeat in repeats[source_name]:
            raise AuditError(f"duplicate repeat {repeat} for {source_name}")
        repeats[source_name].add(repeat)
        state, action = load_arrays(augmented_root, record, names, 15)
        native_state, native_action = native_arrays[source_name]
        if not np.array_equal(state, native_state) or not np.array_equal(action, native_action):
            raise AuditError(f"visual replay changed policy arrays for {source_name} repeat {repeat}")
        if set(record.get("videos", {})) != set(CAMERA_MAP.values()):
            raise AuditError(
                f"visual AllCams record is incomplete for {source_name} repeat {repeat}"
            )
        for source_camera, output_camera in CAMERA_MAP.items():
            verify_video(
                augmented_root / record["videos"][output_camera],
                int(record["length"]),
                15,
                source_camera,
            )
        protected = record.get("protected_pose_max_abs_error", {})
        if set(protected) != {"robot", "coffee_can_right", "coffee_can_left"}:
            raise AuditError(f"incomplete protected-root evidence for {source_name}")
        max_protected_error = max(max_protected_error, *(abs(float(value)) for value in protected.values()))
        randomization = record.get("randomization", {})
        visual_samples["dome_intensity"].append(float(randomization["lighting"]["dome"]["intensity"]))
        visual_samples["wall_rgb"].append(tuple(randomization["background_wall_rgb"]))
        camera_randomization = randomization.get("camera", {})
        if set(camera_randomization) != set(CAMERA_MAP):
            raise AuditError("visual replay must randomize all three policy cameras")
        for source_camera in CAMERA_MAP:
            sample = camera_randomization[source_camera]
            visual_samples[f"{source_camera}_focal"].append(
                float(sample["coupled_focal_scale"])
            )
            visual_samples[f"{source_camera}_rpy"].append(
                tuple(sample["local_rpy_rad"])
            )
        yaw = randomization["coffee_can_texture_yaw"]
        if set(yaw) != {"coffee_can_right", "coffee_can_left"}:
            raise AuditError("both can texture yaw samples are required")
        visual_samples["can_yaw"].extend(float(yaw[name]["rad"]) for name in sorted(yaw))
        total_frames += int(record["length"])
    expected_repeats = set(range(args.augmented_repeats))
    invalid = {source: sorted(values) for source, values in repeats.items() if values != expected_repeats}
    if invalid or set(repeats) != set(native_arrays):
        raise AuditError(f"visual repeat coverage mismatch: {invalid}")
    if max_protected_error != 0.0:
        raise AuditError(f"visual-only augmentation changed rigid roots: {max_protected_error}")
    diversity = {key: len({json.dumps(value) for value in values}) for key, values in visual_samples.items()}
    if any(count < 2 for count in diversity.values()):
        raise AuditError(f"visual axis lacks sampled diversity: {diversity}")
    return {
        "task_id": args.task_id,
        "native_episodes": args.expected_sources,
        "visual_episodes": args.expected_sources * args.augmented_repeats,
        "merged_episode_count": args.expected_sources * (args.augmented_repeats + 1),
        "total_staged_frames": total_frames,
        "policy_dimension": policy_dimension,
        "policy_names": list(names),
        "cameras": list(CAMERA_MAP.values()),
        "visual_axis_unique_counts": diversity,
        "max_protected_root_error": max_protected_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", choices=("000001", "000002"), required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--expected-sources", type=int, default=50)
    parser.add_argument("--augmented-repeats", type=int, default=8)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    result = audit(arguments)
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
