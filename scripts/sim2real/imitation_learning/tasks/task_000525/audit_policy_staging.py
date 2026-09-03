#!/usr/bin/env python3
"""Audit Task525 native and visual-augmentation policy staging datasets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from policy_staging_common import (
    CAMERA_ROTATION_DEG,
    CANONICAL_CAMERA_MAP,
    CANONICAL_CAMERA_SHAPES,
    POLICY_TARGET_OBJECT_NAME,
)


class AuditError(RuntimeError):
    pass


TASK000525_CAN_NAMES = (
    "coffee_can_black",
    "coffee_can_brown",
    "coffee_can_green",
    "coffee_can_orange",
)
PROTECTED_ROOT_NAMES = ("robot", *TASK000525_CAN_NAMES)
DISTRACTOR_OBJECT_APPEARANCES = {
    "coffee_can_black": "black",
    "coffee_can_brown": "brown",
    "coffee_can_green": "green",
}


def validate_distractor_appearance(randomization: dict[str, Any]) -> None:
    """Require a realized black/brown/green permutation while protecting orange."""

    evidence = randomization.get("coffee_can_distractor_appearance")
    if not isinstance(evidence, dict):
        raise AuditError("missing coffee-can distractor appearance evidence")
    expected_target = {
        "object_name": POLICY_TARGET_OBJECT_NAME,
        "appearance": "orange",
    }
    if evidence.get("protected_target") != expected_target:
        raise AuditError(
            "distractor appearance evidence does not protect canonical orange target"
        )
    mapping = evidence.get("distractor_mapping")
    if not isinstance(mapping, dict) or set(mapping) != set(
        DISTRACTOR_OBJECT_APPEARANCES
    ):
        raise AuditError(
            "distractor appearance mapping must cover exactly black, brown, and green cans"
        )

    sampled_appearances = []
    for object_name, authored_appearance in DISTRACTOR_OBJECT_APPEARANCES.items():
        sample = mapping[object_name]
        if not isinstance(sample, dict):
            raise AuditError(f"{object_name}: appearance evidence is not a mapping")
        if sample.get("authored_appearance") != authored_appearance:
            raise AuditError(
                f"{object_name}: authored appearance evidence is inconsistent"
            )
        sampled_appearance = sample.get("sampled_appearance")
        if sampled_appearance not in DISTRACTOR_OBJECT_APPEARANCES.values():
            raise AuditError(
                f"{object_name}: sampled appearance is not black/brown/green"
            )
        material_path = sample.get("bound_material_path")
        expected_suffix = f"/{object_name}/Looks/{sampled_appearance}"
        if (
            not isinstance(material_path, str)
            or not material_path.startswith("/")
            or not material_path.endswith(expected_suffix)
        ):
            raise AuditError(
                f"{object_name}: bound material path disagrees with sampled appearance"
            )
        sampled_appearances.append(sampled_appearance)
    if set(sampled_appearances) != set(DISTRACTOR_OBJECT_APPEARANCES.values()):
        raise AuditError(
            "sampled distractor appearances are not an exact black/brown/green permutation"
        )


def validate_visual_randomization(record: dict[str, Any]) -> None:
    """Validate sampled label yaw and appearance-only root invariance."""

    randomization = record.get("randomization", {})
    if not isinstance(randomization, dict):
        raise AuditError("randomization evidence is not a mapping")
    validate_distractor_appearance(randomization)
    samples = randomization.get("coffee_can_visual_yaw", {})
    if set(samples) != set(TASK000525_CAN_NAMES):
        raise AuditError(
            "coffee visual-yaw samples do not cover exactly the four Task525 cans"
        )
    for name, sample in samples.items():
        rad = float(sample["rad"])
        deg = float(sample["deg"])
        if not math.isfinite(rad) or not math.isfinite(deg):
            raise AuditError(f"{name}: non-finite coffee visual yaw")
        if not -math.pi <= rad <= math.pi:
            raise AuditError(f"{name}: coffee visual yaw is outside [-pi, pi]: {rad}")
        if not math.isclose(deg, math.degrees(rad), rel_tol=0.0, abs_tol=1.0e-4):
            raise AuditError(f"{name}: inconsistent visual yaw rad/deg pair")

    protected = record.get("protected_pose_max_abs_error", {})
    if set(protected) != set(PROTECTED_ROOT_NAMES):
        raise AuditError("protected root evidence is incomplete")
    changed = {
        name: float(error)
        for name, error in protected.items()
        if float(error) != 0.0
    }
    if changed:
        raise AuditError(f"visual randomization changed protected rigid roots: {changed}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(root: Path, expected_count: int) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise AuditError(f"missing manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise AuditError(f"missing episode records: {path}")
    if manifest.get("episode_count") != expected_count or len(records) != expected_count:
        raise AuditError(
            f"episode count mismatch at {path}: "
            f"manifest={manifest.get('episode_count')} records={len(records)} "
            f"expected={expected_count}"
        )
    indices = [int(record["episode_index"]) for record in records]
    if indices != list(range(expected_count)):
        raise AuditError(f"episode indices are not contiguous at {path}")
    return manifest


def validate_camera_contract(
    root: Path, manifest: dict[str, Any], label: str
) -> dict[str, str]:
    """Validate one non-empty canonical camera subset and every episode file."""

    camera_map = manifest.get("camera_map")
    if not isinstance(camera_map, dict) or not camera_map:
        raise AuditError(f"{label}: camera_map must be a non-empty mapping")
    unsupported = [name for name in camera_map if name not in CANONICAL_CAMERA_MAP]
    expected_camera_map = (
        {}
        if unsupported
        else {name: CANONICAL_CAMERA_MAP[name] for name in camera_map}
    )
    if unsupported or camera_map != expected_camera_map:
        raise AuditError(
            f"{label}: camera_map is not an exact canonical subset: {camera_map}"
        )
    if len(camera_map) != len(set(camera_map.values())):
        raise AuditError(f"{label}: camera_map output names are not unique")

    expected_shapes = {
        name: list(CANONICAL_CAMERA_SHAPES[name]) for name in camera_map
    }
    expected_rotations = {name: CAMERA_ROTATION_DEG[name] for name in camera_map}
    if manifest.get("camera_shapes_h_w") != expected_shapes:
        raise AuditError(f"{label}: camera_shapes_h_w is not canonical")
    if manifest.get("camera_rotation_deg") != expected_rotations:
        raise AuditError(f"{label}: camera_rotation_deg is not canonical")

    output_cameras = tuple(camera_map.values())
    for record in manifest["episodes"]:
        episode_index = int(record["episode_index"])
        expected_videos = {
            camera: (
                f"videos/episode_{episode_index:06d}/" f"{camera}.mp4"
            )
            for camera in output_cameras
        }
        if record.get("videos") != expected_videos:
            raise AuditError(
                f"{label} episode {episode_index}: video map does not exactly "
                "match camera_map"
            )
        episode_directory = root / "videos" / f"episode_{episode_index:06d}"
        if not episode_directory.is_dir():
            raise AuditError(
                f"{label} episode {episode_index}: missing video directory"
            )
        actual_files = {
            path.name for path in episode_directory.iterdir() if path.is_file()
        }
        expected_files = {f"{camera}.mp4" for camera in output_cameras}
        if actual_files != expected_files:
            raise AuditError(
                f"{label} episode {episode_index}: video files "
                f"{sorted(actual_files)} != {sorted(expected_files)}"
            )
    return dict(camera_map)


def validate_matching_camera_contracts(
    native_root: Path,
    native: dict[str, Any],
    augmented_root: Path,
    augmented: dict[str, Any],
) -> dict[str, str]:
    native_cameras = validate_camera_contract(native_root, native, "native")
    augmented_cameras = validate_camera_contract(
        augmented_root, augmented, "augmented"
    )
    if native_cameras != augmented_cameras:
        raise AuditError("native/augmented camera subsets do not match exactly")
    return native_cameras


def load_arrays(
    root: Path, record: dict[str, Any], fps: float
) -> tuple[np.ndarray, np.ndarray]:
    path = root / str(record["arrays"])
    if not path.is_file() or sha256(path) != record["array_sha256"]:
        raise AuditError(f"missing or hash-mismatched array archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["observation_state"])
        action = np.asarray(archive["action"])
        timestamp = np.asarray(archive["timestamp_s"])
    length = int(record["length"])
    if state.shape != (length, 22) or action.shape != (length, 22):
        raise AuditError(
            f"bad policy array shape at {path}: state={state.shape}, action={action.shape}"
        )
    if timestamp.shape != (length,):
        raise AuditError(f"bad timestamp shape at {path}: {timestamp.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise AuditError(f"non-finite state/action values at {path}")
    expected_time = np.arange(length, dtype=np.float64) / fps
    if not np.allclose(timestamp, expected_time, rtol=0.0, atol=1.0e-12):
        raise AuditError(f"timestamps differ from the {fps:g} Hz contract at {path}")
    if len(record["state_names"]) != 22 or len(record["action_names"]) != 22:
        raise AuditError(f"bad policy name count at {path}")
    if list(record["state_names"]) != list(record["action_names"]):
        raise AuditError(f"state/action name order differs at {path}")
    return state, action


def audit(args: argparse.Namespace) -> dict[str, Any]:
    native_root = args.native.resolve()
    augmented_root = args.augmented.resolve()
    native = load_manifest(native_root, args.expected_sources)
    augmented = load_manifest(
        augmented_root, args.expected_sources * args.augmented_repeats
    )
    native_cameras = validate_matching_camera_contracts(
        native_root, native, augmented_root, augmented
    )
    if native.get("policy") != args.policy or augmented.get("policy") != args.policy:
        raise AuditError("manifest policy does not match --policy")
    for label, manifest in (("native", native), ("augmented", augmented)):
        target_object = manifest.get("target_object_name")
        if target_object != POLICY_TARGET_OBJECT_NAME:
            raise AuditError(
                f"{label} manifest target {target_object!r} != {POLICY_TARGET_OBJECT_NAME!r}"
            )
    if float(native["fps"]) != float(augmented["fps"]):
        raise AuditError("native/augmented fps mismatch")
    fps = float(native["fps"])

    native_by_source: dict[str, tuple[dict[str, Any], np.ndarray, np.ndarray]] = {}
    native_frames = 0
    for record in native["episodes"]:
        source = str(record["source_episode"])
        if source in native_by_source:
            raise AuditError(f"duplicate native source episode: {source}")
        state, action = load_arrays(native_root, record, fps)
        expected_tasks = (
            [0, 1]
            if args.policy == "pick"
            else ([0, 1, 2, 3, 4] if args.policy == "all" else [2])
        )
        if list(record.get("source_task_ids", [])) != expected_tasks:
            raise AuditError(
                f"{source}: native task IDs {record.get('source_task_ids')} "
                f"!= {expected_tasks}"
            )
        if args.policy == "mobile_ccw":
            evidence = record.get("selection_evidence", {})
            if not evidence.get("is_counterclockwise"):
                raise AuditError(f"{source}: not marked counterclockwise")
            if float(evidence.get("first_nonzero_angular_z_radps", 0.0)) <= 0.0:
                raise AuditError(f"{source}: first angular-z command is not positive")
        native_by_source[source] = (record, state, action)
        native_frames += int(record["length"])

    repeat_counts: Counter[str] = Counter()
    repeat_indices: dict[str, set[int]] = {
        source: set() for source in native_by_source
    }
    augmented_frames = 0
    max_state_error = 0.0
    max_pose_error = {
        "root_position_m": 0.0,
        "root_angle_rad": 0.0,
        "target_position_m": 0.0,
        "target_angle_rad": 0.0,
    }
    for record in augmented["episodes"]:
        source = str(record["source_episode"])
        if source not in native_by_source:
            raise AuditError(f"augmented record has unexpected source: {source}")
        repeat = int(record["repeat_index"])
        repeat_counts[source] += 1
        repeat_indices[source].add(repeat)
        native_record, native_state, native_action = native_by_source[source]
        state, action = load_arrays(augmented_root, record, fps)
        if (
            int(record["length"]) != int(native_record["length"])
            or int(record["source_segment_start"])
            != int(native_record["source_segment_start"])
            or int(record["source_segment_end"])
            != int(native_record["source_segment_end"])
        ):
            raise AuditError(f"{source} repeat {repeat}: temporal crop differs from native")
        if not np.array_equal(action, native_action):
            raise AuditError(f"{source} repeat {repeat}: action differs from native")
        state_error = float(np.max(np.abs(state - native_state), initial=0.0))
        max_state_error = max(max_state_error, state_error)
        if state_error > args.state_atol:
            raise AuditError(
                f"{source} repeat {repeat}: state max error {state_error} "
                f"> {args.state_atol}"
            )
        validate_visual_randomization(record)
        errors = record.get("trajectory_replay_max_error", {})
        for key in max_pose_error:
            max_pose_error[key] = max(max_pose_error[key], float(errors.get(key, 0.0)))
        augmented_frames += int(record["length"])

    expected_repeats = set(range(args.augmented_repeats))
    for source in native_by_source:
        if repeat_counts[source] != args.augmented_repeats:
            raise AuditError(
                f"{source}: {repeat_counts[source]} repeats != {args.augmented_repeats}"
            )
        if repeat_indices[source] != expected_repeats:
            raise AuditError(f"{source}: repeat indices are incomplete")
    if max_pose_error["root_position_m"] > 1.0e-5:
        raise AuditError(f"root position replay error too high: {max_pose_error}")
    if max_pose_error["target_position_m"] > 1.0e-5:
        raise AuditError(f"target position replay error too high: {max_pose_error}")
    if max_pose_error["root_angle_rad"] > 2.0e-3:
        raise AuditError(f"root angular replay error too high: {max_pose_error}")
    if max_pose_error["target_angle_rad"] > 2.0e-3:
        raise AuditError(f"target angular replay error too high: {max_pose_error}")

    return {
        "policy": args.policy,
        "sources": len(native_by_source),
        "native_episodes": len(native["episodes"]),
        "native_frames": native_frames,
        "augmented_repeats_per_source": args.augmented_repeats,
        "augmented_episodes": len(augmented["episodes"]),
        "augmented_frames": augmented_frames,
        "max_state_abs_error": max_state_error,
        "max_trajectory_replay_error": max_pose_error,
        "action_match": "exact",
        "camera_map": native_cameras,
        "coffee_visual_yaw_samples": "valid",
        "coffee_distractor_appearance": "valid_non_target_permutation",
        "protected_rigid_roots": "bit-identical",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--augmented", type=Path, required=True)
    parser.add_argument("--policy", choices=("pick", "mobile_ccw", "all"), required=True)
    parser.add_argument("--expected-sources", type=int, required=True)
    parser.add_argument("--augmented-repeats", type=int, required=True)
    parser.add_argument("--state-atol", type=float, default=1.0e-6)
    return parser.parse_args()


if __name__ == "__main__":
    print(
        "TASK525_STAGING_AUDIT="
        + json.dumps(audit(parse_args()), sort_keys=True),
        flush=True,
    )
