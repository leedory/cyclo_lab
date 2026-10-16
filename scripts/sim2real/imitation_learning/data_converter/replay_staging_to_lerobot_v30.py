#!/usr/bin/env python3
"""Validate named replay staging and write a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Mapping, Sequence

DATA_CONVERTER_DIR = str(Path(__file__).resolve().parent)
if DATA_CONVERTER_DIR not in sys.path:
    sys.path.insert(0, DATA_CONVERTER_DIR)
from replay_staging_contract import (
    ReplayStagingContractError,
    STAGING_SCHEMA,
    validate_runtime_rate_provenance,
)

SUPPORTED_ACTION_SEMANTICS = frozenset(
    {
        "pre_step_raw_absolute_joint_position_command",
        "pre_step_joint_position_19_plus_body_velocity_3",
        # Current Task000300 staging uses an explicit zero-body-velocity
        # spelling for the same 19-joint action contract.
        "pre_step_joint_position_19_plus_constant_zero_body_velocity_3",
        "pre_step_named_joint_position_plus_optional_body_velocity",
    }
)
SIDECAR_NAME = "isaac_action_replay_provenance.json"

CAMERA_SHAPE_CONTRACTS: dict[str, dict[str, tuple[int, int]]] = {
    "ffw_sg2_rev1": {
        # camera: (height, width). Conversion never rotates or resizes video.
        "cam_left_head": (376, 672),
        "cam_left_wrist": (640, 480),
        "cam_right_wrist": (640, 480),
    }
}


class ConversionError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_reorder_indices(source_names: Sequence[str], target_names: Sequence[str]) -> list[int]:
    source = [str(name) for name in source_names]
    target = [str(name) for name in target_names]
    if len(source) != len(set(source)) or len(target) != len(set(target)):
        raise ConversionError("joint name lists must not contain duplicates")
    missing = [name for name in target if name not in source]
    extra = [name for name in source if name not in target]
    if missing or extra:
        raise ConversionError(f"joint-name mismatch: missing={missing}, extra={extra}")
    lookup = {name: index for index, name in enumerate(source)}
    return [lookup[name] for name in target]


def build_projection_indices(
    source_names: Sequence[str],
    requested_names: Sequence[str] | None,
    *,
    field: str,
) -> tuple[list[str], list[int]]:
    """Resolve a non-empty, ordered subset of a named source vector."""

    source = [str(name) for name in source_names]
    selected = (
        source
        if requested_names is None
        else [str(name) for name in requested_names]
    )
    if not source:
        raise ConversionError(f"source {field} names must not be empty")
    if len(source) != len(set(source)):
        raise ConversionError(f"source {field} names must not contain duplicates")
    if not selected:
        raise ConversionError(f"requested {field} names must not be empty")
    if len(selected) != len(set(selected)):
        raise ConversionError(f"requested {field} names must not contain duplicates")
    missing = [name for name in selected if name not in source]
    if missing:
        raise ConversionError(f"requested {field} names not found in source: {missing}")
    lookup = {name: index for index, name in enumerate(source)}
    return selected, [lookup[name] for name in selected]


def verify_video(path: Path, expected_frames: int, expected_fps: int) -> dict[str, Any]:
    if not path.is_file():
        raise ConversionError(f"missing video: {path}")
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise ConversionError(f"video must have one stream: {path}")
    stream = streams[0]
    frames = int(stream["nb_read_frames"])
    if frames != expected_frames:
        raise ConversionError(f"video frame mismatch {path}: {frames} != {expected_frames}")
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    fps = numerator / denominator
    if not math.isclose(fps, expected_fps, rel_tol=0.0, abs_tol=1.0e-6):
        raise ConversionError(f"video fps mismatch {path}: {fps} != {expected_fps}")
    return {
        "path": str(path),
        "codec": stream["codec_name"],
        "frames": frames,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "sha256": sha256(path),
        "validation": "full",
    }


def unchecked_video_metadata(
    path: Path,
    expected_frames: int,
    expected_fps: int,
    robot_type: str,
    camera: str,
) -> dict[str, Any]:
    """Record expected metadata without decoding non-sampled videos."""

    if not path.is_file():
        raise ConversionError(f"missing video: {path}")
    height, width = CAMERA_SHAPE_CONTRACTS[robot_type][camera]
    return {
        "path": str(path),
        "codec": None,
        "frames": expected_frames,
        "width": width,
        "height": height,
        "fps": expected_fps,
        "sha256": None,
        "validation": "not_sampled",
    }


def validate_camera_shape(
    robot_type: str,
    camera: str,
    *,
    input_height: int,
    input_width: int,
) -> None:
    """Require canonical pixels without rotating, resizing, or relabeling."""

    contract = CAMERA_SHAPE_CONTRACTS.get(robot_type, {}).get(camera)
    if contract is None:
        raise ConversionError(
            f"no camera shape contract for robot_type={robot_type!r}, camera={camera!r}"
        )
    if (input_height, input_width) != contract:
        raise ConversionError(
            f"camera shape mismatch for {robot_type}/{camera}: "
            f"input={(input_height, input_width)}, expected={contract}; "
            "rotation and resizing are not permitted"
        )


def validate_vision_transform(
    staging: Path, payload: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate a producer supplied, task agnostic LUT contract."""

    transform = payload.get("vision_transform")
    if not isinstance(transform, dict):
        raise ConversionError("vision_transform must be an object")
    required_strings = (
        "schema", "profile", "input_color_order", "output_color_order",
        "interpolation", "application_stage",
    )
    missing = [
        key for key in required_strings
        if not isinstance(transform.get(key), str) or not transform[key]
    ]
    if missing:
        raise ConversionError(f"vision LUT contract lacks fields: {missing}")
    if not isinstance(transform.get("lut_size"), int) or transform["lut_size"] < 2:
        raise ConversionError("vision LUT contract requires lut_size >= 2")

    cameras = transform.get("cameras")
    expected_cameras = set(payload.get("camera_map", {}).values())
    if not isinstance(cameras, dict) or set(cameras) != expected_cameras:
        raise ConversionError(
            "vision LUT cameras do not match staging camera_map: "
            f"got={sorted(cameras) if isinstance(cameras, dict) else cameras}, "
            f"expected={sorted(expected_cameras)}"
        )

    root = staging.resolve()
    for camera in sorted(expected_cameras):
        item = cameras[camera]
        expected_sha = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(expected_sha, str) or len(expected_sha) != 64:
            raise ConversionError(f"vision LUT hash contract is invalid for {camera}")
        relative = Path(str(item.get("path", "")))
        if relative.is_absolute():
            raise ConversionError(f"vision LUT path must be relative: {relative}")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ConversionError(f"vision LUT escapes staging: {relative}") from error
        if not path.is_file() or sha256(path) != expected_sha:
            raise ConversionError(f"missing or changed vision LUT for {camera}: {path}")
    return transform


def copy_vision_assets(
    staging: Path, meta_dir: Path, transform: Mapping[str, Any]
) -> dict[str, Any]:
    result = json.loads(json.dumps(transform))
    profile = str(result["profile"])
    destination = meta_dir / "vision_tuning"
    destination.mkdir(parents=True, exist_ok=True)
    for camera, item in result["cameras"].items():
        source = staging / str(item["path"])
        suffix = source.suffix or ".npy"
        target = destination / f"{camera}_{profile}{suffix}"
        shutil.copy2(source, target)
        if sha256(target) != item["sha256"]:
            raise ConversionError(f"copied {profile} LUT hash mismatch for {camera}")
        item["dataset_path"] = str(target.relative_to(meta_dir.parent))
    return result


def load_manifest(staging: Path, expected_episodes: int | None) -> dict[str, Any]:
    path = staging / "manifest.json"
    if not path.is_file():
        raise ConversionError(f"missing staging manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGING_SCHEMA:
        raise ConversionError(f"unsupported staging schema: {payload.get('schema')!r}")
    if payload.get("action_semantics") not in SUPPORTED_ACTION_SEMANTICS:
        raise ConversionError(f"unsupported action semantics: {payload.get('action_semantics')!r}")
    try:
        validate_runtime_rate_provenance(payload)
    except ReplayStagingContractError as error:
        raise ConversionError(str(error)) from error
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ConversionError("staging manifest contains no episodes")
    if payload.get("vision_transform") is not None:
        validate_vision_transform(staging, payload)
    old_head = [
        int(item.get("episode_index", index))
        for index, item in enumerate(episodes)
        if item.get("diagnostic_old_head_pose") is True
    ]
    if old_head:
        raise ConversionError(
            f"staging has incompatible head-pose diagnostic episodes {old_head}"
        )
    assets = payload.get("scene_assets")
    if assets is not None and (
        not isinstance(assets, dict)
        or any(not isinstance(value, str) or len(value) != 64 for value in assets.values())
    ):
        raise ConversionError("scene_assets must contain SHA-256 strings")
    if payload.get("episode_count") != len(episodes):
        raise ConversionError("manifest episode_count does not match episode records")
    if expected_episodes is not None and len(episodes) != expected_episodes:
        raise ConversionError(f"expected {expected_episodes} episodes, got {len(episodes)}")
    indices = [int(item["episode_index"]) for item in episodes]
    if indices != list(range(len(episodes))):
        raise ConversionError("episode indices must be contiguous and ordered from zero")
    fps = float(payload.get("fps", 0.0))
    if not fps.is_integer() or fps <= 0.0:
        raise ConversionError(f"LeRobot v3 writer requires a positive integer fps, got {fps}")
    return payload


def load_episode_arrays(
    staging: Path,
    record: Mapping[str, Any],
    requested_state_names: Sequence[str] | None = None,
    requested_action_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    import numpy as np

    path = staging / str(record["arrays"])
    if not path.is_file() or sha256(path) != record["array_sha256"]:
        raise ConversionError(f"missing or changed policy array archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        state = np.asarray(archive["observation_state"], dtype=np.float32)
        action = np.asarray(archive["action"], dtype=np.float32)
        timestamp = np.asarray(archive["timestamp_s"], dtype=np.float64).reshape(-1)
    length = int(record["length"])
    state_names = [str(name) for name in record["state_names"]]
    action_names = [str(name) for name in record["action_names"]]
    if state.shape != (length, len(state_names)):
        raise ConversionError(f"bad state shape at {path}: {state.shape}")
    if action.shape != (length, len(action_names)):
        raise ConversionError(f"bad action shape at {path}: {action.shape}")
    if timestamp.shape != (length,):
        raise ConversionError(f"bad timestamp shape at {path}: {timestamp.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all() or not np.isfinite(timestamp).all():
        raise ConversionError(f"non-finite policy data at {path}")
    if length > 1 and np.any(np.diff(timestamp) <= 0.0):
        raise ConversionError(f"timestamps are not strictly increasing at {path}")
    selected_state_names, state_indices = build_projection_indices(
        state_names,
        requested_state_names,
        field="state",
    )
    selected_action_names, action_indices = build_projection_indices(
        action_names,
        requested_action_names,
        field="action",
    )
    return (
        state[:, state_indices],
        action[:, action_indices],
        timestamp,
        selected_state_names,
        selected_action_names,
    )


def write_dataset(
    staging: Path,
    output: Path,
    repo_id: str,
    robot_type: str,
    cyclo_source_root: Path,
    expected_episodes: int | None,
    *,
    state_names: Sequence[str] | None = None,
    action_names: Sequence[str] | None = None,
    video_validation_sample_size: int | None = None,
) -> dict[str, Any]:
    staging = staging.resolve()
    output = output.resolve()
    if output.exists():
        raise ConversionError(f"refusing existing output directory: {output}")
    manifest = load_manifest(staging, expected_episodes)
    fps = int(manifest["fps"])
    cameras = list(manifest["camera_map"].values())
    if len(cameras) != len(set(cameras)) or not cameras:
        raise ConversionError("camera output names must be unique and non-empty")

    sys.path[:0] = [
        str(cyclo_source_root / "cyclo_data"),
        str(cyclo_source_root / "shared"),
    ]
    try:
        from cyclo_data.converter.base_converter import EpisodeData
        from cyclo_data.converter.to_lerobot_v30 import (
            RosbagToLerobotV30Converter,
            V30ConversionConfig,
        )
    except ImportError as error:
        raise ConversionError(f"Cyclo Intelligence v3 writer import failed: {error}") from error

    records = manifest["episodes"]
    if video_validation_sample_size is None or video_validation_sample_size >= len(records):
        validation_indices = set(range(len(records)))
    elif video_validation_sample_size <= 0:
        raise ConversionError("video validation sample size must be positive")
    elif video_validation_sample_size == 1:
        validation_indices = {0}
    else:
        validation_indices = {
            round(index * (len(records) - 1) / (video_validation_sample_size - 1))
            for index in range(video_validation_sample_size)
        }
    print(
        "VIDEO_VALIDATION_SAMPLE="
        + json.dumps(sorted(validation_indices)),
        flush=True,
    )

    episodes = []
    input_video_checks = []
    policy_video_checks = []
    state_names_reference = None
    action_names_reference = None
    for record_index, record in enumerate(records):
        (
            state,
            action,
            timestamp,
            episode_state_names,
            episode_action_names,
        ) = load_episode_arrays(
            staging,
            record,
            requested_state_names=state_names,
            requested_action_names=action_names,
        )
        if state_names_reference is None:
            state_names_reference = episode_state_names
        elif episode_state_names != state_names_reference:
            raise ConversionError("observation state names change across episodes")
        if action_names_reference is None:
            action_names_reference = episode_action_names
        elif episode_action_names != action_names_reference:
            raise ConversionError("action names change across episodes")
        videos = {}
        input_checks = {}
        output_checks = {}
        for camera in cameras:
            if camera not in record["videos"]:
                raise ConversionError(
                    f"episode {record['episode_index']} is missing camera {camera}"
                )
            video_path = staging / str(record["videos"][camera])
            if record_index in validation_indices:
                input_check = verify_video(video_path, len(timestamp), fps)
            else:
                input_check = unchecked_video_metadata(
                    video_path,
                    len(timestamp),
                    fps,
                    robot_type,
                    camera,
                )
            input_checks[camera] = input_check
            validate_camera_shape(
                robot_type,
                camera,
                input_height=int(input_check["height"]),
                input_width=int(input_check["width"]),
            )
            output_checks[camera] = {
                **input_check,
                "rotation_applied_deg": 0,
            }
            videos[camera] = video_path
        input_video_checks.append(
            {
                "episode_index": int(record["episode_index"]),
                "cameras": input_checks,
            }
        )
        policy_video_checks.append(
            {
                "episode_index": int(record["episode_index"]),
                "cameras": output_checks,
            }
        )
        task_text = str(manifest.get("task_instruction") or manifest["task"])
        episodes.append(
            EpisodeData(
                episode_index=int(record["episode_index"]),
                timestamps=timestamp.tolist(),
                observation_state=[row for row in state],
                action=[row for row in action],
                video_files=videos,
                tasks=[task_text],
                length=len(timestamp),
                source_path=staging / "manifest.json",
                task_name=task_text,
                observation_state_names=episode_state_names,
                action_names=episode_action_names,
            )
        )
        if record_index in validation_indices:
            print(
                f"VIDEO_VALIDATION_OK episode={record['episode_index']} "
                f"cameras={len(cameras)} frames={len(timestamp)}",
                flush=True,
            )

    config = V30ConversionConfig(
        repo_id=repo_id,
        output_dir=output,
        fps=fps,
        robot_type=robot_type,
        use_videos=True,
        selected_cameras=cameras,
        camera_rotations={camera: 0 for camera in cameras},
        source_rosbags=[],
        apply_trim=False,
        apply_exclude_regions=False,
        data_file_size_in_mb=100,
        video_file_size_in_mb=200,
    )
    writer = RosbagToLerobotV30Converter(config)
    if not writer.write_from_episodes(episodes):
        raise ConversionError("Cyclo Intelligence write_from_episodes returned false")

    meta_dir = output / "meta"
    vision_transform = manifest.get("vision_transform")
    if vision_transform is not None:
        vision_transform = copy_vision_assets(
            staging,
            meta_dir,
            validate_vision_transform(staging, manifest),
        )
    provenance = {
        "schema": "cyclo.isaac_action_replay_lerobot_v30_provenance.v1",
        "repo_id": repo_id,
        "robot_type": robot_type,
        "episode_count": len(episodes),
        "total_frames": sum(episode.length for episode in episodes),
        "fps": fps,
        "source_hdf": manifest["source_hdf"],
        "source_hdf_sha256": manifest["source_hdf_sha256"],
        "source_episode_count": manifest["source_episode_count"],
        "repeats": manifest["repeats"],
        "randomization_profile": manifest["randomization_profile"],
        "action_semantics": manifest["action_semantics"],
        "observation_semantics": manifest["observation_semantics"],
        "mobile_tail": manifest.get("mobile_tail"),
        "render_contract": manifest["render_contract"],
        "scene_assets": manifest.get("scene_assets"),
        "vision_transform": vision_transform,
        "physical_runtime_rate_provenance": manifest.get(
            "physical_runtime_rate_provenance"
        ),
        "camera_names": cameras,
        "joint_names": state_names_reference,
        "state_names": state_names_reference,
        "action_names": action_names_reference,
        "source_state_names": [
            str(name) for name in manifest["episodes"][0]["state_names"]
        ],
        "source_action_names": [
            str(name) for name in manifest["episodes"][0]["action_names"]
        ],
        "writer": (
            "cyclo_data.converter.to_lerobot_v30."
            "RosbagToLerobotV30Converter.write_from_episodes"
        ),
        "source_staging_manifest": str((staging / "manifest.json").resolve()),
        "source_staging_manifest_sha256": sha256(staging / "manifest.json"),
        "input_video_checks": input_video_checks,
        "policy_video_checks": policy_video_checks,
        "video_validation_sample_indices": sorted(validation_indices),
        "camera_shape_contract_h_w": {
            camera: {
                "height": int(size[0]), "width": int(size[1])
            }
            for camera, size in CAMERA_SHAPE_CONTRACTS.get(robot_type, {}).items()
            if camera in cameras
        },
        "episodes": manifest["episodes"],
    }
    (meta_dir / SIDECAR_NAME).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(staging / "manifest.json", meta_dir / "source_replay_manifest.json")
    if manifest.get("physical_runtime_rate_provenance") is not None:
        (meta_dir / "physical_runtime_rate_provenance.json").write_text(
            json.dumps(
                manifest["physical_runtime_rate_provenance"], indent=2
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "dataset": str(output),
        "repo_id": repo_id,
        "episodes": len(episodes),
        "frames": provenance["total_frames"],
        "provenance": str(meta_dir / SIDECAR_NAME),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--robot-type", default="ffw_sg2_rev1")
    parser.add_argument("--expected-episodes", type=int)
    parser.add_argument(
        "--video-validation-sample-size",
        type=int,
        help="Fully decode an evenly spaced sample; check existence only for the rest",
    )
    parser.add_argument(
        "--state-names",
        nargs="+",
        help="Ordered non-empty subset of source observation.state names",
    )
    parser.add_argument(
        "--action-names",
        nargs="+",
        help="Ordered non-empty subset of source action names",
    )
    parser.add_argument(
        "--cyclo-source-root",
        type=Path,
        required=True,
        help="Cyclo Intelligence source root containing cyclo_data/ and shared/",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = write_dataset(
        args.staging,
        args.output,
        args.repo_id,
        args.robot_type,
        args.cyclo_source_root,
        args.expected_episodes,
        state_names=args.state_names,
        action_names=args.action_names,
        video_validation_sample_size=args.video_validation_sample_size,
    )
    print("LEROBOT_V30=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
