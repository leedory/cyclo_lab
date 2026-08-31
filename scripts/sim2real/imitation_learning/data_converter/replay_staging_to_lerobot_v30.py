#!/usr/bin/env python3
"""Validate reviewed replay staging and write a LeRobot v3 dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence



STAGING_SCHEMA = "cyclo.isaac_action_replay_staging.v1"
ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"
SIDECAR_NAME = "isaac_action_replay_provenance.json"

# Staging videos already contain the captured camera pixel arrays. Preserve
# their HxW=480x640 layout for ACT instead of applying the SG2 runtime camera
# rotation a second time during LeRobot conversion.
CAMERA_ORIENTATION_CONTRACTS: dict[
    str, dict[str, tuple[tuple[int, int], int]]
] = {
    "ffw_sg2_rev1": {
        # camera: ((height, width), rotation convention used by video_sync)
        "cam_left_head": ((480, 640), 0),
        "cam_left_wrist": ((480, 640), 0),
        "cam_right_wrist": ((480, 640), 0),
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
    }


def required_camera_rotation(
    robot_type: str,
    camera: str,
    *,
    input_height: int,
    input_width: int,
) -> int:
    """Return the rotation needed to satisfy a robot camera shape contract.

    A video that already has the required dimensions is left unchanged. This
    makes conversion idempotent for recorder videos whose rotation was already
    baked into the MP4.
    """
    contract = CAMERA_ORIENTATION_CONTRACTS.get(robot_type, {}).get(camera)
    if contract is None:
        return 0

    (target_height, target_width), rotation_deg = contract
    if (input_height, input_width) == (target_height, target_width):
        return 0

    output_height, output_width = input_height, input_width
    if rotation_deg % 180:
        output_height, output_width = input_width, input_height
    if (output_height, output_width) != (target_height, target_width):
        raise ConversionError(
            f"camera shape cannot satisfy {robot_type} contract for {camera}: "
            f"input={(input_height, input_width)}, "
            f"rotation={rotation_deg}, "
            f"expected={(target_height, target_width)}"
        )
    return rotation_deg


def load_manifest(staging: Path, expected_episodes: int | None) -> dict[str, Any]:
    path = staging / "manifest.json"
    if not path.is_file():
        raise ConversionError(f"missing staging manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != STAGING_SCHEMA:
        raise ConversionError(f"unsupported staging schema: {payload.get('schema')!r}")
    if payload.get("action_semantics") != ACTION_SEMANTICS:
        raise ConversionError(f"unsupported action semantics: {payload.get('action_semantics')!r}")
    episodes = payload.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise ConversionError("staging manifest contains no episodes")
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


def load_episode_arrays(staging: Path, record: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    action = action[:, build_reorder_indices(action_names, state_names)]
    return state, action, timestamp


def write_dataset(
    staging: Path,
    output: Path,
    repo_id: str,
    robot_type: str,
    cyclo_source_root: Path,
    expected_episodes: int | None,
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
        from cyclo_data.converter.video_sync import remux_selected_frames
    except ImportError as error:
        raise ConversionError(f"Cyclo Intelligence v3 writer import failed: {error}") from error

    episodes = []
    input_video_checks = []
    policy_video_checks = []
    state_names_reference = None
    orientation_tmp = tempfile.TemporaryDirectory(
        prefix="cyclo_lerobot_oriented_videos_"
    )
    try:
        orientation_root = Path(orientation_tmp.name)
        for record in manifest["episodes"]:
            state, action, timestamp = load_episode_arrays(staging, record)
            state_names = [str(name) for name in record["state_names"]]
            if state_names_reference is None:
                state_names_reference = state_names
            elif state_names != state_names_reference:
                raise ConversionError("observation state names change across episodes")
            videos = {}
            input_checks = {}
            output_checks = {}
            for camera in cameras:
                if camera not in record["videos"]:
                    raise ConversionError(
                        f"episode {record['episode_index']} is missing camera {camera}"
                    )
                video_path = staging / str(record["videos"][camera])
                input_check = verify_video(video_path, len(timestamp), fps)
                input_checks[camera] = input_check
                rotation_deg = required_camera_rotation(
                    robot_type,
                    camera,
                    input_height=int(input_check["height"]),
                    input_width=int(input_check["width"]),
                )
                policy_video_path = video_path
                if rotation_deg:
                    policy_video_path = (
                        orientation_root
                        / f"episode_{int(record['episode_index']):06d}"
                        / f"{camera}.mp4"
                    )
                    sync_result = remux_selected_frames(
                        video_path,
                        range(len(timestamp)),
                        policy_video_path,
                        target_fps=fps,
                        rotation_deg=rotation_deg,
                    )
                    if int(sync_result.frame_count) != len(timestamp):
                        raise ConversionError(
                            f"rotated video frame mismatch {policy_video_path}: "
                            f"{sync_result.frame_count} != {len(timestamp)}"
                        )
                policy_check = verify_video(
                    policy_video_path, len(timestamp), fps
                )
                contract = CAMERA_ORIENTATION_CONTRACTS.get(robot_type, {}).get(camera)
                if contract is not None:
                    expected_size, _ = contract
                    actual_size = (
                        int(policy_check["height"]),
                        int(policy_check["width"]),
                    )
                    if actual_size != expected_size:
                        raise ConversionError(
                            f"policy camera shape mismatch for {camera}: "
                            f"{actual_size} != {expected_size}"
                        )
                output_checks[camera] = {
                    **policy_check,
                    "rotation_applied_deg": rotation_deg,
                }
                videos[camera] = policy_video_path
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
                    observation_state_names=state_names,
                    action_names=state_names,
                )
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
    finally:
        orientation_tmp.cleanup()

    provenance = {
        "schema": "cyclo.isaac_action_replay_lerobot_v30_provenance.v1",
        "review_status": "unreviewed",
        "training_ready": False,
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
        "render_contract": manifest["render_contract"],
        "camera_names": cameras,
        "joint_names": state_names_reference,
        "writer": (
            "cyclo_data.converter.to_lerobot_v30."
            "RosbagToLerobotV30Converter.write_from_episodes"
        ),
        "source_staging_manifest": str((staging / "manifest.json").resolve()),
        "source_staging_manifest_sha256": sha256(staging / "manifest.json"),
        "input_video_checks": input_video_checks,
        "policy_video_checks": policy_video_checks,
        "camera_orientation_contract": {
            camera: {
                "height": int(size[0]),
                "width": int(size[1]),
                "rotation_deg_when_transposed": int(rotation),
            }
            for camera, (size, rotation) in CAMERA_ORIENTATION_CONTRACTS.get(
                robot_type, {}
            ).items()
            if camera in cameras
        },
        "episodes": manifest["episodes"],
    }
    meta_dir = output / "meta"
    (meta_dir / SIDECAR_NAME).write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    shutil.copy2(staging / "manifest.json", meta_dir / "source_replay_manifest.json")
    (output / "REVIEW_REQUIRED.txt").write_text(
        "This action-replay dataset is structurally validated but has not been visually reviewed.\n",
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
        "--cyclo-source-root",
        type=Path,
        default=Path("/root/ros2_ws/src/cyclo_intelligence"),
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
    )
    print("LEROBOT_V30=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
