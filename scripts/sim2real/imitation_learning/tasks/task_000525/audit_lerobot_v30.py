#!/usr/bin/env python3
"""Audit a final Task525 LeRobot v3 dataset, including every video chunk."""

from __future__ import annotations

import argparse
import hashlib
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any, Sequence


JOINT_NAMES = [
    "arm_l_joint1",
    "arm_l_joint2",
    "arm_l_joint3",
    "arm_l_joint4",
    "arm_l_joint5",
    "arm_l_joint6",
    "arm_l_joint7",
    "gripper_l_joint1",
    "arm_r_joint1",
    "arm_r_joint2",
    "arm_r_joint3",
    "arm_r_joint4",
    "arm_r_joint5",
    "arm_r_joint6",
    "arm_r_joint7",
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    "linear_x",
    "linear_y",
    "angular_z",
]
CAMERAS = ("cam_left_head", "cam_left_wrist", "cam_right_wrist")
ACTION_SEMANTICS = "pre_step_joint_position_19_plus_body_velocity_3"
CAMERA_SHAPES_H_W = {
    "cam_left_head": (376, 672),
    "cam_left_wrist": (640, 480),
    "cam_right_wrist": (640, 480),
}


class AuditError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_expected_subset(
    values: Sequence[str], canonical: Sequence[str], label: str
) -> tuple[str, ...]:
    selected = tuple(str(value) for value in values)
    if not selected:
        raise AuditError(f"{label} must not be empty")
    if len(selected) != len(set(selected)):
        raise AuditError(f"{label} contains duplicates")
    unknown = [value for value in selected if value not in canonical]
    if unknown:
        raise AuditError(f"{label} contains non-canonical names: {unknown}")
    selected_set = set(selected)
    canonical_order = tuple(value for value in canonical if value in selected_set)
    if selected != canonical_order:
        raise AuditError(f"{label} must preserve canonical order: {canonical_order}")
    return selected


def validate_video_check_records(
    records: Any,
    expected_cameras: Sequence[str],
    expected_episodes: int,
    expected_frames: int,
    *,
    label: str,
) -> None:
    if not isinstance(records, list) or len(records) != expected_episodes:
        raise AuditError(f"{label}: expected {expected_episodes} episode checks")
    frame_sums = {camera: 0 for camera in expected_cameras}
    for episode_index, record in enumerate(records):
        if int(record.get("episode_index", -1)) != episode_index:
            raise AuditError(f"{label}: episode indices are not contiguous")
        checks = record.get("cameras")
        if not isinstance(checks, dict) or set(checks) != set(expected_cameras):
            raise AuditError(f"{label}: episode {episode_index} camera set mismatch")
        for camera in expected_cameras:
            check = checks[camera]
            if not isinstance(check, dict):
                raise AuditError(
                    f"{label}: episode {episode_index} {camera} check is not a mapping"
                )
            height, width = CAMERA_SHAPES_H_W[camera]
            if (
                int(check.get("height", -1)) != height
                or int(check.get("width", -1)) != width
                or float(check.get("fps", -1.0)) != 15.0
                or int(check.get("frames", -1)) <= 0
            ):
                raise AuditError(
                    f"{label}: episode {episode_index} {camera} contract mismatch"
                )
            frame_sums[camera] += int(check["frames"])
    if any(total != expected_frames for total in frame_sums.values()):
        raise AuditError(f"{label}: per-camera frame totals mismatch {frame_sums}")


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise AuditError(f"missing JSON file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def probe_video(
    path: Path, expected_fps: int, expected_shape_h_w: tuple[int, int]
) -> int:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    streams = json.loads(result.stdout).get("streams", [])
    if len(streams) != 1:
        raise AuditError(f"{path}: expected one video stream")
    stream = streams[0]
    actual = {
        "codec_name": stream.get("codec_name"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(stream["avg_frame_rate"])),
    }
    expected_height, expected_width = expected_shape_h_w
    expected = {
        "codec_name": "h264",
        "pix_fmt": "yuv420p",
        "width": expected_width,
        "height": expected_height,
        "fps": float(expected_fps),
    }
    if actual != expected:
        raise AuditError(f"{path}: video contract mismatch {actual} != {expected}")
    frames = int(stream["nb_read_frames"])
    if frames <= 0:
        raise AuditError(f"{path}: no video frames")
    return frames


def audit(args: argparse.Namespace) -> dict[str, Any]:
    import numpy as np
    import pyarrow.parquet as pq

    root = args.dataset.resolve()
    expected_state_names = validate_expected_subset(
        args.expected_state_names, JOINT_NAMES, "expected state names"
    )
    expected_action_names = validate_expected_subset(
        args.expected_action_names, JOINT_NAMES, "expected action names"
    )
    expected_cameras = validate_expected_subset(
        args.expected_cameras, CAMERAS, "expected cameras"
    )
    info = load_json(root / "meta" / "info.json")
    root_info = load_json(root / "info.json")
    root_config = root_info.get("conversion_config", {})
    expected_root_config = {
        "robot_type": "ffw_sg2_rev1",
        "task_name": args.expected_task,
        "fps": 15,
        "selected_cameras": list(expected_cameras),
    }
    root_mismatches = {
        key: (root_config.get(key), value)
        for key, value in expected_root_config.items()
        if root_config.get(key) != value
    }
    if root_mismatches:
        raise AuditError(f"root writer info mismatch: {root_mismatches}")
    expected_info = {
        "codebase_version": "v3.0",
        "robot_type": "ffw_sg2_rev1",
        "total_episodes": args.expected_episodes,
        "total_frames": args.expected_frames,
        "total_tasks": 1,
        "fps": 15,
        "splits": {"train": f"0:{args.expected_episodes}"},
    }
    mismatches = {
        key: (info.get(key), value)
        for key, value in expected_info.items()
        if info.get(key) != value
    }
    if mismatches:
        raise AuditError(f"info.json mismatch: {mismatches}")

    features = info.get("features", {})
    expected_vector_features = {
        "observation.state": expected_state_names,
        "action": expected_action_names,
    }
    for feature_name, expected_names in expected_vector_features.items():
        feature = features.get(feature_name, {})
        if feature.get("dtype") != "float32":
            raise AuditError(f"{feature_name}: expected float32")
        if (
            feature.get("shape") != [len(expected_names)]
            or feature.get("names") != list(expected_names)
        ):
            raise AuditError(
                f"{feature_name}: expected exact {len(expected_names)}D named contract"
            )

    camera_prefix = "observation.images.rgb."
    actual_camera_features = {
        name.removeprefix(camera_prefix)
        for name in features
        if name.startswith(camera_prefix)
    }
    if actual_camera_features != set(expected_cameras):
        raise AuditError(
            "image feature set mismatch: "
            f"actual={sorted(actual_camera_features)}, "
            f"expected={list(expected_cameras)}"
        )
    for camera in expected_cameras:
        feature = features[f"{camera_prefix}{camera}"]
        video_info = feature.get("info", {})
        height, width = CAMERA_SHAPES_H_W[camera]
        if feature.get("dtype") != "video" or feature.get("shape") != [
            3,
            height,
            width,
        ]:
            raise AuditError(f"{camera}: bad image feature")
        expected_video_info = {
            "video.fps": 15.0,
            "video.height": height,
            "video.width": width,
            "video.channels": 3,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "has_audio": False,
        }
        if video_info != expected_video_info:
            raise AuditError(f"{camera}: bad info.json video contract")

    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise AuditError("no data parquet files")
    parquet_rows = 0
    expected_global_index = 0
    for path in data_files:
        table = pq.read_table(path)
        parquet_rows += table.num_rows
        for feature_name, expected_names in expected_vector_features.items():
            column = table.column(feature_name).combine_chunks()
            if column.null_count:
                raise AuditError(f"{path}: null values in {feature_name}")
            values = column.values.to_numpy(zero_copy_only=False)
            if (
                values.size != table.num_rows * len(expected_names)
                or not np.isfinite(values).all()
            ):
                raise AuditError(f"{path}: invalid values in {feature_name}")
        indices = table.column("index").combine_chunks().to_numpy(zero_copy_only=False)
        expected_indices = np.arange(
            expected_global_index,
            expected_global_index + table.num_rows,
            dtype=indices.dtype,
        )
        if not np.array_equal(indices, expected_indices):
            raise AuditError(f"{path}: global index is not contiguous")
        expected_global_index += table.num_rows
    if parquet_rows != args.expected_frames:
        raise AuditError(f"parquet rows {parquet_rows} != {args.expected_frames}")

    episode_files = sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet"))
    if not episode_files:
        raise AuditError("no episode metadata parquet")
    episodes = pq.read_table(episode_files)
    if episodes.num_rows != args.expected_episodes:
        raise AuditError(f"episode metadata rows {episodes.num_rows} != {args.expected_episodes}")
    episode_indices = episodes.column("episode_index").combine_chunks().to_numpy()
    if not np.array_equal(episode_indices, np.arange(args.expected_episodes)):
        raise AuditError("episode metadata indices are not contiguous")
    lengths = episodes.column("length").combine_chunks().to_numpy()
    if int(lengths.sum()) != args.expected_frames:
        raise AuditError("episode length sum differs from total frames")
    starts = episodes.column("dataset_from_index").combine_chunks().to_numpy()
    stops = episodes.column("dataset_to_index").combine_chunks().to_numpy()
    data_file_indices = (
        episodes.column("data/file_index").combine_chunks().to_numpy()
    )
    if not np.array_equal(stops - starts, lengths):
        raise AuditError("episode dataset ranges do not match episode lengths")
    for file_index in np.unique(data_file_indices):
        rows = np.flatnonzero(data_file_indices == file_index)
        if not np.array_equal(rows, np.arange(rows[0], rows[-1] + 1)):
            raise AuditError(f"data file {file_index}: episode rows are not contiguous")
        if starts[rows[0]] != 0:
            raise AuditError(f"data file {file_index}: first dataset offset is not zero")
        if len(rows) > 1 and not np.array_equal(
            starts[rows[1:]], stops[rows[:-1]]
        ):
            raise AuditError(f"data file {file_index}: dataset offsets have gaps")
        if int(stops[rows[-1]]) != int(lengths[rows].sum()):
            raise AuditError(f"data file {file_index}: final dataset offset is wrong")
    tasks = episodes.column("tasks").to_pylist()
    if any(value != [args.expected_task] for value in tasks):
        raise AuditError("episode task text mismatch")

    tasks_table = pq.read_table(root / "meta" / "tasks.parquet")
    if tasks_table.num_rows != 1 or args.expected_task not in str(tasks_table.to_pydict()):
        raise AuditError("tasks.parquet mismatch")

    provenance = load_json(root / "meta" / "isaac_action_replay_provenance.json")
    expected_provenance = {
        "schema": "cyclo.isaac_action_replay_lerobot_v30_provenance.v1",
        "robot_type": "ffw_sg2_rev1",
        "episode_count": args.expected_episodes,
        "total_frames": args.expected_frames,
        "fps": 15,
        "action_semantics": ACTION_SEMANTICS,
        "camera_names": list(expected_cameras),
        "joint_names": list(expected_state_names),
        "state_names": list(expected_state_names),
        "action_names": list(expected_action_names),
        "source_state_names": JOINT_NAMES,
        "source_action_names": JOINT_NAMES,
        "camera_shape_contract_h_w": {
            camera: {
                "height": CAMERA_SHAPES_H_W[camera][0],
                "width": CAMERA_SHAPES_H_W[camera][1],
            }
            for camera in expected_cameras
        },
    }
    provenance_mismatches = {
        key: (provenance.get(key), value)
        for key, value in expected_provenance.items()
        if provenance.get(key) != value
    }
    if provenance_mismatches:
        raise AuditError(f"provenance contract mismatch: {provenance_mismatches}")
    if (
        len(provenance.get("episodes", [])) != args.expected_episodes
        or not provenance.get("source_hdf")
        or not provenance.get("source_hdf_sha256")
    ):
        raise AuditError("provenance source/count fields are incomplete")
    validate_video_check_records(
        provenance.get("input_video_checks"),
        expected_cameras,
        args.expected_episodes,
        args.expected_frames,
        label="input_video_checks",
    )
    validate_video_check_records(
        provenance.get("policy_video_checks"),
        expected_cameras,
        args.expected_episodes,
        args.expected_frames,
        label="policy_video_checks",
    )

    source_manifest_path = root / "meta" / "source_replay_manifest.json"
    source_manifest = load_json(source_manifest_path)
    source_manifest_expected = {
        "schema": "cyclo.isaac_action_replay_staging.v1",
        "action_semantics": ACTION_SEMANTICS,
        "episode_count": args.expected_episodes,
        "total_frames": args.expected_frames,
    }
    source_manifest_mismatches = {
        key: (source_manifest.get(key), value)
        for key, value in source_manifest_expected.items()
        if source_manifest.get(key) != value
    }
    if source_manifest_mismatches:
        raise AuditError(
            f"source replay manifest mismatch: {source_manifest_mismatches}"
        )
    if provenance.get("source_staging_manifest_sha256") != sha256(source_manifest_path):
        raise AuditError("source replay manifest hash differs from provenance")
    camera_map = source_manifest.get("camera_map")
    if not isinstance(camera_map, dict) or list(camera_map.values()) != list(
        expected_cameras
    ):
        raise AuditError("source replay manifest camera_map mismatch")
    camera_shapes = source_manifest.get("camera_shapes_h_w")
    camera_rotations = source_manifest.get("camera_rotation_deg")
    if (
        not isinstance(camera_shapes, dict)
        or set(camera_shapes) != set(camera_map)
        or not isinstance(camera_rotations, dict)
        or set(camera_rotations) != set(camera_map)
    ):
        raise AuditError("source replay manifest camera metadata keys mismatch")
    for source_camera, output_camera in camera_map.items():
        if (
            tuple(camera_shapes[source_camera]) != CAMERA_SHAPES_H_W[output_camera]
            or int(camera_rotations[source_camera]) != 0
        ):
            raise AuditError(
                f"source replay manifest {source_camera} camera contract mismatch"
            )
    source_episodes = source_manifest.get("episodes")
    if not isinstance(source_episodes, list) or len(source_episodes) != args.expected_episodes:
        raise AuditError("source replay manifest episode records mismatch")
    if any(
        set(record.get("videos", {})) != set(expected_cameras)
        for record in source_episodes
    ):
        raise AuditError("source replay manifest episode camera set mismatch")

    video_root = root / "videos"
    if not video_root.is_dir():
        raise AuditError(f"missing video directory: {video_root}")
    actual_video_cameras = {
        path.name.removeprefix("observation.images.rgb.")
        for path in video_root.iterdir()
        if path.is_dir() and path.name.startswith("observation.images.rgb.")
    }
    if actual_video_cameras != set(expected_cameras):
        raise AuditError(
            "video directory camera set mismatch: "
            f"actual={sorted(actual_video_cameras)}, expected={list(expected_cameras)}"
        )
    camera_frames = {}
    camera_files = {}
    for camera in expected_cameras:
        paths = sorted(
            (root / "videos" / f"observation.images.rgb.{camera}").glob(
                "chunk-*/file-*.mp4"
            )
        )
        if not paths:
            raise AuditError(f"{camera}: no video chunks")
        frames = sum(
            probe_video(path, 15, CAMERA_SHAPES_H_W[camera]) for path in paths
        )
        if frames != args.expected_frames:
            raise AuditError(f"{camera}: frame sum {frames} != {args.expected_frames}")
        camera_frames[camera] = frames
        camera_files[camera] = len(paths)

    if not (root / "meta" / "stats.json").is_file():
        raise AuditError("missing stats.json")
    return {
        "dataset": str(root),
        "episodes": args.expected_episodes,
        "frames": args.expected_frames,
        "parquet_files": len(data_files),
        "camera_video_files": camera_files,
        "camera_frames": camera_frames,
        "state_dim": len(expected_state_names),
        "action_dim": len(expected_action_names),
        "cameras": list(expected_cameras),
        "fps": 15,
        "video_contract": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "camera_shapes_h_w": {
                camera: CAMERA_SHAPES_H_W[camera] for camera in expected_cameras
            },
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-task", required=True)
    parser.add_argument(
        "--expected-state-names",
        nargs="+",
        default=list(JOINT_NAMES),
        metavar="NAME",
        help="Exact canonical observation.state names in order (default: full 22D)",
    )
    parser.add_argument(
        "--expected-action-names",
        nargs="+",
        default=list(JOINT_NAMES),
        metavar="NAME",
        help="Exact canonical action names in order (default: full 22D)",
    )
    parser.add_argument(
        "--expected-cameras",
        nargs="+",
        default=list(CAMERAS),
        metavar="CAMERA",
        help="Exact canonical LeRobot camera feature set (default: all three)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    print(
        "TASK525_LEROBOT_V30_AUDIT="
        + json.dumps(audit(parse_args()), sort_keys=True),
        flush=True,
    )
