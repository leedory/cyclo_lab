#!/usr/bin/env python3
"""Audit a final Task525 LeRobot v3 dataset, including every video chunk."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pyarrow.parquet as pq


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
CAMERA_SHAPES_H_W = {
    "cam_left_head": (376, 672),
    "cam_left_wrist": (640, 480),
    "cam_right_wrist": (640, 480),
}


class AuditError(RuntimeError):
    pass


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
    root = args.dataset.resolve()
    info = load_json(root / "meta" / "info.json")
    root_info = load_json(root / "info.json")
    root_config = root_info.get("conversion_config", {})
    expected_root_config = {
        "robot_type": "ffw_sg2_rev1",
        "task_name": args.expected_task,
        "fps": 15,
        "selected_cameras": list(CAMERAS),
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
    for feature_name in ("observation.state", "action"):
        feature = features.get(feature_name, {})
        if feature.get("dtype") != "float32":
            raise AuditError(f"{feature_name}: expected float32")
        if feature.get("shape") != [22] or feature.get("names") != JOINT_NAMES:
            raise AuditError(f"{feature_name}: bad 22D joint contract")
    for camera in CAMERAS:
        feature = features.get(f"observation.images.rgb.{camera}", {})
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
        for feature_name in ("observation.state", "action"):
            column = table.column(feature_name).combine_chunks()
            if column.null_count:
                raise AuditError(f"{path}: null values in {feature_name}")
            values = column.values.to_numpy(zero_copy_only=False)
            if values.size != table.num_rows * 22 or not np.isfinite(values).all():
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
    if (
        provenance.get("episode_count") != args.expected_episodes
        or provenance.get("total_frames") != args.expected_frames
        or len(provenance.get("episodes", [])) != args.expected_episodes
        or len(provenance.get("input_video_checks", [])) != args.expected_episodes
        or len(provenance.get("policy_video_checks", [])) != args.expected_episodes
    ):
        raise AuditError("provenance counts are inconsistent")

    camera_frames = {}
    camera_files = {}
    for camera in CAMERAS:
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
        "state_action_dim": 22,
        "fps": 15,
        "video_contract": {
            "codec": "h264",
            "pixel_format": "yuv420p",
            "camera_shapes_h_w": CAMERA_SHAPES_H_W,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, required=True)
    parser.add_argument("--expected-frames", type=int, required=True)
    parser.add_argument("--expected-task", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    print(
        "TASK525_LEROBOT_V30_AUDIT="
        + json.dumps(audit(parse_args()), sort_keys=True),
        flush=True,
    )
