#!/usr/bin/env python3
"""Compose Task000458 original/before/after replay videos for visual review."""

from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np


class ComparisonError(RuntimeError):
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-staging", type=Path, required=True)
    parser.add_argument("--before-staging", type=Path, required=True)
    parser.add_argument("--after-staging", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-episode",
        action="append",
        required=True,
        help="Source episode name to compare. May be repeated, for example demo_8.",
    )
    parser.add_argument("--repeat-index", type=int, default=0)
    parser.add_argument("--camera", default="cam_left_head")
    parser.add_argument("--label-height", type=int, default=40)
    parser.add_argument(
        "--font-file",
        type=Path,
        default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    parser.add_argument("--ffmpeg-bin", default="ffmpeg")
    parser.add_argument("--ffprobe-bin", default="ffprobe")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_manifest(staging: Path) -> dict[str, Any]:
    path = staging.resolve() / "manifest.json"
    if not path.is_file():
        raise ComparisonError(f"missing staging manifest: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = payload.get("episodes")
    if not isinstance(episodes, list):
        raise ComparisonError(f"manifest has no episode list: {path}")
    fps = float(payload.get("fps", 0.0))
    if fps <= 0.0:
        raise ComparisonError(f"manifest has invalid fps: {path}")
    return payload


def resolve_record(
    manifest: Mapping[str, Any], source_episode: str, repeat_index: int, label: str
) -> Mapping[str, Any]:
    records = [
        record
        for record in manifest["episodes"]
        if str(record.get("source_episode")) == source_episode
        and int(record.get("repeat_index", 0)) == repeat_index
    ]
    if len(records) != 1:
        raise ComparisonError(
            f"{label} expected one record for ({source_episode}, repeat={repeat_index}), "
            f"found {len(records)}"
        )
    return records[0]


def resolve_video(staging: Path, record: Mapping[str, Any], camera: str) -> Path:
    videos = record.get("videos")
    if not isinstance(videos, Mapping):
        raise ComparisonError(f"episode {record.get('episode_index')} has no video map")
    candidates = []
    for key, relative_path in videos.items():
        key = str(key)
        if key == camera or key.endswith(f".{camera}"):
            candidates.append(staging.resolve() / str(relative_path))
    if len(candidates) != 1:
        raise ComparisonError(
            f"episode {record.get('episode_index')} expected one camera matching {camera!r}, "
            f"found {len(candidates)} in {sorted(str(key) for key in videos)}"
        )
    path = candidates[0]
    if not path.is_file():
        raise ComparisonError(f"missing episode video: {path}")
    return path


def load_action(staging: Path, record: Mapping[str, Any]) -> np.ndarray:
    path = staging.resolve() / str(record.get("arrays", ""))
    if not path.is_file():
        raise ComparisonError(f"missing policy array archive: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if "action" not in archive:
            raise ComparisonError(f"policy array archive has no action: {path}")
        return np.asarray(archive["action"], dtype=np.float32)


def probe_video(ffprobe_bin: str, path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [
            ffprobe_bin,
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,profile,pix_fmt,width,height,avg_frame_rate,nb_read_frames",
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
        raise ComparisonError(f"expected one video stream: {path}")
    stream = streams[0]
    numerator, denominator = (int(value) for value in stream["avg_frame_rate"].split("/"))
    return {
        "codec": str(stream.get("codec_name", "")),
        "profile": str(stream.get("profile", "")),
        "pix_fmt": str(stream.get("pix_fmt", "")),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(Fraction(numerator, denominator)),
        "frames": int(stream["nb_read_frames"]),
    }


def validate_inputs(
    manifests: Sequence[Mapping[str, Any]],
    records: Sequence[Mapping[str, Any]],
    actions: Sequence[np.ndarray],
    probes: Sequence[Mapping[str, Any]],
) -> tuple[int, float, int, int]:
    lengths = [int(record.get("length", -1)) for record in records]
    if len(set(lengths)) != 1 or lengths[0] <= 0:
        raise ComparisonError(f"episode length mismatch: {lengths}")
    fps_values = [float(manifest["fps"]) for manifest in manifests]
    if len(set(fps_values)) != 1:
        raise ComparisonError(f"manifest fps mismatch: {fps_values}")
    if any(action.shape != actions[0].shape or not np.array_equal(action, actions[0]) for action in actions[1:]):
        raise ComparisonError("reference/before/after action arrays are not byte-identical float32 values")
    if records[1].get("random_seed") != records[2].get("random_seed"):
        raise ComparisonError("before/after random seeds differ")
    if records[1].get("randomization") != records[2].get("randomization"):
        raise ComparisonError("before/after randomization snapshots differ")
    dimensions = [(int(probe["width"]), int(probe["height"])) for probe in probes]
    if len(set(dimensions)) != 1:
        raise ComparisonError(f"input video dimensions differ: {dimensions}")
    for probe in probes:
        if int(probe["frames"]) != lengths[0]:
            raise ComparisonError(
                f"input video frame count {probe['frames']} does not match episode length {lengths[0]}"
            )
        if abs(float(probe["fps"]) - fps_values[0]) > 1.0e-6:
            raise ComparisonError(
                f"input video fps {probe['fps']} does not match manifest fps {fps_values[0]}"
            )
    return lengths[0], fps_values[0], dimensions[0][0], dimensions[0][1]


def filter_graph(font_file: Path, label_height: int) -> str:
    if label_height < 24 or label_height % 2:
        raise ComparisonError("--label-height must be an even integer of at least 24")
    font = str(font_file).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    labels = ("ORIGINAL HDF", "BEFORE - ROOT DRIFT", "AFTER - ROOT STABLE")
    parts = []
    for index, label in enumerate(labels):
        parts.append(
            f"[{index}:v]setpts=PTS-STARTPTS,"
            f"pad=iw:ih+{label_height}:0:{label_height}:color=black,"
            f"drawtext=fontfile='{font}':text='{label}':"
            f"x=(w-text_w)/2:y=({label_height}-text_h)/2:fontsize=20:fontcolor=white[p{index}]"
        )
    parts.append("[p0][p1][p2]hstack=inputs=3[v]")
    return ";".join(parts)


def compose_video(
    args: argparse.Namespace,
    source_episode: str,
    videos: Sequence[Path],
    expected_frames: int,
    expected_fps: float,
    input_width: int,
    input_height: int,
) -> tuple[Path, dict[str, Any]]:
    if not args.font_file.is_file():
        raise ComparisonError(f"missing font file: {args.font_file}")
    safe_episode = source_episode.replace("/", "_")
    output = args.output_dir.resolve() / (
        f"{safe_episode}_repeat_{args.repeat_index}_original_before_after_{args.camera}.mp4"
    )
    if output.exists() and not args.overwrite:
        raise ComparisonError(f"refusing existing output without --overwrite: {output}")
    command = [args.ffmpeg_bin, "-y" if args.overwrite else "-n"]
    for video in videos:
        command.extend(["-i", str(video)])
    command.extend(
        [
            "-filter_complex",
            filter_graph(args.font_file, args.label_height),
            "-map",
            "[v]",
            "-c:v",
            "libx264",
            "-profile:v",
            "baseline",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-an",
            str(output),
        ]
    )
    subprocess.run(command, check=True)
    output_probe = probe_video(args.ffprobe_bin, output)
    expected_width = input_width * 3
    expected_height = input_height + args.label_height
    required = {
        "codec": "h264",
        "profile": "Constrained Baseline",
        "pix_fmt": "yuv420p",
        "width": expected_width,
        "height": expected_height,
        "frames": expected_frames,
    }
    mismatches = {
        key: (output_probe.get(key), value)
        for key, value in required.items()
        if output_probe.get(key) != value
    }
    if abs(float(output_probe["fps"]) - expected_fps) > 1.0e-6:
        mismatches["fps"] = (output_probe["fps"], expected_fps)
    if mismatches:
        raise ComparisonError(f"output video contract mismatch at {output}: {mismatches}")
    subprocess.run(
        [
            args.ffmpeg_bin,
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:v:0",
            "-f",
            "null",
            "-",
        ],
        check=True,
    )
    return output, output_probe


def main() -> int:
    if args.repeat_index < 0:
        raise ComparisonError("--repeat-index must be non-negative")
    stagings = [
        args.reference_staging.resolve(),
        args.before_staging.resolve(),
        args.after_staging.resolve(),
    ]
    manifests = [load_manifest(staging) for staging in stagings]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for source_episode in args.source_episode:
        records = [
            resolve_record(manifest, source_episode, args.repeat_index, label)
            for manifest, label in zip(manifests, ("reference", "before", "after"), strict=True)
        ]
        videos = [
            resolve_video(staging, record, args.camera)
            for staging, record in zip(stagings, records, strict=True)
        ]
        actions = [
            load_action(staging, record)
            for staging, record in zip(stagings, records, strict=True)
        ]
        probes = [probe_video(args.ffprobe_bin, video) for video in videos]
        expected_frames, expected_fps, width, height = validate_inputs(
            manifests, records, actions, probes
        )
        output, output_probe = compose_video(
            args,
            source_episode,
            videos,
            expected_frames,
            expected_fps,
            width,
            height,
        )
        results.append(
            {
                "source_episode": source_episode,
                "repeat_index": args.repeat_index,
                "camera": args.camera,
                "output": str(output),
                "video": output_probe,
                "fully_decoded": True,
            }
        )
    print("REPLAY_STAGING_VIDEO_COMPARISON=" + json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    args = parse_args()
    raise SystemExit(main())
