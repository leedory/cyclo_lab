#!/usr/bin/env python3
"""Merge compatible replay-staging datasets without rerunning Isaac or video encoding.

Use this after native export or randomized action replay when reviewed episode
sets should become one LeRobot conversion input. Arrays and videos are linked
when possible, copied otherwise, and recorded in a new provenance manifest.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
from typing import Any, Mapping


SCHEMA = "cyclo.isaac_action_replay_staging.v1"
ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"


class MergeError(RuntimeError):
    pass


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_manifest(staging: Path) -> dict[str, Any]:
    path = staging / "manifest.json"
    if not path.is_file():
        raise MergeError(f"missing manifest: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA:
        raise MergeError(f"unsupported staging schema at {path}")
    if manifest.get("action_semantics") != ACTION_SEMANTICS:
        raise MergeError(f"unsupported action semantics at {path}")
    episodes = manifest.get("episodes")
    if (
        not isinstance(episodes, list)
        or not episodes
        or manifest.get("episode_count") != len(episodes)
    ):
        raise MergeError(f"incomplete staging manifest: {path}")
    if [int(item["episode_index"]) for item in episodes] != list(range(len(episodes))):
        raise MergeError(f"non-contiguous episode indices: {path}")
    return manifest


def link_or_copy(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise MergeError(f"missing source file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def merge(
    inputs: list[Path],
    output: Path,
    exclude_source_episodes: set[str] | None = None,
) -> dict[str, Any]:
    sources = [path.resolve() for path in inputs]
    destination = output.resolve()
    if not sources:
        raise MergeError("at least one --input staging directory is required")
    excluded_sources = set(exclude_source_episodes or ())
    if destination.exists():
        raise MergeError(f"refusing existing output directory: {destination}")
    manifests = [load_manifest(path) for path in sources]
    selected_count = sum(
        str(record.get("source_episode")) not in excluded_sources
        for manifest in manifests
        for record in manifest["episodes"]
    )
    if selected_count == 0:
        raise MergeError("episode selection is empty")
    reference = manifests[0]
    for path, manifest in zip(sources[1:], manifests[1:]):
        for key in ("fps", "camera_map", "action_semantics", "task_instruction"):
            if manifest.get(key) != reference.get(key):
                raise MergeError(f"incompatible {key} at {path}")
        first_names = reference["episodes"][0]["state_names"]
        if manifest["episodes"][0]["state_names"] != first_names:
            raise MergeError(f"incompatible state joint names at {path}")

    destination.mkdir(parents=True)
    (destination / "policy_arrays").mkdir()
    records: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    transfer_modes = {"hardlink": 0, "copy": 0}

    for source_index, (staging, manifest) in enumerate(zip(sources, manifests)):
        source_records.append(
            {
                "source_index": source_index,
                "staging": str(staging),
                "manifest_sha256": sha256(staging / "manifest.json"),
                "episode_count": len(manifest["episodes"]),
                "total_frames": int(manifest["total_frames"]),
                "source_hdf": manifest.get("source_hdf"),
                "source_hdf_sha256": manifest.get("source_hdf_sha256"),
                "randomization_profile": manifest.get("randomization_profile"),
            }
        )
        for source_record in manifest["episodes"]:
            if str(source_record.get("source_episode")) in excluded_sources:
                continue
            episode_index = len(records)
            record = deepcopy(source_record)
            source_array = staging / str(source_record["arrays"])
            destination_array = (
                destination / "policy_arrays" / f"episode_{episode_index:06d}.npz"
            )
            mode = link_or_copy(source_array, destination_array)
            transfer_modes[mode] += 1
            if sha256(destination_array) != source_record["array_sha256"]:
                raise MergeError(f"array hash mismatch after transfer: {destination_array}")

            videos: dict[str, str] = {}
            for camera, relative_path in source_record["videos"].items():
                source_video = staging / str(relative_path)
                destination_video = (
                    destination
                    / "videos"
                    / f"episode_{episode_index:06d}"
                    / f"{camera}.mp4"
                )
                mode = link_or_copy(source_video, destination_video)
                transfer_modes[mode] += 1
                videos[camera] = str(destination_video.relative_to(destination))

            record["episode_index"] = episode_index
            record["arrays"] = str(destination_array.relative_to(destination))
            record["videos"] = videos
            record["combined_source"] = {
                "source_index": source_index,
                "source_staging": str(staging),
                "source_episode_index": int(source_record["episode_index"]),
            }
            records.append(record)

    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": utc_now(),
        "source_hdf": [item.get("source_hdf") for item in manifests],
        "source_hdf_sha256": [item.get("source_hdf_sha256") for item in manifests],
        "source_episode_count": len(records),
        "source_episodes": [
            f"source_{record['combined_source']['source_index']}:"
            f"{record.get('source_episode', record['episode_index'])}"
            for record in records
        ],
        "source_attrs": reference.get("source_attrs", {}),
        "task": reference.get("task", ""),
        "task_instruction": reference.get("task_instruction", ""),
        "target_object_name": reference.get("target_object_name", ""),
        "randomization_profile": (
            "filtered_replay_staging" if excluded_sources else "combined_replay_staging"
        ),
        "randomization_profile_values": {"excluded_source_episodes": sorted(excluded_sources)},
        "repeats": None,
        "seed": None,
        "fps": reference["fps"],
        "camera_map": reference["camera_map"],
        "action_semantics": ACTION_SEMANTICS,
        "observation_semantics": "pre_step sources combined without temporal changes",
        "render_contract": "compatible native and replay staging episodes combined in input order",
        "episodes": records,
        "batches": [],
        "gpu_samples": [],
        "episode_count": len(records),
        "total_frames": sum(int(record["length"]) for record in records),
        "source_manifests": source_records,
        "transfer_modes": transfer_modes,
    }
    write_json(destination / "manifest.json", manifest)
    return {
        "staging": str(destination),
        "episodes": manifest["episode_count"],
        "frames": manifest["total_frames"],
        "transfer_modes": transfer_modes,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--exclude-source-episode",
        action="append",
        default=[],
        help="Source episode name to omit. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = merge(
        args.input, args.output, set(args.exclude_source_episode)
    )
    print("MERGED_REPLAY_STAGING=" + json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
