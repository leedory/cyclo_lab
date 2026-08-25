#!/usr/bin/env python3
"""Generate a validated SG2 runtime camera profile from a raw CameraInfo record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from cyclo_lab.robot_specs.ffw.sg2 import load_sg2_camera_profile


def _camera_document(record: dict, role: str, serial: str | None) -> dict:
    try:
        capture = record["cameras"][role]
        camera_info = capture["camera_info"]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Raw record is missing cameras.{role}.camera_info.") from exc
    if not capture.get("stable_across_samples"):
        raise ValueError(f"Raw record camera role {role!r} is not stable across its samples.")

    return {
        "topic": capture["topic"],
        "frame_id": camera_info["frame_id"],
        "serial": serial,
        "width": camera_info["width"],
        "height": camera_info["height"],
        "distortion_model": camera_info["distortion_model"],
        "d": camera_info["d"],
        "k": camera_info["k"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True, help="Raw sg2-camera-intrinsics-v1 JSON record.")
    parser.add_argument("--output", type=Path, required=True, help="Destination runtime YAML profile.")
    parser.add_argument("--profile-id", required=True, help="Immutable, versioned profile identifier.")
    parser.add_argument("--ssh-alias", required=True, help="Robot SSH alias, for example 1050.")
    parser.add_argument("--robot-hostname", help="Override hostname recorded in the raw capture.")
    parser.add_argument("--head-serial", help="Optional head-camera serial.")
    parser.add_argument("--wrist-left-serial", required=True)
    parser.add_argument("--wrist-right-serial", required=True)
    parser.add_argument("--experiment", help="Experiment directory name containing the raw evidence.")
    parser.add_argument("--ai-worker-image")
    parser.add_argument("--ai-worker-commit")
    parser.add_argument("--force", action="store_true", help="Replace an existing output profile.")
    args = parser.parse_args()

    record_bytes = args.record.read_bytes()
    record = json.loads(record_bytes)
    if record.get("schema") != "sg2-camera-intrinsics-v1":
        raise ValueError(f"Unsupported raw capture schema: {record.get('schema')!r}.")
    if not record.get("complete"):
        raise ValueError("Raw CameraInfo capture is incomplete.")

    robot_hostname = args.robot_hostname or record.get("hostname")
    if not robot_hostname:
        raise ValueError("Robot hostname is absent; pass --robot-hostname.")

    document = {
        "schema_version": 1,
        "profile_id": args.profile_id,
        "robot": {
            "ssh_alias": str(args.ssh_alias),
            "hostname": robot_hostname,
        },
        "cameras": {
            "head": _camera_document(record, "head", args.head_serial),
            "wrist_left": _camera_document(record, "wrist_left", args.wrist_left_serial),
            "wrist_right": _camera_document(record, "wrist_right", args.wrist_right_serial),
        },
        "provenance": {
            "captured_at_utc": record.get("captured_at_utc"),
            "capture_schema": record["schema"],
            "capture_sha256": hashlib.sha256(record_bytes).hexdigest(),
            "experiment": args.experiment,
            "ai_worker_image": args.ai_worker_image,
            "ai_worker_commit": args.ai_worker_commit,
            "note": "Runtime CameraInfo capture; not a checkerboard recalibration.",
        },
    }
    document["provenance"] = {key: value for key, value in document["provenance"].items() if value is not None}

    if args.output.exists() and not args.force:
        raise FileExistsError(f"Output already exists: {args.output}. Pass --force to replace it.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_output.write_text(
        yaml.safe_dump(document, sort_keys=False, width=120),
        encoding="utf-8",
    )
    validated = load_sg2_camera_profile(temporary_output.resolve())
    temporary_output.replace(args.output)

    print(
        json.dumps(
            {
                "output": str(args.output),
                "profile_id": validated.profile_id,
                "robot_hostname": validated.robot_hostname,
                "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
