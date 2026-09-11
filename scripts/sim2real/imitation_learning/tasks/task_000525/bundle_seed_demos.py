#!/usr/bin/env python3
"""Validate and bundle Task525 A/B-left and C/D-right seed recordings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


REGION_KEYS = ("A", "B", "C", "D")
REGION_TO_SIDE = {"A": "left", "B": "left", "C": "right", "D": "right"}
CAN_NAMES = (
    "coffee_can_black",
    "coffee_can_brown",
    "coffee_can_green",
    "coffee_can_orange",
)
TARGET_OBJECT = "coffee_can_orange"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Seed HDF5 input. Pass once for A/B and once for C/D.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def attr_text(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def main() -> None:
    args = parse_args()
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(output_path)
    if len(args.input) != 2:
        raise ValueError("Task525 seed bundling requires exactly two --input files")

    handles: list[h5py.File] = []
    episodes: dict[str, tuple[h5py.File, str, Path]] = {}
    try:
        contract_id = None
        for input_path in args.input:
            resolved = input_path.resolve()
            handle = h5py.File(resolved, "r")
            handles.append(handle)
            if "data" not in handle:
                raise ValueError(f"{resolved}: missing /data group")
            data = handle["data"]
            dataset_target = attr_text(data.attrs.get("target_object_name", ""))
            if dataset_target != TARGET_OBJECT:
                raise ValueError(
                    f"{resolved}: Task525 requires target {TARGET_OBJECT}, "
                    f"got {dataset_target!r}"
                )
            current_contract = attr_text(
                data.attrs.get("robot_contract_id", "")
            )
            if contract_id is None:
                contract_id = current_contract
            elif current_contract != contract_id:
                raise ValueError(
                    f"seed action contract mismatch: {contract_id} vs {current_contract}"
                )
            for demo_name, demo in data.items():
                if not bool(demo.attrs.get("success", False)):
                    raise ValueError(f"{resolved}:{demo_name} is not marked successful")
                episode_target = attr_text(
                    demo.attrs.get("target_object_name", "")
                )
                if episode_target != TARGET_OBJECT:
                    raise ValueError(
                        f"{resolved}:{demo_name} requires target {TARGET_OBJECT}, "
                        f"got {episode_target!r}"
                    )
                region = attr_text(demo.attrs.get("task525_target_region", "")).upper()
                side = attr_text(demo.attrs.get("task525_manipulation_side", ""))
                if region not in REGION_TO_SIDE:
                    raise ValueError(
                        f"{resolved}:{demo_name} has invalid target region {region!r}"
                    )
                try:
                    region_to_object = json.loads(
                        attr_text(demo.attrs.get("task525_region_to_object", ""))
                    )
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"{resolved}:{demo_name} has invalid arrangement JSON"
                    ) from error
                if (
                    not isinstance(region_to_object, dict)
                    or set(region_to_object) != set(REGION_KEYS)
                    or set(region_to_object.values()) != set(CAN_NAMES)
                    or region_to_object.get(region) != TARGET_OBJECT
                ):
                    raise ValueError(
                        f"{resolved}:{demo_name} has invalid orange-target arrangement"
                    )
                if side != REGION_TO_SIDE[region]:
                    raise ValueError(
                        f"{resolved}:{demo_name} region {region} requires "
                        f"{REGION_TO_SIDE[region]}, got {side!r}"
                    )
                if region in episodes:
                    raise ValueError(f"duplicate Task525 seed region {region}")
                episodes[region] = (handle, demo_name, resolved)

        missing = [region for region in REGION_KEYS if region not in episodes]
        if missing:
            raise ValueError(f"Task525 seed inputs are missing regions: {missing}")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output_path, "x") as output:
            output_data = output.create_group("data")
            copy_attrs(handles[0]["data"].attrs, output_data.attrs)
            total_frames = 0
            for index, region in enumerate(REGION_KEYS):
                source, source_name, source_path = episodes[region]
                destination_name = f"demo_{index}"
                source.copy(
                    source[f"data/{source_name}"],
                    output_data,
                    name=destination_name,
                )
                destination = output_data[destination_name]
                destination.attrs["source_file"] = str(source_path)
                destination.attrs["source_demo_id"] = source_name
                frames = int(
                    destination.attrs.get(
                        "num_samples",
                        destination["actions"].shape[0],
                    )
                )
                total_frames += frames

            output_data.attrs["total"] = total_frames
            output_data.attrs["target_object_name"] = TARGET_OBJECT
            output_data.attrs["target_side"] = "region_conditioned"
            output_data.attrs["task525_seed_bundle_version"] = 2
            output_data.attrs["task525_seed_regions"] = ",".join(REGION_KEYS)
            output_data.attrs["task525_region_side_policy"] = json.dumps(
                REGION_TO_SIDE,
                sort_keys=True,
            )
            output_data.attrs["source_files"] = json.dumps(
                [str(path.resolve()) for path in args.input]
            )
    finally:
        for handle in handles:
            handle.close()

    print(
        f"Bundled Task525 regions A-D into {output_path} "
        f"with policy {REGION_TO_SIDE}"
    )


if __name__ == "__main__":
    main()
