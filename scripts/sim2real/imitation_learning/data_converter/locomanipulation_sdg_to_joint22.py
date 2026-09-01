#!/usr/bin/env python3
"""Convert Task525 SDG EEF22 HDF5 into causal policy joint22 HDF5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


POLICY_CONTRACT_ID = "ffw_sg2_rev1_mobile_22d_v1"
SDG_CONTRACT_TOKEN = "locomanipulation_sdg_eef"


def _decode_attr(value):
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _write_attr(group, key: str, value) -> None:
    if isinstance(value, (dict, list, tuple)):
        value = json.dumps(value)
    group.attrs[key] = value


def _copy_attrs(source, destination) -> None:
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def _copy_cropped(source, destination, source_frames: int) -> None:
    """Copy a group and crop datasets whose first dimension is source time."""

    _copy_attrs(source, destination)
    for key, item in source.items():
        if isinstance(item, h5py.Group):
            _copy_cropped(item, destination.create_group(key), source_frames)
            continue
        values = np.asarray(item)
        if values.ndim > 0 and values.shape[0] == source_frames:
            values = values[:-1]
        kwargs = {"compression": "gzip"} if values.ndim > 0 else {}
        dataset = destination.create_dataset(key, data=values, **kwargs)
        _copy_attrs(item, dataset)


def _replace_dataset(group, path: str, values: np.ndarray) -> None:
    parent_path, _, name = path.rpartition("/")
    parent = group[parent_path] if parent_path else group
    if name in parent:
        del parent[name]
    parent.create_dataset(name, data=values, compression="gzip")


def _require_demo_contract(demo, demo_name: str) -> int:
    required = (
        "actions",
        "obs/actions",
        "obs/joint_pos",
        "obs/joint_pos_target",
        "obs/base_velocity_body",
    )
    missing = [path for path in required if path not in demo]
    if missing:
        raise ValueError(f"{demo_name}: missing SDG datasets: {missing}")
    frames = int(demo["actions"].shape[0])
    shapes = {
        "actions": (frames, 22),
        "obs/actions": (frames, 22),
        "obs/joint_pos": (frames, 19),
        "obs/joint_pos_target": (frames, 19),
        "obs/base_velocity_body": (frames, 3),
    }
    for path, expected in shapes.items():
        actual = tuple(demo[path].shape)
        if actual != expected:
            raise ValueError(f"{demo_name}: {path} is {actual}, expected {expected}")
    if frames < 2:
        raise ValueError(f"{demo_name}: at least two frames are required")
    return frames


def _convert_demo(source_demo, output_demo, demo_name: str) -> int:
    frames = _require_demo_contract(source_demo, demo_name)
    raw_sdg_actions = np.asarray(source_demo["actions"], dtype=np.float32)
    raw_last_actions = np.asarray(source_demo["obs/actions"], dtype=np.float32)
    joint_pos = np.asarray(source_demo["obs/joint_pos"], dtype=np.float32)
    joint_targets = np.asarray(
        source_demo["obs/joint_pos_target"], dtype=np.float32
    )

    # Action t produces the joint target visible in pre-step observation t+1.
    policy_actions = np.concatenate(
        (joint_targets[1:], raw_sdg_actions[:-1, 19:22]), axis=-1
    )
    previous_policy_actions = np.empty_like(policy_actions)
    previous_policy_actions[0, :19] = joint_pos[0]
    previous_policy_actions[0, 19:22] = 0.0
    previous_policy_actions[1:] = policy_actions[:-1]

    if not np.isfinite(policy_actions).all():
        raise ValueError(f"{demo_name}: derived policy actions contain non-finite values")

    _copy_cropped(source_demo, output_demo, frames)
    _replace_dataset(output_demo, "source_sdg_actions", raw_sdg_actions[:-1])
    _replace_dataset(output_demo, "obs/source_sdg_actions", raw_last_actions[:-1])
    _replace_dataset(
        output_demo,
        "obs/source_joint_pos_target_pre_step",
        joint_targets[:-1],
    )
    if "processed_actions" in output_demo:
        source_processed = np.asarray(output_demo["processed_actions"])
        _replace_dataset(output_demo, "source_sdg_processed_actions", source_processed)

    _replace_dataset(output_demo, "actions", policy_actions)
    _replace_dataset(output_demo, "processed_actions", policy_actions)
    _replace_dataset(output_demo, "obs/actions", previous_policy_actions)
    _replace_dataset(output_demo, "obs/joint_pos_target", joint_targets[1:])
    output_demo.attrs["num_samples"] = frames - 1

    if not np.allclose(output_demo["actions"][:, :19], joint_targets[1:]):
        raise AssertionError(f"{demo_name}: joint target causal alignment failed")
    if not np.allclose(
        output_demo["actions"][:, 19:22], raw_sdg_actions[:-1, 19:22]
    ):
        raise AssertionError(f"{demo_name}: base command alignment failed")
    return frames - 1


def convert(input_path: Path, output_path: Path) -> None:
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("input and output paths must differ")
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(input_path, "r") as source, h5py.File(
            output_path, "x"
        ) as output:
            if "data" not in source:
                raise ValueError("input HDF5 has no /data group")
            source_data = source["data"]
            contract_id = str(
                _decode_attr(source_data.attrs.get("robot_contract_id", ""))
            )
            demo_names = sorted(source_data.keys())
            if not demo_names:
                raise ValueError("input HDF5 has no demos")
            has_sdg_group = any(
                "locomanipulation_sdg_output_data" in source_data[name]
                for name in demo_names
            )
            if SDG_CONTRACT_TOKEN not in contract_id and not has_sdg_group:
                raise ValueError(
                    "input is not marked as Task525 locomanipulation_sdg EEF22"
                )

            action_names = _decode_attr(
                source_data.attrs.get("observation_state_names")
            )
            action_units = _decode_attr(
                source_data.attrs.get("observation_state_units")
            )
            if not isinstance(action_names, list) or len(action_names) != 22:
                raise ValueError("source observation_state_names must declare joint22")
            if not isinstance(action_units, list) or len(action_units) != 22:
                raise ValueError("source observation_state_units must declare 22 units")

            output_data = output.create_group("data")
            _copy_attrs(source_data, output_data)
            total = 0
            for demo_name in demo_names:
                total += _convert_demo(
                    source_data[demo_name],
                    output_data.create_group(demo_name),
                    demo_name,
                )

            _write_attr(output_data, "source_robot_contract_id", contract_id)
            _write_attr(output_data, "robot_contract_id", POLICY_CONTRACT_ID)
            _write_attr(output_data, "action_names", action_names)
            _write_attr(output_data, "action_units", action_units)
            _write_attr(
                output_data,
                "action_semantics",
                "derived_pre_step_absolute_joint_position_19_plus_body_velocity_3",
            )
            _write_attr(
                output_data,
                "causal_alignment",
                "state=source_obs_t; joint_action=source_joint_pos_target_t_plus_1; "
                "base_action=source_raw_action_t",
            )
            _write_attr(
                output_data,
                "obs_last_action_semantics",
                "previous_derived_policy_action; row0=initial_joint_hold_plus_zero_base",
            )
            _write_attr(
                output_data,
                "processed_actions_semantics",
                "derived_policy_action_copy; raw values preserved under source_sdg_*",
            )
            output_data.attrs["total"] = total
    except Exception:
        if output_path.exists():
            output_path.unlink()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    convert(args.input, args.output)
    print(f"Wrote causal Task525 joint22 HDF5: {args.output.resolve()}")


if __name__ == "__main__":
    main()
