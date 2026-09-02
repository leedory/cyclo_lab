#!/usr/bin/env python3
"""Audit Task000525 trajectories and optionally merge only valid episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


QUALITY_GATE_ID = "task525_generation_quality_gate_v2"
DEFAULT_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M = 0.005
EXPECTED_TASK_SEQUENCE = [0, 1, 2, 3, 4]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=0)
    parser.add_argument(
        "--max_pre_nav_root_xy_displacement_m",
        type=float,
        default=DEFAULT_MAX_PRE_NAV_ROOT_XY_DISPLACEMENT_M,
    )
    return parser.parse_args()


def demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def ordered_unique(values: np.ndarray) -> list[int]:
    result: list[int] = []
    for value in values.reshape(-1):
        item = int(value)
        if not result or result[-1] != item:
            result.append(item)
    return result


def audit_demo(
    demo: h5py.Group,
    *,
    max_root_displacement_m: float,
) -> dict:
    required = (
        "actions",
        "obs/robot_root_pose_world",
        "locomanipulation_sdg_output_data/task",
    )
    missing = [path for path in required if path not in demo]
    if missing:
        return {"passed": False, "reasons": [f"missing datasets: {missing}"]}

    actions = np.asarray(demo["actions"])
    root = np.asarray(demo["obs/robot_root_pose_world"])
    task = np.asarray(demo["locomanipulation_sdg_output_data/task"]).reshape(-1)
    reasons: list[str] = []
    if actions.ndim != 2 or actions.shape[1] != 22:
        reasons.append(f"actions shape is {actions.shape}, expected [T, 22]")
    if root.ndim != 2 or root.shape[1] != 7:
        reasons.append(f"root pose shape is {root.shape}, expected [T, 7]")
    if not (len(actions) == len(root) == len(task)):
        reasons.append(
            f"frame counts differ: actions={len(actions)}, root={len(root)}, task={len(task)}"
        )
    if reasons:
        return {"passed": False, "reasons": reasons}

    success = bool(demo.attrs.get("success", False))
    if not success:
        reasons.append("episode success attribute is not true")
    if int(demo.attrs.get("num_samples", -1)) != len(actions):
        reasons.append("num_samples attribute does not match frame count")
    if not np.isfinite(actions).all() or not np.isfinite(root).all():
        reasons.append("actions or root pose contains non-finite values")

    sequence = ordered_unique(task)
    if sequence != EXPECTED_TASK_SEQUENCE:
        reasons.append(f"task sequence is {sequence}, expected {EXPECTED_TASK_SEQUENCE}")

    navigation_indices = np.flatnonzero(task == 2)
    if len(navigation_indices) == 0:
        navigation_start = None
        pre_nav_end = len(task)
        reasons.append("task 2 navigation phase is absent")
    else:
        navigation_start = int(navigation_indices[0])
        # Include the first task-2 observation. It is the pose measured by the
        # online carry gate immediately before its first navigation command.
        pre_nav_end = navigation_start + 1

    root_delta_xy = root[:pre_nav_end, :2] - root[0, :2]
    root_displacement = np.linalg.norm(root_delta_xy, axis=1)
    max_root_displacement = float(np.max(root_displacement))
    final_root_delta = root[pre_nav_end - 1, :2] - root[0, :2]
    if max_root_displacement > max_root_displacement_m:
        reasons.append(
            "pre-navigation root XY displacement "
            f"{max_root_displacement:.6f} m exceeds {max_root_displacement_m:.6f} m"
        )

    stationary_actions = actions[: navigation_start or 0, 19:22]
    stationary_command_peak = (
        float(np.max(np.abs(stationary_actions))) if len(stationary_actions) else 0.0
    )
    if stationary_command_peak > 1e-7:
        reasons.append(
            f"pre-navigation base command peak {stationary_command_peak:.3e} is non-zero"
        )

    first_navigation_wz = None
    if navigation_start is not None:
        navigation_wz = actions[navigation_start:, 21]
        nonzero_wz = np.flatnonzero(np.abs(navigation_wz) > 1e-6)
        if len(nonzero_wz) == 0:
            reasons.append("navigation has no non-zero angular-z command")
        else:
            first_navigation_wz = float(navigation_wz[nonzero_wz[0]])
            if first_navigation_wz <= 0.0:
                reasons.append(
                    f"first navigation angular-z {first_navigation_wz:.6f} is not CCW"
                )

    return {
        "passed": not reasons,
        "reasons": reasons,
        "frames": int(len(actions)),
        "task_sequence": sequence,
        "navigation_start_frame": navigation_start,
        "pre_navigation_base_command_peak_abs": stationary_command_peak,
        "pre_navigation_root_dx_m": float(final_root_delta[0]),
        "pre_navigation_root_dy_m": float(final_root_delta[1]),
        "pre_navigation_root_xy_displacement_m": float(
            np.linalg.norm(final_root_delta)
        ),
        "pre_navigation_root_xy_max_displacement_m": max_root_displacement,
        "first_navigation_angular_z_radps": first_navigation_wz,
        "source_success_criterion_id": str(
            demo.attrs.get("success_criterion_id", "")
        ),
    }


def copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def main() -> None:
    args = parse_args()
    if args.count < 0:
        raise ValueError("--count must be non-negative")
    if args.output is not None and args.count <= 0:
        raise ValueError("--count must be positive when --output is used")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)
    if args.manifest.exists():
        raise FileExistsError(args.manifest)

    records: list[dict] = []
    selected: list[dict] = []
    handles: list[h5py.File] = []
    try:
        for input_path in args.inputs:
            resolved = input_path.resolve()
            handle = h5py.File(resolved, "r")
            handles.append(handle)
            if "data" not in handle:
                raise KeyError(f"{resolved}: missing /data group")
            for name in sorted(handle["data"].keys(), key=demo_sort_key):
                result = audit_demo(
                    handle["data"][name],
                    max_root_displacement_m=args.max_pre_nav_root_xy_displacement_m,
                )
                record = {
                    "source_file": str(resolved),
                    "source_demo": name,
                    **result,
                }
                records.append(record)
                if result["passed"] and (
                    args.count <= 0 or len(selected) < args.count
                ):
                    selected.append(record)

        passed = sum(bool(record["passed"]) for record in records)
        if args.output is not None and len(selected) != args.count:
            raise RuntimeError(
                f"only {len(selected)} valid episodes available, requested {args.count}"
            )

        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(args.output, "x") as output:
                output_data = output.create_group("data")
                copy_attrs(handles[0]["data"].attrs, output_data.attrs)
                total_frames = 0
                for index, record in enumerate(selected):
                    source_index = next(
                        i
                        for i, path in enumerate(args.inputs)
                        if str(path.resolve()) == record["source_file"]
                    )
                    source_demo = handles[source_index]["data"][record["source_demo"]]
                    destination_name = f"demo_{index}"
                    handles[source_index].copy(
                        source_demo, output_data, name=destination_name
                    )
                    destination = output_data[destination_name]
                    destination.attrs["success"] = True
                    destination.attrs["success_criterion_id"] = QUALITY_GATE_ID
                    destination.attrs["failure_reason"] = ""
                    destination.attrs["quality_carry_root_dx_m"] = record[
                        "pre_navigation_root_dx_m"
                    ]
                    destination.attrs["quality_carry_root_dy_m"] = record[
                        "pre_navigation_root_dy_m"
                    ]
                    destination.attrs["quality_carry_root_xy_displacement_m"] = record[
                        "pre_navigation_root_xy_displacement_m"
                    ]
                    destination.attrs["quality_carry_root_xy_max_displacement_m"] = record[
                        "pre_navigation_root_xy_max_displacement_m"
                    ]
                    destination.attrs["quality_carry_root_xy_limit_m"] = (
                        args.max_pre_nav_root_xy_displacement_m
                    )
                    record["destination_demo"] = destination_name
                    total_frames += int(record["frames"])
                output_data.attrs["total"] = total_frames
                output_data.attrs["trajectory_selection_contract_id"] = QUALITY_GATE_ID
                output_data.attrs["max_pre_navigation_root_xy_displacement_m"] = (
                    args.max_pre_nav_root_xy_displacement_m
                )
                output_data.attrs["source_files"] = json.dumps(
                    [str(path.resolve()) for path in args.inputs]
                )

        report = {
            "quality_gate_id": QUALITY_GATE_ID,
            "max_pre_navigation_root_xy_displacement_m": (
                args.max_pre_nav_root_xy_displacement_m
            ),
            "input_episode_count": len(records),
            "passed_episode_count": passed,
            "failed_episode_count": len(records) - passed,
            "selected_episode_count": len(selected),
            "output": str(args.output.resolve()) if args.output is not None else None,
            "records": records,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps({key: value for key, value in report.items() if key != "records"}, indent=2))
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    main()
