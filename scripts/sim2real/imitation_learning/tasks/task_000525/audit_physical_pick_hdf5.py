#!/usr/bin/env python3
"""Fail-closed audit for a completed Task000525 physical pick HDF5."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import sys
from typing import Any

import h5py
import numpy as np

from policy_staging_common import (
    CANONICAL_CAMERA_SHAPES,
    SDG_SOURCE_FORMAT,
    Task525PolicyDataError,
    jsonable,
    validate_source,
)


REGIONS = ("A", "B", "C", "D")
REGION_TO_SIDE = {"A": "left", "B": "left", "C": "right", "D": "right"}
TARGET_OBJECT = "coffee_can_orange"
COFFEE_CAN_OBJECTS = {
    "coffee_can_black",
    "coffee_can_brown",
    "coffee_can_green",
    "coffee_can_orange",
}
SUCCESS_CRITERION = "task525_pick_quality_gate_v3"
ACCEPTANCE_SCOPE = "pick"
EXPECTED_SCHEMA = "cyclo_lab_hdf5_v1"
EXPECTED_FPS = 15

# TASK000525_PHYSICAL_TRAJECTORY_GENERATION and the online carry gate.
ROOT_NOMINAL_X_M = -1.47138
ROOT_NOMINAL_Y_M = 0.775837960613148
ROOT_XY_RANDOMIZATION_LIMIT_M = 0.030
ROOT_YAW_RANDOMIZATION_LIMIT_RAD = math.radians(2.5)
CARRY_DISTANCE_MIN_M = 0.080
CARRY_DISTANCE_TOLERANCE_M = 0.025
CARRY_OBJECT_DISPLACEMENT_MIN_M = 0.200
CARRY_HOME_ERROR_MAX = 0.150
CARRY_ROOT_XY_MAX_M = 0.005
PICK_SENTINEL_ROWS = 2

# Reviewed Task525 layout B center sampling rectangles.
TARGET_X_BOUNDS_M = (-2.2861267375946044, -2.1361267852783204)
TARGET_Y_BOUNDS_M = {
    "A": (0.4980692815780638, 0.5609115815162657),
    "B": (0.6609115815162657, 0.7237538814544676),
    "C": (0.8237538814544676, 0.8865961813926695),
    "D": (0.9865961813926695, 1.0494384813308714),
}

REQUIRED_QUALITY_METRICS = (
    "quality_carry_eef_object_distance_m",
    "quality_carry_source_eef_object_distance_m",
    "quality_carry_eef_object_distance_tolerance_m",
    "quality_carry_eef_object_distance_limit_m",
    "quality_carry_object_displacement_m",
    "quality_carry_home_joint_max_error_rad_or_m",
    "quality_carry_root_dx_m",
    "quality_carry_root_dy_m",
    "quality_carry_root_xy_displacement_m",
    "quality_carry_root_xy_max_displacement_m",
    "quality_carry_root_xy_limit_m",
    "quality_pick_navigate_sentinel_rows",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_hdf5", type=Path)
    parser.add_argument("--expected-total", type=int, required=True)
    parser.add_argument("--expected-per-region", type=int, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def _text(value: Any) -> str:
    value = jsonable(value)
    return value if isinstance(value, str) else str(value)


def _close(actual: float, expected: float, *, atol: float = 1.0e-6) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=atol))


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _ordered_unique(values: np.ndarray) -> list[int]:
    result: list[int] = []
    for value in values.reshape(-1):
        item = int(value)
        if not result or result[-1] != item:
            result.append(item)
    return result


def _require_scalar_metrics(demo: h5py.Group, reasons: list[str]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    missing = [name for name in REQUIRED_QUALITY_METRICS if name not in demo.attrs]
    if missing:
        reasons.append(f"missing required quality metadata: {missing}")
    for name, raw_value in demo.attrs.items():
        if not name.startswith("quality_"):
            continue
        value = np.asarray(raw_value)
        if value.size != 1:
            reasons.append(f"{name} is not scalar")
            continue
        try:
            number = float(value.reshape(-1)[0])
        except (TypeError, ValueError):
            reasons.append(f"{name} is not numeric")
            continue
        if not math.isfinite(number):
            reasons.append(f"{name} is not finite")
            continue
        metrics[name] = number
    return metrics


def _metric(metrics: dict[str, float], name: str) -> float | None:
    return metrics.get(name)


def _check_gate_metrics(metrics: dict[str, float], reasons: list[str]) -> None:
    if any(name not in metrics for name in REQUIRED_QUALITY_METRICS):
        return

    current = metrics["quality_carry_eef_object_distance_m"]
    source = metrics["quality_carry_source_eef_object_distance_m"]
    tolerance = metrics["quality_carry_eef_object_distance_tolerance_m"]
    limit = metrics["quality_carry_eef_object_distance_limit_m"]
    displacement = metrics["quality_carry_object_displacement_m"]
    home_error = metrics["quality_carry_home_joint_max_error_rad_or_m"]
    root_dx = metrics["quality_carry_root_dx_m"]
    root_dy = metrics["quality_carry_root_dy_m"]
    root_displacement = metrics["quality_carry_root_xy_displacement_m"]
    root_max = metrics["quality_carry_root_xy_max_displacement_m"]
    root_limit = metrics["quality_carry_root_xy_limit_m"]
    sentinel_rows = metrics["quality_pick_navigate_sentinel_rows"]

    expected_limit = max(CARRY_DISTANCE_MIN_M, source + CARRY_DISTANCE_TOLERANCE_M)
    if source < 0.0 or current < 0.0:
        reasons.append("carry EEF/object distances must be non-negative")
    if not _close(tolerance, CARRY_DISTANCE_TOLERANCE_M, atol=1.0e-9):
        reasons.append(f"carry tolerance {tolerance} is not {CARRY_DISTANCE_TOLERANCE_M}")
    if not _close(limit, expected_limit):
        reasons.append(f"carry distance limit {limit} is not source-relative {expected_limit}")
    if current > limit + 1.0e-9:
        reasons.append(f"carry EEF/object distance {current} exceeds {limit}")
    if displacement < CARRY_OBJECT_DISPLACEMENT_MIN_M:
        reasons.append(f"carry object displacement {displacement} is below 0.2 m")
    if home_error < 0.0 or home_error > CARRY_HOME_ERROR_MAX:
        reasons.append(f"carry home error {home_error} is outside [0.0, 0.15]")
    if not _close(root_limit, CARRY_ROOT_XY_MAX_M, atol=1.0e-9):
        reasons.append(f"carry root limit {root_limit} is not 0.005 m")
    if not _close(root_displacement, math.hypot(root_dx, root_dy)):
        reasons.append("carry root displacement is inconsistent with dx/dy")
    if root_displacement > root_limit + 1.0e-9 or root_max > root_limit + 1.0e-9:
        reasons.append("carry root displacement exceeds its gate limit")
    if root_max + 1.0e-9 < root_displacement:
        reasons.append("carry root maximum is below checkpoint displacement")
    if not _close(sentinel_rows, PICK_SENTINEL_ROWS, atol=1.0e-9):
        reasons.append(f"pick sentinel metric {sentinel_rows} is not 2")


def _check_observed_gate_metrics(
    demo: h5py.Group,
    metrics: dict[str, float],
    task: np.ndarray,
    frames: int,
    reasons: list[str],
) -> None:
    """Cross-check stored carry metrics at the first task-2 pre-step observation."""
    if any(name not in metrics for name in REQUIRED_QUALITY_METRICS):
        return
    root_initial_path = "initial_state/articulation/robot/root_pose"
    target_initial_path = f"initial_state/rigid_object/{TARGET_OBJECT}/root_pose"
    paths = (
        root_initial_path,
        target_initial_path,
        "obs/robot_root_pose_world",
        "obs/target_object_pose_world",
    )
    if any(path not in demo for path in paths):
        return
    if (
        demo[root_initial_path].shape != (1, 7)
        or demo[target_initial_path].shape != (1, 7)
        or demo["obs/robot_root_pose_world"].shape != (frames, 7)
        or demo["obs/target_object_pose_world"].shape != (frames, 7)
    ):
        return
    gate_indices = np.flatnonzero(task == 2)
    if len(gate_indices) != PICK_SENTINEL_ROWS:
        return
    gate_index = int(gate_indices[0])
    initial_root_xy = np.asarray(demo[root_initial_path])[0, :2]
    initial_target_xyz = np.asarray(demo[target_initial_path])[0, :3]
    root_xy = np.asarray(demo["obs/robot_root_pose_world"])[:, :2]
    target_xyz = np.asarray(demo["obs/target_object_pose_world"])[gate_index, :3]
    root_delta = root_xy[gate_index] - initial_root_xy
    observed = {
        "quality_carry_object_displacement_m": float(
            np.linalg.norm(target_xyz - initial_target_xyz)
        ),
        "quality_carry_root_dx_m": float(root_delta[0]),
        "quality_carry_root_dy_m": float(root_delta[1]),
        "quality_carry_root_xy_displacement_m": float(np.linalg.norm(root_delta)),
        "quality_carry_root_xy_max_displacement_m": float(
            np.max(
                np.linalg.norm(
                    root_xy[: gate_index + 1] - initial_root_xy,
                    axis=1,
                )
            )
        ),
    }
    for name, value in observed.items():
        if not _close(metrics[name], value):
            reasons.append(
                f"{name}={metrics[name]} disagrees with first task-2 observation {value}"
            )


def _pose_record(demo: h5py.Group, region: str, reasons: list[str]) -> dict[str, float]:
    root_path = "initial_state/articulation/robot/root_pose"
    target_path = f"initial_state/rigid_object/{TARGET_OBJECT}/root_pose"
    missing = [path for path in (root_path, target_path) if path not in demo]
    if missing:
        reasons.append(f"missing initial pose datasets: {missing}")
        return {}
    root = np.asarray(demo[root_path])
    target = np.asarray(demo[target_path])
    if root.shape != (1, 7) or target.shape != (1, 7):
        reasons.append(f"initial root/target poses must each have shape (1, 7), got {root.shape}/{target.shape}")
        return {}
    if not np.isfinite(root).all() or not np.isfinite(target).all():
        reasons.append("initial root/target pose contains non-finite values")
        return {}
    if not _close(float(np.linalg.norm(root[0, 3:7])), 1.0, atol=1.0e-3):
        reasons.append("initial robot root quaternion is not normalized")
    if not _close(float(np.linalg.norm(target[0, 3:7])), 1.0, atol=1.0e-3):
        reasons.append("initial target quaternion is not normalized")

    root_x, root_y = float(root[0, 0]), float(root[0, 1])
    target_x, target_y = float(target[0, 0]), float(target[0, 1])
    root_dx = root_x - ROOT_NOMINAL_X_M
    root_dy = root_y - ROOT_NOMINAL_Y_M
    root_yaw_world = _yaw_wxyz(root[0, 3:7])
    root_yaw_delta = _wrap_pi(root_yaw_world - math.pi)
    target_yaw_world = _yaw_wxyz(target[0, 3:7])
    target_center_x = sum(TARGET_X_BOUNDS_M) / 2.0
    target_center_y = sum(TARGET_Y_BOUNDS_M[region]) / 2.0

    if abs(root_dx) > ROOT_XY_RANDOMIZATION_LIMIT_M + 1.0e-6:
        reasons.append(f"initial root dx {root_dx} exceeds +/-0.03 m")
    if abs(root_dy) > ROOT_XY_RANDOMIZATION_LIMIT_M + 1.0e-6:
        reasons.append(f"initial root dy {root_dy} exceeds +/-0.03 m")
    if abs(root_yaw_delta) > ROOT_YAW_RANDOMIZATION_LIMIT_RAD + 1.0e-6:
        reasons.append(f"initial root yaw delta {root_yaw_delta} exceeds +/-2.5 deg")
    x_min, x_max = TARGET_X_BOUNDS_M
    y_min, y_max = TARGET_Y_BOUNDS_M[region]
    if not x_min - 1.0e-6 <= target_x <= x_max + 1.0e-6:
        reasons.append(f"initial target x {target_x} is outside region {region} bounds")
    if not y_min - 1.0e-6 <= target_y <= y_max + 1.0e-6:
        reasons.append(f"initial target y {target_y} is outside region {region} bounds")

    record = {
        "target_world_x_m": target_x,
        "target_world_y_m": target_y,
        "target_region_center_dx_m": target_x - target_center_x,
        "target_region_center_dy_m": target_y - target_center_y,
        "target_yaw_world_rad": target_yaw_world,
        "root_world_x_m": root_x,
        "root_world_y_m": root_y,
        "root_nominal_dx_m": root_dx,
        "root_nominal_dy_m": root_dy,
        "root_yaw_world_rad": root_yaw_world,
        "root_yaw_delta_rad": root_yaw_delta,
        "target_root_dx_m": target_x - root_x,
        "target_root_dy_m": target_y - root_y,
    }
    duplicate_attrs = {
        "quality_initial_target_x_m": "target_world_x_m",
        "quality_initial_target_y_m": "target_world_y_m",
        "quality_initial_robot_root_x_m": "root_world_x_m",
        "quality_initial_robot_root_y_m": "root_world_y_m",
        "quality_initial_target_root_dx_m": "target_root_dx_m",
        "quality_initial_target_root_dy_m": "target_root_dy_m",
    }
    for attr_name, record_name in duplicate_attrs.items():
        if attr_name in demo.attrs and not _close(float(demo.attrs[attr_name]), record[record_name]):
            reasons.append(f"{attr_name} disagrees with immutable initial_state pose")
    return record


def audit_demo(name: str, demo: h5py.Group) -> dict[str, Any]:
    reasons: list[str] = []
    region = _text(demo.attrs.get("task525_target_region", ""))
    if region not in REGIONS:
        reasons.append(f"invalid or missing target region {region!r}")
        region = ""
    if jsonable(demo.attrs.get("success", False)) is not True:
        reasons.append("success attribute is not true")
    if _text(demo.attrs.get("success_criterion_id", "")) != SUCCESS_CRITERION:
        reasons.append("success_criterion_id is not task525_pick_quality_gate_v3")
    if _text(demo.attrs.get("task525_acceptance_scope", "")) != ACCEPTANCE_SCOPE:
        reasons.append("task525_acceptance_scope is not pick")
    if _text(demo.attrs.get("target_object_name", "")) != TARGET_OBJECT:
        reasons.append("target_object_name is not coffee_can_orange")
    if region and _text(demo.attrs.get("task525_manipulation_side", "")) != REGION_TO_SIDE[region]:
        reasons.append(f"region {region} has the wrong manipulation side")
    try:
        arrangement = json.loads(_text(demo.attrs.get("task525_region_to_object", "")))
        if set(arrangement) != set(REGIONS) or arrangement.get(region) != TARGET_OBJECT:
            reasons.append("task525_region_to_object does not place orange in the target region")
        if set(arrangement.values()) != COFFEE_CAN_OBJECTS:
            reasons.append("task525_region_to_object does not contain the four canonical cans")
    except (AttributeError, json.JSONDecodeError, TypeError):
        reasons.append("task525_region_to_object is not valid JSON")

    required = (
        "actions",
        "obs/joint_pos",
        "obs/joint_pos_target",
        "obs/base_velocity_body",
        "obs/robot_root_pose_world",
        "obs/target_object_pose_world",
        "locomanipulation_sdg_output_data/task",
        "locomanipulation_sdg_output_data/recording_step",
        "locomanipulation_sdg_output_data/base_velocity_target",
    )
    missing = [path for path in required if path not in demo]
    frames = -1
    task = np.asarray([], dtype=np.int64)
    if missing:
        reasons.append(f"missing required episode datasets: {missing}")
    else:
        frames = int(demo["actions"].shape[0])
        expected_widths = {
            "actions": 22,
            "obs/joint_pos": 19,
            "obs/joint_pos_target": 19,
            "obs/base_velocity_body": 3,
            "obs/robot_root_pose_world": 7,
            "obs/target_object_pose_world": 7,
            "locomanipulation_sdg_output_data/task": 1,
            "locomanipulation_sdg_output_data/recording_step": 1,
            "locomanipulation_sdg_output_data/base_velocity_target": 3,
        }
        for path, width in expected_widths.items():
            shape = demo[path].shape
            if len(shape) != 2 or shape != (frames, width):
                reasons.append(f"{path} shape {shape} is not ({frames}, {width})")
        if int(demo.attrs.get("num_samples", -1)) != frames:
            reasons.append("num_samples does not match actions frame count")
        for path in (
            "actions",
            "obs/joint_pos",
            "obs/joint_pos_target",
            "obs/base_velocity_body",
            "obs/robot_root_pose_world",
            "obs/target_object_pose_world",
        ):
            if not np.isfinite(np.asarray(demo[path])).all():
                reasons.append(f"{path} contains non-finite values")
        task = np.asarray(demo["locomanipulation_sdg_output_data/task"]).reshape(-1)
        if not np.issubdtype(task.dtype, np.integer):
            reasons.append("phase metadata is not integer-valued")
        if _ordered_unique(task) != [0, 1, 2]:
            reasons.append(f"phase sequence {_ordered_unique(task)} is not [0, 1, 2]")
        sentinel = task[-PICK_SENTINEL_ROWS:] if len(task) >= PICK_SENTINEL_ROWS else task
        if len(task) < PICK_SENTINEL_ROWS or int(np.count_nonzero(task == 2)) != PICK_SENTINEL_ROWS or not np.all(sentinel == 2):
            reasons.append("episode does not end in exactly two task-2 sentinel rows")
        recording_step = np.asarray(demo["locomanipulation_sdg_output_data/recording_step"]).reshape(-1)
        if len(recording_step) >= PICK_SENTINEL_ROWS and not np.all(recording_step[-PICK_SENTINEL_ROWS:] == recording_step[-1]):
            reasons.append("pick sentinel rows do not share one source recording_step")
        actions = np.asarray(demo["actions"])
        base_target = np.asarray(demo["locomanipulation_sdg_output_data/base_velocity_target"])
        if actions.ndim == 2 and actions.shape[1] == 22 and len(actions) >= PICK_SENTINEL_ROWS and (
            np.max(np.abs(actions[-PICK_SENTINEL_ROWS:, 19:22])) > 1.0e-7
            or np.max(np.abs(base_target[-PICK_SENTINEL_ROWS:])) > 1.0e-7
        ):
            reasons.append("pick sentinel rows contain non-zero base commands")

    camera_shapes: dict[str, list[int]] = {}
    for camera, (height, width) in CANONICAL_CAMERA_SHAPES.items():
        path = f"obs/{camera}"
        if path not in demo:
            reasons.append(f"missing canonical camera raster {path}")
            continue
        dataset = demo[path]
        camera_shapes[camera] = [int(value) for value in dataset.shape[1:]]
        if dataset.shape != (frames, height, width, 3):
            reasons.append(f"{path} shape {dataset.shape} is not ({frames}, {height}, {width}, 3)")
        if dataset.dtype != np.dtype("uint8"):
            reasons.append(f"{path} dtype {dataset.dtype} is not uint8")

    metrics = _require_scalar_metrics(demo, reasons)
    _check_gate_metrics(metrics, reasons)
    if frames >= 0:
        _check_observed_gate_metrics(demo, metrics, task, frames, reasons)
    poses = _pose_record(demo, region, reasons) if region else {}
    return {
        "demo": name,
        "accepted": not reasons,
        "region": region or None,
        "frames": frames,
        "reasons": reasons,
        "camera_rasters_h_w_c": camera_shapes,
        "poses": poses,
        "gate_metrics": {name.removeprefix("quality_"): _metric(metrics, name) for name in REQUIRED_QUALITY_METRICS},
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None, "values": []}
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
        "values": values,
    }


def _distributions(records: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for region in REGIONS:
        poses = [record["poses"] for record in records if record["accepted"] and record["region"] == region]
        names = sorted({name for pose in poses for name in pose})
        result[region] = {
            "accepted_count": len(poses),
            "metrics": {name: _stats([float(pose[name]) for pose in poses]) for name in names},
        }
    return result


def audit_file(data: h5py.Group, *, source_path: Path, expected_total: int, expected_per_region: int) -> dict[str, Any]:
    errors: list[str] = []
    try:
        contract = validate_source(data)
        if contract["source_format"] != SDG_SOURCE_FORMAT:
            errors.append(f"source format {contract['source_format']!r} is not the physical generator contract")
        if contract["fps"] != EXPECTED_FPS:
            errors.append(f"source frequency {contract['fps']} Hz is not 15 Hz")
    except (Task525PolicyDataError, KeyError, TypeError, ValueError) as error:
        contract = None
        errors.append(f"source contract: {error}")
    if _text(data.attrs.get("schema_version", "")) != EXPECTED_SCHEMA:
        errors.append(f"schema_version is not {EXPECTED_SCHEMA}")

    names = sorted(data.keys())
    if len(names) != expected_total:
        errors.append(f"demo count {len(names)} does not equal expected {expected_total}")
    expected_names = {f"demo_{index}" for index in range(expected_total)}
    if set(names) != expected_names:
        errors.append("demo names are not the complete contiguous expected demo_0..demo_N set")
    records = [audit_demo(name, data[name]) for name in names]
    recorded_total = int(data.attrs.get("total", -1))
    observed_frames = sum(record["frames"] for record in records)
    if recorded_total != observed_frames:
        errors.append(f"data total frames {recorded_total} does not equal observed {observed_frames}")
    region_counts = Counter(record["region"] for record in records)
    expected_counts = {region: expected_per_region for region in REGIONS}
    observed_counts = {region: int(region_counts[region]) for region in REGIONS}
    if observed_counts != expected_counts:
        errors.append(f"region counts {observed_counts} do not equal expected {expected_counts}")
    rejected = [record["demo"] for record in records if not record["accepted"]]
    if rejected:
        errors.append(f"{len(rejected)} episode(s) failed audit: {rejected}")

    passed = not errors
    return {
        "audit_schema": "task525.physical_pick_hdf5_audit.v1",
        "passed": passed,
        "source_hdf5": str(source_path.resolve()),
        "expected_demo_count": expected_total,
        "observed_demo_count": len(names),
        "expected_region_counts": expected_counts,
        "observed_region_counts": observed_counts,
        "accepted_demo_count": sum(record["accepted"] for record in records),
        "rejected_demo_count": len(rejected),
        "errors": errors,
        "source_contract": contract,
        "camera_contract_h_w": {name: list(shape) for name, shape in CANONICAL_CAMERA_SHAPES.items()},
        "accepted_pose_distributions_by_region": _distributions(records),
        "episodes": records,
    }


def _print_human(report: dict[str, Any], output_json: Path) -> None:
    status = "PASS" if report["passed"] else "FAIL"
    if "observed_demo_count" not in report:
        print(f"{status} Task000525 physical pick audit")
        for error in report["errors"]:
            print(f"  ERROR: {error}")
        print(f"JSON: {output_json.resolve()}")
        return
    print(
        f"{status} Task000525 physical pick audit: "
        f"demos={report['observed_demo_count']}/{report['expected_demo_count']} "
        f"regions={report['observed_region_counts']} accepted={report['accepted_demo_count']}"
    )
    for region, distribution in report["accepted_pose_distributions_by_region"].items():
        metrics = distribution["metrics"]
        if not metrics:
            continue
        tx, ty = metrics["target_world_x_m"], metrics["target_world_y_m"]
        rdx, rdy = metrics["root_nominal_dx_m"], metrics["root_nominal_dy_m"]
        yaw = metrics["root_yaw_delta_rad"]
        print(
            f"  {region}: n={distribution['accepted_count']} "
            f"target_x=[{tx['min']:.4f},{tx['max']:.4f}] "
            f"target_y=[{ty['min']:.4f},{ty['max']:.4f}] "
            f"root_dx=[{rdx['min']:.4f},{rdx['max']:.4f}] "
            f"root_dy=[{rdy['min']:.4f},{rdy['max']:.4f}] "
            f"root_yaw_delta_deg=[{math.degrees(yaw['min']):.2f},{math.degrees(yaw['max']):.2f}]"
        )
    for error in report["errors"]:
        print(f"  ERROR: {error}")
    print(f"JSON: {output_json.resolve()}")


def main() -> int:
    args = parse_args()
    if args.expected_total <= 0 or args.expected_per_region <= 0:
        raise SystemExit("expected counts must be positive")
    if args.expected_total != len(REGIONS) * args.expected_per_region:
        raise SystemExit("--expected-total must equal 4 * --expected-per-region")
    if args.input_hdf5.resolve() == args.output_json.resolve():
        raise SystemExit("input HDF5 and output JSON paths must differ")

    try:
        with h5py.File(args.input_hdf5, "r") as handle:
            if "data" not in handle:
                raise KeyError("missing /data group")
            report = audit_file(
                handle["data"],
                source_path=args.input_hdf5,
                expected_total=args.expected_total,
                expected_per_region=args.expected_per_region,
            )
    except (OSError, KeyError, ValueError) as error:
        report = {
            "audit_schema": "task525.physical_pick_hdf5_audit.v1",
            "passed": False,
            "source_hdf5": str(args.input_hdf5.resolve()),
            "errors": [f"could not audit HDF5: {error}"],
        }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _print_human(report, args.output_json)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
