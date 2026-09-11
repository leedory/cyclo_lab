#!/usr/bin/env python3
"""Fail-closed audit and selection for Coffee Transport physical trajectories.

The audit is intentionally Isaac-free.  It recomputes immutable spawn and final
placement checks from HDF5 arrays, validates the task-axis route, and reports
grasp-relative metrics without using them as an acceptance gate.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from plot_success_initial_poses import (
    EpisodePose,
    build_summary as build_pose_summary,
    row_for_episode,
    save_plot,
    write_csv as write_pose_csv,
)
from runtime_rate_attestation import (
    RuntimeRateAttestationError,
    hdf5_rate_attributes,
    load_runtime_rate_attestation,
)


AUDIT_CONTRACT_ID = "coffee_transport_physical_audit_v2"
TASK_IDS = ("000001", "000002")
CAN_NAMES = ("coffee_can_right", "coffee_can_left")
EXPECTED_SCHEMA = "cyclo_lab_hdf5_v1"
EXPECTED_ORIGIN = "coffee_transport_physical_trajectory_generation"
EXPECTED_METHODS = {
    "000001": "coffee_two_can_endpoint_lift_targets_pending_validation_v13",
    "000002": "coffee_two_can_production_realized_initial_commanded_joint_hold_v12",
}
EXPECTED_INITIAL_NORMALIZATION_CONTRACTS = {
    "000001": "coffee_realized_pose_endpoint_lift_targets_v13",
    "000002": "coffee_production_realized_pose_commanded_joint_hold_v12",
}
EXPECTED_GENERATION_PROFILES = {
    "000001": "coffee_physical_trajectory_generation_endpoint_lift_targets_v13",
    "000002": "coffee_physical_trajectory_generation_commanded_joint_hold_v12",
}
EXPECTED_PRODUCTION_PROMOTIONS = {
    "000001": "coffee_endpoint_lift_targets_v13_reviewed",
    "000002": "coffee_commanded_joint_hold_v12_promoted_20260908",
}
EXPECTED_COLLISION_PROFILE = "cylinder_r33p5_h90_v1"
EXPECTED_COLLISION_ASSET_SHA256 = (
    "fe340f411e93fd3fd54bc06b1b4ed1e6dec24214179e69ae5dd6ce1dd50d4a82"
)
EXPECTED_GRASP_CORRECTION = {
    "grasp_correction_can_x_m": -0.020,
    "grasp_correction_can_z_m": 0.030,
    "grasp_correction_blend_steps": 30.0,
}
EXPECTED_SUCCESS_CRITERION = "coffee_two_can_carry_and_placement_quality_gate_v3"
EXPECTED_PHASE_VERSIONS = {"000001": 7, "000002": 6}
EXPECTED_FPS = 15.0
EXPECTED_TASK_AXIS_ACTION_CONTRACTS = {
    "000001": "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1",
    "000002": "base_body_velocity_feedback_v1",
}

ROBOT_NOMINAL_XY_M = (-1.47138, 0.80011813)
ROBOT_NOMINAL_YAW_RAD = math.pi
ROBOT_HOME_XY_LIMIT_M = 0.020
ROBOT_HOME_YAW_LIMIT_RAD = math.radians(1.0)
LIFT_RANGE_M = (-0.28, 0.0)
LIFT_ACTION_ENDPOINT_TOLERANCE_M = 1.0e-6
CABINET_RIGHT_DELTA_Y_M = 0.815072039386852
SOURCE_CAN_Z_M = 1.350128932
DESTINATION_LOWER_CAN_Z_M = 1.062119288
PLACEMENT_XY_TOLERANCE_M = 0.006
PLACEMENT_Z_TOLERANCE_M = 0.010
PLACEMENT_TILT_LIMIT_RAD = math.radians(10.0)
PLACEMENT_LINEAR_SPEED_LIMIT_MPS = 0.030
PLACEMENT_ANGULAR_SPEED_LIMIT_RADPS = 0.250
RELEASED_EEF_DISTANCE_MIN_M = 0.100
HEAD_STATIC_LIMIT_RAD = 0.005
TASK_AXIS_ENDPOINT_TOLERANCE_M = 0.025
ROOT_ROUTE_X_LIMIT_M = 0.030
ROOT_ROUTE_Y_MARGIN_M = 0.030
ROOT_ROUTE_YAW_LIMIT_RAD = math.radians(2.0)
ROOT_TILT_LIMIT_RAD = math.radians(3.0)
ROOT_Z_LIMIT_M = 0.010
CAN_MOVEMENT_ONSET_M = 0.030

# A large reset discontinuity is suspicious only when it is also much larger
# than the following render deltas. Exact previous-final reuse is always fatal.
FRAME0_JUMP_ABS_LIMIT = 45.0
FRAME0_JUMP_RATIO_LIMIT = 4.0

SOURCE_RANGES_M = {
    "coffee_can_right": (-2.20137, -2.14137, 0.93394, 1.05394),
    "coffee_can_left": (-2.20137, -2.14137, 0.52042, 0.64042),
}
CAMERA_SHAPES = {
    "cam_head": (376, 672, 3),
    "cam_wrist_left": (640, 480, 3),
    "cam_wrist_right": (640, 480, 3),
}
EEF_ACTION_NAMES = (
    "left_eef_x", "left_eef_y", "left_eef_z",
    "left_eef_qw", "left_eef_qx", "left_eef_qy", "left_eef_qz",
    "gripper_l_joint1",
    "right_eef_x", "right_eef_y", "right_eef_z",
    "right_eef_qw", "right_eef_qx", "right_eef_qy", "right_eef_qz",
    "gripper_r_joint1", "head_joint1", "head_joint2", "lift_joint",
)
BASE_ACTION_NAMES = ("linear_x", "linear_y", "angular_z")
FAILURE_RESPONSE_PROTOCOL = (
    {
        "order": 1,
        "trigger": "physical augmentation failure",
        "action_ko": "현재 contact/collision을 유지하고 동기화 grasp correction을 먼저 적용한다.",
    },
    {
        "order": 2,
        "trigger": "step 1 remains insufficient",
        "action_ko": "격리된 12각기둥 can main-body collision mesh A/B를 실행한다. 원본은 덮어쓰지 않고 variant ID, seed, 성공률을 기록한다.",
    },
    {
        "order": 3,
        "trigger": "trajectory augmentation has failed for about 8 hours",
        "action_ko": "아침에 변경된 contact 방식의 롤백을 검토한다. 8시간 전에는 롤백하지 않는다.",
    },
)

REQUIRED_GRASP_DIAGNOSTICS = (
    "close_onset_step",
    "close_frame_count",
    "open_gripper_value",
    "closed_gripper_value",
    "diagnostic_eef_can_distance_m",
    "diagnostic_can_local_eef_x_m",
    "diagnostic_can_local_eef_y_m",
    "diagnostic_can_local_eef_z_m",
    "diagnostic_source_can_local_eef_x_m",
    "diagnostic_source_can_local_eef_y_m",
    "diagnostic_source_can_local_eef_z_m",
    "diagnostic_can_local_error_x_m",
    "diagnostic_can_local_error_y_m",
    "diagnostic_can_local_error_z_m",
    "diagnostic_orientation_error_rad",
)
REQUIRED_CARRY_METRICS = (
    "eef_object_distance_m",
    "source_eef_object_distance_m",
    "tcp_z_relative_to_can_m",
    "can_displacement_m",
    "home_joint_max_error",
    "can_linear_speed_mps",
    "can_angular_speed_radps",
)
REQUIRED_PLACEMENT_METRICS = (
    "center_x_m",
    "center_y_m",
    "origin_z_m",
    "expected_origin_z_m",
    "upright_cosine",
    "can_linear_speed_mps",
    "can_angular_speed_radps",
    "active_eef_distance_m",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--task-id", choices=TASK_IDS, required=True)
    parser.add_argument("--expected-successes", type=int, default=50)
    parser.add_argument("--output-hdf5", type=Path)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--runtime-rate-attestation",
        type=Path,
        required=True,
        help=(
            "Reviewed, versioned manifest attesting the exact raw HDF hashes and "
            "their actual physics/control/camera process rates."
        ),
    )
    parser.add_argument(
        "--allow-rejected",
        action="store_true",
        help="Allow rejected attempts when at least the requested number pass.",
    )
    parser.add_argument(
        "--require-fixed-base-stationarity",
        action="store_true",
        help=(
            "Task000001 policy curation only: reject root XY excursions over 3 cm "
            "because the 17D arms+grippers+lift policy excludes base state/action."
        ),
    )
    return parser.parse_args()


def _scalar(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.generic):
        return value.item()
    return value


def _text(value: Any) -> str:
    return str(_scalar(value))


def _close(actual: float, expected: float, atol: float = 1.0e-6) -> bool:
    return bool(np.isclose(actual, expected, rtol=0.0, atol=atol))


def _within(value: float, lower: float, upper: float, tolerance: float = 1.0e-6) -> bool:
    return lower - tolerance <= value <= upper + tolerance


def _wrap_pi(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _yaw_wxyz(quaternion: np.ndarray) -> float:
    w, x, y, z = (float(value) for value in quaternion)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _upright_cosine_wxyz(quaternion: np.ndarray) -> float:
    q = np.asarray(quaternion, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if not math.isfinite(norm) or norm <= 0.0:
        return float("nan")
    _, x, y, _ = q / norm
    return float(1.0 - 2.0 * (x * x + y * y))


def _root_tilt_rad(quaternions: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternions, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    normalized = values / np.maximum(norms, 1.0e-12)
    x = normalized[:, 1]
    y = normalized[:, 2]
    upright = np.clip(1.0 - 2.0 * (x * x + y * y), -1.0, 1.0)
    return np.arccos(upright)


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Rotate one vector by one WXYZ quaternion without an Isaac dependency."""

    quaternion = np.asarray(quaternion, dtype=np.float64)
    quaternion /= max(float(np.linalg.norm(quaternion)), 1.0e-12)
    scalar = quaternion[0]
    imaginary = quaternion[1:]
    return (
        vector
        + 2.0 * scalar * np.cross(imaginary, vector)
        + 2.0 * np.cross(imaginary, np.cross(imaginary, vector))
    )


def _dataset_finite(dataset: h5py.Dataset, block_rows: int = 128) -> bool:
    if not (
        np.issubdtype(dataset.dtype, np.floating)
        or np.issubdtype(dataset.dtype, np.complexfloating)
    ):
        return True
    if not dataset.shape:
        return bool(np.isfinite(np.asarray(dataset)[()]).all())
    for start in range(0, dataset.shape[0], block_rows):
        if not np.isfinite(np.asarray(dataset[start : start + block_rows])).all():
            return False
    return True


def _frame_hash(frame: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _frame0_camera_metrics(dataset: h5py.Dataset) -> dict[str, Any]:
    count = min(int(dataset.shape[0]), 8)
    frames = [np.asarray(dataset[index]) for index in range(count)]
    first = frames[0]
    deltas = [
        float(np.mean(np.abs(frames[index + 1].astype(np.float32) - frames[index].astype(np.float32))))
        for index in range(len(frames) - 1)
    ]
    baseline = float(np.median(deltas[1:])) if len(deltas) > 1 else 0.0
    ratio = deltas[0] / max(baseline, 1.0e-6) if deltas else 0.0
    return {
        "frame0_hash": _frame_hash(first),
        "last_frame_hash": _frame_hash(np.asarray(dataset[-1])),
        "frame0_mean": float(np.mean(first)),
        "frame0_std": float(np.std(first)),
        "frame0_range": int(np.max(first)) - int(np.min(first)),
        "frame0_to_frame1_mad": deltas[0] if deltas else 0.0,
        "following_delta_median": baseline,
        "frame0_jump_ratio": ratio,
    }


def _required_numeric_attrs(
    demo: h5py.Group,
    names: tuple[str, ...],
    reasons: list[str],
) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in names:
        if name not in demo.attrs:
            reasons.append(f"missing numeric episode attribute {name}")
            continue
        try:
            value = float(np.asarray(demo.attrs[name]).reshape(-1)[0])
        except (TypeError, ValueError):
            reasons.append(f"episode attribute {name} is not numeric")
            continue
        if not math.isfinite(value):
            reasons.append(f"episode attribute {name} is not finite")
            continue
        values[name] = value
    return values


def _first_movement_frame(poses: np.ndarray, initial_xyz: np.ndarray) -> int | None:
    displacement = np.linalg.norm(poses[:, :3] - initial_xyz[:3], axis=1)
    indices = np.flatnonzero(displacement > CAN_MOVEMENT_ONSET_M)
    return int(indices[0]) if len(indices) else None


def _check_common_schema(demo: h5py.Group, task_id: str, reasons: list[str]) -> int:
    action_width = 19 if task_id == "000001" else 22
    required_shapes = {
        "actions": action_width,
        "obs/actions": action_width,
        "processed_actions": action_width,
        "obs/joint_pos": 19,
        "obs/joint_pos_target": 19,
        "obs/left_eef_pose": 7,
        "obs/right_eef_pose": 7,
        "states/articulation/robot/joint_position": 31,
        "states/articulation/robot/joint_velocity": 31,
        "states/articulation/robot/root_pose": 7,
        "states/articulation/robot/root_velocity": 6,
    }
    for can_name in CAN_NAMES:
        required_shapes[f"states/rigid_object/{can_name}/root_pose"] = 7
        required_shapes[f"states/rigid_object/{can_name}/root_velocity"] = 6
    missing = [path for path in required_shapes if path not in demo]
    if missing:
        reasons.append(f"missing required episode datasets: {missing}")
        return -1
    actions = demo["actions"]
    frames = int(actions.shape[0]) if actions.ndim == 2 else -1
    for path, width in required_shapes.items():
        shape = demo[path].shape
        if len(shape) != 2 or shape != (frames, width):
            reasons.append(f"{path} shape {shape} is not ({frames}, {width})")
    if frames < 3:
        reasons.append(f"episode has only {frames} frames")
    if int(demo.attrs.get("num_samples", -1)) != frames:
        reasons.append("num_samples does not match actions frame count")

    initial_shapes = {
        "initial_state/articulation/robot/root_pose": (1, 7),
        "initial_state/articulation/robot/root_velocity": (1, 6),
        "initial_state/articulation/robot/joint_position": (1, 31),
        "initial_state/articulation/robot/joint_velocity": (1, 31),
        "initial_state/rigid_object/coffee_can_right/root_pose": (1, 7),
        "initial_state/rigid_object/coffee_can_right/root_velocity": (1, 6),
        "initial_state/rigid_object/coffee_can_left/root_pose": (1, 7),
        "initial_state/rigid_object/coffee_can_left/root_velocity": (1, 6),
    }
    for path, shape in initial_shapes.items():
        if path not in demo:
            reasons.append(f"missing required initial-state dataset {path}")
        elif demo[path].shape != shape:
            reasons.append(f"{path} shape {demo[path].shape} is not {shape}")

    def visit(name: str, item: Any) -> None:
        if not isinstance(item, h5py.Dataset):
            return
        if not name.startswith("initial_state/") and item.shape and item.shape[0] != frames:
            reasons.append(f"time-aligned dataset {name} has {item.shape[0]} rows, expected {frames}")
        if not _dataset_finite(item):
            reasons.append(f"dataset {name} contains non-finite values")

    demo.visititems(visit)
    return frames


def _check_cameras(
    demo: h5py.Group,
    frames: int,
    previous_head_last_hash: str | None,
    reasons: list[str],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for camera, expected_hwc in CAMERA_SHAPES.items():
        path = f"obs/{camera}"
        if path not in demo:
            if camera == "cam_head":
                reasons.append("missing required head camera obs/cam_head")
            continue
        dataset = demo[path]
        expected = (frames, *expected_hwc)
        if dataset.shape != expected:
            reasons.append(f"{path} shape {dataset.shape} is not {expected}")
            continue
        if dataset.dtype != np.dtype("uint8"):
            reasons.append(f"{path} dtype {dataset.dtype} is not uint8")
            continue
        camera_metrics = _frame0_camera_metrics(dataset)
        metrics[camera] = camera_metrics
        if camera_metrics["frame0_range"] < 8 or camera_metrics["frame0_std"] < 2.0:
            reasons.append(f"{path} frame 0 is blank or nearly constant")
    head = metrics.get("cam_head")
    if head is not None:
        if previous_head_last_hash and head["frame0_hash"] == previous_head_last_hash:
            reasons.append("cam_head frame 0 exactly reuses the previous episode's last frame")
        if (
            head["frame0_to_frame1_mad"] > FRAME0_JUMP_ABS_LIMIT
            and head["frame0_jump_ratio"] > FRAME0_JUMP_RATIO_LIMIT
        ):
            reasons.append(
                "cam_head frame 0 to 1 jump is an isolated reset outlier "
                f"(MAD={head['frame0_to_frame1_mad']:.3f}, ratio={head['frame0_jump_ratio']:.2f})"
            )
    return metrics


def _check_episode_identity(demo: h5py.Group, task_id: str, reasons: list[str]) -> None:
    checks = {
        "task_id": task_id,
        "generation_method": EXPECTED_METHODS[task_id],
        "success_criterion_id": EXPECTED_SUCCESS_CRITERION,
        "initial_normalization_contract_id": EXPECTED_INITIAL_NORMALIZATION_CONTRACTS[task_id],
        "generation_profile_id": EXPECTED_GENERATION_PROFILES[task_id],
        "production_promotion_id": EXPECTED_PRODUCTION_PROMOTIONS[task_id],
        "coffee_can_collision_profile_id": EXPECTED_COLLISION_PROFILE,
        "coffee_can_collision_asset_sha256": EXPECTED_COLLISION_ASSET_SHA256,
        "task_axis_action_contract": EXPECTED_TASK_AXIS_ACTION_CONTRACTS[task_id],
    }
    for key, expected in checks.items():
        if _text(demo.attrs.get(key, "")) != expected:
            reasons.append(f"{key} is not {expected!r}")
    if _scalar(demo.attrs.get("success", False)) is not True:
        reasons.append("success attribute is not true")
    if _scalar(demo.attrs.get("production_approved", False)) is not True:
        reasons.append("production_approved attribute is not true")
    if task_id == "000001" and _text(
        demo.attrs.get("production_validation_status", "")
    ) != "reviewed_endpoint_lift_targets_v13":
        reasons.append(
            "production_validation_status is not "
            "'reviewed_endpoint_lift_targets_v13'"
        )
    if _scalar(demo.attrs.get("coffee_can_collision_12gon_applied", True)) is not False:
        reasons.append("coffee_can_collision_12gon_applied attribute is not false")
    if _text(demo.attrs.get("failure_reason", "")):
        reasons.append("failure_reason is not empty")
    expected_phase_version = EXPECTED_PHASE_VERSIONS[task_id]
    if int(demo.attrs.get("coffee_phase_contract_version", -1)) != expected_phase_version:
        reasons.append(
            f"coffee_phase_contract_version is not {expected_phase_version}"
        )
    for key, expected in EXPECTED_GRASP_CORRECTION.items():
        try:
            actual = float(demo.attrs[key])
        except (KeyError, TypeError, ValueError):
            reasons.append(f"missing or invalid {key}")
            continue
        if not math.isfinite(actual) or not _close(actual, expected, 1.0e-9):
            reasons.append(f"{key}={actual} is not campaign value {expected}")


def _pose_and_randomization(
    demo: h5py.Group,
    task_id: str,
    reasons: list[str],
) -> dict[str, float]:
    required = (
        "robot_spawn_x_m",
        "robot_spawn_y_m",
        "robot_spawn_yaw_rad",
        "robot_home_x_m",
        "robot_home_y_m",
        "robot_home_yaw_rad",
        "lift_spawn_m",
        "base_route_spawn_offset_m",
        *(f"{can}_{axis}" for can in CAN_NAMES for axis in ("x_m", "y_m", "yaw_rad")),
    )
    values = _required_numeric_attrs(demo, tuple(required), reasons)
    if len(values) != len(required):
        return values

    root = np.asarray(demo["initial_state/articulation/robot/root_pose"])[0]
    if not _close(float(np.linalg.norm(root[3:])), 1.0, 1.0e-3):
        reasons.append("initial robot root quaternion is not normalized")
    observed_root = (float(root[0]), float(root[1]), _yaw_wxyz(root[3:]))
    for key, observed in zip(
        ("robot_spawn_x_m", "robot_spawn_y_m", "robot_spawn_yaw_rad"), observed_root
    ):
        delta = _wrap_pi(values[key] - observed) if key.endswith("yaw_rad") else values[key] - observed
        if abs(delta) > 1.0e-5:
            reasons.append(f"{key} disagrees with immutable initial robot pose")

    home_dx = values["robot_home_x_m"] - ROBOT_NOMINAL_XY_M[0]
    home_dy = values["robot_home_y_m"] - ROBOT_NOMINAL_XY_M[1]
    home_dyaw = _wrap_pi(values["robot_home_yaw_rad"] - ROBOT_NOMINAL_YAW_RAD)
    values["robot_home_delta_x_m"] = home_dx
    values["robot_home_delta_y_m"] = home_dy
    values["robot_home_delta_yaw_rad"] = home_dyaw
    if abs(home_dx) > ROBOT_HOME_XY_LIMIT_M + 1.0e-6:
        reasons.append("robot HOME X is outside nominal +/-2 cm")
    if abs(home_dy) > ROBOT_HOME_XY_LIMIT_M + 1.0e-6:
        reasons.append("robot HOME Y is outside nominal +/-2 cm")
    if abs(home_dyaw) > ROBOT_HOME_YAW_LIMIT_RAD + 1.0e-6:
        reasons.append("robot HOME yaw is outside nominal +/-1 degree")

    for can_name in CAN_NAMES:
        pose = np.asarray(demo[f"initial_state/rigid_object/{can_name}/root_pose"])[0]
        if not _close(float(np.linalg.norm(pose[3:])), 1.0, 1.0e-3):
            reasons.append(f"{can_name} initial quaternion is not normalized")
        observed = (float(pose[0]), float(pose[1]), _yaw_wxyz(pose[3:]))
        for key, actual in zip(
            (f"{can_name}_x_m", f"{can_name}_y_m", f"{can_name}_yaw_rad"), observed
        ):
            delta = _wrap_pi(values[key] - actual) if key.endswith("yaw_rad") else values[key] - actual
            if abs(delta) > 1.0e-5:
                reasons.append(f"{key} disagrees with immutable initial can pose")
        x_min, x_max, y_min, y_max = SOURCE_RANGES_M[can_name]
        if not _within(observed[0], x_min, x_max):
            reasons.append(f"{can_name} initial X is outside the active 6 cm depth")
        if not _within(observed[1], y_min, y_max):
            reasons.append(f"{can_name} initial Y is outside its 12 cm source range")
        if abs(float(pose[2]) - SOURCE_CAN_Z_M) > PLACEMENT_Z_TOLERANCE_M:
            reasons.append(f"{can_name} initial Z is not on the source support")

    spawn_x, spawn_y = values["robot_spawn_x_m"], values["robot_spawn_y_m"]
    spawn_yaw = values["robot_spawn_yaw_rad"]
    home_x, home_y = values["robot_home_x_m"], values["robot_home_y_m"]
    home_yaw = values["robot_home_yaw_rad"]
    route_offset = values["base_route_spawn_offset_m"]
    if task_id == "000001":
        if not _within(values["lift_spawn_m"], *LIFT_RANGE_M):
            reasons.append("lift spawn is outside [-0.28, 0] m")
        if abs(route_offset) > 1.0e-6:
            reasons.append("fixed-base task has a non-zero route spawn offset")
        if math.hypot(spawn_x - home_x, spawn_y - home_y) > 1.0e-5 or abs(_wrap_pi(spawn_yaw - home_yaw)) > 1.0e-5:
            reasons.append("Task000001 robot spawn does not equal episode HOME")
    else:
        if abs(values["lift_spawn_m"]) > 0.005:
            reasons.append("Task000002 lift spawn is not nominal")
        if not _within(route_offset, 0.0, CABINET_RIGHT_DELTA_Y_M):
            reasons.append("base route spawn offset is outside [HOME, right cabinet]")
        if abs(spawn_x - home_x) > 1.0e-5:
            reasons.append("Task000002 route spawn changes world X")
        if abs((spawn_y - home_y) - route_offset) > 1.0e-5:
            reasons.append("base_route_spawn_offset_m disagrees with spawn minus HOME Y")
        if abs(_wrap_pi(spawn_yaw - home_yaw)) > 1.0e-5:
            reasons.append("Task000002 route spawn yaw differs from episode HOME")
    return values


def _quality_metrics(
    demo: h5py.Group,
    reasons: list[str],
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    grasp_names = tuple(
        f"quality_{can}_grasp_{metric}"
        for can in CAN_NAMES
        for metric in REQUIRED_GRASP_DIAGNOSTICS
    )
    carry_names = tuple(
        f"quality_{can}_carry_{metric}"
        for can in CAN_NAMES
        for metric in REQUIRED_CARRY_METRICS
    )
    placement_names = tuple(
        f"quality_{can}_placement_{metric}"
        for can in CAN_NAMES
        for metric in REQUIRED_PLACEMENT_METRICS
    )
    grasp = _required_numeric_attrs(demo, grasp_names, reasons)
    carry = _required_numeric_attrs(demo, carry_names, reasons)
    placement = _required_numeric_attrs(demo, placement_names, reasons)

    # Carry stability is diagnostic in the working generator. Geometry, clear,
    # and HOME posture remain its actual acceptance contract.
    for can in CAN_NAMES:
        prefix = f"quality_{can}_carry_"
        if all(prefix + name in carry for name in REQUIRED_CARRY_METRICS):
            current = carry[prefix + "eef_object_distance_m"]
            source = carry[prefix + "source_eef_object_distance_m"]
            limit = min(0.100, max(0.080, source + 0.025))
            if current < 0.0 or current > limit + 1.0e-9:
                reasons.append(f"{can} carry EEF distance {current:.6f} exceeds {limit:.6f}")
            tcp_z = carry[prefix + "tcp_z_relative_to_can_m"]
            if not _within(tcp_z, -0.030, 0.050):
                reasons.append(f"{can} carry TCP Z is outside the body band")
            if carry[prefix + "can_displacement_m"] < 0.200:
                reasons.append(f"{can} did not clear the source by 0.2 m")
            home_error = carry[prefix + "home_joint_max_error"]
            if not _within(home_error, 0.0, 0.150):
                reasons.append(f"{can} carry HOME joint error is outside [0, 0.15]")
    return grasp, carry, placement


def _check_final_placements(
    demo: h5py.Group,
    task_id: str,
    placement: dict[str, float],
    reasons: list[str],
) -> dict[str, float]:
    result: dict[str, float] = {}
    destination_dy = 0.0 if task_id == "000001" else CABINET_RIGHT_DELTA_Y_M
    expected_z = DESTINATION_LOWER_CAN_Z_M if task_id == "000001" else SOURCE_CAN_Z_M
    for can in CAN_NAMES:
        pose = np.asarray(demo[f"states/rigid_object/{can}/root_pose"])[-1]
        velocity = np.asarray(demo[f"states/rigid_object/{can}/root_velocity"])[-1]
        x, y, z = (float(value) for value in pose[:3])
        upright = _upright_cosine_wxyz(pose[3:])
        linear_speed = float(np.linalg.norm(velocity[:3]))
        angular_speed = float(np.linalg.norm(velocity[3:]))
        prefix = f"{can}_final_"
        result.update(
            {
                prefix + "x_m": x,
                prefix + "y_m": y,
                prefix + "z_m": z,
                prefix + "upright_cosine": upright,
                prefix + "linear_speed_mps": linear_speed,
                prefix + "angular_speed_radps": angular_speed,
            }
        )
        x_min, x_max, y_min, y_max = SOURCE_RANGES_M[can]
        y_min += destination_dy
        y_max += destination_dy
        if not (
            _within(x, x_min, x_max, PLACEMENT_XY_TOLERANCE_M)
            and _within(y, y_min, y_max, PLACEMENT_XY_TOLERANCE_M)
        ):
            reasons.append(f"{can} final center is outside its named destination region")
        if abs(z - expected_z) > PLACEMENT_Z_TOLERANCE_M:
            reasons.append(f"{can} final Z is not on the destination support")
        if upright < math.cos(PLACEMENT_TILT_LIMIT_RAD):
            reasons.append(f"{can} final pose is not upright")
        if linear_speed > PLACEMENT_LINEAR_SPEED_LIMIT_MPS:
            reasons.append(f"{can} final linear speed is not settled")
        if angular_speed > PLACEMENT_ANGULAR_SPEED_LIMIT_RADPS:
            reasons.append(f"{can} final angular speed is not settled")

        attr_prefix = f"quality_{can}_placement_"
        released = placement.get(attr_prefix + "active_eef_distance_m")
        if released is not None and released < RELEASED_EEF_DISTANCE_MIN_M:
            reasons.append(f"{can} active EEF was not released at placement")
    return result


def _check_placement_checkpoints(
    demo: h5py.Group,
    task_id: str,
    placement: dict[str, float],
    reasons: list[str],
) -> dict[str, Any]:
    """Align stored placement metrics to their recorder post-step checkpoint.

    ``placement_quality`` samples scene state immediately after ``step_checked``.
    That scene state is HDF row ``t`` while the corresponding pre-step EEF
    observation is row ``t + 1``. The episode continues through arm retreat and
    lift/base return, so comparing these attrs with the final row is invalid.
    """

    result: dict[str, Any] = {}
    destination_dy = 0.0 if task_id == "000001" else CABINET_RIGHT_DELTA_Y_M
    expected_z = DESTINATION_LOWER_CAN_Z_M if task_id == "000001" else SOURCE_CAN_Z_M
    root = np.asarray(demo["states/articulation/robot/root_pose"])
    for can in CAN_NAMES:
        prefix = f"quality_{can}_placement_"
        if any(prefix + metric not in placement for metric in REQUIRED_PLACEMENT_METRICS):
            continue
        pose = np.asarray(demo[f"states/rigid_object/{can}/root_pose"])
        velocity = np.asarray(demo[f"states/rigid_object/{can}/root_velocity"])
        target_xyz = np.asarray(
            [
                placement[prefix + "center_x_m"],
                placement[prefix + "center_y_m"],
                placement[prefix + "origin_z_m"],
            ],
            dtype=np.float64,
        )
        stored_upright = placement[prefix + "upright_cosine"]
        stored_linear = placement[prefix + "can_linear_speed_mps"]
        stored_angular = placement[prefix + "can_angular_speed_radps"]
        stored_eef_distance = placement[prefix + "active_eef_distance_m"]
        upright = np.asarray([_upright_cosine_wxyz(q) for q in pose[:, 3:]])
        linear_speed = np.linalg.norm(velocity[:, :3], axis=1)
        angular_speed = np.linalg.norm(velocity[:, 3:], axis=1)
        candidate_mask = (
            (np.max(np.abs(pose[:, :3] - target_xyz), axis=1) <= 1.0e-5)
            & (np.abs(upright - stored_upright) <= 1.0e-5)
            & (np.abs(linear_speed - stored_linear) <= 1.0e-5)
            & (np.abs(angular_speed - stored_angular) <= 1.0e-5)
        )
        # state[t] is paired with pre-step observation[t + 1].
        candidate_mask[-1] = False
        candidates = np.flatnonzero(candidate_mask)
        side = "right" if can.endswith("right") else "left"
        eef_relative = np.asarray(demo[f"obs/{side}_eef_pose"])
        matches: list[tuple[int, float]] = []
        for index in candidates:
            world_eef = root[index, :3] + _quat_rotate_wxyz(
                root[index, 3:], eef_relative[index + 1, :3]
            )
            observed_distance = float(np.linalg.norm(world_eef - pose[index, :3]))
            if abs(observed_distance - stored_eef_distance) <= 1.0e-4:
                matches.append((int(index), observed_distance))
        result[can] = {
            "matching_post_step_frames": [index for index, _ in matches],
            "selected_post_step_frame": matches[-1][0] if matches else None,
            "paired_eef_pre_step_frame": matches[-1][0] + 1 if matches else None,
            "recomputed_active_eef_distance_m": matches[-1][1] if matches else None,
        }
        if not matches:
            reasons.append(
                f"{can} placement metadata cannot be aligned to any state[t]/EEF[t+1] checkpoint"
            )

        x_min, x_max, y_min, y_max = SOURCE_RANGES_M[can]
        y_min += destination_dy
        y_max += destination_dy
        if not (
            _within(target_xyz[0], x_min, x_max, PLACEMENT_XY_TOLERANCE_M)
            and _within(target_xyz[1], y_min, y_max, PLACEMENT_XY_TOLERANCE_M)
        ):
            reasons.append(f"{can} placement checkpoint is outside its named region")
        if not _close(placement[prefix + "expected_origin_z_m"], expected_z, 1.0e-6):
            reasons.append(f"{can} placement checkpoint has the wrong expected support Z")
        if abs(target_xyz[2] - expected_z) > PLACEMENT_Z_TOLERANCE_M:
            reasons.append(f"{can} placement checkpoint is not on destination support")
        if stored_upright < math.cos(PLACEMENT_TILT_LIMIT_RAD):
            reasons.append(f"{can} placement checkpoint is not upright")
        if stored_linear > PLACEMENT_LINEAR_SPEED_LIMIT_MPS:
            reasons.append(f"{can} placement checkpoint linear speed is not settled")
        if stored_angular > PLACEMENT_ANGULAR_SPEED_LIMIT_RADPS:
            reasons.append(f"{can} placement checkpoint angular speed is not settled")
        if stored_eef_distance < RELEASED_EEF_DISTANCE_MIN_M:
            reasons.append(f"{can} active EEF is not released at the placement checkpoint")
    return result


def _check_head_static(demo: h5py.Group, reasons: list[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for path in ("actions", "obs/joint_pos", "obs/joint_pos_target"):
        values = np.asarray(demo[path])[:, 16:18]
        peak = float(np.max(np.abs(values - values[0])))
        result[path.replace("/", "_") + "_head_peak_delta_rad"] = peak
        if peak > HEAD_STATIC_LIMIT_RAD:
            reasons.append(f"{path} head changes by {peak:.6f} rad, over {HEAD_STATIC_LIMIT_RAD}")
    return result


def _check_lift_action_endpoints(
    demo: h5py.Group,
    reasons: list[str],
) -> dict[str, dict[str, float | int]]:
    """Reject Task000001 labels that teach intermediate absolute targets."""

    result: dict[str, dict[str, float | int]] = {}
    for path, start_frame in (
        ("actions", 0),
        ("processed_actions", 0),
        # ACT pairs state[t] with joint_pos_target[t + 1]. The reset target at
        # index 0 is not a policy label.
        ("obs/joint_pos_target", 1),
    ):
        values = np.asarray(demo[path])[start_frame:, 18].astype(
            np.float64, copy=False
        )
        endpoint_distance = np.minimum(
            np.abs(values),
            np.abs(values - LIFT_RANGE_M[0]),
        )
        invalid = endpoint_distance > LIFT_ACTION_ENDPOINT_TOLERANCE_M
        result[path.replace("/", "_")] = {
            "invalid_count": int(np.count_nonzero(invalid)),
            "max_endpoint_error_m": float(np.max(endpoint_distance)),
        }
        if np.any(invalid):
            first = int(np.flatnonzero(invalid)[0]) + start_frame
            reasons.append(
                f"{path} lift action contains an intermediate absolute target "
                f"at frame {first}: {float(demo[path][first, 18]):.9f}"
            )
    return result


def _check_task_route(
    demo: h5py.Group,
    task_id: str,
    poses: dict[str, float],
    reasons: list[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    initial_can = {
        can: np.asarray(demo[f"initial_state/rigid_object/{can}/root_pose"])[0]
        for can in CAN_NAMES
    }
    can_poses = {
        can: np.asarray(demo[f"states/rigid_object/{can}/root_pose"])
        for can in CAN_NAMES
    }
    onsets = {can: _first_movement_frame(can_poses[can], initial_can[can]) for can in CAN_NAMES}
    result["can_movement_onset_frames"] = onsets
    right = onsets["coffee_can_right"]
    left = onsets["coffee_can_left"]
    if right is None:
        reasons.append("right can never moves 3 cm from its initial pose")
    if left is None:
        reasons.append("left can never moves 3 cm from its initial pose")
    if right is None or left is None:
        return result
    if right >= left:
        reasons.append("can movement order is not right then left")

    joint = np.asarray(demo["obs/joint_pos"])
    root = np.asarray(demo["states/articulation/robot/root_pose"])
    home_x = poses.get("robot_home_x_m", float("nan"))
    home_y = poses.get("robot_home_y_m", float("nan"))
    home_yaw = poses.get("robot_home_yaw_rad", float("nan"))
    root_xy_distance_home = np.linalg.norm(root[:, :2] - np.array((home_x, home_y)), axis=1)
    root_yaw_error = np.asarray([abs(_wrap_pi(_yaw_wxyz(q) - home_yaw)) for q in root[:, 3:]])
    home_mask = (root_xy_distance_home <= TASK_AXIS_ENDPOINT_TOLERANCE_M) & (
        root_yaw_error <= ROOT_ROUTE_YAW_LIMIT_RAD
    )

    if task_id == "000001":
        axis = joint[:, 18]
        destination_mask = np.abs(axis + 0.28) <= TASK_AXIS_ENDPOINT_TOLERANCE_M
        source_mask = np.abs(axis) <= TASK_AXIS_ENDPOINT_TOLERANCE_M
        result.update(
            {
                "lift_min_m": float(np.min(axis)),
                "lift_max_m": float(np.max(axis)),
                "lift_final_m": float(axis[-1]),
            }
        )
        if abs(float(axis[0]) - poses.get("lift_spawn_m", float("nan"))) > 0.010:
            reasons.append("first observed lift does not match lift_spawn_m")
        if not np.any(source_mask[:right]):
            reasons.append("lift does not normalize to 0 before the right-can pickup")
        if not np.any(destination_mask[right:left]):
            reasons.append("lift does not reach -0.28 m for the right-can cycle")
        if not np.any(source_mask[right:left]):
            reasons.append("lift does not return to 0 between can cycles")
        if not np.any(destination_mask[left:]):
            reasons.append("lift does not reach -0.28 m for the left-can cycle")
        if not source_mask[-1]:
            reasons.append("lift does not finish at 0")
        initial_root_xy = np.asarray(
            demo["initial_state/articulation/robot/root_pose"]
        )[0, :2]
        root_xy_from_initial = np.linalg.norm(root[:, :2] - initial_root_xy, axis=1)
        root_xy_peak = float(np.max(root_xy_distance_home))
        result.update(
            {
                "fixed_base_root_xy_peak_from_home_m": root_xy_peak,
                "fixed_base_root_xy_final_from_home_m": float(root_xy_distance_home[-1]),
                "fixed_base_root_xy_peak_from_initial_m": float(np.max(root_xy_from_initial)),
                "fixed_base_root_drift_reference_m": ROOT_ROUTE_X_LIMIT_M,
                "fixed_base_root_drift_reference_exceeded": root_xy_peak > ROOT_ROUTE_X_LIMIT_M,
                "fixed_base_root_stationarity_is_acceptance_gate": False,
                "fixed_base_return_gate_applicable": False,
            }
        )
    else:
        destination_y = home_y + CABINET_RIGHT_DELTA_Y_M
        destination_mask = (
            np.abs(root[:, 1] - destination_y) <= TASK_AXIS_ENDPOINT_TOLERANCE_M
        ) & (np.abs(root[:, 0] - home_x) <= ROOT_ROUTE_X_LIMIT_M)
        result.update(
            {
                "root_x_peak_from_home_m": float(np.max(np.abs(root[:, 0] - home_x))),
                "root_y_min_from_home_m": float(np.min(root[:, 1] - home_y)),
                "root_y_max_from_home_m": float(np.max(root[:, 1] - home_y)),
                "root_final_xy_error_from_home_m": float(root_xy_distance_home[-1]),
                "root_yaw_peak_error_from_home_rad": float(np.max(root_yaw_error)),
                "root_tilt_peak_rad": float(np.max(_root_tilt_rad(root[:, 3:]))),
                "root_z_peak_abs_m": float(np.max(np.abs(root[:, 2]))),
            }
        )
        if not np.any(home_mask[:right]):
            reasons.append("base does not normalize to episode HOME before right pickup")
        if not np.any(destination_mask[right:left]):
            reasons.append("base does not reach the right cabinet for the right-can cycle")
        if not np.any(home_mask[right:left]):
            reasons.append("base does not return HOME between can cycles")
        if not np.any(destination_mask[left:]):
            reasons.append("base does not reach the right cabinet for the left-can cycle")
        if not home_mask[-1]:
            reasons.append("base does not finish at episode HOME")
        if result["root_x_peak_from_home_m"] > ROOT_ROUTE_X_LIMIT_M:
            reasons.append("base route strays more than 3 cm in world X")
        if result["root_y_min_from_home_m"] < -ROOT_ROUTE_Y_MARGIN_M:
            reasons.append("base route moves left of HOME by more than 3 cm")
        if result["root_y_max_from_home_m"] > CABINET_RIGHT_DELTA_Y_M + ROOT_ROUTE_Y_MARGIN_M:
            reasons.append("base route overshoots the right cabinet by more than 3 cm")
        if result["root_yaw_peak_error_from_home_rad"] > ROOT_ROUTE_YAW_LIMIT_RAD:
            reasons.append("base yaw changes more than 2 degrees on the lateral route")
        if result["root_tilt_peak_rad"] > ROOT_TILT_LIMIT_RAD:
            reasons.append("base roll/pitch tilt exceeds 3 degrees")
        if result["root_z_peak_abs_m"] > ROOT_Z_LIMIT_M:
            reasons.append("base root Z excursion exceeds 1 cm")
    return result


def audit_demo(
    name: str,
    demo: h5py.Group,
    *,
    task_id: str,
    previous_head_last_hash: str | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    _check_episode_identity(demo, task_id, reasons)
    frames = _check_common_schema(demo, task_id, reasons)
    if frames < 0:
        return {"demo": name, "accepted": False, "frames": frames, "reasons": reasons}
    cameras = _check_cameras(demo, frames, previous_head_last_hash, reasons)
    poses = _pose_and_randomization(demo, task_id, reasons)
    grasp, carry, placement = _quality_metrics(demo, reasons)
    placement_checkpoints = _check_placement_checkpoints(
        demo, task_id, placement, reasons
    )
    final = _check_final_placements(demo, task_id, placement, reasons)
    head = _check_head_static(demo, reasons)
    lift_action_endpoints = (
        _check_lift_action_endpoints(demo, reasons)
        if task_id == "000001"
        else {}
    )
    route = _check_task_route(demo, task_id, poses, reasons)
    return {
        "demo": name,
        "accepted": not reasons,
        "frames": frames,
        "reasons": reasons,
        "generation_seed": int(demo.attrs.get("generation_seed", -1)),
        "generation_attempt": int(demo.attrs.get("generation_attempt", -1)),
        "grasp_correction": {key: float(demo.attrs.get(key, float("nan"))) for key in EXPECTED_GRASP_CORRECTION},
        "poses": poses,
        "final": final,
        "placement_checkpoints": placement_checkpoints,
        "grasp_diagnostics": grasp,
        "carry_metrics": carry,
        "placement_metrics": placement,
        "head_static": head,
        "lift_action_endpoints": lift_action_endpoints,
        "task_route": route,
        "camera_sanity": cameras,
    }


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return int(name.rsplit("_", 1)[1]), name
    except (IndexError, ValueError):
        return 2**31 - 1, name


def _check_file_contract(data: h5py.Group, task_id: str) -> list[str]:
    errors: list[str] = []
    checks = {
        "schema_version": EXPECTED_SCHEMA,
        "dataset_origin": EXPECTED_ORIGIN,
        "generation_method": EXPECTED_METHODS[task_id],
        "success_criterion_id": EXPECTED_SUCCESS_CRITERION,
        "initial_normalization_contract_id": EXPECTED_INITIAL_NORMALIZATION_CONTRACTS[task_id],
        "generation_profile_id": EXPECTED_GENERATION_PROFILES[task_id],
        "production_promotion_id": EXPECTED_PRODUCTION_PROMOTIONS[task_id],
        "coffee_can_collision_profile_id": EXPECTED_COLLISION_PROFILE,
        "coffee_can_collision_asset_sha256": EXPECTED_COLLISION_ASSET_SHA256,
        "task_axis_action_contract": EXPECTED_TASK_AXIS_ACTION_CONTRACTS[task_id],
        "task_id": task_id,
        "observation_semantics": "pre_step",
        "scene_state_semantics": "post_step",
        "action_semantics": "pre_step_dual_eef_pose_wxyz_grippers_head_lift_plus_optional_body_velocity",
    }
    for key, expected in checks.items():
        if _text(data.attrs.get(key, "")) != expected:
            errors.append(f"/data {key} is not {expected!r}")
    if _scalar(data.attrs.get("production_approved", False)) is not True:
        errors.append("/data production_approved is not true")
    if task_id == "000001" and _text(
        data.attrs.get("production_validation_status", "")
    ) != "reviewed_endpoint_lift_targets_v13":
        errors.append(
            "/data production_validation_status is not "
            "'reviewed_endpoint_lift_targets_v13'"
        )
    if _scalar(data.attrs.get("coffee_can_collision_12gon_applied", True)) is not False:
        errors.append("/data coffee_can_collision_12gon_applied is not false")
    for key in ("control_hz", "camera_hz"):
        if not _close(float(data.attrs.get(key, float("nan"))), EXPECTED_FPS):
            errors.append(f"/data {key} is not {EXPECTED_FPS}")
    expected_action_names = EEF_ACTION_NAMES + (BASE_ACTION_NAMES if task_id == "000002" else ())
    try:
        action_names = tuple(json.loads(_text(data.attrs.get("action_names", ""))))
    except (TypeError, json.JSONDecodeError):
        action_names = ()
    if action_names != expected_action_names:
        errors.append("/data action_names do not match the canonical EEF action order")
    names = sorted(data.keys(), key=_demo_sort_key)
    if any(not name.startswith("demo_") for name in names):
        errors.append("/data contains non-demo children")
    return errors


def audit_inputs(
    paths: list[Path],
    task_id: str,
    expected_successes: int,
    allow_rejected: bool,
    *,
    require_fixed_base_stationarity: bool = False,
) -> dict[str, Any]:
    if require_fixed_base_stationarity and task_id != "000001":
        raise ValueError("--require-fixed-base-stationarity applies only to Task000001")
    records: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    previous_head_last_hash: str | None = None
    for path in paths:
        resolved = path.resolve()
        with h5py.File(resolved, "r") as handle:
            if "data" not in handle:
                file_reports.append({"source_file": str(resolved), "errors": ["missing /data group"]})
                continue
            data = handle["data"]
            file_errors = _check_file_contract(data, task_id)
            observed_total = 0
            for name in sorted(data.keys(), key=_demo_sort_key):
                record = audit_demo(
                    name,
                    data[name],
                    task_id=task_id,
                    previous_head_last_hash=previous_head_last_hash,
                )
                record["source_file"] = str(resolved)
                records.append(record)
                observed_total += max(0, int(record.get("frames", 0)))
                previous_head_last_hash = (
                    record.get("camera_sanity", {})
                    .get("cam_head", {})
                    .get("last_frame_hash")
                )
            if int(data.attrs.get("total", -1)) != observed_total:
                file_errors.append(
                    f"/data total={data.attrs.get('total', -1)} does not equal observed {observed_total}"
                )
            file_reports.append({"source_file": str(resolved), "errors": file_errors})

    # Duplicate seeds mean independently randomized successes are not auditable.
    accepted_records = [record for record in records if record.get("accepted")]
    seed_counts = Counter(record["generation_seed"] for record in accepted_records)
    duplicate_seeds = sorted(seed for seed, count in seed_counts.items() if seed < 0 or count > 1)
    if duplicate_seeds:
        for record in accepted_records:
            if record["generation_seed"] in duplicate_seeds:
                record["accepted"] = False
                record["reasons"].append(
                    f"duplicate or invalid generation_seed {record['generation_seed']}"
                )
    policy_excluded_count = 0
    if require_fixed_base_stationarity:
        for record in records:
            if record.get("accepted") and record.get("task_route", {}).get(
                "fixed_base_root_drift_reference_exceeded", False
            ):
                record["accepted"] = False
                record["reasons"].append(
                    "Task000001 root XY exceeds 3 cm; excluded because the 17D "
                    "arms+grippers+lift policy excludes base state/action"
                )
                policy_excluded_count += 1

    accepted_records = [record for record in records if record.get("accepted")]
    rejected_records = [record for record in records if not record.get("accepted")]
    selected = accepted_records[:expected_successes]
    errors = [
        f"{item['source_file']}: {error}"
        for item in file_reports
        for error in item["errors"]
    ]
    if allow_rejected:
        if len(accepted_records) < expected_successes:
            errors.append(
                f"accepted success count is {len(accepted_records)}, "
                f"expected at least {expected_successes} with --allow-rejected"
            )
    else:
        if len(accepted_records) != expected_successes:
            errors.append(
                f"accepted success count is {len(accepted_records)}, "
                f"expected exactly {expected_successes}"
            )
        if rejected_records:
            errors.append(f"found {len(rejected_records)} rejected/malformed episodes")
    if len(selected) != expected_successes:
        errors.append(f"selected success count is {len(selected)}, expected {expected_successes}")

    diagnostic_names = sorted(
        {name for record in accepted_records for name in record.get("grasp_diagnostics", {})}
    )
    diagnostic_summary = {
        name: _stats(
            [float(record["grasp_diagnostics"][name]) for record in accepted_records if name in record["grasp_diagnostics"]]
        )
        for name in diagnostic_names
    }
    reason_histogram = Counter(reason for record in rejected_records for reason in record.get("reasons", []))
    checkpoint_records = [
        checkpoint
        for record in accepted_records
        for checkpoint in record.get("placement_checkpoints", {}).values()
    ]
    task_diagnostics_summary: dict[str, Any] = {
        "placement_checkpoint_count": len(checkpoint_records),
        "placement_checkpoint_aligned_count": sum(
            bool(checkpoint.get("matching_post_step_frames"))
            for checkpoint in checkpoint_records
        ),
        "placement_alignment": "post-step state[t] with pre-step EEF observation[t+1]",
    }
    if task_id == "000001" and records:
        route_records = [record["task_route"] for record in records if record.get("task_route")]
        task_diagnostics_summary.update(
            {
                "fixed_base_root_stationarity_is_acceptance_gate": require_fixed_base_stationarity,
                "fixed_base_root_policy_dimension": 17,
                "fixed_base_root_policy_exclusion_reason": "17D arms+grippers+lift policy excludes base state/action",
                "fixed_base_root_policy_excluded_count": policy_excluded_count,
                "fixed_base_root_drift_reference_m": ROOT_ROUTE_X_LIMIT_M,
                "fixed_base_root_drift_reference_exceeded_count": sum(
                    bool(route["fixed_base_root_drift_reference_exceeded"])
                    for route in route_records
                ),
                "fixed_base_root_xy_peak_from_home_max_m": max(
                    route["fixed_base_root_xy_peak_from_home_m"]
                    for route in route_records
                ),
                "fixed_base_root_xy_final_from_home_max_m": max(
                    route["fixed_base_root_xy_final_from_home_m"]
                    for route in route_records
                ),
            }
        )
    return {
        "audit_contract_id": AUDIT_CONTRACT_ID,
        "failure_response_protocol": list(FAILURE_RESPONSE_PROTOCOL),
        "task_id": task_id,
        "allow_rejected": allow_rejected,
        "selection_rule": "first N accepted episodes in input order, then numeric demo order",
        "require_fixed_base_stationarity": require_fixed_base_stationarity,
        "policy_selection_contract": (
            "Task000001 17D arms+grippers+lift excludes base state/action; "
            "root XY must stay within 3 cm"
            if require_fixed_base_stationarity
            else "physical trajectory validity only; fixed-base root drift is diagnostic"
        ),
        "summary_ko": [
            f"Task {task_id} 물리 trajectory 감사 결과는 {'통과' if not errors else '실패'}입니다.",
            f"입력 {len(records)}개 중 성공 {len(accepted_records)}개, 거부 {len(rejected_records)}개이며 {len(selected)}개를 선택했습니다.",
            "grasp can-local XYZ와 EEF-can 거리는 진단 전용이며 성공 gate가 아닙니다.",
        ],
        "expected_successes": expected_successes,
        "passed": not errors,
        "errors": errors,
        "input_episode_count": len(records),
        "accepted_episode_count": len(accepted_records),
        "rejected_episode_count": len(rejected_records),
        "selected_episode_count": len(selected),
        "total_selected_frames": sum(record["frames"] for record in selected),
        "file_reports": file_reports,
        "rejection_reason_histogram": dict(sorted(reason_histogram.items())),
        "grasp_diagnostics_are_acceptance_gate": False,
        "grasp_diagnostic_summary": diagnostic_summary,
        "task_diagnostics_summary": task_diagnostics_summary,
        "records": records,
        "selected_records": selected,
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
    }


def _copy_attrs(source: h5py.AttributeManager, destination: h5py.AttributeManager) -> None:
    for key, value in source.items():
        destination[key] = value


def write_selected_hdf5(report: dict[str, Any], output_path: Path) -> None:
    if "runtime_rate_attestation" not in report:
        raise RuntimeRateAttestationError(
            "refusing curated HDF without reviewed runtime-rate attestation"
        )
    if output_path.exists():
        raise FileExistsError(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected = report["selected_records"]
    with h5py.File(output_path, "x") as destination:
        output_data = destination.create_group("data")
        first_path = Path(selected[0]["source_file"])
        with h5py.File(first_path, "r") as first:
            _copy_attrs(first["data"].attrs, output_data.attrs)
        handles: dict[str, h5py.File] = {}
        try:
            attestation = report["runtime_rate_attestation"]
            attested_by_source = {
                item["source_file"]: item for item in attestation["inputs"]
            }
            for index, record in enumerate(selected):
                path = record["source_file"]
                if path not in handles:
                    handles[path] = h5py.File(path, "r")
                destination_name = f"demo_{index}"
                handles[path].copy(
                    handles[path][f"data/{record['demo']}"],
                    output_data,
                    name=destination_name,
                )
                output_data[destination_name].attrs["physical_audit_contract_id"] = AUDIT_CONTRACT_ID
                output_data[destination_name].attrs["physical_audit_source_file"] = path
                output_data[destination_name].attrs["physical_audit_source_demo"] = record["demo"]
                output_data[destination_name].attrs["physical_audit_selection_rule"] = report["selection_rule"]
                output_data[destination_name].attrs[
                    "require_fixed_base_stationarity"
                ] = bool(report["require_fixed_base_stationarity"])
                output_data[destination_name].attrs[
                    "policy_selection_contract"
                ] = report["policy_selection_contract"]
                source_attestation = attested_by_source[path]
                output_data[destination_name].attrs[
                    "runtime_rate_attestation_source_hdf_sha256"
                ] = source_attestation["hdf5_sha256"]
                output_data[destination_name].attrs[
                    "runtime_rate_attestation_source_process_started_utc"
                ] = source_attestation["actual_process_start_evidence"][
                    "process_started_utc"
                ]
                output_data[destination_name].attrs[
                    "runtime_rate_attestation_source_generation_log_sha256"
                ] = source_attestation["actual_process_start_evidence"][
                    "generation_log_sha256"
                ]
                record["destination_demo"] = destination_name
            output_data.attrs["total"] = report["total_selected_frames"]
            output_data.attrs["physical_audit_contract_id"] = AUDIT_CONTRACT_ID
            output_data.attrs["physical_audit_selected_count"] = len(selected)
            output_data.attrs["physical_audit_selection_rule"] = report["selection_rule"]
            output_data.attrs["physical_audit_allow_rejected"] = bool(report["allow_rejected"])
            output_data.attrs["require_fixed_base_stationarity"] = bool(
                report["require_fixed_base_stationarity"]
            )
            output_data.attrs["policy_selection_contract"] = report[
                "policy_selection_contract"
            ]
            output_data.attrs["physical_audit_source_files"] = json.dumps(
                sorted(handles), ensure_ascii=True
            )
            for key, value in hdf5_rate_attributes(attestation).items():
                output_data.attrs[key] = value
        finally:
            for handle in handles.values():
                handle.close()


def _flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "source_file": record.get("source_file"),
        "source_demo": record.get("demo"),
        "destination_demo": record.get("destination_demo", ""),
        "accepted": record.get("accepted"),
        "reasons": " | ".join(record.get("reasons", [])),
        "frames": record.get("frames"),
        "generation_seed": record.get("generation_seed"),
        "generation_attempt": record.get("generation_attempt"),
    }
    for section in (
        "poses",
        "grasp_correction",
        "final",
        "placement_checkpoints",
        "grasp_diagnostics",
        "carry_metrics",
        "placement_metrics",
        "head_static",
        "task_route",
    ):
        for key, value in record.get(section, {}).items():
            if isinstance(value, dict):
                row[f"{section}.{key}"] = json.dumps(value, sort_keys=True)
            else:
                row[f"{section}.{key}"] = value
    head_camera = record.get("camera_sanity", {}).get("cam_head", {})
    for key, value in head_camera.items():
        if not key.endswith("_hash"):
            row[f"camera_sanity.cam_head.{key}"] = value
    return row


def write_episode_csv(records: list[dict[str, Any]], path: Path) -> None:
    rows = [_flatten_record(record) for record in records]
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(report: dict[str, Any], path: Path) -> None:
    lines = [
        f"# Coffee Transport {report['task_id']} Physical Audit",
        "",
        f"- Contract: `{report['audit_contract_id']}`",
        f"- Result: **{'PASS' if report['passed'] else 'FAIL'}**",
        f"- Episodes: input {report['input_episode_count']}, accepted {report['accepted_episode_count']}, rejected {report['rejected_episode_count']}",
        f"- Selected: {report['selected_episode_count']} / expected {report['expected_successes']}",
        f"- Selected frames: {report['total_selected_frames']}",
        f"- Selection: `{report['selection_rule']}`",
        "- Grasp relative-pose metrics are diagnostic only and never reject an episode.",
        "",
        "## 한국어 요약",
        "",
        f"- 감사 결과: **{'통과' if report['passed'] else '실패'}**",
        f"- 입력 {report['input_episode_count']}개 중 성공 {report['accepted_episode_count']}개, 거부 {report['rejected_episode_count']}개입니다.",
        f"- 요청한 {report['expected_successes']}개 중 {report['selected_episode_count']}개를 선택했습니다.",
        "- grasp의 can-local XYZ와 EEF-can 거리는 분포만 보고하며 성공 판정에는 사용하지 않습니다.",
        "- init pose map은 감사에 통과한 trajectory만 포함합니다.",
        f"- placement checkpoint {report['task_diagnostics_summary']['placement_checkpoint_aligned_count']} / {report['task_diagnostics_summary']['placement_checkpoint_count']}개를 state[t]와 EEF[t+1]에서 재현했습니다.",
        *(
            [
                "- Task000001 비구동 base의 3 cm reference 초과는 "
                f"{report['task_diagnostics_summary']['fixed_base_root_drift_reference_exceeded_count']}개이며, "
                + (
                    "base state/action을 제외한 17D policy selection에서 제외했습니다."
                    if report["require_fixed_base_stationarity"]
                    else "물리 lift 성공 gate와 분리한 진단값으로 기록했습니다."
                )
            ]
            if report["task_id"] == "000001"
            else []
        ),
        "",
        "## 실패 대응 순서",
        "",
    ]
    lines.extend(
        f"{step['order']}. {step['action_ko']} (조건: {step['trigger']})"
        for step in report["failure_response_protocol"]
    )
    lines.append("")
    if report["errors"]:
        lines.extend(("## Errors", ""))
        lines.extend(f"- {error}" for error in report["errors"])
        lines.append("")
    lines.extend(("## Episode Results", "", "| Source | Demo | Frames | Accepted | Reasons |", "| --- | --- | ---: | :---: | --- |"))
    for record in report["records"]:
        reasons = "; ".join(record.get("reasons", [])) or "-"
        lines.append(
            f"| `{Path(record['source_file']).name}` | `{record['demo']}` | {record['frames']} | "
            f"{'yes' if record['accepted'] else 'no'} | {reasons} |"
        )
    lines.extend(("", "## Grasp Diagnostics", ""))
    lines.append("These are aggregate observations, not thresholds.")
    lines.extend(("", "| Metric | Count | Min | Mean | Max |", "| --- | ---: | ---: | ---: | ---: |"))
    for name, stats in report["grasp_diagnostic_summary"].items():
        lines.append(
            f"| `{name}` | {stats['count']} | {stats['min']:.6g} | {stats['mean']:.6g} | {stats['max']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_pose_artifacts(report: dict[str, Any], inputs: list[Path], artifact_dir: Path) -> None:
    task_id = report["task_id"]
    episodes = [
        EpisodePose(
            episode_id=f"{Path(record['source_file']).name}:{record['demo']}",
            task_id=task_id,
            generation_seed=record["generation_seed"],
            values={key: float(value) for key, value in record["poses"].items() if key in {
                "robot_spawn_x_m", "robot_spawn_y_m", "robot_spawn_yaw_rad",
                "robot_home_x_m", "robot_home_y_m", "robot_home_yaw_rad",
                "lift_spawn_m", "base_route_spawn_offset_m",
                "coffee_can_right_x_m", "coffee_can_right_y_m", "coffee_can_right_yaw_rad",
                "coffee_can_left_x_m", "coffee_can_left_y_m", "coffee_can_left_yaw_rad",
            }},
        )
        for record in report["selected_records"]
    ]
    plot = artifact_dir / f"task_{task_id}_success_initial_poses.png"
    pose_csv = artifact_dir / f"task_{task_id}_success_initial_poses.csv"
    summary_json = artifact_dir / f"task_{task_id}_success_initial_poses_summary.json"
    save_plot(episodes, plot)
    write_pose_csv([row_for_episode(episode) for episode in episodes], pose_csv)
    summary = build_pose_summary(episodes, inputs)
    summary["summary_ko"] = [
        f"Task {task_id} 감사 통과 trajectory {len(episodes)}개의 초기 pose map입니다.",
        "캔은 robot-side 절반 깊이인 6 x 12 cm 범위, robot HOME은 XY +/-2 cm와 yaw +/-1 degree 범위를 표시합니다.",
        "plot, CSV, JSON은 동일한 선택 episode 목록에서 생성되었습니다.",
    ]
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def _public_report(report: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in report.items() if key != "selected_records"}


def main() -> None:
    args = parse_args()
    if args.expected_successes <= 0:
        raise ValueError("--expected-successes must be positive")
    inputs = [path.resolve() for path in args.inputs]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    json_path = artifact_dir / "physical_audit.json"
    csv_path = artifact_dir / "physical_episode_audit.csv"
    markdown_path = artifact_dir / "physical_audit_report.md"
    for path in (json_path, csv_path, markdown_path):
        if path.exists():
            raise FileExistsError(path)

    report = audit_inputs(
        inputs,
        args.task_id,
        args.expected_successes,
        args.allow_rejected,
        require_fixed_base_stationarity=args.require_fixed_base_stationarity,
    )
    try:
        report["runtime_rate_attestation"] = load_runtime_rate_attestation(
            args.runtime_rate_attestation,
            inputs,
            args.task_id,
        )
    except RuntimeRateAttestationError as error:
        report["errors"].append(f"runtime-rate attestation: {error}")
        report["passed"] = False
    if report["passed"] and args.output_hdf5 is not None:
        write_selected_hdf5(report, args.output_hdf5.resolve())
        report["output_hdf5"] = str(args.output_hdf5.resolve())
    if report["passed"]:
        write_pose_artifacts(report, inputs, artifact_dir)
    write_episode_csv(report["records"], csv_path)
    write_markdown_report(report, markdown_path)
    json_path.write_text(
        json.dumps(_public_report(report), indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    summary = {
        "passed": report["passed"],
        "task_id": report["task_id"],
        "accepted": report["accepted_episode_count"],
        "rejected": report["rejected_episode_count"],
        "selected": report["selected_episode_count"],
        "artifact_dir": str(artifact_dir),
        "output_hdf5": report.get("output_hdf5"),
    }
    print(json.dumps(summary, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
