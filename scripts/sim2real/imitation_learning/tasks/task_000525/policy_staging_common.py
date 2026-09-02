"""Shared contracts for Task000525 phase-specific ACT datasets.

The source generation file stores a hybrid right-EEF action.  ACT must never
consume that field directly.  This module derives the causal joint19+base3
policy action and selects the requested contiguous task phase without copying
the 51 GiB source HDF5.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping

import h5py
import numpy as np


STAGING_SCHEMA = "cyclo.isaac_action_replay_staging.v1"
STAGING_ACTION_SEMANTICS = "pre_step_raw_absolute_joint_position_command"
POLICY_ACTION_SEMANTICS = (
    "derived_pre_step_absolute_joint_position_19_plus_body_velocity_3"
)
POLICY_CONTRACT_ID = "ffw_sg2_rev1_mobile_22d_v1"
TASK_PATH = "locomanipulation_sdg_output_data/task"
CANONICAL_CAMERA_SHAPES = {
    "cam_head": (376, 672),
    "cam_wrist_left": (640, 480),
    "cam_wrist_right": (640, 480),
}
CAMERA_ROTATION_DEG = {
    "cam_head": 0,
    "cam_wrist_left": 0,
    "cam_wrist_right": 0,
}
POLICY_INSTRUCTIONS = {
    "pick": "Pick the green coffee can out of the cabinet and carry it to the home pose.",
    "mobile_ccw": "Navigate counterclockwise to the dining table while carrying the green coffee can.",
    "all": (
        "Pick the green coffee can out of the cabinet, carry it to the dining table, "
        "place it on the mat, and return to the home pose."
    ),
}


class Task525PolicyDataError(RuntimeError):
    """Raised when the source does not satisfy the phase-policy contract."""


def jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def parse_names(value: Any, label: str) -> list[str]:
    value = jsonable(value)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise Task525PolicyDataError(f"{label} must be a JSON name list") from error
    if not isinstance(value, list) or not value or not all(isinstance(item, str) for item in value):
        raise Task525PolicyDataError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise Task525PolicyDataError(f"{label} contains duplicate names")
    return value


def natural_key(value: str) -> tuple[Any, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"(\d+)", value))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_fps(data: h5py.Group) -> int:
    value = float(data.attrs.get("control_hz", 0.0))
    if value <= 0.0:
        env_args = jsonable(data.attrs.get("env_args", ""))
        if isinstance(env_args, str):
            try:
                env_args = json.loads(env_args)
            except json.JSONDecodeError as error:
                raise Task525PolicyDataError("data/env_args is not valid JSON") from error
        try:
            sim_args = env_args["sim_args"]
            value = 1.0 / (float(sim_args["dt"]) * int(sim_args["decimation"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
            raise Task525PolicyDataError("cannot derive source control frequency") from error
    rounded = int(round(value))
    if rounded <= 0 or not np.isclose(value, rounded, atol=1.0e-6):
        raise Task525PolicyDataError(f"LeRobot requires integer fps, got {value}")
    return rounded


def validate_source(data: h5py.Group) -> dict[str, Any]:
    contract_id = str(jsonable(data.attrs.get("robot_contract_id", "")))
    if "locomanipulation_sdg_eef" not in contract_id:
        raise Task525PolicyDataError(
            f"source is not Task525 locomanipulation SDG EEF data: {contract_id!r}"
        )
    state_names = parse_names(
        data.attrs.get("observation_state_names"), "observation_state_names"
    )
    if len(state_names) != 22:
        raise Task525PolicyDataError(
            f"Task525 policy state must be joint19+base3, got {len(state_names)}"
        )
    return {
        "source_contract_id": contract_id,
        "state_names": state_names,
        "action_names": list(state_names),
        "fps": source_fps(data),
    }


def derive_policy_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "actions": 22,
        "obs/joint_pos": 19,
        "obs/joint_pos_target": 19,
        "obs/base_velocity_body": 3,
    }
    missing = [path for path in (*required, TASK_PATH) if path not in group]
    if missing:
        raise Task525PolicyDataError(f"{group.name}: missing datasets {missing}")
    frames = int(group["actions"].shape[0])
    if frames < 2:
        raise Task525PolicyDataError(f"{group.name}: at least two source frames are required")
    values: dict[str, np.ndarray] = {}
    for path, width in required.items():
        values[path] = np.asarray(group[path], dtype=np.float32)
        if values[path].shape != (frames, width):
            raise Task525PolicyDataError(
                f"{group.name}/{path}: {values[path].shape} != {(frames, width)}"
            )
    tasks = np.asarray(group[TASK_PATH], dtype=np.int64).reshape(-1)
    if tasks.shape != (frames,):
        raise Task525PolicyDataError(
            f"{group.name}/{TASK_PATH}: {tasks.shape} != {(frames,)}"
        )

    # Source action t produces the joint target observed at pre-step t+1.
    action = np.concatenate(
        (values["obs/joint_pos_target"][1:], values["actions"][:-1, 19:22]),
        axis=-1,
    ).astype(np.float32, copy=False)
    state = np.concatenate(
        (values["obs/joint_pos"][:-1], values["obs/base_velocity_body"][:-1]),
        axis=-1,
    ).astype(np.float32, copy=False)
    tasks = tasks[:-1]
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise Task525PolicyDataError(f"{group.name}: non-finite policy array")
    return state, action, tasks


def phase_bounds(tasks: np.ndarray, policy: str) -> tuple[int, int]:
    if policy == "pick":
        selected = np.isin(tasks, (0, 1))
    elif policy == "mobile_ccw":
        selected = tasks == 2
    elif policy == "all":
        selected = np.isin(tasks, (0, 1, 2, 3, 4))
    else:
        raise Task525PolicyDataError(f"unsupported policy {policy!r}")
    indices = np.flatnonzero(selected)
    if not len(indices):
        raise Task525PolicyDataError(f"no selected frames for policy={policy}")
    start = int(indices[0])
    end = int(indices[-1]) + 1
    if not selected[start:end].all() or selected[:start].any() or selected[end:].any():
        raise Task525PolicyDataError(
            f"policy={policy} phase is not one contiguous interval"
        )
    if policy == "pick":
        if start != 0 or set(np.unique(tasks[start:end]).tolist()) != {0, 1}:
            raise Task525PolicyDataError("pick must contain task 0 then task 1 from frame zero")
        if end >= len(tasks) or int(tasks[end]) != 2:
            raise Task525PolicyDataError("pick must end immediately before task 2 navigation")
    elif policy == "all":
        if start != 0 or end != len(tasks):
            raise Task525PolicyDataError("all must cover the complete causal episode")
        if set(np.unique(tasks).tolist()) != {0, 1, 2, 3, 4}:
            raise Task525PolicyDataError("all must contain task IDs 0 through 4")
        if np.any(np.diff(tasks) < 0):
            raise Task525PolicyDataError("all task IDs must be monotonic")
    return start, end


def navigation_evidence(action: np.ndarray, tasks: np.ndarray) -> dict[str, Any]:
    start, end = phase_bounds(tasks, "mobile_ccw")
    angular_z = action[start:end, 21]
    nonzero = np.flatnonzero(np.abs(angular_z) > 1.0e-5)
    if not len(nonzero):
        raise Task525PolicyDataError("navigation has no non-zero angular-z command")
    first = float(angular_z[int(nonzero[0])])
    return {
        "first_nonzero_angular_z_radps": first,
        "angular_z_sum": float(angular_z.sum()),
        "positive_command_frames": int(np.count_nonzero(angular_z > 1.0e-5)),
        "negative_command_frames": int(np.count_nonzero(angular_z < -1.0e-5)),
        "is_counterclockwise": first > 0.0,
    }


def selected_episode_names(data: h5py.Group, policy: str) -> tuple[list[str], dict[str, dict[str, Any]]]:
    names = sorted(data.keys(), key=natural_key)
    evidence: dict[str, dict[str, Any]] = {}
    if policy in ("pick", "all"):
        for name in names:
            _state, action, tasks = derive_policy_arrays(data[name])
            start, end = phase_bounds(tasks, policy)
            evidence[name] = {"segment_start": start, "segment_end": end}
        return names, evidence

    selected = []
    for name in names:
        _state, action, tasks = derive_policy_arrays(data[name])
        item = navigation_evidence(action, tasks)
        start, end = phase_bounds(tasks, policy)
        item.update({"segment_start": start, "segment_end": end})
        evidence[name] = item
        if item["is_counterclockwise"]:
            selected.append(name)
    return selected, evidence


def crop_policy_episode(
    group: h5py.Group, policy: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[int, int]]:
    state, action, tasks = derive_policy_arrays(group)
    start, end = phase_bounds(tasks, policy)
    return state[start:end], action[start:end], tasks[start:end], (start, end)


def canonicalize_camera_frame(source_camera: str, frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame[..., :3], dtype=np.uint8)
    expected_shape = CANONICAL_CAMERA_SHAPES.get(source_camera)
    if expected_shape is None:
        raise Task525PolicyDataError(f"unsupported policy camera: {source_camera}")
    if rgb.shape[:2] != expected_shape:
        raise Task525PolicyDataError(
            f"{source_camera}: canonical frame HxW must be {expected_shape}, "
            f"got {rgb.shape[:2]}; rotation and resizing are not permitted"
        )
    return np.ascontiguousarray(rgb)
