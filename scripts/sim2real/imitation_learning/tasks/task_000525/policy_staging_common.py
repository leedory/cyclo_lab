"""Shared contracts for Task000525 phase-specific ACT datasets.

The active pipeline has two exact source formats: canonical joint22 seed data,
and the dual-EEF output of the current physical trajectory generator.  This
module dispatches between those formats and always exposes aligned canonical
joint19+body-velocity3 state/action rows to downstream staging.
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
POLICY_CONTRACT_ID = "ffw_sg2_rev1_mobile_22d_v1"
POLICY_ACTION_SEMANTICS = "pre_step_joint_position_19_plus_body_velocity_3"
STAGING_ACTION_SEMANTICS = POLICY_ACTION_SEMANTICS
CANONICAL_SOURCE_FORMAT = "canonical_joint22"
CANONICAL_TASK_PATH = "obs/task525_demo_phase"
SDG_SOURCE_FORMAT = "task525_dual_eef_generator"
SDG_CONTRACT_ID = "ffw_sg2_task525_locomanipulation_sdg_eef22_v1"
SDG_ACTION_SEMANTICS = (
    "pre_step_dual_eef_pose16_plus_passive_joint3_plus_body_velocity3"
)
SDG_TASK_PATH = "locomanipulation_sdg_output_data/task"
CANONICAL_STATE_ACTION_NAMES = (
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
)
SDG_ACTION_NAMES = (
    "left_eef_x_robot_root",
    "left_eef_y_robot_root",
    "left_eef_z_robot_root",
    "left_eef_qw_robot_root",
    "left_eef_qx_robot_root",
    "left_eef_qy_robot_root",
    "left_eef_qz_robot_root",
    "gripper_l_joint1",
    "right_eef_x_robot_root",
    "right_eef_y_robot_root",
    "right_eef_z_robot_root",
    "right_eef_qw_robot_root",
    "right_eef_qx_robot_root",
    "right_eef_qy_robot_root",
    "right_eef_qz_robot_root",
    "gripper_r_joint1",
    "head_joint1",
    "head_joint2",
    "lift_joint",
    "linear_x",
    "linear_y",
    "angular_z",
)
CANONICAL_CAMERA_SHAPES = {
    "cam_head": (376, 672),
    "cam_wrist_left": (640, 480),
    "cam_wrist_right": (640, 480),
}
CANONICAL_CAMERA_MAP = {
    "cam_head": "cam_left_head",
    "cam_wrist_left": "cam_left_wrist",
    "cam_wrist_right": "cam_right_wrist",
}
CAMERA_ROTATION_DEG = {
    "cam_head": 0,
    "cam_wrist_left": 0,
    "cam_wrist_right": 0,
}
POLICY_TARGET_OBJECT_NAME = "coffee_can_orange"
POLICY_INSTRUCTIONS = {
    "pick": "Pick the orange coffee can out of the cabinet and carry it to the home pose.",
    "mobile_ccw": "Navigate counterclockwise to the dining table while carrying the orange coffee can.",
    "all": (
        "Pick the orange coffee can out of the cabinet, carry it to the dining table, "
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


def source_format(data: h5py.Group) -> str:
    """Identify one of the two exact source formats in the active pipeline."""

    contract_id = str(jsonable(data.attrs.get("robot_contract_id", "")))
    action_semantics = str(jsonable(data.attrs.get("action_semantics", "")))
    pair = (contract_id, action_semantics)
    if pair == (POLICY_CONTRACT_ID, POLICY_ACTION_SEMANTICS):
        return CANONICAL_SOURCE_FORMAT
    if pair == (SDG_CONTRACT_ID, SDG_ACTION_SEMANTICS):
        return SDG_SOURCE_FORMAT
    raise Task525PolicyDataError(
        "Task525 source must match an active exact contract/semantics pair; "
        f"got robot_contract_id={contract_id!r}, action_semantics={action_semantics!r}"
    )


def validate_source(data: h5py.Group) -> dict[str, Any]:
    format_name = source_format(data)
    target_object = str(jsonable(data.attrs.get("target_object_name", "")))
    if format_name == CANONICAL_SOURCE_FORMAT:
        if target_object != POLICY_TARGET_OBJECT_NAME:
            raise Task525PolicyDataError(
                f"Task525 canonical source requires dataset target "
                f"{POLICY_TARGET_OBJECT_NAME}, got {target_object!r}"
            )
    else:
        if target_object and target_object != POLICY_TARGET_OBJECT_NAME:
            raise Task525PolicyDataError(
                f"Task525 generator source has unexpected dataset target "
                f"{target_object!r}"
            )
        episode_targets = {
            name: str(jsonable(data[name].attrs.get("target_object_name", "")))
            for name in data
        }
        invalid_targets = {
            name: target
            for name, target in episode_targets.items()
            if target != POLICY_TARGET_OBJECT_NAME
        }
        if invalid_targets:
            raise Task525PolicyDataError(
                "Task525 generator episodes must declare the orange target: "
                f"{invalid_targets}"
            )
    contract_id = str(jsonable(data.attrs["robot_contract_id"]))
    action_semantics = str(jsonable(data.attrs.get("action_semantics", "")))
    state_names = parse_names(
        data.attrs.get("observation_state_names"), "observation_state_names"
    )
    action_names = parse_names(data.attrs.get("action_names"), "action_names")
    expected_names = list(CANONICAL_STATE_ACTION_NAMES)
    if state_names != expected_names:
        raise Task525PolicyDataError(
            "Task525 observation_state_names do not match the canonical "
            f"joint22 order: {state_names}"
        )
    expected_source_action_names = (
        expected_names
        if format_name == CANONICAL_SOURCE_FORMAT
        else list(SDG_ACTION_NAMES)
    )
    if action_names != expected_source_action_names:
        raise Task525PolicyDataError(
            f"Task525 {format_name} action_names do not match their exact order: "
            f"{action_names}"
        )
    return {
        "source_format": format_name,
        "source_contract_id": contract_id,
        "source_action_semantics": action_semantics,
        "source_action_names": action_names,
        "state_names": state_names,
        "action_names": expected_names,
        "fps": source_fps(data),
    }


def derive_policy_arrays(group: h5py.Group) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    format_name = source_format(group.parent)
    required = {
        "actions": 22,
        "obs/joint_pos": 19,
        "obs/base_velocity_body": 3,
    }
    task_path = (
        CANONICAL_TASK_PATH
        if format_name == CANONICAL_SOURCE_FORMAT
        else SDG_TASK_PATH
    )
    if format_name == SDG_SOURCE_FORMAT:
        required["obs/joint_pos_target"] = 19
    missing = [path for path in (*required, task_path) if path not in group]
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
    tasks = np.asarray(group[task_path], dtype=np.int64).reshape(-1)
    if tasks.shape != (frames,):
        raise Task525PolicyDataError(
            f"{group.name}/{task_path}: {tasks.shape} != {(frames,)}"
        )

    if format_name == CANONICAL_SOURCE_FORMAT:
        action = values["actions"].astype(np.float32, copy=False)
        state = np.concatenate(
            (values["obs/joint_pos"], values["obs/base_velocity_body"]),
            axis=-1,
        ).astype(np.float32, copy=False)
    else:
        action = np.concatenate(
            (
                values["obs/joint_pos_target"][1:],
                values["actions"][:-1, 19:22],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        state = np.concatenate(
            (
                values["obs/joint_pos"][:-1],
                values["obs/base_velocity_body"][:-1],
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        tasks = tasks[:-1]
    if not np.isfinite(action).all() or not np.isfinite(state).all():
        raise Task525PolicyDataError(f"{group.name}: non-finite policy array")
    return state, action, tasks


def select_camera_map(camera_names: list[str] | tuple[str, ...]) -> dict[str, str]:
    """Return a validated camera subset in canonical camera order."""

    names = list(camera_names)
    if not names:
        raise Task525PolicyDataError("camera_names must be non-empty")
    if len(names) != len(set(names)):
        raise Task525PolicyDataError("camera_names contains duplicates")
    unsupported = [name for name in names if name not in CANONICAL_CAMERA_MAP]
    if unsupported:
        raise Task525PolicyDataError(
            f"unsupported Task525 cameras: {unsupported}; "
            f"choose from {list(CANONICAL_CAMERA_MAP)}"
        )
    selected = set(names)
    return {
        name: output_name
        for name, output_name in CANONICAL_CAMERA_MAP.items()
        if name in selected
    }


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
        if np.any(np.diff(tasks[start:end]) < 0):
            raise Task525PolicyDataError("pick task IDs must be monotonic 0 then 1")
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
