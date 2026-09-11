"""Named causal policy contract for Coffee Transport ACT datasets."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

try:
    import h5py
except ModuleNotFoundError:
    h5py = None
import numpy as np

from runtime_rate_attestation import (
    RuntimeRateAttestationError,
    physical_runtime_rate_provenance_from_attrs,
)


STAGING_SCHEMA = "cyclo.isaac_action_replay_staging.v1"
STAGING_ACTION_SEMANTICS = "pre_step_named_joint_position_plus_optional_body_velocity"
SOURCE_ACTION_SEMANTICS = (
    "pre_step_dual_eef_pose_wxyz_grippers_head_lift_plus_optional_body_velocity"
)
SOURCE_ORIGIN = "coffee_transport_physical_trajectory_generation"
# Camera contract, aligned with tasks/task_000525/policy_staging_common.py and
# with the real robot: profile 1050.yaml streams head 672x376 and wrists
# 640x480 landscape; the simulator renders the wrists axis-swapped so the stored
# frame is upright 480x640, which is the shape real LeRobot datasets carry
# (observation.images.rgb.cam_left_wrist -> [3, 640, 480]).
CAMERA_MAP = {
    "cam_head": "cam_left_head",
    "cam_wrist_left": "cam_left_wrist",
    "cam_wrist_right": "cam_right_wrist",
}
CAMERA_SHAPES = {
    "cam_head": (376, 672),
    "cam_wrist_left": (640, 480),
    "cam_wrist_right": (640, 480),
}
CAMERA_ROTATION_DEG = {
    "cam_head": 0,
    "cam_wrist_left": 0,
    "cam_wrist_right": 0,
}
POLICY_CAMERA_NAMES = tuple(CAMERA_MAP)

JOINT19_NAMES = (
    "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4",
    "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
    "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4",
    "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
    "head_joint1", "head_joint2", "lift_joint",
)
TASK_POLICY_NAMES = {
    "000001": (*JOINT19_NAMES[:16], "lift_joint"),
    "000002": (*JOINT19_NAMES[:16], "linear_x", "linear_y", "angular_z"),
}
TASK000001_LIFT_ENDPOINTS_M = np.asarray((0.0, -0.28), dtype=np.float32)
TASK000001_LIFT_ENDPOINT_TOLERANCE_M = 1.0e-6
TASK_INSTRUCTIONS = {
    "000001": (
        "Move the right orange coffee can with the right hand and the left orange "
        "coffee can with the left hand to the lower cabinet shelf."
    ),
    "000002": (
        "Move the right orange coffee can with the right hand and the left orange "
        "coffee can with the left hand to the matching cabinet on the right."
    ),
}

# This is the policy-training trust boundary, not the generator's current
# defaults.  A physical audit may promote a generated campaign only after its
# task-specific contract has been reviewed.  Keeping the accepted metadata
# here prevents callers from bypassing that audit by invoking staging directly.
REVIEWED_SOURCE_METADATA = {
    "000001": {
        "generation_method": (
            "coffee_two_can_endpoint_lift_targets_pending_validation_v13"
        ),
        "initial_normalization_contract_id": (
            "coffee_realized_pose_endpoint_lift_targets_v13"
        ),
        "generation_profile_id": (
            "coffee_physical_trajectory_generation_endpoint_lift_targets_v13"
        ),
        "production_promotion_id": "coffee_endpoint_lift_targets_v13_reviewed",
        "production_validation_status": "reviewed_endpoint_lift_targets_v13",
        "task_axis_action_contract": (
            "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1"
        ),
    },
    "000002": {
        "generation_method": (
            "coffee_two_can_production_realized_initial_commanded_joint_hold_v12"
        ),
        "initial_normalization_contract_id": (
            "coffee_production_realized_pose_commanded_joint_hold_v12"
        ),
        "generation_profile_id": (
            "coffee_physical_trajectory_generation_commanded_joint_hold_v12"
        ),
        "production_promotion_id": (
            "coffee_commanded_joint_hold_v12_promoted_20260908"
        ),
        "task_axis_action_contract": "base_body_velocity_feedback_v1",
    },
}
REVIEWED_COMMON_SOURCE_METADATA = {
    "production_approved": True,
    "coffee_can_collision_profile_id": "cylinder_r33p5_h90_v1",
    "coffee_can_collision_asset_sha256": (
        "fe340f411e93fd3fd54bc06b1b4ed1e6dec24214179e69ae5dd6ce1dd50d4a82"
    ),
    "coffee_can_collision_12gon_applied": False,
}


class CoffeePolicyDataError(RuntimeError):
    """Raised when Coffee source or staging data violates its exact contract."""


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
            raise CoffeePolicyDataError(f"{label} must be a JSON name list") from error
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise CoffeePolicyDataError(f"{label} must be a non-empty string list")
    if len(value) != len(set(value)):
        raise CoffeePolicyDataError(f"{label} contains duplicate names")
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
    if h5py is None:
        raise CoffeePolicyDataError("h5py is required for HDF5 source operations")
    value = float(data.attrs.get("control_hz", 0.0))
    rounded = int(round(value))
    if rounded <= 0 or not np.isclose(value, rounded, atol=1.0e-6):
        raise CoffeePolicyDataError(f"positive integer control_hz required, got {value}")
    return rounded


def source_action_names(task_id: str) -> tuple[str, ...]:
    names = (
        "left_eef_x", "left_eef_y", "left_eef_z", "left_eef_qw",
        "left_eef_qx", "left_eef_qy", "left_eef_qz", "gripper_l_joint1",
        "right_eef_x", "right_eef_y", "right_eef_z", "right_eef_qw",
        "right_eef_qx", "right_eef_qy", "right_eef_qz", "gripper_r_joint1",
        "head_joint1", "head_joint2", "lift_joint",
    )
    return names if task_id == "000001" else (*names, "linear_x", "linear_y", "angular_z")


def validate_reviewed_source_metadata(
    metadata: Mapping[str, Any], task_id: str
) -> None:
    """Reject unreviewed or superseded physical-generation contracts."""

    expected = {
        **REVIEWED_COMMON_SOURCE_METADATA,
        **REVIEWED_SOURCE_METADATA[task_id],
    }
    mismatches = {
        key: (jsonable(metadata.get(key)), value)
        for key, value in expected.items()
        if jsonable(metadata.get(key)) != value
    }
    if mismatches:
        raise CoffeePolicyDataError(
            f"source is not an approved task {task_id} physical dataset: {mismatches}"
        )


def validate_source(data: h5py.Group, task_id: str) -> dict[str, Any]:
    if task_id not in TASK_POLICY_NAMES:
        raise CoffeePolicyDataError(f"unsupported Coffee task: {task_id}")
    expected = {
        "task_id": task_id,
        "dataset_origin": SOURCE_ORIGIN,
        "action_semantics": SOURCE_ACTION_SEMANTICS,
    }
    mismatches = {
        key: (jsonable(data.attrs.get(key)), value)
        for key, value in expected.items()
        if jsonable(data.attrs.get(key)) != value
    }
    if mismatches:
        raise CoffeePolicyDataError(f"source metadata mismatch: {mismatches}")
    validate_reviewed_source_metadata(data.attrs, task_id)
    actual_names = parse_names(data.attrs.get("action_names"), "action_names")
    if actual_names != list(source_action_names(task_id)):
        raise CoffeePolicyDataError(f"source action_names mismatch for task {task_id}")
    names = sorted(data.keys(), key=natural_key)
    if not names:
        raise CoffeePolicyDataError("source contains no episodes")
    failed = [name for name in names if not bool(data[name].attrs.get("success", False))]
    if failed:
        raise CoffeePolicyDataError(f"source contains unsuccessful episodes: {failed}")
    try:
        runtime_rate_provenance = physical_runtime_rate_provenance_from_attrs(
            data.attrs
        )
    except RuntimeRateAttestationError as error:
        raise CoffeePolicyDataError(str(error)) from error
    return {
        "task_id": task_id,
        "fps": source_fps(data),
        "source_action_names": actual_names,
        "state_names": list(TASK_POLICY_NAMES[task_id]),
        "action_names": list(TASK_POLICY_NAMES[task_id]),
        "episode_names": names,
        "physical_runtime_rate_provenance": runtime_rate_provenance,
    }


def _pre_step_body_twist(group: h5py.Group, frames: int) -> np.ndarray:
    """Derive canonical body-frame [linear_x, linear_y, angular_z]."""

    root_pose_post = np.asarray(group["states/articulation/robot/root_pose"], dtype=np.float64)
    root_velocity_post = np.asarray(
        group["states/articulation/robot/root_velocity"], dtype=np.float64
    )
    initial_pose = np.asarray(group["initial_state/articulation/robot/root_pose"])[0]
    initial_velocity = np.asarray(group["initial_state/articulation/robot/root_velocity"])[0]
    pose = np.concatenate((initial_pose[None], root_pose_post[:-1]), axis=0)
    velocity = np.concatenate((initial_velocity[None], root_velocity_post[:-1]), axis=0)
    if pose.shape != (frames, 7) or velocity.shape != (frames, 6):
        raise CoffeePolicyDataError(f"{group.name}: invalid causal root-state shape")
    # Isaac stores root twist in world coordinates. Match
    # asset.data.root_{lin,ang}_vel_b by applying R(q)^T; root q is WXYZ.
    w, x, y, z = (pose[:, index] for index in range(3, 7))
    linear_world = velocity[:, :3]
    angular_world = velocity[:, 3:]
    linear_x = (
        (1.0 - 2.0 * (y * y + z * z)) * linear_world[:, 0]
        + 2.0 * (x * y + w * z) * linear_world[:, 1]
        + 2.0 * (x * z - w * y) * linear_world[:, 2]
    )
    linear_y = (
        2.0 * (x * y - w * z) * linear_world[:, 0]
        + (1.0 - 2.0 * (x * x + z * z)) * linear_world[:, 1]
        + 2.0 * (y * z + w * x) * linear_world[:, 2]
    )
    angular_z = (
        2.0 * (x * z + w * y) * angular_world[:, 0]
        + 2.0 * (y * z - w * x) * angular_world[:, 1]
        + (1.0 - 2.0 * (x * x + y * y)) * angular_world[:, 2]
    )
    return np.stack((linear_x, linear_y, angular_z), axis=-1).astype(np.float32)


def derive_policy_arrays(group: h5py.Group, task_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Convert EEF-generator output to aligned, deployable ACT rows."""

    frames = int(group["actions"].shape[0])
    required = {
        "obs/joint_pos": (frames, 19),
        "obs/joint_pos_target": (frames, 19),
    }
    for path, shape in required.items():
        if path not in group or group[path].shape != shape:
            actual = None if path not in group else group[path].shape
            raise CoffeePolicyDataError(f"{group.name}/{path}: {actual} != {shape}")
    if frames < 2:
        raise CoffeePolicyDataError(f"{group.name}: at least two frames required")
    joint = np.asarray(group["obs/joint_pos"], dtype=np.float32)
    target = np.asarray(group["obs/joint_pos_target"], dtype=np.float32)
    if task_id == "000001":
        indices = [*range(16), 18]
        state = joint[:-1, indices]
        action = target[1:, indices]
        lift_action = action[:, -1]
        endpoint_error = np.min(
            np.abs(
                lift_action[:, None]
                - TASK000001_LIFT_ENDPOINTS_M[None, :]
            ),
            axis=1,
        )
        invalid = np.flatnonzero(
            endpoint_error > TASK000001_LIFT_ENDPOINT_TOLERANCE_M
        )
        if len(invalid):
            policy_frame = int(invalid[0])
            source_target_frame = policy_frame + 1
            raise CoffeePolicyDataError(
                f"{group.name}/obs/joint_pos_target[{source_target_frame},18] "
                f"is {lift_action[policy_frame]:.9f}; Task000001 ACT lift actions "
                "must be absolute endpoints 0.0 or -0.28 m"
            )
    elif task_id == "000002":
        source_action = np.asarray(group["actions"], dtype=np.float32)
        if source_action.shape != (frames, 22):
            raise CoffeePolicyDataError(f"{group.name}/actions must be {(frames, 22)}")
        body_state = _pre_step_body_twist(group, frames)
        state = np.concatenate((joint[:-1, :16], body_state[:-1]), axis=-1)
        # Preserve all recorded body commands, including near-zero X and yaw.
        action = np.concatenate((target[1:, :16], source_action[:-1, 19:22]), axis=-1)
    else:
        raise CoffeePolicyDataError(f"unsupported task {task_id}")
    expected_width = len(TASK_POLICY_NAMES[task_id])
    if state.shape != action.shape or state.shape[1] != expected_width:
        raise CoffeePolicyDataError(f"{group.name}: policy shape mismatch {state.shape}/{action.shape}")
    if not np.isfinite(state).all() or not np.isfinite(action).all():
        raise CoffeePolicyDataError(f"{group.name}: non-finite policy values")
    return state.astype(np.float32, copy=False), action.astype(np.float32, copy=False)


def canonicalize_camera_frame(source_camera: str, frame: np.ndarray) -> np.ndarray:
    """Validate and normalise one rendered frame for ``source_camera``.

    Signature matches tasks/task_000525/policy_staging_common.py so both task
    families share one contract. Rotation is 0 for every SG2 camera; the entry
    is kept so a future recalibration has one place to change.
    """
    if source_camera not in CAMERA_SHAPES:
        raise CoffeePolicyDataError(f"unknown policy camera: {source_camera}")
    rgb = np.asarray(frame[..., :3], dtype=np.uint8)
    if rgb.shape[:2] != CAMERA_SHAPES[source_camera]:
        raise CoffeePolicyDataError(
            f"{source_camera} canonical HxW must be {CAMERA_SHAPES[source_camera]}, "
            f"got {rgb.shape[:2]}"
        )
    if CAMERA_ROTATION_DEG[source_camera]:
        raise CoffeePolicyDataError(
            f"{source_camera} declares a non-zero rotation; the replay writer does not rotate"
        )
    return np.ascontiguousarray(rgb)


def selected_names(values: Sequence[str], limit: int | None = None) -> list[str]:
    result = sorted(values, key=natural_key)
    return result if limit is None else result[:limit]
