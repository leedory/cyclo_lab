"""Validated camera profiles for physical FFW F2 robots."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import re
from typing import Any

import yaml


DEFAULT_F2_CAMERA_PROFILE = "9004"
F2_CAMERA_PROFILE_SCHEMA_VERSION = 1
F2_CAMERA_ROLES = ("head", "wrist_left", "wrist_right")
_PROFILE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_PACKAGED_PROFILE_DIR = Path(__file__).with_name("profiles")


@dataclass(frozen=True)
class F2CameraCalibration:
    role: str
    topic: str
    width: int
    height: int
    intrinsic_matrix: tuple[float, ...]
    distortion_model: str
    distortion: tuple[float, ...]
    frame_id: str
    serial: str


@dataclass(frozen=True)
class F2CameraProfile:
    profile_id: str
    robot_model: str
    robot_ssh_alias: str
    cameras: dict[str, F2CameraCalibration]
    provenance: dict[str, Any]
    source_path: Path
    source_sha256: str

    def camera(self, role: str) -> F2CameraCalibration:
        try:
            return self.cameras[role]
        except KeyError as exc:
            raise KeyError(f"Camera role {role!r} is not present in profile {self.profile_id!r}.") from exc


def _require_mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{field} must be a mapping.")
    return value


def _require_string(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string.")
    return value.strip()


def _require_positive_int(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer.")
    return value


def _require_finite_vector(value: Any, field: str, *, length: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, list) or (length is not None and len(value) != length):
        expected = f" with {length} entries" if length is not None else ""
        raise ValueError(f"{field} must be a list{expected}.")
    result = tuple(float(item) for item in value)
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError(f"{field} must contain only numeric entries.")
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{field} must be finite.")
    return result


def _profile_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.is_absolute() or candidate.parent != Path("."):
        return candidate.expanduser().resolve()
    profile_name = candidate.name
    if not _PROFILE_NAME_PATTERN.fullmatch(profile_name):
        raise ValueError(f"Invalid F2 camera profile name: {profile_name!r}.")
    if not profile_name.endswith((".yaml", ".yml")):
        profile_name += ".yaml"
    return _PACKAGED_PROFILE_DIR / profile_name


def _parse_camera(role: str, value: Any) -> F2CameraCalibration:
    data = _require_mapping(value, f"cameras.{role}")
    intrinsic_matrix = _require_finite_vector(data.get("k"), f"cameras.{role}.k", length=9)
    if intrinsic_matrix[0] <= 0.0 or intrinsic_matrix[4] <= 0.0:
        raise ValueError(f"cameras.{role}.k must contain positive fx and fy.")
    if any(abs(intrinsic_matrix[index]) > 1e-9 for index in (1, 3, 6, 7)):
        raise ValueError(f"cameras.{role}.k must have canonical pinhole matrix structure.")
    if abs(intrinsic_matrix[8] - 1.0) > 1e-9:
        raise ValueError(f"cameras.{role}.k must have canonical pinhole matrix structure.")
    return F2CameraCalibration(
        role=role,
        topic=_require_string(data.get("topic"), f"cameras.{role}.topic"),
        width=_require_positive_int(data.get("width"), f"cameras.{role}.width"),
        height=_require_positive_int(data.get("height"), f"cameras.{role}.height"),
        intrinsic_matrix=intrinsic_matrix,
        distortion_model=_require_string(data.get("distortion_model"), f"cameras.{role}.distortion_model"),
        distortion=_require_finite_vector(data.get("d"), f"cameras.{role}.d"),
        frame_id=_require_string(data.get("frame_id"), f"cameras.{role}.frame_id"),
        serial=_require_string(data.get("serial"), f"cameras.{role}.serial"),
    )


def load_f2_camera_profile(profile: str | Path = DEFAULT_F2_CAMERA_PROFILE) -> F2CameraProfile:
    """Load and validate a packaged F2 profile name or explicit YAML path."""

    source_path = _profile_path(profile)
    try:
        source_bytes = source_path.read_bytes()
    except FileNotFoundError as exc:
        raise ValueError(f"F2 camera profile does not exist: {source_path}") from exc
    root = _require_mapping(yaml.safe_load(source_bytes), "profile")
    if root.get("schema_version") != F2_CAMERA_PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported F2 camera profile schema {root.get('schema_version')!r}; "
            f"expected {F2_CAMERA_PROFILE_SCHEMA_VERSION}."
        )
    robot = _require_mapping(root.get("robot"), "robot")
    camera_values = _require_mapping(root.get("cameras"), "cameras")
    if set(camera_values) != set(F2_CAMERA_ROLES):
        raise ValueError(f"cameras must contain exactly {F2_CAMERA_ROLES}.")
    cameras = {role: _parse_camera(role, camera_values[role]) for role in F2_CAMERA_ROLES}
    serials = {camera.serial for camera in cameras.values()}
    if len(serials) != len(cameras):
        raise ValueError("F2 camera roles must map to different physical serials.")
    return F2CameraProfile(
        profile_id=_require_string(root.get("profile_id"), "profile_id"),
        robot_model=_require_string(robot.get("model"), "robot.model"),
        robot_ssh_alias=_require_string(robot.get("ssh_alias"), "robot.ssh_alias"),
        cameras=cameras,
        provenance=_require_mapping(root.get("provenance"), "provenance"),
        source_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
    )
