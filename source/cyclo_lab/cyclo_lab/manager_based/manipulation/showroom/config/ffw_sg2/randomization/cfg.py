"""User-facing randomization settings for the peanut take-out dataset.

All values are maximum deviations sampled continuously at reset. Packet
coordinates use showroom world axes: X is shelf depth, Y is sideways along the
shelf, and Z is vertical. Target Z is never randomized.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


WALL_RGB_MODE = "rgb"
WALL_WHITE_MODE = "near_white"

@dataclass(frozen=True)
class TargetPoseRandomizationCfg:
    enabled: bool = False
    lateral_y_max_m: float = 0.010
    yaw_max_rad: float = math.radians(10.0)


@dataclass(frozen=True)
class RobotRootRandomizationCfg:
    # Disabled by default until the reduced range is replay-validated.
    enabled: bool = False
    depth_x_max_m: float = 0.010
    lateral_y_max_m: float = 0.010
    yaw_max_rad: float = math.radians(1.0)


@dataclass(frozen=True)
class PresenceRandomizationCfg:
    enabled: bool = False
    object_names: tuple[str, ...] = ()
    disappearance_probability: float = 0.50


@dataclass(frozen=True)
class LightingRandomizationCfg:
    # The dome is shared by all parallel environments and is sampled once per run.
    enabled: bool = False
    dome_intensity_range: tuple[float, float] = (2400.0, 3600.0)
    dome_rgb_range: tuple[tuple[float, float], ...] = (
        (0.705, 0.795),
        (0.705, 0.795),
        (0.705, 0.795),
    )
    weak_key_intensity_range: tuple[float, float] = (0.0, 5000.0)


@dataclass(frozen=True)
class ShelfAppearanceRandomizationCfg:
    enabled: bool = False
    brightness_range: tuple[float, float] = (0.85, 1.15)
    channel_tint_max: float = 0.08


@dataclass(frozen=True)
class WallAppearanceRandomizationCfg:
    enabled: bool = False
    # Both task-facing walls receive the same sampled color.
    mode: str = WALL_RGB_MODE
    rgb_range: tuple[float, float] = (0.0, 1.0)
    near_white_range: tuple[float, float] = (0.65, 1.0)


@dataclass(frozen=True)
class CameraRandomizationCfg:
    enabled: bool = False
    camera_names: tuple[str, ...] = ()
    coupled_focal_scale_range: tuple[float, float] = (0.95, 1.05)
    local_roll_max_rad: float = math.radians(5.0)
    local_pitch_max_rad: float = math.radians(5.0)
    local_yaw_max_rad: float = math.radians(5.0)


@dataclass(frozen=True)
class ShowroomGenerationRandomizationCfg:
    """Task-generation profile with optional pose and appearance axes."""

    target_pose: TargetPoseRandomizationCfg = TargetPoseRandomizationCfg()
    robot_root: RobotRootRandomizationCfg = RobotRootRandomizationCfg()
    presence: PresenceRandomizationCfg = PresenceRandomizationCfg()
    lighting: LightingRandomizationCfg = LightingRandomizationCfg()
    shelf: ShelfAppearanceRandomizationCfg = ShelfAppearanceRandomizationCfg()
    wall: WallAppearanceRandomizationCfg = WallAppearanceRandomizationCfg()
    camera: CameraRandomizationCfg = CameraRandomizationCfg()


# Compatibility alias for datasets and scripts that still use the original type name.
Task458RandomizationCfg = ShowroomGenerationRandomizationCfg


@dataclass(frozen=True)
class ObjectPoseRandomizationCfg:
    """Pose range for an explicit list of showroom objects."""

    enabled: bool = False
    object_names: tuple[str, ...] = ()
    x_max_m: float = 0.0
    y_max_m: float = 0.0
    yaw_max_rad: float = 0.0


@dataclass(frozen=True)
class ShowroomRandomizationCfg:
    """Task-neutral robot and object pose profile."""

    robot_root: RobotRootRandomizationCfg = RobotRootRandomizationCfg()
    objects: ObjectPoseRandomizationCfg = ObjectPoseRandomizationCfg()


def validate_randomization_cfg(
    cfg: ShowroomGenerationRandomizationCfg | ShowroomRandomizationCfg,
) -> None:
    """Reject invalid ranges before Isaac Sim starts."""
    if isinstance(cfg, ShowroomRandomizationCfg):
        if min(
            cfg.robot_root.depth_x_max_m,
            cfg.robot_root.lateral_y_max_m,
            cfg.robot_root.yaw_max_rad,
            cfg.objects.x_max_m,
            cfg.objects.y_max_m,
            cfg.objects.yaw_max_rad,
        ) < 0.0:
            raise ValueError("showroom pose randomization ranges must be non-negative")
        return
    if cfg.target_pose.lateral_y_max_m < 0.0:
        raise ValueError("target lateral range must be non-negative")
    if cfg.target_pose.yaw_max_rad < 0.0:
        raise ValueError("target yaw range must be non-negative")
    if min(
        cfg.robot_root.depth_x_max_m,
        cfg.robot_root.lateral_y_max_m,
        cfg.robot_root.yaw_max_rad,
    ) < 0.0:
        raise ValueError("robot-root ranges must be non-negative")
    if not 0.0 <= cfg.presence.disappearance_probability <= 1.0:
        raise ValueError("disappearance_probability must be in [0, 1]")
    if cfg.wall.mode not in (WALL_RGB_MODE, WALL_WHITE_MODE):
        raise ValueError(f"unsupported wall mode: {cfg.wall.mode}")
    if cfg.camera.enabled and not cfg.camera.camera_names:
        raise ValueError("camera randomization requires at least one camera")
