"""Authoritative randomization profiles for showroom Task000525.

Every Task525-specific axis lives here.  Consumers select a complete profile
instead of enabling B-region or coffee-label randomization with separate
environment booleans.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    RobotRootRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    WallAppearanceRandomizationCfg,
    validate_randomization_cfg,
)
from .layout import TASK000525_CAN_NAMES, TASK000525_SELECTED_LAYOUT_KEY
from .spec import TASK_000525_SPEC


@dataclass(frozen=True)
class CoffeeRegionRandomizationCfg:
    """Task525 coffee-can center sampling inside one reviewed layout."""

    enabled: bool = False
    layout_key: str = TASK000525_SELECTED_LAYOUT_KEY


@dataclass(frozen=True)
class CoffeeVisualYawRandomizationCfg:
    """Appearance-only label yaw; rigid pose and collisions stay unchanged."""

    enabled: bool = False
    object_names: tuple[str, ...] = TASK000525_CAN_NAMES
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class Task000525RandomizationCfg(ShowroomGenerationRandomizationCfg):
    """Shared showroom axes plus Task525's two coffee-can axes."""

    coffee_positions: CoffeeRegionRandomizationCfg = CoffeeRegionRandomizationCfg()
    coffee_visual_yaw: CoffeeVisualYawRandomizationCfg = (
        CoffeeVisualYawRandomizationCfg()
    )


def validate_task000525_randomization_cfg(
    profile: Task000525RandomizationCfg,
) -> None:
    """Validate common and Task525-specific profile fields."""

    validate_randomization_cfg(profile)
    if profile.coffee_positions.layout_key != TASK000525_SELECTED_LAYOUT_KEY:
        raise ValueError(
            "Task525 coffee positions must use the reviewed layout "
            f"{TASK000525_SELECTED_LAYOUT_KEY!r}, got "
            f"{profile.coffee_positions.layout_key!r}."
        )
    yaw_min, yaw_max = profile.coffee_visual_yaw.yaw_range_rad
    if not (-math.pi <= yaw_min <= yaw_max <= math.pi):
        raise ValueError(
            "Task525 coffee visual yaw must be ordered inside [-pi, pi], "
            f"got {profile.coffee_visual_yaw.yaw_range_rad}."
        )
    unknown = set(profile.coffee_visual_yaw.object_names) - set(TASK000525_CAN_NAMES)
    if unknown:
        raise ValueError(f"Unknown Task525 coffee visual objects: {sorted(unknown)}")


TASK000525_DETERMINISTIC = Task000525RandomizationCfg()

TASK000525_RECORD_RANDOMIZED = Task000525RandomizationCfg(
    coffee_positions=CoffeeRegionRandomizationCfg(enabled=True),
    coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True),
)

# Physical trajectory generation: root pose and all four can centers move.
# Visual axes are deliberately disabled for the later replay augmentation pass.
TASK000525_PHYSICAL_TRAJECTORY_GENERATION = Task000525RandomizationCfg(
    robot_root=RobotRootRandomizationCfg(
        enabled=True,
        depth_x_max_m=0.030,
        lateral_y_max_m=0.030,
        yaw_max_rad=math.radians(2.5),
    ),
    coffee_positions=CoffeeRegionRandomizationCfg(enabled=True),
)

# Visual replay augmentation: no robot, rigid-object, furniture, or collision
# changes.  The can label direction is varied by rotating only its visual mesh.
TASK000525_VISUAL_REPLAY_AUGMENTATION = Task000525RandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000525_SPEC.policy_cameras,
    ),
    coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True),
)
