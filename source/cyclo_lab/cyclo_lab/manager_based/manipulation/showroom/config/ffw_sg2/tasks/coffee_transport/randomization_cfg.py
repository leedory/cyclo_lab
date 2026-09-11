"""Typed randomization axes shared by the coffee-transport task family."""

from __future__ import annotations

from dataclasses import dataclass
import math

from ...randomization.cfg import (
    ShowroomGenerationRandomizationCfg,
    validate_randomization_cfg,
)
from .coffee_transport_common import (
    CABINET_RIGHT_DELTA_Y_M,
    COFFEE_CAN_NAMES,
    LIFT_TRAVEL_COMMAND_M,
    ROBOT_SPAWN_XY_RANDOM_M,
    ROBOT_SPAWN_YAW_RANDOM_RAD,
)


@dataclass(frozen=True)
class CoffeePositionRandomizationCfg:
    """Independent X/Y center sampling inside each can's reviewed square."""

    enabled: bool = False
    object_names: tuple[str, ...] = COFFEE_CAN_NAMES


@dataclass(frozen=True)
class CoffeeVisualYawRandomizationCfg:
    """Appearance-only orange-label yaw; rigid state remains unchanged."""

    enabled: bool = False
    object_names: tuple[str, ...] = COFFEE_CAN_NAMES
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class EpisodeHomeRandomizationCfg:
    """Episode-local HOME around the authored robot pose."""

    enabled: bool = False
    depth_x_max_m: float = ROBOT_SPAWN_XY_RANDOM_M
    lateral_y_max_m: float = ROBOT_SPAWN_XY_RANDOM_M
    yaw_max_rad: float = ROBOT_SPAWN_YAW_RANDOM_RAD


@dataclass(frozen=True)
class LiftStartRandomizationCfg:
    """Task000001 lift start; the policy first returns it to zero."""

    enabled: bool = False
    position_range_m: tuple[float, float] = (LIFT_TRAVEL_COMMAND_M, 0.0)


@dataclass(frozen=True)
class BaseRouteStartRandomizationCfg:
    """Task000002 start offset from sampled HOME along world +Y."""

    enabled: bool = False
    offset_range_m: tuple[float, float] = (0.0, CABINET_RIGHT_DELTA_Y_M)


def validate_coffee_transport_randomization_cfg(
    profile: ShowroomGenerationRandomizationCfg,
    *,
    coffee_positions: CoffeePositionRandomizationCfg,
    coffee_visual_yaw: CoffeeVisualYawRandomizationCfg,
    episode_home: EpisodeHomeRandomizationCfg,
) -> None:
    """Validate shared axes and reject generic pose axes that would double-reset."""

    validate_randomization_cfg(profile)
    if profile.target_pose.enabled or profile.robot_root.enabled or profile.presence.enabled:
        raise ValueError(
            "Coffee transport uses coffee_positions and episode_home instead of "
            "generic target/root/presence pose randomization."
        )
    if tuple(coffee_positions.object_names) != COFFEE_CAN_NAMES:
        raise ValueError(
            "Coffee position randomization must own exactly the two task cans "
            f"in canonical order: {COFFEE_CAN_NAMES}."
        )
    if tuple(coffee_visual_yaw.object_names) != COFFEE_CAN_NAMES:
        raise ValueError(
            "Coffee visual-yaw randomization must own exactly the two orange "
            f"task cans in canonical order: {COFFEE_CAN_NAMES}."
        )
    yaw_min, yaw_max = coffee_visual_yaw.yaw_range_rad
    if not (-math.pi <= yaw_min <= yaw_max <= math.pi):
        raise ValueError(
            "Coffee visual-yaw range must be ordered and inside [-pi, pi], "
            f"got {coffee_visual_yaw.yaw_range_rad}."
        )
    if min(
        episode_home.depth_x_max_m,
        episode_home.lateral_y_max_m,
        episode_home.yaw_max_rad,
    ) < 0.0:
        raise ValueError("Episode HOME randomization ranges must be non-negative.")
    if episode_home.depth_x_max_m > ROBOT_SPAWN_XY_RANDOM_M:
        raise ValueError("Episode HOME depth range exceeds the reviewed 2 cm bound.")
    if episode_home.lateral_y_max_m > ROBOT_SPAWN_XY_RANDOM_M:
        raise ValueError("Episode HOME lateral range exceeds the reviewed 2 cm bound.")
    if episode_home.yaw_max_rad > ROBOT_SPAWN_YAW_RANDOM_RAD + math.ulp(1.0):
        raise ValueError("Episode HOME yaw range exceeds the reviewed 1 degree bound.")


def validate_lift_start_cfg(cfg: LiftStartRandomizationCfg) -> None:
    lower, upper = cfg.position_range_m
    if not LIFT_TRAVEL_COMMAND_M <= lower <= upper <= 0.0:
        raise ValueError(
            "Lift start range must stay inside the reviewed [-0.28, 0.0] m interval."
        )


def validate_base_route_start_cfg(cfg: BaseRouteStartRandomizationCfg) -> None:
    lower, upper = cfg.offset_range_m
    if not 0.0 <= lower <= upper <= CABINET_RIGHT_DELTA_Y_M:
        raise ValueError(
            "Base start offset must stay on the HOME-to-right-cabinet route."
        )
