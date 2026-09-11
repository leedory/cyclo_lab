"""Authoritative randomization profiles for horizontal coffee transport."""

from __future__ import annotations

from dataclasses import dataclass

from ....randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    WallAppearanceRandomizationCfg,
)
from ..randomization_cfg import (
    BaseRouteStartRandomizationCfg,
    CoffeePositionRandomizationCfg,
    CoffeeVisualYawRandomizationCfg,
    EpisodeHomeRandomizationCfg,
    validate_base_route_start_cfg,
    validate_coffee_transport_randomization_cfg,
)
from .spec import TASK_000002_SPEC


@dataclass(frozen=True)
class Task000002RandomizationCfg(ShowroomGenerationRandomizationCfg):
    coffee_positions: CoffeePositionRandomizationCfg = CoffeePositionRandomizationCfg()
    coffee_visual_yaw: CoffeeVisualYawRandomizationCfg = (
        CoffeeVisualYawRandomizationCfg()
    )
    episode_home: EpisodeHomeRandomizationCfg = EpisodeHomeRandomizationCfg()
    base_route_start: BaseRouteStartRandomizationCfg = BaseRouteStartRandomizationCfg()


def validate_task000002_randomization_cfg(
    profile: Task000002RandomizationCfg,
) -> None:
    validate_coffee_transport_randomization_cfg(
        profile,
        coffee_positions=profile.coffee_positions,
        coffee_visual_yaw=profile.coffee_visual_yaw,
        episode_home=profile.episode_home,
    )
    validate_base_route_start_cfg(profile.base_route_start)


TASK000002_DETERMINISTIC = Task000002RandomizationCfg()
TASK000002_SEED = Task000002RandomizationCfg()

TASK000002_RECORD_RANDOMIZED = Task000002RandomizationCfg(
    coffee_positions=CoffeePositionRandomizationCfg(enabled=True),
    episode_home=EpisodeHomeRandomizationCfg(enabled=True),
    base_route_start=BaseRouteStartRandomizationCfg(enabled=True),
)

# HOME is sampled first; route offset is then measured from that HOME.
TASK000002_PHYSICAL_TRAJECTORY_GENERATION = Task000002RandomizationCfg(
    coffee_positions=CoffeePositionRandomizationCfg(enabled=True),
    episode_home=EpisodeHomeRandomizationCfg(enabled=True),
    base_route_start=BaseRouteStartRandomizationCfg(enabled=True),
)

# Production generator profile promoted after the commanded-target hold probes.
TASK000002_PRODUCTION_TRAJECTORY_GENERATION = (
    TASK000002_PHYSICAL_TRAJECTORY_GENERATION
)

# Replay augmentation cannot alter the lateral route or collision clearance.
TASK000002_VISUAL_REPLAY_AUGMENTATION = Task000002RandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000002_SPEC.policy_cameras,
    ),
    coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True),
)
