"""Authoritative randomization profiles for vertical coffee transport."""

from __future__ import annotations

from dataclasses import dataclass

from ....randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    WallAppearanceRandomizationCfg,
)
from ..randomization_cfg import (
    CoffeePositionRandomizationCfg,
    CoffeeVisualYawRandomizationCfg,
    EpisodeHomeRandomizationCfg,
    LiftStartRandomizationCfg,
    validate_coffee_transport_randomization_cfg,
    validate_lift_start_cfg,
)
from .spec import TASK_000001_SPEC


@dataclass(frozen=True)
class Task000001RandomizationCfg(ShowroomGenerationRandomizationCfg):
    coffee_positions: CoffeePositionRandomizationCfg = CoffeePositionRandomizationCfg()
    coffee_visual_yaw: CoffeeVisualYawRandomizationCfg = (
        CoffeeVisualYawRandomizationCfg()
    )
    episode_home: EpisodeHomeRandomizationCfg = EpisodeHomeRandomizationCfg()
    lift_start: LiftStartRandomizationCfg = LiftStartRandomizationCfg()


def validate_task000001_randomization_cfg(
    profile: Task000001RandomizationCfg,
) -> None:
    validate_coffee_transport_randomization_cfg(
        profile,
        coffee_positions=profile.coffee_positions,
        coffee_visual_yaw=profile.coffee_visual_yaw,
        episode_home=profile.episode_home,
    )
    validate_lift_start_cfg(profile.lift_start)


TASK000001_DETERMINISTIC = Task000001RandomizationCfg()
TASK000001_SEED = Task000001RandomizationCfg()

TASK000001_RECORD_RANDOMIZED = Task000001RandomizationCfg(
    coffee_positions=CoffeePositionRandomizationCfg(enabled=True),
    episode_home=EpisodeHomeRandomizationCfg(enabled=True),
    lift_start=LiftStartRandomizationCfg(enabled=True),
)

# Physical generation changes only collision-relevant initial state.
TASK000001_PHYSICAL_TRAJECTORY_GENERATION = Task000001RandomizationCfg(
    coffee_positions=CoffeePositionRandomizationCfg(enabled=True),
    episode_home=EpisodeHomeRandomizationCfg(enabled=True),
    lift_start=LiftStartRandomizationCfg(enabled=True),
)

# Production generator profile promoted after the commanded-target hold probes.
TASK000001_PRODUCTION_TRAJECTORY_GENERATION = (
    TASK000001_PHYSICAL_TRAJECTORY_GENERATION
)

# Replay augmentation preserves the recorded robot, cans, and cabinet geometry.
TASK000001_VISUAL_REPLAY_AUGMENTATION = Task000001RandomizationCfg(
    lighting=LightingRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000001_SPEC.policy_cameras,
    ),
    coffee_visual_yaw=CoffeeVisualYawRandomizationCfg(enabled=True),
)
