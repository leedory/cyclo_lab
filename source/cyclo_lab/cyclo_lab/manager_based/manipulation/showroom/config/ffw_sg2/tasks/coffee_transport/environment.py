"""Reusable scene and reset wiring for coffee-transport environments."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import EventTermCfg as EventTerm

from cyclo_lab.assets.environments.robotis_showroom import (
    iter_robotis_showroom_object_cfgs,
)

from .coffee_transport_common import (
    COFFEE_TRANSPORT_HOME_POS,
    COFFEE_TRANSPORT_HOME_ROT,
    SOURCE_SPAWN_SQUARES,
    iter_static_task_jelly_bag_cfgs,
    make_orange_coffee_can_cfg,
)
from .reset_events import (
    randomize_coffee_can_positions,
    randomize_coffee_can_visual_yaw,
    reset_episode_home_and_base_start,
)


def add_coffee_transport_cans(scene: Any) -> None:
    """Add the two task-owned cans before the parent scene is finalized."""

    for square in SOURCE_SPAWN_SQUARES:
        setattr(scene, square.name, make_orange_coffee_can_cfg(square))


def configure_coffee_transport_scene(scene: Any) -> None:
    """Apply the shared robot HOME and static furniture layout."""

    scene.robot.init_state.pos = COFFEE_TRANSPORT_HOME_POS
    scene.robot.init_state.rot = COFFEE_TRANSPORT_HOME_ROT
    for object_name, _ in iter_robotis_showroom_object_cfgs():
        if hasattr(scene, object_name):
            delattr(scene, object_name)
    for object_name, object_cfg in iter_static_task_jelly_bag_cfgs():
        setattr(scene, object_name, object_cfg)


def configure_coffee_randomization_events(events: Any, profile: Any) -> None:
    """Configure reset events shared by every coffee-transport task."""

    coffee_positions = profile.coffee_positions
    if coffee_positions.enabled:
        events.randomize_coffee_positions = EventTerm(
            func=randomize_coffee_can_positions,
            mode="reset",
            params={"object_names": coffee_positions.object_names},
        )

    coffee_visual_yaw = profile.coffee_visual_yaw
    if coffee_visual_yaw.enabled:
        events.randomize_coffee_visual_yaw = EventTerm(
            func=randomize_coffee_can_visual_yaw,
            mode="reset",
            params={
                "object_names": coffee_visual_yaw.object_names,
                "yaw_range_rad": coffee_visual_yaw.yaw_range_rad,
            },
        )


def make_episode_home_reset_event(
    episode_home: Any,
    *,
    base_route_start: Any | None = None,
) -> EventTerm:
    """Build the shared HOME reset, optionally with a mobile route offset."""

    params = {
        "home_enabled": episode_home.enabled,
        "depth_x_max_m": episode_home.depth_x_max_m,
        "lateral_y_max_m": episode_home.lateral_y_max_m,
        "yaw_max_rad": episode_home.yaw_max_rad,
    }
    if base_route_start is not None:
        params.update(
            route_enabled=base_route_start.enabled,
            route_offset_range_m=base_route_start.offset_range_m,
        )
    return EventTerm(
        func=reset_episode_home_and_base_start,
        mode="reset",
        params=params,
    )
