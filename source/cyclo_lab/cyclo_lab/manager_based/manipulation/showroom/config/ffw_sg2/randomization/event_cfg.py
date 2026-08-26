"""Build reset-event configurations directly from selected profiles."""

from __future__ import annotations

from typing import Any

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg

from . import appearance_events
from . import events as pose_events
from . import task_pose_events
from .cfg import (
    ShowroomGenerationRandomizationCfg,
    ShowroomRandomizationCfg,
    validate_randomization_cfg,
)


def configure_profiled_reset_events(
    event_cfg: Any,
    profile: ShowroomRandomizationCfg | ShowroomGenerationRandomizationCfg,
    *,
    target_object: str | None = None,
) -> Any:
    """Replace optional reset terms using one authoritative profile."""
    validate_randomization_cfg(profile)
    if isinstance(profile, ShowroomRandomizationCfg):
        return _configure_pose_events(event_cfg, profile)
    if isinstance(profile, ShowroomGenerationRandomizationCfg):
        if not target_object:
            raise ValueError("A target object is required for task-generation randomization.")
        return _configure_generation_events(event_cfg, profile, target_object)
    raise TypeError(f"Unsupported showroom randomization profile: {type(profile)!r}")


def validate_profile_scene_entities(
    scene_cfg: Any,
    profile: ShowroomRandomizationCfg | ShowroomGenerationRandomizationCfg,
    *,
    target_object: str | None = None,
) -> None:
    """Reject enabled profile entities that are absent from the configured scene."""
    object_names: tuple[str, ...]
    if isinstance(profile, ShowroomRandomizationCfg):
        object_names = profile.objects.object_names if profile.objects.enabled else ()
    elif isinstance(profile, ShowroomGenerationRandomizationCfg):
        if (
            target_object
            and profile.presence.enabled
            and target_object in profile.presence.object_names
        ):
            raise ValueError("The task target cannot also be randomized as non-target presence.")
        target_names = (target_object,) if profile.target_pose.enabled and target_object else ()
        presence_names = profile.presence.object_names if profile.presence.enabled else ()
        object_names = (*target_names, *presence_names)
    else:
        raise TypeError(f"Unsupported showroom randomization profile: {type(profile)!r}")

    missing = tuple(name for name in object_names if not hasattr(scene_cfg, name))
    if missing:
        raise ValueError(f"Randomized showroom objects are missing from the scene: {missing}")
    if isinstance(profile, ShowroomGenerationRandomizationCfg) and profile.camera.enabled:
        missing_cameras = tuple(
            name for name in profile.camera.camera_names if not hasattr(scene_cfg, name)
        )
        if missing_cameras:
            raise ValueError(
                f"Randomized showroom cameras are missing from the scene: {missing_cameras}"
            )


def _configure_pose_events(event_cfg: Any, profile: ShowroomRandomizationCfg) -> Any:
    robot = profile.robot_root
    event_cfg.randomize_robot_root_pose = (
        EventTerm(
            func=pose_events.randomize_root_pose_in_xy_box,
            mode="reset",
            params={
                "x_max": robot.depth_x_max_m,
                "y_max": robot.lateral_y_max_m,
                "yaw_max": robot.yaw_max_rad,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        if robot.enabled
        else None
    )

    objects = profile.objects
    event_cfg.randomize_selected_objects = (
        EventTerm(
            func=pose_events.randomize_root_poses_in_xy_box,
            mode="reset",
            params={
                "object_names": objects.object_names,
                "x_max": objects.x_max_m,
                "y_max": objects.y_max_m,
                "yaw_max": objects.yaw_max_rad,
            },
        )
        if objects.enabled
        else None
    )
    return event_cfg


def _configure_generation_events(
    event_cfg: Any,
    profile: ShowroomGenerationRandomizationCfg,
    target_object: str,
) -> Any:
    target = profile.target_pose
    event_cfg.refresh_shelf_support = (
        EventTerm(func=task_pose_events.refresh_shelf_support_collider, mode="reset")
        if target.enabled
        else None
    )
    event_cfg.randomize_target_pose = (
        EventTerm(
            func=task_pose_events.randomize_target_pose,
            mode="reset",
            params={
                "lateral_y_max_m": target.lateral_y_max_m,
                "yaw_max_rad": target.yaw_max_rad,
                "asset_cfg": SceneEntityCfg(target_object),
            },
        )
        if target.enabled
        else None
    )

    robot = profile.robot_root
    event_cfg.randomize_robot_root = (
        EventTerm(
            func=task_pose_events.randomize_robot_root,
            mode="reset",
            params={
                "depth_x_max_m": robot.depth_x_max_m,
                "lateral_y_max_m": robot.lateral_y_max_m,
                "yaw_max_rad": robot.yaw_max_rad,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        if robot.enabled
        else None
    )

    presence = profile.presence
    event_cfg.randomize_non_target_presence = (
        EventTerm(
            func=appearance_events.randomize_non_target_presence,
            mode="reset",
            params={
                "object_names": presence.object_names,
                "disappearance_probability": presence.disappearance_probability,
            },
        )
        if presence.enabled
        else None
    )

    lighting = profile.lighting
    event_cfg.randomize_lighting = (
        EventTerm(
            func=appearance_events.randomize_dome_and_weak_keys,
            mode="reset",
            params={
                "dome_intensity_range": lighting.dome_intensity_range,
                "dome_rgb_range": lighting.dome_rgb_range,
                "weak_key_intensity_range": lighting.weak_key_intensity_range,
            },
        )
        if lighting.enabled
        else None
    )

    shelf = profile.shelf
    event_cfg.randomize_shelf_appearance = (
        EventTerm(
            func=appearance_events.randomize_shelf_texture_scale,
            mode="reset",
            params={
                "brightness_range": shelf.brightness_range,
                "channel_tint_max": shelf.channel_tint_max,
            },
        )
        if shelf.enabled
        else None
    )

    wall = profile.wall
    event_cfg.randomize_wall_color = (
        EventTerm(
            func=appearance_events.randomize_wall_solid_rgb,
            mode="reset",
            params={
                "mode": wall.mode,
                "rgb_range": wall.rgb_range,
                "near_white_range": wall.near_white_range,
            },
        )
        if wall.enabled
        else None
    )

    camera = profile.camera
    event_cfg.randomize_cameras = (
        EventTerm(
            func=appearance_events.randomize_policy_cameras,
            mode="reset",
            params={
                "camera_names": camera.camera_names,
                "coupled_focal_scale_range": camera.coupled_focal_scale_range,
                "local_roll_max_rad": camera.local_roll_max_rad,
                "local_pitch_max_rad": camera.local_pitch_max_rad,
                "local_yaw_max_rad": camera.local_yaw_max_rad,
            },
        )
        if camera.enabled
        else None
    )
    return event_cfg
