"""Randomization profiles owned by showroom Task000458."""

import math

from ...randomization.cfg import (
    CameraRandomizationCfg,
    LightingRandomizationCfg,
    ObjectPoseRandomizationCfg,
    PresenceRandomizationCfg,
    RobotRootRandomizationCfg,
    ShelfAppearanceRandomizationCfg,
    ShowroomGenerationRandomizationCfg,
    ShowroomRandomizationCfg,
    TargetPoseRandomizationCfg,
    WallAppearanceRandomizationCfg,
)
from .spec import TASK_000458_SPEC


TASK000458_PRESENCE_CANDIDATES = (
    "peanut_mix_bag",
    *(f"peanut_mix_bag_{index:02d}" for index in range(1, 6)),
    *(
        "roasted_chestnut_bag"
        if index == 0
        else f"roasted_chestnut_bag_{index:02d}"
        for index in range(6)
    ),
    *(f"jelly_bag_{index:02d}" for index in range(1, 13)),
)
TASK000458_NON_TARGET_OBJECTS = tuple(
    name
    for name in TASK000458_PRESENCE_CANDIDATES
    if name != TASK_000458_SPEC.target_object
)

TASK000458_RECORD_DETERMINISTIC = ShowroomRandomizationCfg()

TASK000458_RECORD_RANDOM = ShowroomRandomizationCfg(
    robot_root=RobotRootRandomizationCfg(
        enabled=True,
        depth_x_max_m=0.030,
        lateral_y_max_m=0.030,
        yaw_max_rad=math.radians(5.0),
    ),
    objects=ObjectPoseRandomizationCfg(
        enabled=True,
        object_names=(TASK_000458_SPEC.target_object,),
        x_max_m=0.010,
        y_max_m=0.010,
        yaw_max_rad=math.radians(10.0),
    ),
)

TASK000458_MIMIC_SEED = ShowroomGenerationRandomizationCfg()

TASK000458_MIMIC_GENERATION = ShowroomGenerationRandomizationCfg(
    target_pose=TargetPoseRandomizationCfg(enabled=True),
    presence=PresenceRandomizationCfg(
        enabled=True,
        object_names=TASK000458_NON_TARGET_OBJECTS,
    ),
    lighting=LightingRandomizationCfg(enabled=True),
    shelf=ShelfAppearanceRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000458_SPEC.policy_cameras,
    ),
)

TASK000458_AUGMENT_RANDOM = ShowroomGenerationRandomizationCfg(
    presence=PresenceRandomizationCfg(
        enabled=True,
        object_names=TASK000458_NON_TARGET_OBJECTS,
    ),
    lighting=LightingRandomizationCfg(enabled=True),
    shelf=ShelfAppearanceRandomizationCfg(enabled=True),
    wall=WallAppearanceRandomizationCfg(enabled=True),
    camera=CameraRandomizationCfg(
        enabled=True,
        camera_names=TASK_000458_SPEC.policy_cameras,
    ),
)
