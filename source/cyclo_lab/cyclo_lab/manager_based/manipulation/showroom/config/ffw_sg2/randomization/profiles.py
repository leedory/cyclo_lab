"""User-facing SG2 showroom randomization profiles.

Edit ``DIGITAL_TWIN_SELECTED_OBJECTS`` and ``DIGITAL_TWIN_RANDOM`` to change
which entities move when the Random Continuous showroom is reset.
"""

import math

from .cfg import (
    ObjectPoseRandomizationCfg,
    RobotRootRandomizationCfg,
    ShowroomRandomizationCfg,
)


DIGITAL_TWIN_SELECTED_OBJECTS = ("peanut_mix_bag_02",)

NO_RANDOMIZATION = ShowroomRandomizationCfg()

DIGITAL_TWIN_RANDOM = ShowroomRandomizationCfg(
    robot_root=RobotRootRandomizationCfg(
        enabled=True,
        depth_x_max_m=0.030,
        lateral_y_max_m=0.030,
        yaw_max_rad=math.radians(5.0),
    ),
    objects=ObjectPoseRandomizationCfg(
        enabled=True,
        object_names=DIGITAL_TWIN_SELECTED_OBJECTS,
        x_max_m=0.010,
        y_max_m=0.010,
        yaw_max_rad=math.radians(10.0),
    ),
)

# Task-specific profiles live below tasks/task_000xxx/profiles.py.
