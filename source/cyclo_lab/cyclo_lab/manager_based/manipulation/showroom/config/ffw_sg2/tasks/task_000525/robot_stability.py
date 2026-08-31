"""Task000525-only SG2 arm hold tuning."""

from __future__ import annotations

import math


TASK000525_ARM_HOLD_ACTUATOR_NAMES = ("DY_80", "DY_70", "DP-42")
TASK000525_ARM_STIFFNESS_SCALE = 2.0
TASK000525_ARM_DAMPING_SCALE = math.sqrt(TASK000525_ARM_STIFFNESS_SCALE)


def apply_task000525_arm_hold_tuning(robot_cfg) -> None:
    """Reduce gravity settle without changing SG2 dynamics in other tasks.

    Scaling damping with the square root of stiffness approximately preserves
    each implicit PD actuator's damping ratio. The effort limits are left
    unchanged so this does not grant the robot additional peak joint torque.
    """
    missing = [
        name
        for name in TASK000525_ARM_HOLD_ACTUATOR_NAMES
        if name not in robot_cfg.actuators
    ]
    if missing:
        raise KeyError(f"Task000525 SG2 arm actuators are missing: {missing}")

    for name in TASK000525_ARM_HOLD_ACTUATOR_NAMES:
        actuator = robot_cfg.actuators[name]
        if not isinstance(actuator.stiffness, (int, float)):
            raise TypeError(f"Task000525 actuator {name} stiffness must be numeric")
        if not isinstance(actuator.damping, (int, float)):
            raise TypeError(f"Task000525 actuator {name} damping must be numeric")
        actuator.stiffness *= TASK000525_ARM_STIFFNESS_SCALE
        actuator.damping *= TASK000525_ARM_DAMPING_SCALE
