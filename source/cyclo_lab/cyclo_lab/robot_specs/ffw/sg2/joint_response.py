"""Measured SG2 arm and gripper command-response profile."""

from __future__ import annotations

from dataclasses import dataclass

from .joints import (
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_RIGHT_ARM_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
)


SG2_MEASURED_RESPONSE_PROFILE_ID = "sg2-1050-joint-response-2026-08-24"
SG2_MEASURED_RESPONSE_VARIATION_STD_FRACTION = 0.05
SG2_MEASURED_RESPONSE_VARIATION_SCALE_BOUNDS = (0.85, 1.15)


@dataclass(frozen=True)
class SG2JointResponseGroup:
    """Nominal closed-loop response shared by a hardware-related joint group."""

    name: str
    joint_names: tuple[str, ...]
    delay_seconds: float
    filter_time_constant_seconds: float


SG2_MEASURED_RESPONSE_GROUPS = (
    SG2JointResponseGroup(
        name="arm joints 1 to 6",
        joint_names=(*FFW_SG2_LEFT_ARM_JOINT_NAMES[:6], *FFW_SG2_RIGHT_ARM_JOINT_NAMES[:6]),
        delay_seconds=0.085,
        filter_time_constant_seconds=0.070,
    ),
    SG2JointResponseGroup(
        name="arm joint 7",
        joint_names=(FFW_SG2_LEFT_ARM_JOINT_NAMES[6], FFW_SG2_RIGHT_ARM_JOINT_NAMES[6]),
        delay_seconds=0.070,
        filter_time_constant_seconds=0.035,
    ),
    SG2JointResponseGroup(
        name="gripper",
        joint_names=(*FFW_SG2_LEFT_GRIPPER_JOINT_NAMES, *FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES),
        delay_seconds=0.010,
        filter_time_constant_seconds=0.050,
    ),
)


# These simulator-side offsets cancel the stock model's HOME gravity sag. They are
# not encoder-zero corrections and must never be added to real-robot commands.
SG2_MEASURED_TARGET_OFFSETS_RAD = {
    "arm_l_joint1": -0.017326655,
    "arm_l_joint2": 0.009596735,
    "arm_l_joint3": 0.009360122,
    "arm_l_joint4": -0.013356014,
    "arm_l_joint5": -0.004394775,
    "arm_l_joint6": -0.002288955,
    "arm_l_joint7": -0.000011080,
    "gripper_l_joint1": 0.000000741,
    "arm_r_joint1": -0.017124539,
    "arm_r_joint2": -0.009704820,
    "arm_r_joint3": -0.009417301,
    "arm_r_joint4": -0.013170099,
    "arm_r_joint5": 0.004368097,
    "arm_r_joint6": -0.002604828,
    "arm_r_joint7": 0.000033451,
    "gripper_r_joint1": -0.000398407,
}
