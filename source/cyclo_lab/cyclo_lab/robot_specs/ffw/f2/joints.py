"""FFW F2 joint groups and public joint orders."""

FFW_F2_LEFT_ARM_JOINT_NAMES = tuple(f"arm_l_joint{index}" for index in range(1, 8))
FFW_F2_RIGHT_ARM_JOINT_NAMES = tuple(f"arm_r_joint{index}" for index in range(1, 8))
FFW_F2_LEFT_GRIPPER_JOINT_NAMES = ("gripper_l_joint1",)
FFW_F2_RIGHT_GRIPPER_JOINT_NAMES = ("gripper_r_joint1",)
FFW_F2_LEFT_GRIPPER_PHYSICAL_JOINT_NAMES = tuple(
    f"gripper_l_joint{index}" for index in range(1, 5)
)
FFW_F2_RIGHT_GRIPPER_PHYSICAL_JOINT_NAMES = tuple(
    f"gripper_r_joint{index}" for index in range(1, 5)
)
FFW_F2_HEAD_JOINT_NAMES = ("head_joint1", "head_joint2")
FFW_F2_LIFT_JOINT_NAME = "lift_joint"
FFW_F2_LIFT_JOINT_NAMES = (FFW_F2_LIFT_JOINT_NAME,)

FFW_F2_PUBLISHED_JOINT_NAMES = (
    *FFW_F2_LEFT_ARM_JOINT_NAMES,
    *FFW_F2_LEFT_GRIPPER_JOINT_NAMES,
    *FFW_F2_RIGHT_ARM_JOINT_NAMES,
    *FFW_F2_RIGHT_GRIPPER_JOINT_NAMES,
    *FFW_F2_HEAD_JOINT_NAMES,
    *FFW_F2_LIFT_JOINT_NAMES,
)
FFW_F2_ACTION_JOINT_NAMES = FFW_F2_PUBLISHED_JOINT_NAMES

# Canonical limits from ROBOTIS-GIT/ai_worker tag 2.2.5.
FFW_F2_JOINT_POSITION_LIMITS = {
    "arm_l_joint1": (-3.14, 3.14),
    "arm_l_joint2": (0.0, 3.14),
    "arm_l_joint3": (-3.14, 3.14),
    "arm_l_joint4": (-2.9361, 1.0786),
    "arm_l_joint5": (-3.14, 3.14),
    "arm_l_joint6": (-1.57, 1.57),
    "arm_l_joint7": (-1.5804, 1.8201),
    "gripper_l_joint1": (0.0, 1.1),
    "arm_r_joint1": (-3.14, 3.14),
    "arm_r_joint2": (-3.14, 0.0),
    "arm_r_joint3": (-3.14, 3.14),
    "arm_r_joint4": (-2.9361, 1.0786),
    "arm_r_joint5": (-3.14, 3.14),
    "arm_r_joint6": (-1.57, 1.57),
    "arm_r_joint7": (-1.8201, 1.5804),
    "gripper_r_joint1": (0.0, 1.1),
    "head_joint1": (-0.2317, 0.6951),
    "head_joint2": (-0.35, 0.35),
    "lift_joint": (-0.5, 0.0),
}
