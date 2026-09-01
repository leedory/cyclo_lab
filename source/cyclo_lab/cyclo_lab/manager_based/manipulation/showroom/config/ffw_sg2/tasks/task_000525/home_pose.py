"""Task000525 home pose shared by Isaac reset and the active A3 leader.

The arm/gripper values exactly mirror ``left/right_save_pose_3`` in the
running A3 right-only controller. R and N use the complete neutral pose. G
uses the same arm target but deliberately overrides head, lift, and the
carrying right gripper in its task-local state machine.
"""

TASK000525_SAVE_POSE_3_JOINT_POSITIONS = {
    "arm_l_joint1": 0.0005,
    "arm_l_joint2": 0.6040,
    "arm_l_joint3": -0.2963,
    "arm_l_joint4": -2.5052,
    "arm_l_joint5": 0.5672,
    "arm_l_joint6": 0.4926,
    "arm_l_joint7": 0.7391,
    "gripper_l_joint1": 0.0,
    "arm_r_joint1": 0.0005,
    "arm_r_joint2": -0.6040,
    "arm_r_joint3": 0.2963,
    "arm_r_joint4": -2.5052,
    "arm_r_joint5": -0.5672,
    "arm_r_joint6": 0.4926,
    "arm_r_joint7": -0.7391,
    "gripper_r_joint1": 0.0,
    # R/N reset looks slightly downward; G overrides pitch to maximum-down.
    "head_joint1": 0.2,
    "head_joint2": 0.0,
    "lift_joint": 0.0,
}
