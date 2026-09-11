"""Head-camera ACT deployment contract for Coffee Transport Task 000002."""

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES


_ARM_AND_GRIPPER_NAMES = FFW_SG2_ACTION_JOINT_NAMES[:16]

POLICY_CONTRACT = {
    "task": "task_000002",
    "robot": "ffw_sg2_rev1",
    "policy_hz": 15,
    "state_components": ("arm_left", "arm_right", "mobile"),
    "action_components": ("arm_left", "arm_right", "mobile"),
    "state_names": (
        *_ARM_AND_GRIPPER_NAMES,
        "linear_x",
        "linear_y",
        "angular_z",
    ),
    "action_names": (
        *_ARM_AND_GRIPPER_NAMES,
        "linear_x",
        "linear_y",
        "angular_z",
    ),
    "inactive_actions": {"head": "hold_current", "lift": "hold_current"},
    "cameras": {"cam_left_head": {"width": 672, "height": 376}},
    "simulation": {
        "environment": "Cyclo-Real-Showroom-Task000002-FFW-SG2-v0",
        "randomized_environment": (
            "Cyclo-Real-Showroom-Task000002-Random-FFW-SG2-v0"
        ),
        "default_reset": "deterministic",
    },
}
