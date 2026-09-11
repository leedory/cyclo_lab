"""Head-camera ACT deployment contract for Coffee Transport Task 000001."""

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES


_ARM_AND_GRIPPER_NAMES = FFW_SG2_ACTION_JOINT_NAMES[:16]

POLICY_CONTRACT = {
    "task": "task_000001",
    "robot": "ffw_sg2_rev1",
    "policy_hz": 15,
    "state_components": ("arm_left", "arm_right", "lift"),
    "action_components": ("arm_left", "arm_right", "lift"),
    "state_names": (*_ARM_AND_GRIPPER_NAMES, "lift_joint"),
    "action_names": (*_ARM_AND_GRIPPER_NAMES, "lift_joint"),
    "inactive_actions": {"head": "hold_current", "mobile": "hold_current"},
    "cameras": {"cam_left_head": {"width": 672, "height": 376}},
    "simulation": {
        "environment": "Cyclo-Real-Showroom-Task000001-FFW-SG2-v0",
        "randomized_environment": (
            "Cyclo-Real-Showroom-Task000001-Random-FFW-SG2-v0"
        ),
        "default_reset": "deterministic",
    },
}
