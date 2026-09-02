"""Default learned-policy contract for Task000525 deployment."""

from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_MOBILE_ACTION_NAMES


POLICY_CONTRACT = {
    "task": "task_000525",
    "robot": "ffw_sg2_rev1",
    "policy_hz": 15,
    "state_components": ("arm_left", "arm_right", "head", "lift", "mobile"),
    "action_components": ("arm_left", "arm_right", "head", "lift", "mobile"),
    "state_names": FFW_SG2_MOBILE_ACTION_NAMES,
    "action_names": FFW_SG2_MOBILE_ACTION_NAMES,
    "inactive_actions": {},
    "cameras": {
        "cam_left_head": {"width": 672, "height": 376},
        "cam_left_wrist": {"width": 480, "height": 640},
        "cam_right_wrist": {"width": 480, "height": 640},
    },
    "simulation": {
        "environment": "Cyclo-Real-Showroom-Task000525-FFW-SG2-v0",
        "randomized_environment": "Cyclo-Real-Showroom-Task000525-Random-FFW-SG2-v0",
        "default_reset": "deterministic",
    },
}
