"""Task-neutral FFW SG2 action configuration for Isaac Lab Mimic."""

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg

from cyclo_lab.manager_based.actions.ffw_sg2 import make_ffw_sg2_joint_position_action_cfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
)


def configure_ffw_sg2_mimic_ik_actions(actions) -> None:
    """Configure the stable 19D dual-arm EEF/gripper/lift/head action layout."""
    for side in ("l", "r"):
        setattr(
            actions,
            f"arm_{side}_action",
            DifferentialInverseKinematicsActionCfg(
                asset_name="robot",
                joint_names=[f"arm_{side}_joint[1-7]"],
                body_name=f"arm_{side}_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False,
                ),
                body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(
                    pos=[0.0, 0.0, -0.2]
                ),
            ),
        )

    actions.gripper_l_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_LEFT_GRIPPER_JOINT_NAMES
    )
    actions.gripper_r_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES
    )
    actions.head_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_HEAD_JOINT_NAMES
    )
    actions.lift_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_LIFT_JOINT_NAMES
    )
