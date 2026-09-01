"""Task-neutral FFW SG2 action configuration for Isaac Lab Mimic."""

from __future__ import annotations

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.utils import configclass

from cyclo_lab.manager_based.actions.ffw_sg2 import make_ffw_sg2_joint_position_action_cfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
    FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES,
)


class PostureBiasedDifferentialInverseKinematicsAction(
    DifferentialInverseKinematicsAction
):
    """Track an EEF pose while resolving the 7-DoF null space toward home.

    A Cartesian pose constrains six dimensions, so the SG2 seven-joint arm can
    otherwise return to the correct TCP pose with a visibly different elbow
    posture. The extra update is projected into the Jacobian null space and
    therefore does not replace the demonstrated Cartesian trajectory.
    """

    cfg: "PostureBiasedDifferentialInverseKinematicsActionCfg"

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]
        if ee_quat_curr.norm() != 0:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(
                ee_pos_curr, ee_quat_curr, jacobian, joint_pos
            )

            jacobian_t = torch.transpose(jacobian, dim0=1, dim1=2)
            damping = float(self.cfg.controller.ik_params["lambda_val"])
            task_eye = torch.eye(
                jacobian.shape[1], device=self.device, dtype=jacobian.dtype
            ).unsqueeze(0)
            jacobian_pinv = jacobian_t @ torch.linalg.inv(
                jacobian @ jacobian_t + damping**2 * task_eye
            )
            joint_eye = torch.eye(
                jacobian.shape[2], device=self.device, dtype=jacobian.dtype
            ).unsqueeze(0)
            null_projector = joint_eye - jacobian_pinv @ jacobian
            home_joint_pos = self._asset.data.default_joint_pos[:, self._joint_ids]
            posture_delta = self.cfg.posture_gain * (
                null_projector @ (home_joint_pos - joint_pos).unsqueeze(-1)
            ).squeeze(-1)
            posture_delta = torch.clamp(
                posture_delta,
                min=-self.cfg.posture_step_max_rad,
                max=self.cfg.posture_step_max_rad,
            )
            joint_pos_des = joint_pos_des + posture_delta
        else:
            joint_pos_des = joint_pos.clone()
        self._asset.set_joint_position_target(joint_pos_des, self._joint_ids)


@configclass
class PostureBiasedDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for the posture-biased seven-joint IK action."""

    class_type: type = PostureBiasedDifferentialInverseKinematicsAction
    posture_gain: float = 0.20
    posture_step_max_rad: float = 0.03


def configure_ffw_sg2_mimic_ik_actions(
    actions, posture_bias_sides: tuple[str, ...] = ()
) -> None:
    """Configure the EEF19 prefix while preserving an existing base3 term.

    Stationary tasks therefore remain 19D. Task525 already owns a
    ``base_velocity`` action term, so it becomes EEF19 + base3 = 22D.
    """
    for side in ("l", "r"):
        action_cfg_type = (
            PostureBiasedDifferentialInverseKinematicsActionCfg
            if side in posture_bias_sides
            else DifferentialInverseKinematicsActionCfg
        )
        setattr(
            actions,
            f"arm_{side}_action",
            action_cfg_type(
                asset_name="robot",
                joint_names=[f"arm_{side}_joint[1-7]"],
                body_name=f"arm_{side}_link7",
                controller=DifferentialIKControllerCfg(
                    command_type="pose",
                    ik_params={"lambda_val": 0.05},
                    ik_method="dls",
                    use_relative_mode=False,
                ),
                body_offset=action_cfg_type.OffsetCfg(
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
