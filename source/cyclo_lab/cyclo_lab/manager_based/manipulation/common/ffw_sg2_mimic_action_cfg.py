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

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._external_joint_position_hold_target: torch.Tensor | None = None

    @property
    def external_joint_position_hold_active(self) -> bool:
        """Whether an explicit joint target is replacing IK output."""

        return self._external_joint_position_hold_target is not None

    def begin_external_joint_position_hold(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Hold the existing command and return it with the current actual pose."""

        if self.external_joint_position_hold_active:
            raise RuntimeError("external joint-position hold is already active")
        commanded_target = (
            self._asset.data.joint_pos_target[:, self._joint_ids]
            .detach()
            .clone()
        )
        actual_joint_pos = (
            self._asset.data.joint_pos[:, self._joint_ids].detach().clone()
        )
        if not bool(torch.isfinite(commanded_target).all().item()):
            raise ValueError("external joint-position hold target is not finite")
        if not bool(torch.isfinite(actual_joint_pos).all().item()):
            raise ValueError("external joint-position hold actual pose is not finite")
        self._external_joint_position_hold_target = commanded_target
        self._asset.set_joint_position_target(commanded_target, self._joint_ids)
        return commanded_target.clone(), actual_joint_pos

    def end_external_joint_position_hold(self) -> None:
        """Restore normal IK target generation on the next physics step."""

        self._external_joint_position_hold_target = None

    def apply_actions(self):
        if self._external_joint_position_hold_target is not None:
            self._asset.set_joint_position_target(
                self._external_joint_position_hold_target, self._joint_ids
            )
            return

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

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids=env_ids)
        self.end_external_joint_position_hold()


class InitialJointHoldDifferentialInverseKinematicsAction(
    DifferentialInverseKinematicsAction
):
    """Keep the passive Mimic arm at its configured reset joint pose."""

    def apply_actions(self):
        self._asset.set_joint_position_target(
            self._asset.data.default_joint_pos[:, self._joint_ids], self._joint_ids
        )


@configclass
class InitialJointHoldDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    class_type: type = InitialJointHoldDifferentialInverseKinematicsAction


@configclass
class PostureBiasedDifferentialInverseKinematicsActionCfg(
    DifferentialInverseKinematicsActionCfg
):
    """Configuration for the posture-biased seven-joint IK action."""

    class_type: type = PostureBiasedDifferentialInverseKinematicsAction
    posture_gain: float = 0.20
    posture_step_max_rad: float = 0.03


def configure_ffw_sg2_mimic_ik_actions(
    actions,
    posture_bias_sides: tuple[str, ...] = (),
    *,
    joint_hold_sides: tuple[str, ...] = (),
    include_head: bool = True,
    include_lift: bool = True,
    eef_offset_z_m: float = -0.2,
) -> None:
    """Configure dual-arm EEF control and optional passive platform joints.

    Existing stationary tasks remain EEF19 by default. Task525 already owns a
    ``base_velocity`` action term, so its default becomes EEF19 + base3 = 22D.
    Tasks whose public contract excludes head/lift can request EEF16 without
    adding those action terms.
    """
    if set(joint_hold_sides) & set(posture_bias_sides):
        raise ValueError("An arm cannot use both joint hold and posture-biased IK")
    for side in ("l", "r"):
        action_cfg_type = (
            InitialJointHoldDifferentialInverseKinematicsActionCfg
            if side in joint_hold_sides
            else PostureBiasedDifferentialInverseKinematicsActionCfg
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
                    pos=[0.0, 0.0, eef_offset_z_m]
                ),
            ),
        )

    actions.gripper_l_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_LEFT_GRIPPER_JOINT_NAMES
    )
    actions.gripper_r_action = make_ffw_sg2_joint_position_action_cfg(
        FFW_SG2_RIGHT_GRIPPER_JOINT_NAMES
    )
    if include_head:
        actions.head_action = make_ffw_sg2_joint_position_action_cfg(
            FFW_SG2_HEAD_JOINT_NAMES
        )
    if include_lift:
        actions.lift_action = make_ffw_sg2_joint_position_action_cfg(
            FFW_SG2_LIFT_JOINT_NAMES
        )
