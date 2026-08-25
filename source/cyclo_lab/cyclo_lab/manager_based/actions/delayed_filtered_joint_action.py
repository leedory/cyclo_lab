"""Joint-position action with measured delay and first-order response."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from cyclo_lab.utils.joint_command_response import CommandResponseGroup, JointCommandResponse


@configclass
class JointCommandResponseGroupCfg:
    """Readable configuration for joints that share one command response."""

    name: str = MISSING
    joint_names: tuple[str, ...] = ()
    delay_seconds: float = 0.0
    filter_time_constant_seconds: float = 0.0
    variation_std_fraction: float = 0.0
    variation_scale_bounds: tuple[float, float] = (1.0, 1.0)


class DelayedFilteredJointPositionAction(JointPositionAction):
    """Condition absolute targets while retaining the existing implicit simulator drive."""

    cfg: DelayedFilteredJointPositionActionCfg

    def __init__(self, cfg: DelayedFilteredJointPositionActionCfg, env) -> None:
        super().__init__(cfg, env)
        groups = []
        covered_names: set[str] = set()
        for group_cfg in cfg.response_groups:
            unknown_names = sorted(set(group_cfg.joint_names).difference(self._joint_names))
            if unknown_names:
                raise ValueError(f"Response group {group_cfg.name!r} has unknown joints: {unknown_names}.")
            duplicate_names = sorted(covered_names.intersection(group_cfg.joint_names))
            if duplicate_names:
                raise ValueError(f"Response groups overlap at joints: {duplicate_names}.")
            covered_names.update(group_cfg.joint_names)
            groups.append(
                CommandResponseGroup(
                    name=group_cfg.name,
                    joint_indices=tuple(self._joint_names.index(name) for name in group_cfg.joint_names),
                    delay_seconds=group_cfg.delay_seconds,
                    filter_time_constant_seconds=group_cfg.filter_time_constant_seconds,
                    variation_std_fraction=group_cfg.variation_std_fraction,
                    variation_scale_bounds=group_cfg.variation_scale_bounds,
                )
            )
        missing_names = sorted(set(self._joint_names).difference(covered_names))
        if missing_names:
            raise ValueError(f"Response groups do not cover joints: {missing_names}.")

        offsets = torch.zeros(self.action_dim, device=self.device)
        if isinstance(cfg.target_offset, (int, float)):
            offsets.fill_(float(cfg.target_offset))
        elif isinstance(cfg.target_offset, dict):
            unknown_offsets = sorted(set(cfg.target_offset).difference(self._joint_names))
            if unknown_offsets:
                raise ValueError(f"Target offsets have unknown joints: {unknown_offsets}.")
            for name, value in cfg.target_offset.items():
                offsets[self._joint_names.index(name)] = float(value)
        else:
            raise TypeError("target_offset must be a number or a joint-name mapping.")

        physics_dt = float(getattr(env, "physics_dt", 0.0))
        self._command_response = JointCommandResponse(
            num_envs=self.num_envs,
            num_joints=self.action_dim,
            physics_dt=physics_dt,
            groups=groups,
            target_offsets=offsets,
            device=self.device,
        )
        self._has_reset = False

    @property
    def conditioned_actions(self) -> torch.Tensor:
        """Targets currently sent to the articulation after delay, filtering, and offset."""
        return self._command_response.conditioned_targets

    @property
    def command_response(self) -> JointCommandResponse:
        """Runtime response state exposed for diagnostics and recording."""
        return self._command_response

    def apply_actions(self) -> None:
        if not self._has_reset:
            self.reset()
        targets = self._command_response.update(self.processed_actions)
        self._asset.set_joint_position_target(targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        current_targets = self._asset.data.joint_pos[:, self._joint_ids].detach()
        self._command_response.reset(current_targets, env_ids)
        self._has_reset = True


@configclass
class DelayedFilteredJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for delayed and filtered absolute joint-position targets."""

    class_type: type[ActionTerm] = DelayedFilteredJointPositionAction
    response_groups: tuple[JointCommandResponseGroupCfg, ...] = ()
    target_offset: float | dict[str, float] = 0.0
