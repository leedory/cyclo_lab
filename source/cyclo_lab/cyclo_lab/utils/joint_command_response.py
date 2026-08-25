"""Small, stateful command-response model for joint-position targets."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class CommandResponseGroup:
    """Joints that share one delayed first-order command response."""

    name: str
    joint_indices: tuple[int, ...]
    delay_seconds: float
    filter_time_constant_seconds: float
    variation_std_fraction: float = 0.0
    variation_scale_bounds: tuple[float, float] = (1.0, 1.0)


class JointCommandResponse:
    """Delay and smooth absolute joint targets before they reach a simulator drive."""

    def __init__(
        self,
        *,
        num_envs: int,
        num_joints: int,
        physics_dt: float,
        groups: Sequence[CommandResponseGroup],
        target_offsets: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        if num_envs <= 0 or num_joints <= 0:
            raise ValueError("num_envs and num_joints must be positive.")
        if not math.isfinite(physics_dt) or physics_dt <= 0.0:
            raise ValueError("physics_dt must be finite and positive.")

        self.num_envs = int(num_envs)
        self.num_joints = int(num_joints)
        self.physics_dt = float(physics_dt)
        self.device = torch.device(device)
        self.groups = tuple(groups)
        self._validate_groups()

        max_delay_seconds = max(
            group.delay_seconds * group.variation_scale_bounds[1] for group in self.groups
        )
        self._history_length = math.ceil(max_delay_seconds / self.physics_dt) + 1
        self._history = torch.zeros(
            self._history_length,
            self.num_envs,
            self.num_joints,
            device=self.device,
        )
        self._history_index = 0
        self._filtered_targets = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self._conditioned_targets = torch.zeros_like(self._filtered_targets)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._all_initialized = False
        self._batch_indices = torch.arange(self.num_envs, device=self.device)
        self._group_joint_indices = {
            group.name: torch.tensor(group.joint_indices, dtype=torch.long, device=self.device)
            for group in self.groups
        }
        self._response_scales = {
            group.name: torch.ones(self.num_envs, device=self.device) for group in self.groups
        }
        self._delay_seconds = {
            group.name: torch.full((self.num_envs,), group.delay_seconds, device=self.device)
            for group in self.groups
        }
        self._delay_steps = {
            group.name: torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
            for group in self.groups
        }
        self._filter_time_constants = {
            group.name: torch.full(
                (self.num_envs,), group.filter_time_constant_seconds, device=self.device
            )
            for group in self.groups
        }

        if target_offsets is None:
            self._target_offsets = torch.zeros(1, self.num_joints, device=self.device)
        else:
            offsets = torch.as_tensor(target_offsets, dtype=torch.float32, device=self.device)
            if offsets.shape not in ((self.num_joints,), (1, self.num_joints)):
                raise ValueError(
                    f"target_offsets must have shape ({self.num_joints},) or (1, {self.num_joints})."
                )
            self._target_offsets = offsets.reshape(1, self.num_joints)

    @property
    def conditioned_targets(self) -> torch.Tensor:
        return self._conditioned_targets

    @property
    def initialized(self) -> torch.Tensor:
        return self._initialized

    @property
    def response_scales(self) -> dict[str, torch.Tensor]:
        return {name: values.clone() for name, values in self._response_scales.items()}

    @property
    def delay_seconds(self) -> dict[str, torch.Tensor]:
        return {name: values.clone() for name, values in self._delay_seconds.items()}

    @property
    def filter_time_constants(self) -> dict[str, torch.Tensor]:
        return {name: values.clone() for name, values in self._filter_time_constants.items()}

    @property
    def realized_delay_seconds(self) -> dict[str, torch.Tensor]:
        return {
            name: steps.clone() * self.physics_dt for name, steps in self._delay_steps.items()
        }

    def reset(
        self,
        initial_targets: torch.Tensor,
        env_ids: Sequence[int] | torch.Tensor | slice | None = None,
    ) -> None:
        """Clear history and sample one response variation for each reset environment."""
        targets = torch.as_tensor(initial_targets, dtype=torch.float32, device=self.device)
        expected_shape = (self.num_envs, self.num_joints)
        if targets.shape != expected_shape:
            raise ValueError(f"initial_targets must have shape {expected_shape}, got {tuple(targets.shape)}.")

        ids = self._resolve_env_ids(env_ids)
        self._history[:, ids] = targets[ids].unsqueeze(0)
        self._filtered_targets[ids] = targets[ids]
        self._conditioned_targets[ids] = targets[ids] + self._target_offsets
        self._sample_response_parameters(ids)
        self._initialized[ids] = True
        if ids.numel() == self.num_envs:
            self._all_initialized = True

    def update(self, targets: torch.Tensor) -> torch.Tensor:
        """Advance one physics step and return the targets for the existing joint drive."""
        values = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        expected_shape = (self.num_envs, self.num_joints)
        if values.shape != expected_shape:
            raise ValueError(f"targets must have shape {expected_shape}, got {tuple(values.shape)}.")
        if not self._all_initialized:
            raise RuntimeError("JointCommandResponse.reset() must be called before update().")

        self._history_index = (self._history_index + 1) % self._history_length
        self._history[self._history_index] = values

        for group in self.groups:
            joint_indices = self._group_joint_indices[group.name]
            history_indices = torch.remainder(
                self._history_index - self._delay_steps[group.name], self._history_length
            )
            delayed = self._history[history_indices, self._batch_indices][:, joint_indices]

            time_constant = self._filter_time_constants[group.name]
            alpha = torch.ones_like(time_constant)
            active = time_constant > 0.0
            alpha[active] = 1.0 - torch.exp(-self.physics_dt / time_constant[active])
            current = self._filtered_targets[:, joint_indices]
            self._filtered_targets[:, joint_indices] = current + alpha.unsqueeze(1) * (delayed - current)

        self._conditioned_targets = self._filtered_targets + self._target_offsets
        return self._conditioned_targets

    def _validate_groups(self) -> None:
        if not self.groups:
            raise ValueError("At least one command-response group is required.")
        names: set[str] = set()
        covered_indices: set[int] = set()
        for group in self.groups:
            if not group.name or group.name in names:
                raise ValueError(f"Command-response group names must be unique and non-empty: {group.name!r}.")
            names.add(group.name)
            if not group.joint_indices:
                raise ValueError(f"Command-response group {group.name!r} has no joints.")
            if group.delay_seconds < 0.0 or group.filter_time_constant_seconds < 0.0:
                raise ValueError(f"Command-response group {group.name!r} has a negative time value.")
            if group.variation_std_fraction < 0.0:
                raise ValueError(f"Command-response group {group.name!r} has negative variation.")
            lower, upper = group.variation_scale_bounds
            if lower <= 0.0 or upper < lower or not (lower <= 1.0 <= upper):
                raise ValueError(
                    f"Command-response group {group.name!r} must have positive bounds containing 1.0."
                )
            for joint_index in group.joint_indices:
                if joint_index < 0 or joint_index >= self.num_joints:
                    raise ValueError(f"Joint index {joint_index} is outside [0, {self.num_joints}).")
                if joint_index in covered_indices:
                    raise ValueError(f"Joint index {joint_index} appears in more than one response group.")
                covered_indices.add(joint_index)
        if covered_indices != set(range(self.num_joints)):
            missing = sorted(set(range(self.num_joints)).difference(covered_indices))
            raise ValueError(f"Command-response groups do not cover joint indices {missing}.")

    def _resolve_env_ids(
        self, env_ids: Sequence[int] | torch.Tensor | slice | None
    ) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device).reshape(-1)

    def _sample_response_parameters(self, env_ids: torch.Tensor) -> None:
        for group in self.groups:
            if group.variation_std_fraction == 0.0:
                scale = torch.ones(len(env_ids), device=self.device)
            else:
                scale = 1.0 + torch.randn(len(env_ids), device=self.device) * group.variation_std_fraction
                scale.clamp_(*group.variation_scale_bounds)
            self._response_scales[group.name][env_ids] = scale
            self._delay_seconds[group.name][env_ids] = group.delay_seconds * scale
            self._delay_steps[group.name][env_ids] = torch.round(
                self._delay_seconds[group.name][env_ids] / self.physics_dt
            ).to(dtype=torch.long)
            self._filter_time_constants[group.name][env_ids] = (
                group.filter_time_constant_seconds * scale
            )
