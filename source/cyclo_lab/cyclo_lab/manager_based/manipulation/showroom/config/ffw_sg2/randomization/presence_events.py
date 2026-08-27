"""Reset-time object presence randomization for SG2 showroom environments.

The asset remains registered so rigid-body tensor views and dataset schemas stay
constant. Absence disables the per-environment rigid-body simulation flag and
hides the asset root. No body is teleported and no collider prim is rewritten.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class PresenceError(RuntimeError):
    """Raised when an asset cannot safely participate in presence toggling."""


@dataclass(frozen=True)
class PresenceChange:
    """Compact result returned after one reset-time presence update."""

    env_count: int
    object_count: int
    present_pairs: int
    absent_pairs: int
    simulation_pairs_updated: int


@dataclass(frozen=True)
class _VisibilityAttribute:
    attribute: Any
    baseline_value: Any
    baseline_authored: bool


def _env_ids(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | None,
) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)


def sample_present_masks(
    *,
    object_names: Sequence[str],
    env_count: int,
    disappearance_probability: float,
    device: str | torch.device,
) -> dict[str, torch.Tensor]:
    """Draw one independent present/absent bit per object/environment pair."""

    if not 0.0 <= disappearance_probability <= 1.0:
        raise ValueError("disappearance_probability must be in [0, 1]")
    if env_count < 0:
        raise ValueError("env_count must be non-negative")
    names = tuple(object_names)
    if len(set(names)) != len(names):
        raise ValueError("object_names must not contain duplicates")
    return {
        name: torch.rand(env_count, device=device) >= disappearance_probability
        for name in names
    }


class SimulationVisibilityPresenceController:
    """Toggle physical and visual presence without changing scene topology.

    Isaac Sim's rigid-body tensor API provides a per-instance
    ``disable_simulation`` flag. It removes an absent body from physics while
    preserving its view index. The controller also preserves each root's
    authored visibility and each body's pre-existing simulation flag so a
    present reset restores the exact baseline state.
    """

    def __init__(self, env: ManagerBasedEnv, object_names: Sequence[str]):
        from isaaclab.sim.utils import find_matching_prims
        from pxr import UsdGeom

        self._env = env
        self._object_names = tuple(object_names)
        if not self._object_names:
            raise ValueError("object_names must not be empty")
        if len(set(self._object_names)) != len(self._object_names):
            raise ValueError("object_names must not contain duplicates")

        self._simulation_baseline: dict[str, torch.Tensor] = {}
        self._visibility: dict[str, tuple[_VisibilityAttribute, ...]] = {}
        for name in self._object_names:
            if name not in env.scene.rigid_objects:
                raise PresenceError(f"presence asset is not a rigid object: {name}")
            asset = env.scene[name]
            flags = asset.root_physx_view.get_disable_simulations().clone()
            if tuple(flags.shape) != (env.num_envs, 1):
                raise PresenceError(
                    f"unexpected disable_simulations shape for {name}: {tuple(flags.shape)}"
                )
            self._simulation_baseline[name] = flags

            roots = find_matching_prims(asset.cfg.prim_path, stage=env.scene.stage)
            roots_by_env = self._roots_by_environment(roots)
            visibility = []
            for root in roots_by_env:
                imageable = UsdGeom.Imageable(root)
                if not imageable:
                    raise PresenceError(f"asset root is not imageable: {root.GetPath()}")
                attribute = imageable.GetVisibilityAttr()
                visibility.append(
                    _VisibilityAttribute(
                        attribute=attribute,
                        baseline_value=attribute.Get(),
                        baseline_authored=attribute.HasAuthoredValueOpinion(),
                    )
                )
            self._visibility[name] = tuple(visibility)

    @property
    def object_names(self) -> tuple[str, ...]:
        return self._object_names

    def _roots_by_environment(self, roots: Sequence[Any]) -> tuple[Any, ...]:
        ordered = []
        for env_path in self._env.scene.env_prim_paths:
            matches = [
                root
                for root in roots
                if str(root.GetPath()).startswith(f"{env_path}/")
            ]
            if len(matches) != 1:
                raise PresenceError(
                    f"expected one asset root below {env_path}, found {len(matches)}"
                )
            ordered.append(matches[0])
        return tuple(ordered)

    def apply_present_masks(
        self,
        env_ids: torch.Tensor | Sequence[int] | None,
        present_masks: Mapping[str, torch.Tensor],
    ) -> PresenceChange:
        """Apply explicit subset masks atomically between simulation steps."""

        from pxr import UsdGeom

        ids = _env_ids(self._env, env_ids)
        ids_cpu = ids.cpu()
        missing = set(self._object_names) - set(present_masks)
        extra = set(present_masks) - set(self._object_names)
        if missing or extra:
            raise ValueError(
                f"presence mask keys differ: missing={sorted(missing)}, "
                f"extra={sorted(extra)}"
            )

        normalized: dict[str, torch.Tensor] = {}
        for name in self._object_names:
            mask = torch.as_tensor(
                present_masks[name], dtype=torch.bool, device=self._env.device
            ).reshape(-1)
            if mask.shape != ids.shape:
                raise ValueError(
                    f"presence mask for {name} must have shape {tuple(ids.shape)}, "
                    f"got {tuple(mask.shape)}"
                )
            normalized[name] = mask

        simulation_before = {}
        visibility_before = []
        try:
            present_pairs = 0
            for name in self._object_names:
                asset = self._env.scene[name]
                mask = normalized[name]
                present_pairs += int(mask.sum().item())

                absent_ids = ids[~mask]
                if len(absent_ids) > 0:
                    asset.write_root_velocity_to_sim(
                        torch.zeros(
                            (len(absent_ids), 6),
                            dtype=asset.data.root_vel_w.dtype,
                            device=asset.device,
                        ),
                        env_ids=absent_ids,
                    )

                flags = asset.root_physx_view.get_disable_simulations().clone()
                simulation_before[name] = flags.clone()
                baseline = self._simulation_baseline[name]
                present_cpu = mask.to(device=flags.device)
                flags[ids_cpu, 0] = torch.where(
                    present_cpu,
                    baseline[ids_cpu, 0],
                    torch.ones_like(flags[ids_cpu, 0]),
                )
                asset.root_physx_view.set_disable_simulations(flags, ids_cpu)

                for row, env_id in enumerate(ids_cpu.tolist()):
                    binding = self._visibility[name][env_id]
                    visibility_before.append(
                        (binding.attribute, binding.attribute.Get())
                    )
                    if bool(present_cpu[row]):
                        if binding.baseline_authored:
                            binding.attribute.Set(binding.baseline_value)
                        else:
                            binding.attribute.Clear()
                    else:
                        binding.attribute.Set(UsdGeom.Tokens.invisible)

            self._remember_masks(ids, normalized)
            pair_count = len(ids) * len(self._object_names)
            return PresenceChange(
                env_count=len(ids),
                object_count=len(self._object_names),
                present_pairs=present_pairs,
                absent_pairs=pair_count - present_pairs,
                simulation_pairs_updated=pair_count,
            )
        except Exception:
            all_ids_cpu = torch.arange(self._env.num_envs, dtype=torch.long)
            for name, previous in simulation_before.items():
                self._env.scene[name].root_physx_view.set_disable_simulations(
                    previous, all_ids_cpu
                )
            for attribute, previous in reversed(visibility_before):
                attribute.Set(previous)
            raise

    def randomize(
        self,
        env_ids: torch.Tensor | Sequence[int] | None,
        disappearance_probability: float,
    ) -> PresenceChange:
        ids = _env_ids(self._env, env_ids)
        masks = sample_present_masks(
            object_names=self._object_names,
            env_count=len(ids),
            disappearance_probability=disappearance_probability,
            device=self._env.device,
        )
        return self.apply_present_masks(ids, masks)

    def _remember_masks(
        self,
        ids: torch.Tensor,
        normalized: Mapping[str, torch.Tensor],
    ) -> None:
        state = getattr(self._env, "_showroom_non_target_presence", None)
        if state is None:
            state = getattr(self._env, "_task458_non_target_presence", {})
        for name, subset_mask in normalized.items():
            full_mask = state.get(name)
            if full_mask is None:
                full_mask = torch.ones(
                    self._env.num_envs,
                    dtype=torch.bool,
                    device=self._env.device,
                )
                state[name] = full_mask
            full_mask[ids] = subset_mask

        # Generic state is canonical. Keep the Task458 alias while the replay
        # staging writer still consumes its original attribute name.
        self._env._showroom_non_target_presence = state
        self._env._task458_non_target_presence = state

    def simulation_state_summary(self) -> dict[str, int]:
        result = {"enabled": 0, "disabled": 0}
        for name in self._object_names:
            values = self._env.scene[name].root_physx_view.get_disable_simulations()
            disabled = int(values.to(dtype=torch.bool).sum().item())
            result["disabled"] += disabled
            result["enabled"] += int(values.numel()) - disabled
        return result

    def visibility_state_summary(self) -> dict[str, int]:
        from pxr import UsdGeom

        result = {"visible": 0, "invisible": 0}
        for per_env in self._visibility.values():
            for binding in per_env:
                value = binding.attribute.Get()
                key = (
                    "invisible"
                    if value == UsdGeom.Tokens.invisible
                    else "visible"
                )
                result[key] += 1
        return result


def randomize_non_target_presence(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: Sequence[str],
    disappearance_probability: float,
) -> PresenceChange:
    """Randomization event shared by Mimic and replay augmentation."""

    names = tuple(object_names)
    controller = getattr(env, "_showroom_presence_controller", None)
    if controller is None:
        controller = SimulationVisibilityPresenceController(env, names)
        env._showroom_presence_controller = controller
    elif controller.object_names != names:
        raise PresenceError(
            "one environment cannot reuse a presence controller with different object_names"
        )
    return controller.randomize(env_ids, disappearance_probability)
