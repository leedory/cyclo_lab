"""Task000525 appearance-only randomization for rotationally symmetric cans."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def randomize_coffee_can_visual_yaw(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: Sequence[str],
    yaw_range_rad: tuple[float, float] = (-math.pi, math.pi),
) -> None:
    """Rotate only each can's visual mesh around its local Z axis.

    The coffee-can visual is an exactly axisymmetric lathe mesh.  Rotating that
    mesh changes the label/texture direction while leaving the rigid root pose,
    velocities, and three collision proxies untouched.  This avoids pretending
    that the physically upright can has a randomized rigid-body orientation.

    Args:
        env: Manager-based Task000525 environment.
        env_ids: Environment indices being reset.
        object_names: Coffee-can scene entity names.
        yaw_range_rad: Inclusive sampling interval in radians.
    """

    yaw_min, yaw_max = yaw_range_rad
    if not (-math.pi <= yaw_min <= yaw_max <= math.pi):
        raise ValueError(
            "Coffee-can visual yaw range must be ordered and inside [-pi, pi], "
            f"got {yaw_range_rad}."
        )
    if len(env_ids) == 0 or not object_names:
        return

    from pxr import UsdGeom

    samples = torch.rand(
        (len(env_ids), len(object_names)), device=env.device, dtype=torch.float32
    )
    yaw_rad = yaw_min + samples * (yaw_max - yaw_min)
    yaw_deg = torch.rad2deg(yaw_rad)

    env_id_values = env_ids.detach().cpu().tolist()
    yaw_rad_values = yaw_rad.detach().cpu().tolist()
    yaw_deg_values = yaw_deg.detach().cpu().tolist()
    sample_cache = getattr(env, "_task000525_coffee_visual_yaw", {})
    for sample_index, env_id in enumerate(env_id_values):
        env_prim_path = env.scene.env_prim_paths[env_id]
        for object_index, object_name in enumerate(object_names):
            visual_path = f"{env_prim_path}/{object_name}/Visual/SharedMesh"
            visual_prim = env.scene.stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid() or not visual_prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(
                    "Task000525 coffee-can visual mesh is missing at "
                    f"{visual_path}."
                )

            xformable = UsdGeom.Xformable(visual_prim)
            rotate_op = next(
                (
                    op
                    for op in xformable.GetOrderedXformOps()
                    if op.GetOpName() == "xformOp:rotateZ"
                ),
                None,
            )
            if rotate_op is None:
                rotate_op = xformable.AddRotateZOp(
                    precision=UsdGeom.XformOp.PrecisionFloat
                )
            sampled_rad = float(yaw_rad_values[sample_index][object_index])
            sampled_deg = float(yaw_deg_values[sample_index][object_index])
            rotate_op.Set(sampled_deg)
            sample_cache.setdefault(int(env_id), {})[object_name] = {
                "rad": sampled_rad,
                "deg": sampled_deg,
            }

    env._task000525_coffee_visual_yaw = sample_cache


def randomize_coffee_can_distractor_appearance(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    object_names: Sequence[str],
    appearance_names: Sequence[str],
    protected_object_name: str,
) -> None:
    """Permute only non-target coffee-can appearance variants per environment.

    coffee_can.usd defines all label materials below ``Looks``.  This event
    authors only the ``Visual/SharedMesh`` material-binding relationship.  It
    deliberately does *not* change the root appearance variant at runtime:
    variant selection recomposes the referenced rigid-body subtree and can
    invalidate an already-created PhysX tensor view even though the variant
    itself contains only a visual binding.  The orange target root is excluded
    and its authored variant selection is verified before and after every
    permutation.

    The realized object-to-appearance mapping and bound material paths are
    cached on the environment for inclusion in visual-replay manifests.
    """

    object_names = tuple(object_names)
    appearance_names = tuple(appearance_names)
    if len(object_names) != len(appearance_names):
        raise ValueError(
            "Coffee-can distractor appearance needs one material per object."
        )
    if len(object_names) != len(set(object_names)):
        raise ValueError("Coffee-can distractor object names must be unique.")
    if len(appearance_names) != len(set(appearance_names)):
        raise ValueError("Coffee-can distractor appearance names must be unique.")
    if protected_object_name in object_names:
        raise ValueError(
            "Coffee-can distractor appearance must not include the protected target."
        )
    expected_appearances = {
        object_name.removeprefix("coffee_can_") for object_name in object_names
    }
    if set(appearance_names) != expected_appearances:
        raise ValueError(
            "Coffee-can distractor appearances must permute the selected objects' "
            "authored labels."
        )
    if len(env_ids) == 0 or not object_names:
        return

    from pxr import Sdf, UsdGeom, UsdShade

    permutation_indices = torch.argsort(
        torch.rand(
            (len(env_ids), len(appearance_names)),
            device=env.device,
            dtype=torch.float32,
        ),
        dim=1,
    )
    env_id_values = env_ids.detach().cpu().tolist()
    permutation_values = permutation_indices.detach().cpu().tolist()
    sample_cache = getattr(env, "_task000525_coffee_distractor_appearance", {})

    for sample_index, env_id in enumerate(env_id_values):
        env_prim_path = env.scene.env_prim_paths[env_id]
        protected_path = f"{env_prim_path}/{protected_object_name}"
        protected_prim = env.scene.stage.GetPrimAtPath(protected_path)
        if not protected_prim.IsValid():
            raise RuntimeError(
                f"Task000525 protected coffee-can target is missing at {protected_path}."
            )
        protected_variants = protected_prim.GetVariantSets().GetVariantSet("appearance")
        protected_selection = protected_variants.GetVariantSelection()
        expected_target_appearance = protected_object_name.removeprefix("coffee_can_")
        if protected_selection != expected_target_appearance:
            raise RuntimeError(
                "Task000525 protected target appearance is not canonical: "
                f"{protected_path} selected {protected_selection!r}, expected "
                f"{expected_target_appearance!r}."
            )

        realized_mapping = {}
        for object_index, object_name in enumerate(object_names):
            appearance_name = appearance_names[
                permutation_values[sample_index][object_index]
            ]
            object_path = f"{env_prim_path}/{object_name}"
            object_prim = env.scene.stage.GetPrimAtPath(object_path)
            if not object_prim.IsValid():
                raise RuntimeError(
                    f"Task000525 distractor coffee can is missing at {object_path}."
                )

            visual_path = f"{object_path}/Visual/SharedMesh"
            visual_prim = env.scene.stage.GetPrimAtPath(visual_path)
            if not visual_prim.IsValid() or not visual_prim.IsA(UsdGeom.Mesh):
                raise RuntimeError(
                    f"Task000525 coffee-can visual mesh is missing at {visual_path}."
                )
            expected_material_path = f"{object_path}/Looks/{appearance_name}"
            material_prim = env.scene.stage.GetPrimAtPath(expected_material_path)
            if not material_prim.IsValid() or not material_prim.IsA(UsdShade.Material):
                raise RuntimeError(
                    f"Task000525 coffee-can material is missing at "
                    f"{expected_material_path}."
                )
            binding_relationship = visual_prim.GetRelationship("material:binding")
            if not binding_relationship.IsValid():
                raise RuntimeError(
                    f"Task000525 coffee-can visual has no material binding at "
                    f"{visual_path}."
                )
            if not binding_relationship.SetTargets([Sdf.Path(expected_material_path)]):
                raise RuntimeError(
                    f"Task000525 could not bind {expected_material_path} at "
                    f"{visual_path}."
                )
            bound_material, _ = UsdShade.MaterialBindingAPI(
                visual_prim
            ).ComputeBoundMaterial()
            if (
                not bound_material
                or str(bound_material.GetPath()) != expected_material_path
            ):
                actual_path = (
                    str(bound_material.GetPath()) if bound_material else "<unbound>"
                )
                raise RuntimeError(
                    "Task000525 distractor material binding did not realize: "
                    f"{visual_path} -> {actual_path}, expected {expected_material_path}."
                )
            realized_mapping[object_name] = {
                "authored_appearance": object_prim.GetVariantSets()
                .GetVariantSet("appearance")
                .GetVariantSelection(),
                "sampled_appearance": appearance_name,
                "bound_material_path": expected_material_path,
            }

        if protected_variants.GetVariantSelection() != protected_selection:
            raise RuntimeError(
                "Task000525 distractor appearance changed the protected target."
            )
        sample_cache[int(env_id)] = {
            "protected_target": {
                "object_name": protected_object_name,
                "appearance": protected_selection,
            },
            "distractor_mapping": realized_mapping,
        }

    env._task000525_coffee_distractor_appearance = sample_cache
