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
    yaw_deg = torch.rad2deg(yaw_min + samples * (yaw_max - yaw_min))

    env_id_values = env_ids.detach().cpu().tolist()
    yaw_values = yaw_deg.detach().cpu().tolist()
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
            rotate_op.Set(float(yaw_values[sample_index][object_index]))
