"""Non-trajectory reset randomizers for Task 000458 generated episodes."""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import torch
from isaaclab.utils import math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


WALL_PRIM_SUFFIXES = (
    "/RobotisShowroom/robotis_showroom/ShowroomShell/BackWall",
    "/RobotisShowroom/robotis_showroom/ShowroomShell/LeftWall",
)
WALL_MATERIAL_SUFFIX = "/Task458RandomizationMaterials/wall"
SHELF_TEXTURE_SUFFIX = (
    "/RobotisShowroom/robotis_showroom/kolbjorn_cabinet_1/AssetFrame/Asset/"
    "Looks/material_0/diffuseTex"
)


def _ids(env: ManagerBasedEnv, env_ids) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, dtype=torch.long, device=env.device)
    return torch.as_tensor(env_ids, dtype=torch.long, device=env.device).reshape(-1)


def randomize_shelf_texture_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    brightness_range: tuple[float, float],
    channel_tint_max: float,
) -> None:
    """Retain the authored shelf texture while changing brightness/tint."""
    from pxr import Gf, Sdf, UsdShade

    ids = _ids(env, env_ids)
    samples = torch.rand((len(ids), 4), device=env.device)
    brightness = brightness_range[0] + samples[:, 0] * (
        brightness_range[1] - brightness_range[0]
    )
    tint = (samples[:, 1:4] * 2.0 - 1.0) * channel_tint_max
    rgb = (brightness[:, None] * (1.0 + tint)).clamp(0.0, 2.0)
    baseline_scales = getattr(env, "_task458_shelf_baseline_scale", {})
    realized = getattr(env, "_task458_shelf_rgb_scale", {})
    for row, env_id in enumerate(ids.detach().cpu().tolist()):
        shader = UsdShade.Shader(
            env.scene.stage.GetPrimAtPath(env.scene.env_prim_paths[env_id] + SHELF_TEXTURE_SUFFIX)
        )
        if not shader:
            raise RuntimeError(f"Task458 shelf shader is missing in env {env_id}")
        scale = shader.GetInput("scale")
        if not scale:
            scale = shader.CreateInput("scale", Sdf.ValueTypeNames.Float4)
            scale.Set(Gf.Vec4f(1.0, 1.0, 1.0, 1.0))
        if env_id not in baseline_scales:
            authored = scale.Get()
            if authored is None:
                authored = Gf.Vec4f(1.0, 1.0, 1.0, 1.0)
                scale.Set(authored)
            baseline = tuple(float(component) for component in authored)
            if len(baseline) == 3:
                baseline = (*baseline, 1.0)
            if len(baseline) != 4:
                raise RuntimeError(
                    f"Task458 shelf scale must have three or four channels, got {baseline}"
                )
            baseline_scales[env_id] = baseline
        baseline = baseline_scales[env_id]
        value = tuple(
            max(0.0, min(2.0, baseline[index] * float(rgb[row, index].cpu())))
            for index in range(3)
        )
        realized_scale = (*value, baseline[3])
        scale.Set(Gf.Vec4f(*realized_scale))
        realized[env_id] = realized_scale
    env._task458_shelf_baseline_scale = baseline_scales
    env._task458_shelf_rgb_scale = realized


def randomize_wall_solid_rgb(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    mode: str,
    rgb_range: tuple[float, float],
    near_white_range: tuple[float, float],
) -> None:
    """Bind one sampled solid RGB to the two task-facing showroom walls.

    Only ``ShowroomShell/BackWall`` (toward peanut_mix_bag_02) and
    ``ShowroomShell/LeftWall`` (toward peanut_mix_bag_03) are changed.  Their
    opposite ``*_01`` walls are intentionally left on the authored material.
    A stronger-than-descendants binding avoids editing the shared
    ``warm_white_wall`` shader used by all four walls.
    """
    from pxr import Gf, Sdf, UsdShade
    from .cfg import WALL_RGB_MODE, WALL_WHITE_MODE

    if mode not in (WALL_RGB_MODE, WALL_WHITE_MODE):
        raise ValueError(f"unsupported Task458 wall mode: {mode}")
    ids = _ids(env, env_ids)
    lower, upper = rgb_range if mode == WALL_RGB_MODE else near_white_range
    colors = lower + torch.rand((len(ids), 3), device=env.device) * (upper - lower)
    realized = getattr(env, "_task458_wall_rgb", {})
    bound_prims = getattr(env, "_task458_wall_bound_prims", {})
    for row, env_id in enumerate(ids.detach().cpu().tolist()):
        env_path = env.scene.env_prim_paths[env_id]
        material_path = env_path + WALL_MATERIAL_SUFFIX
        material = UsdShade.Material.Define(env.scene.stage, material_path)
        shader = UsdShade.Shader.Define(env.scene.stage, material_path + "/PreviewSurface")
        shader.CreateIdAttr("UsdPreviewSurface")
        value = tuple(float(component) for component in colors[row].cpu())
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*value))
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.9)
        shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")

        env_bound_prims = []
        for wall_suffix in WALL_PRIM_SUFFIXES:
            wall_path = env_path + wall_suffix
            wall_prim = env.scene.stage.GetPrimAtPath(wall_path)
            if not wall_prim.IsValid():
                raise RuntimeError(f"Task458 task-facing wall is missing: {wall_path}")
            binding_api = UsdShade.MaterialBindingAPI.Apply(wall_prim)
            try:
                binding_api.Bind(
                    material,
                    bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                    materialPurpose=UsdShade.Tokens.allPurpose,
                )
            except TypeError:
                binding_api.Bind(material, UsdShade.Tokens.strongerThanDescendants)
            bound_material, _ = binding_api.ComputeBoundMaterial()
            if not bound_material or str(bound_material.GetPath()) != material_path:
                raise RuntimeError(
                    f"Task458 wall material binding did not realize: {wall_path} -> {material_path}"
                )
            env_bound_prims.append(wall_path)
        realized[env_id] = value
        bound_prims[env_id] = tuple(env_bound_prims)
    env._task458_wall_mode = mode
    env._task458_wall_rgb = realized
    env._task458_wall_bound_prims = bound_prims


def randomize_policy_cameras(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    camera_names: Sequence[str],
    coupled_focal_scale_range: tuple[float, float],
    local_roll_max_rad: float,
    local_pitch_max_rad: float,
    local_yaw_max_rad: float,
) -> None:
    """Apply independent per-camera samples from stable authored local poses.

    Focal scaling is absolute from the first authored intrinsic matrix and uses
    one coupled ``fx == fy`` scale per camera/environment.  Local translation
    is restored exactly and local R/P/Y is composed from the authored local
    quaternion on every reset, so partial resets cannot accumulate transforms.
    """
    ids = _ids(env, env_ids)
    names = tuple(camera_names)
    if not 0.0 < coupled_focal_scale_range[0] <= coupled_focal_scale_range[1]:
        raise ValueError("Task458 coupled focal scale range must be positive and ordered")
    if not hasattr(env, "_task458_camera_baseline_intrinsics"):
        env._task458_camera_names = names
        env._task458_camera_baseline_intrinsics = {}
        env._task458_camera_baseline_local_translation = {}
        env._task458_camera_baseline_local_quat = {}
        env._task458_camera_focal_scale = {}
        env._task458_camera_local_rpy = {}
        env._task458_camera_local_delta_quat = {}
        for name in names:
            sensor = env.scene.sensors[name]
            if not hasattr(sensor, "_view"):
                raise RuntimeError(f"Task458 camera has no initialized XForm view: {name}")
            local_translation, local_quat = sensor._view.get_local_poses()
            env._task458_camera_baseline_intrinsics[name] = (
                sensor.data.intrinsic_matrices.clone()
            )
            env._task458_camera_baseline_local_translation[name] = torch.as_tensor(
                local_translation, dtype=torch.float32, device=env.device
            ).clone()
            env._task458_camera_baseline_local_quat[name] = torch.as_tensor(
                local_quat, dtype=torch.float32, device=env.device
            ).clone()
            env._task458_camera_focal_scale[name] = torch.ones(
                env.num_envs, device=env.device
            )
            env._task458_camera_local_rpy[name] = torch.zeros(
                (env.num_envs, 3), device=env.device
            )
            identity = torch.zeros((env.num_envs, 4), device=env.device)
            identity[:, 0] = 1.0
            env._task458_camera_local_delta_quat[name] = identity
    elif env._task458_camera_names != names:
        raise RuntimeError("Task458 policy camera names changed after baseline capture")

    for name in names:
        sensor = env.scene.sensors[name]
        samples = torch.rand((len(ids), 4), device=env.device) * 2.0 - 1.0
        focal_scale = coupled_focal_scale_range[0] + (samples[:, 0] + 1.0) * 0.5 * (
            coupled_focal_scale_range[1] - coupled_focal_scale_range[0]
        )
        local_rpy = torch.stack(
            (
                samples[:, 1] * local_roll_max_rad,
                samples[:, 2] * local_pitch_max_rad,
                samples[:, 3] * local_yaw_max_rad,
            ),
            dim=-1,
        )
        delta_quat = math_utils.quat_from_euler_xyz(
            local_rpy[:, 0], local_rpy[:, 1], local_rpy[:, 2]
        )
        baseline_k = env._task458_camera_baseline_intrinsics[name][ids].clone()
        baseline_k[:, 0, 0] *= focal_scale
        baseline_k[:, 1, 1] *= focal_scale
        sensor.set_intrinsic_matrices(baseline_k, env_ids=ids.detach().cpu().tolist())
        requested_local_quat = math_utils.quat_mul(
            env._task458_camera_baseline_local_quat[name][ids], delta_quat
        )
        sensor._view.set_local_poses(
            translations=env._task458_camera_baseline_local_translation[name][ids],
            orientations=requested_local_quat,
            indices=ids,
        )
        env._task458_camera_focal_scale[name][ids] = focal_scale
        env._task458_camera_local_rpy[name][ids] = local_rpy
        env._task458_camera_local_delta_quat[name][ids] = delta_quat


def randomize_dome_and_weak_keys(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    dome_intensity_range: tuple[float, float],
    dome_rgb_range: Sequence[tuple[float, float]],
    weak_key_intensity_range: tuple[float, float],
) -> None:
    """Use a run-shard-global dome plus independently reset weak local keys.

    The first invocation must cover every environment and samples the only dome
    value used for this environment process.  Partial resets only reassert that
    value, so they cannot change lighting in active episodes.  Dome diversity
    therefore comes from separate seeded generation shards/processes.

    Per-env sphere keys are kept close to each 8 m-spaced showroom and shaped
    downward to reduce, though not formally eliminate, cross-environment leak.
    """
    from pxr import Gf, UsdGeom, UsdLux

    ids = _ids(env, env_ids)
    if len(dome_rgb_range) != 3 or any(low > high for low, high in dome_rgb_range):
        raise ValueError("Task458 dome RGB requires three ordered channel ranges")
    if not 0.0 <= dome_intensity_range[0] <= dome_intensity_range[1]:
        raise ValueError("Task458 dome intensity range must be non-negative and ordered")
    if not 0.0 <= weak_key_intensity_range[0] <= weak_key_intensity_range[1]:
        raise ValueError("Task458 weak-key intensity range must be non-negative and ordered")

    dome = UsdLux.DomeLight(env.scene.stage.GetPrimAtPath("/World/Light"))
    if not dome:
        raise RuntimeError("Task458 global DomeLight /World/Light is missing")
    if not hasattr(env, "_task458_dome_sample"):
        all_ids = torch.arange(env.num_envs, dtype=torch.long, device=env.device)
        is_full_reset = len(ids) == env.num_envs and torch.equal(
            torch.sort(ids).values, all_ids
        )
        if not is_full_reset:
            raise RuntimeError(
                "Task458 shard-global dome must initialize on a full-environment reset"
            )
        sample = torch.rand(4, device=env.device)
        dome_intensity = dome_intensity_range[0] + float(sample[0]) * (
            dome_intensity_range[1] - dome_intensity_range[0]
        )
        dome_rgb = tuple(
            low + float(sample[index + 1]) * (high - low)
            for index, (low, high) in enumerate(dome_rgb_range)
        )
        env._task458_dome_sample = {"intensity": dome_intensity, "rgb": dome_rgb}
    else:
        dome_intensity = float(env._task458_dome_sample["intensity"])
        dome_rgb = tuple(env._task458_dome_sample["rgb"])
    dome.CreateIntensityAttr().Set(dome_intensity)
    dome.CreateColorAttr().Set(Gf.Vec3f(*dome_rgb))

    key_samples = torch.rand((len(ids), 4), device=env.device)
    key_intensities = getattr(env, "_task458_weak_key_intensity", {})
    key_states = getattr(env, "_task458_weak_key_state", {})
    for row, env_id in enumerate(ids.detach().cpu().tolist()):
        path = env.scene.env_prim_paths[env_id] + "/Task458WeakKey"
        key = UsdLux.SphereLight.Define(env.scene.stage, path)
        intensity = weak_key_intensity_range[0] + float(key_samples[row, 0]) * (
            weak_key_intensity_range[1] - weak_key_intensity_range[0]
        )
        key.CreateIntensityAttr().Set(intensity)
        key.CreateRadiusAttr().Set(0.20)
        key.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.96, 0.90))
        shaping = UsdLux.ShapingAPI.Apply(key.GetPrim())
        shaping.CreateShapingConeAngleAttr().Set(75.0)
        shaping.CreateShapingConeSoftnessAttr().Set(0.6)
        shaping.CreateShapingFocusAttr().Set(0.0)
        xform = UsdGeom.Xformable(key.GetPrim())
        translate = next(
            (op for op in xform.GetOrderedXformOps() if op.GetOpType() == UsdGeom.XformOp.TypeTranslate),
            None,
        )
        if translate is None:
            translate = xform.AddTranslateOp()
        origin = env.scene.env_origins[env_id].detach().cpu()
        position = (
            float(origin[0]) - 1.72 + 0.50 * (float(key_samples[row, 1]) - 0.5),
            float(origin[1]) + 1.59 + 0.50 * (float(key_samples[row, 2]) - 0.5),
            float(origin[2]) + 1.82 + 0.20 * (float(key_samples[row, 3]) - 0.5),
        )
        translate.Set(Gf.Vec3d(*position))
        key_intensities[env_id] = intensity
        key_states[env_id] = {
            "intensity": intensity,
            "position_w": position,
            "cone_angle_deg": 75.0,
            "cone_softness": 0.6,
        }
    env._task458_weak_key_intensity = key_intensities
    env._task458_weak_key_state = key_states
