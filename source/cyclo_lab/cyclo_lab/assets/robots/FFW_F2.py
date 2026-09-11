"""Isaac Lab articulation configuration for the FFW F2."""

import re
from copy import deepcopy

from isaacsim.core.utils.stage import get_current_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim import ArticulationRootPropertiesCfg, RigidBodyMaterialCfg, RigidBodyPropertiesCfg, UsdFileCfg
from isaaclab.sim.spawners.from_files import from_files
from isaaclab.sim.utils import bind_physics_material, clone, make_uninstanceable

from cyclo_lab.assets.robots import CYCLO_LAB_ASSETS_DATA_DIR
from cyclo_lab.robot_specs.ffw.f2 import (
    F2_VISUAL_MATERIAL_COLORS,
    classify_f2_visual_material,
    F2_SWERVE_DRIVE_DAMPING,
    F2_SWERVE_STEERING_JOINTS,
    F2_SWERVE_WHEEL_JOINTS,
    FFW_F2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
)


_F2_WHEEL_PHYSICS_MATERIAL = RigidBodyMaterialCfg(
    friction_combine_mode="max", restitution_combine_mode="min",
    static_friction=2.0, dynamic_friction=1.8, restitution=0.0,
)
_F2_DISTAL_GRIPPER_PHYSICS_MATERIAL = RigidBodyMaterialCfg(
    friction_combine_mode="max", restitution_combine_mode="min",
    static_friction=1.2, dynamic_friction=1.0, restitution=0.0,
)
_F2_WHEEL_LINKS = (
    "left_wheel_steer_link", "left_wheel_drive_link",
    "right_wheel_steer_link", "right_wheel_drive_link",
    "rear_wheel_steer_link", "rear_wheel_drive_link",
)
_F2_WHEEL_DRIVE_LINKS = ("left_wheel_drive_link", "right_wheel_drive_link", "rear_wheel_drive_link")
_F2_PHYSICS_LIFT_EFFORT_LIMIT = 5_000_000.0
_F2_PHYSICS_LIFT_STIFFNESS = 250_000.0
_F2_PHYSICS_LIFT_DAMPING = 5_000.0


def _make_f2_visual_material(
    stage,
    material_path: str,
    color: tuple[float, float, float],
):
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateOutput("out", Sdf.ValueTypeNames.Token)
    for create_output in (
        material.CreateSurfaceOutput,
        material.CreateVolumeOutput,
        material.CreateDisplacementOutput,
    ):
        create_output("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return material


def _apply_f2_visual_materials(stage, prim_path: str) -> None:
    materials = {
        name: _make_f2_visual_material(stage, f"{prim_path}/Looks/{name}", color)
        for name, color in F2_VISUAL_MATERIAL_COLORS.items()
    }
    for prim in _iter_robot_prims(stage, prim_path):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        material_name = classify_f2_visual_material(str(prim.GetPath()))
        if material_name is not None:
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(materials[material_name])


def _spawn_f2_editable(prim_path, cfg, translation=None, orientation=None, **kwargs):
    prim = from_files.spawn_from_usd(prim_path, cfg, translation, orientation, **kwargs)
    stage = get_current_stage()
    make_uninstanceable(prim_path, stage)
    _apply_f2_visual_materials(stage, prim_path)
    return prim, stage


def _iter_robot_prims(stage, prim_path: str):
    robot_prim = stage.GetPrimAtPath(prim_path)
    return Usd.PrimRange(robot_prim) if robot_prim.IsValid() else ()


def _add_filtered_collision_pairs(stage, source_paths: list[str], target_paths: list[str]) -> None:
    for source_path in source_paths:
        source_prim = stage.GetPrimAtPath(source_path)
        if not source_prim.IsValid():
            continue
        relation = UsdPhysics.FilteredPairsAPI.Apply(source_prim).CreateFilteredPairsRel()
        for target_path in target_paths:
            relation.AddTarget(Sdf.Path(target_path))


def _iter_f2_wheel_drive_collision_prims(stage, prim_path: str):
    link_pattern = "|".join(re.escape(name) for name in _F2_WHEEL_DRIVE_LINKS)
    for prim in _iter_robot_prims(stage, prim_path):
        path = str(prim.GetPath())
        if "/collisions/" in path.lower() and re.search(
            rf"(^|/)({link_pattern})(/|_|$)", path.lower()
        ) and prim.HasAPI(UsdPhysics.CollisionAPI):
            yield prim


def _bind_f2_wheel_physics_material(stage, prim_path: str, material_path: str) -> None:
    for prim in _iter_f2_wheel_drive_collision_prims(stage, prim_path):
        bind_physics_material(str(prim.GetPath()), material_path)


def _bind_f2_distal_gripper_physics_material(stage, prim_path: str, material_path: str) -> None:
    distal_pattern = r"gripper_[lr]_hx2_d1_[rl]2"
    for prim in _iter_robot_prims(stage, prim_path):
        path = str(prim.GetPath())
        if "/collisions/" in path.lower() and re.search(
            rf"(^|/)({distal_pattern})(/|_|$)", path.lower()
        ) and prim.HasAPI(UsdPhysics.CollisionAPI):
            bind_physics_material(path, material_path)


def _filter_f2_base_wheel_collisions(stage, prim_path: str) -> None:
    base_collision_paths: list[str] = []
    wheel_collision_paths: list[str] = []
    wheel_pattern = "|".join(re.escape(name) for name in _F2_WHEEL_LINKS)
    for prim in _iter_robot_prims(stage, prim_path):
        path = str(prim.GetPath())
        if "/collisions/" not in path.lower():
            continue
        if re.search(r"(^|/)base_link/collisions(/|_|$)", path.lower()):
            base_collision_paths.append(path)
        elif re.search(rf"(^|/)({wheel_pattern})(/|_|$)", path.lower()):
            wheel_collision_paths.append(path)
    _add_filtered_collision_pairs(stage, base_collision_paths, wheel_collision_paths)
    _add_filtered_collision_pairs(stage, wheel_collision_paths, base_collision_paths)


@clone
def spawn_f2_with_visuals(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn F2 and restore the canonical URDF material assignments."""

    prim, _ = _spawn_f2_editable(prim_path, cfg, translation, orientation, **kwargs)
    return prim


@clone
def spawn_f2_with_base_physics(prim_path, cfg, translation=None, orientation=None, **kwargs):
    """Spawn F2 with canonical visuals and wheel/gripper contact materials."""

    prim, stage = _spawn_f2_editable(prim_path, cfg, translation, orientation, **kwargs)

    material_path = f"{prim_path}/wheelPhysicsMaterial"
    _F2_WHEEL_PHYSICS_MATERIAL.func(material_path, _F2_WHEEL_PHYSICS_MATERIAL)
    _bind_f2_wheel_physics_material(stage, prim_path, material_path)
    material_path = f"{prim_path}/distalGripperPhysicsMaterial"
    _F2_DISTAL_GRIPPER_PHYSICS_MATERIAL.func(material_path, _F2_DISTAL_GRIPPER_PHYSICS_MATERIAL)
    _bind_f2_distal_gripper_physics_material(stage, prim_path, material_path)
    _filter_f2_base_wheel_collisions(stage, prim_path)
    return prim


FFW_F2_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=f"{CYCLO_LAB_ASSETS_DATA_DIR}/robots/FFW/FFW_F2.usd",
        rigid_props=RigidBodyPropertiesCfg(disable_gravity=True, max_depenetration_velocity=5.0),
        articulation_props=ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **{f"arm_l_joint{index}": 0.0 for index in range(1, 8)},
            **{f"arm_r_joint{index}": 0.0 for index in range(1, 8)},
            **{f"gripper_l_joint{index}": 0.0 for index in range(1, 5)},
            **{f"gripper_r_joint{index}": 0.0 for index in range(1, 5)},
            "head_joint1": 0.0, "head_joint2": 0.0, "lift_joint": 0.0,
        }
    ),
    actuators={
        "lift": ImplicitActuatorCfg(
            joint_names_expr=["lift_joint"], velocity_limit_sim=0.2,
            effort_limit_sim=1_000_000.0, stiffness=10_000.0, damping=100.0,
        ),
        "DY_80": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[1-2]", "arm_r_joint[1-2]"], velocity_limit_sim=15.0,
            effort_limit_sim=61.4, stiffness=600.0, damping=30.0,
        ),
        "DY_70": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint[3-6]", "arm_r_joint[3-6]"], velocity_limit_sim=15.0,
            effort_limit_sim=31.7, stiffness=600.0, damping=20.0,
        ),
        "DP-42": ImplicitActuatorCfg(
            joint_names_expr=["arm_l_joint7", "arm_r_joint7"], velocity_limit_sim=6.0,
            effort_limit_sim=5.1, stiffness=200.0, damping=3.0,
        ),
        "gripper_coupled": ImplicitActuatorCfg(
            joint_names_expr=["gripper_[lr]_joint[1-4]"], velocity_limit_sim=6.5,
            effort_limit_sim=30.0, stiffness=100.0, damping=4.0,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_joint1", "head_joint2"], velocity_limit_sim=2.0,
            effort_limit_sim=30.0, stiffness=150.0, damping=3.0,
        ),
    },
)


FFW_F2_CFG.spawn.func = spawn_f2_with_visuals
FFW_F2_PHYSICS_CFG = deepcopy(FFW_F2_CFG)
FFW_F2_PHYSICS_CFG.spawn.func = spawn_f2_with_base_physics
FFW_F2_PHYSICS_CFG.spawn.rigid_props.linear_damping = 2.0
FFW_F2_PHYSICS_CFG.spawn.rigid_props.angular_damping = 4.0
FFW_F2_PHYSICS_CFG.articulation_root_prim_path = "/base_link"
FFW_F2_PHYSICS_CFG.actuators["lift"].effort_limit_sim = _F2_PHYSICS_LIFT_EFFORT_LIMIT
FFW_F2_PHYSICS_CFG.actuators["lift"].stiffness = _F2_PHYSICS_LIFT_STIFFNESS
FFW_F2_PHYSICS_CFG.actuators["lift"].damping = _F2_PHYSICS_LIFT_DAMPING
FFW_F2_PHYSICS_CFG.init_state.joint_pos.update(
    {joint_name: 0.0 for joint_name in (*F2_SWERVE_STEERING_JOINTS, *F2_SWERVE_WHEEL_JOINTS)}
)
FFW_F2_PHYSICS_CFG.actuators = {
    "base_steer": ImplicitActuatorCfg(
        joint_names_expr=list(F2_SWERVE_STEERING_JOINTS),
        velocity_limit_sim=FFW_F2_SWERVE_STEERING_ANGULAR_VELOCITY_LIMIT,
        effort_limit_sim=100_000.0, stiffness=10_000.0, damping=100.0,
    ),
    "base_drive": ImplicitActuatorCfg(
        joint_names_expr=list(F2_SWERVE_WHEEL_JOINTS), velocity_limit_sim=50.0,
        effort_limit_sim=100_000.0, stiffness=0.0, damping=F2_SWERVE_DRIVE_DAMPING,
    ),
    **FFW_F2_PHYSICS_CFG.actuators,
}
