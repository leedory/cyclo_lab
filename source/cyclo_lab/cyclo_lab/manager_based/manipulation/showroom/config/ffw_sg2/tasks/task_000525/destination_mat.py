"""Task000525 destination-mat placement and dining-table attachment."""

from __future__ import annotations

from copy import deepcopy

from cyclo_lab.assets.object import IVORY_TABLE_MAT_CFG


TASK000525_DESTINATION_MAT_NAME = "destination_mat"
TASK000525_DESTINATION_MAT_DIMENSIONS_M = (0.250, 0.210, 0.002)

# The asset's 250 mm local-X edge is rotated onto world Y. Its near world-X
# edge is 85 mm inboard of the table proxy's cabinet-facing X-min short edge.
# Exact centering on the measured 766.040 mm table width leaves 258.020 mm at
# each world-Y side.
TASK000525_DESTINATION_MAT_POS_M = (
    -0.860735354423523,
    -0.6178486049175262,
    0.7511681447029115,
)
TASK000525_DESTINATION_MAT_ROT_WXYZ = (
    0.7071067811865476,
    0.0,
    0.0,
    0.7071067811865475,
)
TASK000525_DESTINATION_MAT_CABINET_EDGE_CLEARANCE_M = 0.085
TASK000525_DESTINATION_MAT_LATERAL_CLEARANCE_M = 0.25802019238471985

TASK000525_DESTINATION_TABLE_PRIM_SUFFIX = (
    "/RobotisShowroom/robotis_showroom/central_dining_set"
)
TASK000525_DESTINATION_MAT_JOINT_SUFFIX = (
    "/Task000525Joints/destination_mat_to_dining_table"
)


def make_task000525_destination_mat_cfg():
    """Return the task-local rigid mat at its approved table-top pose."""

    cfg = deepcopy(IVORY_TABLE_MAT_CFG)
    cfg.prim_path = f"{{ENV_REGEX_NS}}/{TASK000525_DESTINATION_MAT_NAME}"
    cfg.spawn.semantic_tags = [
        ("class", "destination_mat"),
        ("instance", TASK000525_DESTINATION_MAT_NAME),
        ("color", "ivory"),
    ]
    cfg.init_state.pos = TASK000525_DESTINATION_MAT_POS_M
    cfg.init_state.rot = TASK000525_DESTINATION_MAT_ROT_WXYZ
    return cfg


def _relative_pose(stage, parent_body_path, child_body_path):
    """Return the child body frame expressed in the parent body frame."""

    from pxr import Gf, Usd, UsdGeom

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    parent_world = xform_cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(parent_body_path)
    )
    child_world = xform_cache.GetLocalToWorldTransform(
        stage.GetPrimAtPath(child_body_path)
    )
    relative = Gf.Transform(child_world * parent_world.GetInverse())
    translation = relative.GetTranslation()
    quaternion = relative.GetRotation().GetQuat()
    imaginary = quaternion.GetImaginary()
    return (
        tuple(float(translation[index]) for index in range(3)),
        (
            float(quaternion.GetReal()),
            *(float(imaginary[index]) for index in range(3)),
        ),
    )


def attach_task000525_destination_mat_to_table(env, env_ids=None) -> None:
    """Author one fixed joint from each dining-table body to its mat body."""

    from pxr import Gf, Sdf, UsdPhysics

    if env_ids is None:
        env_paths = env.scene.env_prim_paths
    else:
        env_paths = [env.scene.env_prim_paths[int(env_id)] for env_id in env_ids]

    stage = env.scene.stage
    attached_paths = []
    for env_path in env_paths:
        table_path = Sdf.Path(env_path + TASK000525_DESTINATION_TABLE_PRIM_SUFFIX)
        mat_path = Sdf.Path(f"{env_path}/{TASK000525_DESTINATION_MAT_NAME}")
        joint_path = Sdf.Path(env_path + TASK000525_DESTINATION_MAT_JOINT_SUFFIX)

        table_prim = stage.GetPrimAtPath(table_path)
        mat_prim = stage.GetPrimAtPath(mat_path)
        if not table_prim.IsValid() or not table_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Task000525 destination table is not a rigid body: {table_path}")
        if not mat_prim.IsValid() or not mat_prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Task000525 destination mat is not a rigid body: {mat_path}")

        local_pos0, local_rot0 = _relative_pose(stage, table_path, mat_path)
        stage.DefinePrim(joint_path.GetParentPath(), "Scope")
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([table_path])
        joint.CreateBody1Rel().SetTargets([mat_path])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*local_pos0))
        joint.CreateLocalRot0Attr().Set(
            Gf.Quatf(local_rot0[0], Gf.Vec3f(*local_rot0[1:]))
        )
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, Gf.Vec3f(0.0)))
        joint.CreateCollisionEnabledAttr().Set(False)
        joint.CreateExcludeFromArticulationAttr().Set(True)
        joint.CreateJointEnabledAttr().Set(True)
        attached_paths.append(str(joint_path))

    print(
        f"[Task000525] fixed {len(attached_paths)} ivory destination mat(s) "
        "to central_dining_set."
    )
