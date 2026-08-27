"""Shared SG2 robot and camera construction for showroom environments."""

from copy import deepcopy

from isaaclab.assets.articulation import ArticulationCfg

from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_overhead_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    load_sg2_camera_profile,
)


# Keep the real-aligned reset pose horizontally registered to the snack shelf.
SG2_SHOWROOM_ROBOT_POS = (-1.47138, 1.59091, 0.0)
SG2_SHOWROOM_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
SG2_SHOWROOM_INITIAL_JOINT_POSITIONS = {
    "arm_l_joint1": 0.0005,
    "arm_l_joint2": 0.6040,
    "arm_l_joint3": -0.2963,
    "arm_l_joint4": -2.5052,
    "arm_l_joint5": 0.5672,
    "arm_l_joint6": 0.4926,
    "arm_l_joint7": 0.7391,
    "gripper_l_joint1": 0.0,
    "arm_r_joint1": 0.0005,
    "arm_r_joint2": -0.6040,
    "arm_r_joint3": 0.2963,
    "arm_r_joint4": -2.5052,
    "arm_r_joint5": -0.5672,
    "arm_r_joint6": 0.4926,
    "arm_r_joint7": -0.7391,
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.2,
    "head_joint2": 0.0,
    "lift_joint": 0.0,
}


def make_sg2_showroom_robot_cfg() -> ArticulationCfg:
    """Build the shared real-aligned SG2 articulation configuration."""

    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = SG2_SHOWROOM_ROBOT_POS
    robot_cfg.init_state.rot = SG2_SHOWROOM_ROBOT_ROT
    robot_cfg.init_state.joint_pos.update(SG2_SHOWROOM_INITIAL_JOINT_POSITIONS)
    base_drive_actuator = robot_cfg.actuators.get("base_drive")
    if base_drive_actuator is not None:
        base_drive_actuator.velocity_limit_sim *= FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


def apply_sg2_showroom_camera_profile(env_cfg, profile_name: str) -> None:
    """Apply one physical-robot camera profile to a showroom env config."""

    profile = load_sg2_camera_profile(profile_name)
    head = profile.camera("head")
    wrist_left = profile.camera("wrist_left")
    wrist_right = profile.camera("wrist_right")

    env_cfg.robot_profile = profile_name
    env_cfg.robot_profile_id = profile.profile_id
    env_cfg.robot_profile_sha256 = profile.source_sha256
    env_cfg.robot_profile_source = str(profile.source_path)

    env_cfg.scene.cam_head = make_ffw_sg2_head_camera_cfg(
        update_period=0.0,
        width=head.width,
        height=head.height,
        intrinsic_matrix=head.intrinsic_matrix,
    )
    env_cfg.scene.cam_wrist_left = make_ffw_sg2_wrist_camera_cfg(
        "left",
        update_period=0.0,
        # The physical profile describes the sideways native 640x480 stream.
        # Swap its raster axes so the simulator emits the upright 480x640 frame.
        width=wrist_left.height,
        height=wrist_left.width,
        intrinsic_matrix=wrist_left.intrinsic_matrix,
    )
    env_cfg.scene.cam_wrist_right = make_ffw_sg2_wrist_camera_cfg(
        "right",
        update_period=0.0,
        width=wrist_right.height,
        height=wrist_right.width,
        intrinsic_matrix=wrist_right.intrinsic_matrix,
    )


def enable_sg2_showroom_operator_cameras(env_cfg) -> None:
    """Attach the three external cameras used only by the operator dashboard."""

    env_cfg.scene.cam_overhead_left = make_ffw_sg2_overhead_camera_cfg(
        "left", update_period=0.0
    )
    env_cfg.scene.cam_overhead_center = make_ffw_sg2_overhead_camera_cfg(
        "center", update_period=0.0
    )
    env_cfg.scene.cam_overhead_right = make_ffw_sg2_overhead_camera_cfg(
        "right", update_period=0.0
    )
