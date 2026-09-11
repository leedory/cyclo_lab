"""F2 robot and camera construction for the showroom environment."""

from copy import deepcopy

from isaaclab.assets.articulation import ArticulationCfg

from cyclo_lab.assets.robots import FFW_F2_PHYSICS_CFG
from cyclo_lab.assets.sensors.ffw_f2_cameras import (
    make_ffw_f2_head_camera_cfg,
    make_ffw_f2_overhead_camera_cfg,
    make_ffw_f2_wrist_camera_cfg,
)
from cyclo_lab.robot_specs.ffw.f2 import FFW_F2_SWERVE_DRIVE_SPEED_SCALE, load_f2_camera_profile


F2_SHOWROOM_ROBOT_POS = (-1.47138, 1.59091, 0.0)
F2_SHOWROOM_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
F2_SHOWROOM_INITIAL_JOINT_POSITIONS = {
    **{f"arm_l_joint{index}": 0.0 for index in range(1, 8)},
    "gripper_l_joint1": 0.0,
    **{f"arm_r_joint{index}": 0.0 for index in range(1, 8)},
    "gripper_r_joint1": 0.0,
    "head_joint1": 0.0,
    "head_joint2": 0.0,
    "lift_joint": 0.0,
}


def make_f2_showroom_robot_cfg() -> ArticulationCfg:
    robot_cfg = deepcopy(FFW_F2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = F2_SHOWROOM_ROBOT_POS
    robot_cfg.init_state.rot = F2_SHOWROOM_ROBOT_ROT
    robot_cfg.init_state.joint_pos.update(F2_SHOWROOM_INITIAL_JOINT_POSITIONS)
    robot_cfg.actuators["base_drive"].velocity_limit_sim *= FFW_F2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


def apply_f2_showroom_camera_profile(env_cfg, profile_name: str) -> None:
    profile = load_f2_camera_profile(profile_name)
    head = profile.camera("head")
    wrist_left = profile.camera("wrist_left")
    wrist_right = profile.camera("wrist_right")

    env_cfg.robot_profile = profile_name
    env_cfg.robot_profile_id = profile.profile_id
    env_cfg.robot_profile_sha256 = profile.source_sha256
    env_cfg.robot_profile_source = str(profile.source_path)
    env_cfg.scene.cam_head = make_ffw_f2_head_camera_cfg(
        width=head.width, height=head.height, intrinsic_matrix=head.intrinsic_matrix
    )
    env_cfg.scene.cam_wrist_left = make_ffw_f2_wrist_camera_cfg(
        "left", width=wrist_left.width, height=wrist_left.height,
        intrinsic_matrix=wrist_left.intrinsic_matrix,
    )
    env_cfg.scene.cam_wrist_right = make_ffw_f2_wrist_camera_cfg(
        "right", width=wrist_right.width, height=wrist_right.height,
        intrinsic_matrix=wrist_right.intrinsic_matrix,
    )


def enable_f2_showroom_operator_cameras(env_cfg) -> None:
    env_cfg.scene.cam_overhead_left = make_ffw_f2_overhead_camera_cfg("left")
    env_cfg.scene.cam_overhead_center = make_ffw_f2_overhead_camera_cfg("center")
    env_cfg.scene.cam_overhead_right = make_ffw_f2_overhead_camera_cfg("right")

