"""Joint-position SG2 showroom environment configuration."""

from __future__ import annotations

from copy import deepcopy

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS,
    ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    iter_robotis_showroom_object_cfgs,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.assets.sensors.ffw_sg2_cameras import (
    make_ffw_sg2_head_camera_cfg,
    make_ffw_sg2_overhead_camera_cfg,
    make_ffw_sg2_wrist_camera_cfg,
)
from cyclo_lab.assets.robots import FFW_SG2_PHYSICS_CFG
from cyclo_lab.manager_based.actions import make_ffw_sg2_measured_response_actions_cfg
from cyclo_lab.robot_specs.ffw.sg2 import (
    DEFAULT_SG2_CAMERA_PROFILE,
    FFW_SG2_SWERVE_DRIVE_SPEED_SCALE,
    SG2_MEASURED_RESPONSE_PROFILE_ID,
    load_sg2_camera_profile,
)

from .mdp import ffw_sg2_showroom_events
from .showroom_env_cfg import ActionsCfg, ShowroomEnvCfg


# Keep the real-aligned reset pose horizontally registered to the snack shelf.
SG2_SHOWROOM_ROBOT_POS = (-1.47138, 1.59091, 0.0)
SG2_SHOWROOM_ROBOT_ROT = (0.0, 0.0, 0.0, 1.0)
SG2_SHOWROOM_ROOT_POSITION_RANDOMIZATION_RADIUS = 0.0
SG2_SHOWROOM_ROOT_YAW_RANDOMIZATION = 0.0
SG2_SHOWROOM_WALL_BACKGROUND_ZOOM_RANGE = (1.0, 1.3)
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
    robot_cfg = deepcopy(FFW_SG2_PHYSICS_CFG)
    robot_cfg.spawn.rigid_props.disable_gravity = False
    robot_cfg.init_state.pos = SG2_SHOWROOM_ROBOT_POS
    robot_cfg.init_state.rot = SG2_SHOWROOM_ROBOT_ROT
    robot_cfg.init_state.joint_pos.update(SG2_SHOWROOM_INITIAL_JOINT_POSITIONS)
    base_drive_actuator = robot_cfg.actuators.get("base_drive")
    if base_drive_actuator is not None:
        base_drive_actuator.velocity_limit_sim *= FFW_SG2_SWERVE_DRIVE_SPEED_SCALE
    return robot_cfg


@configclass
class EventCfg:
    """Reset events for the SG2 showroom joint-position task."""

    reset_scene_to_default = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )

    randomize_robot_root_pose = EventTerm(
        func=ffw_sg2_showroom_events.randomize_root_pose_in_radius,
        mode="reset",
        params={
            "max_translation_radius": SG2_SHOWROOM_ROOT_POSITION_RANDOMIZATION_RADIUS,
            "max_yaw": SG2_SHOWROOM_ROOT_YAW_RANDOMIZATION,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    randomize_wall_background = EventTerm(
        func=ffw_sg2_showroom_events.randomize_wall_background,
        mode="reset",
        params={
            "texture_paths": ROBOTIS_SHOWROOM_BACKGROUND_TEXTURE_PATHS,
            "zoom_range": SG2_SHOWROOM_WALL_BACKGROUND_ZOOM_RANGE,
        },
    )

    set_robot_joint_pose = EventTerm(
        func=ffw_sg2_showroom_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": SG2_SHOWROOM_INITIAL_JOINT_POSITIONS,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class FFWSG2ShowroomEnvCfg(ShowroomEnvCfg):
    """Canonical SG2 showroom environment used by the ROS2 topic runner."""

    robot_profile: str = DEFAULT_SG2_CAMERA_PROFILE
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""
    joint_response_profile: str = "ideal"
    joint_response_profile_id: str = "ideal"

    def __post_init__(self):
        if self.joint_response_profile == "ideal":
            self.joint_response_profile_id = "ideal"
        else:
            self._configure_joint_response(self.joint_response_profile)
        super().__post_init__()
        self.events = EventCfg()

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH)
        self.apply_robot_profile(self.robot_profile)

        for object_name, object_cfg in iter_robotis_showroom_object_cfgs():
            setattr(self.scene, object_name, object_cfg)

    def apply_joint_response_profile(self, profile_name: str) -> None:
        """Select the stock or measured actuator response before creating the environment."""
        self._configure_joint_response(profile_name)
        ShowroomEnvCfg.__post_init__(self)

    def _configure_joint_response(self, profile_name: str) -> None:
        if profile_name == "ideal":
            self.actions = ActionsCfg()
            self.joint_response_profile_id = "ideal"
            self.control_hz = 15.0
            self.physics_hz = 30.0
            self.camera_hz = 15.0
        elif profile_name in ("measured", "measured-randomized"):
            randomized = profile_name == "measured-randomized"
            self.actions = make_ffw_sg2_measured_response_actions_cfg(randomized=randomized)
            self.joint_response_profile_id = SG2_MEASURED_RESPONSE_PROFILE_ID
            # The measured response profile was identified at these rates.
            self.control_hz = 100.0
            self.physics_hz = 100.0
            self.camera_hz = 10.0
        else:
            raise ValueError(
                "Unsupported SG2 joint response profile "
                f"{profile_name!r}; expected ideal, measured, or measured-randomized."
            )
        self.joint_response_profile = profile_name

    def apply_robot_profile(self, profile_name: str) -> None:
        """Apply a validated physical-robot camera profile to this task."""

        profile = load_sg2_camera_profile(profile_name)
        head = profile.camera("head")
        wrist_left = profile.camera("wrist_left")
        wrist_right = profile.camera("wrist_right")

        self.robot_profile = profile_name
        self.robot_profile_id = profile.profile_id
        self.robot_profile_sha256 = profile.source_sha256
        self.robot_profile_source = str(profile.source_path)

        # Rendering cadence is owned by sim.render_interval. Sensors expose each
        # newly rendered frame directly to the topic bridge and operator viewer.
        self.scene.cam_head = make_ffw_sg2_head_camera_cfg(
            update_period=0.0,
            width=head.width,
            height=head.height,
            intrinsic_matrix=head.intrinsic_matrix,
        )
        self.scene.cam_wrist_left = make_ffw_sg2_wrist_camera_cfg(
            "left",
            update_period=0.0,
            width=wrist_left.width,
            height=wrist_left.height,
            intrinsic_matrix=wrist_left.intrinsic_matrix,
        )
        self.scene.cam_wrist_right = make_ffw_sg2_wrist_camera_cfg(
            "right",
            update_period=0.0,
            width=wrist_right.width,
            height=wrist_right.height,
            intrinsic_matrix=wrist_right.intrinsic_matrix,
        )

    def enable_operator_preview_cameras(self) -> None:
        """Enable the robot-following cameras used only by the operator dashboard."""
        self.scene.cam_overhead_left = make_ffw_sg2_overhead_camera_cfg(
            "left",
            update_period=0.0,
        )
        self.scene.cam_overhead_center = make_ffw_sg2_overhead_camera_cfg(
            "center",
            update_period=0.0,
        )
        self.scene.cam_overhead_right = make_ffw_sg2_overhead_camera_cfg(
            "right",
            update_period=0.0,
        )
