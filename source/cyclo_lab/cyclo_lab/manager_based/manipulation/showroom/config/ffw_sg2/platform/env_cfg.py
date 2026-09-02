"""Joint-position SG2 showroom environment configuration."""

from __future__ import annotations

from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    iter_robotis_showroom_object_cfgs,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.robot_specs.ffw.sg2 import DEFAULT_SG2_CAMERA_PROFILE

from ..randomization import events as random_events
from ..randomization.cfg import ShowroomRandomizationCfg
from ..randomization.event_cfg import (
    configure_profiled_reset_events,
    validate_profile_scene_entities,
)
from ..randomization.profiles import DIGITAL_TWIN_RANDOM, NO_RANDOMIZATION
from .robot_cfg import (
    SG2_SHOWROOM_INITIAL_JOINT_POSITIONS,
    apply_sg2_showroom_camera_profile,
    enable_sg2_showroom_operator_cameras,
    enable_sg2_showroom_ui_session_camera,
    make_sg2_showroom_robot_cfg,
)
from .scene_cfg import ShowroomEnvCfg



@configclass
class DeterministicResetEventsCfg:
    """Restore the authored scene and SG2 joint pose without randomization."""

    reset_scene_to_default = EventTerm(
        func=reset_scene_to_default,
        mode="reset",
        params={"reset_joint_targets": True},
    )
    set_robot_joint_pose = EventTerm(
        func=random_events.set_default_joint_pose,
        mode="reset",
        params={
            "joint_positions": SG2_SHOWROOM_INITIAL_JOINT_POSITIONS,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    randomize_robot_root_pose = None
    randomize_selected_objects = None


@configclass
class FFWSG2ShowroomEnvCfg(ShowroomEnvCfg):
    """Canonical SG2 showroom environment used by the ROS2 topic runner."""

    robot_profile: str = DEFAULT_SG2_CAMERA_PROFILE
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""
    randomization: ShowroomRandomizationCfg = NO_RANDOMIZATION

    def __post_init__(self):
        super().__post_init__()
        self.events = DeterministicResetEventsCfg()

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH)
        self.apply_robot_profile(self.robot_profile)

        for object_name, object_cfg in iter_robotis_showroom_object_cfgs():
            setattr(self.scene, object_name, object_cfg)
        self.apply_randomization_profile(self.randomization)

    def apply_randomization_profile(self, profile: ShowroomRandomizationCfg) -> None:
        """Apply one pose profile to a fresh deterministic reset configuration."""
        self.randomization = profile
        validate_profile_scene_entities(self.scene, profile)
        self.events = configure_profiled_reset_events(
            DeterministicResetEventsCfg(), profile
        )

    def apply_robot_profile(self, profile_name: str) -> None:
        """Apply a validated physical-robot camera profile to this task."""

        apply_sg2_showroom_camera_profile(self, profile_name)

    def enable_operator_preview_cameras(self) -> None:
        """Enable the robot-following cameras used only by the operator dashboard."""
        enable_sg2_showroom_operator_cameras(self)

    def enable_ui_session_camera(self) -> None:
        """Enable only the external camera streamed to Cyclo Intelligence."""
        enable_sg2_showroom_ui_session_camera(self)


@configclass
class ContinuousShowroomEnvCfg(FFWSG2ShowroomEnvCfg):
    """Public deterministic 22D digital-twin preset."""


@configclass
class ContinuousRandomShowroomEnvCfg(ContinuousShowroomEnvCfg):
    """The same 22D digital twin with reset randomization enabled."""

    env_name: str = "Cyclo-Real-Showroom-Random-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = DIGITAL_TWIN_RANDOM

    def __post_init__(self):
        super().__post_init__()
        self.env_name = "Cyclo-Real-Showroom-Random-FFW-SG2-v0"
