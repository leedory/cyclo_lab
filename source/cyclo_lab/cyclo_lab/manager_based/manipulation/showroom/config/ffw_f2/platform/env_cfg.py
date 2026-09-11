"""Continuous F2 showroom environment configuration."""

from isaaclab.envs.mdp.events import reset_scene_to_default
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH,
    iter_robotis_showroom_object_cfgs,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.robot_specs.ffw.f2 import DEFAULT_F2_CAMERA_PROFILE

from ...ffw_sg2.platform.scene_cfg import ShowroomEnvCfg
from .action_cfg import ContinuousShowroomActionsCfg
from .robot_cfg import (
    apply_f2_showroom_camera_profile,
    enable_f2_showroom_operator_cameras,
    make_f2_showroom_robot_cfg,
)


@configclass
class DeterministicResetEventsCfg:
    reset_scene_to_default = EventTerm(
        func=reset_scene_to_default, mode="reset", params={"reset_joint_targets": True}
    )


@configclass
class ContinuousShowroomEnvCfg(ShowroomEnvCfg):
    """Deterministic 22D F2 showroom template."""

    env_name: str = "Cyclo-Real-Showroom-FFW-F2-v0"
    operator_camera_title: str = "F2 Operator Dashboard"
    actions: ContinuousShowroomActionsCfg = ContinuousShowroomActionsCfg()
    robot_profile: str = DEFAULT_F2_CAMERA_PROFILE
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""

    def __post_init__(self):
        super().__post_init__()
        self.events = DeterministicResetEventsCfg()
        self.scene.robot = make_f2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(ROBOTIS_SHOWROOM_BACKGROUND_USD_PATH)
        self.apply_robot_profile(self.robot_profile)
        for object_name, object_cfg in iter_robotis_showroom_object_cfgs():
            setattr(self.scene, object_name, object_cfg)

    def apply_robot_profile(self, profile_name: str) -> None:
        apply_f2_showroom_camera_profile(self, profile_name)

    def enable_operator_preview_cameras(self) -> None:
        enable_f2_showroom_operator_cameras(self)
