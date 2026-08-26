"""Common episodic recorder shell for SG2 showroom tasks."""

from dataclasses import MISSING, dataclass

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    iter_robotis_showroom_object_cfgs,
    make_robotis_showroom_environment_cfg,
)
from cyclo_lab.robot_specs.ffw.sg2 import (
    DEFAULT_SG2_CAMERA_PROFILE,
    FFW_SG2_PUBLISHED_JOINT_NAMES,
)

from ..platform import observations as showroom_obs
from ..platform.action_cfg import EpisodicShowroomActionsCfg
from ..platform.env_cfg import DeterministicResetEventsCfg
from ..platform.robot_cfg import (
    apply_sg2_showroom_camera_profile,
    enable_sg2_showroom_operator_cameras,
    make_sg2_showroom_robot_cfg,
)
from ..randomization.cfg import (
    ShowroomGenerationRandomizationCfg,
    ShowroomRandomizationCfg,
)
from ..randomization.event_cfg import (
    configure_profiled_reset_events,
    validate_profile_scene_entities,
)
from ..randomization.profiles import NO_RANDOMIZATION
from ..platform.scene_cfg import (
    OPERATOR_CAMERA_ROTATIONS,
    OPERATOR_CAMERA_ROWS,
    ShowroomSceneCfg,
)


@dataclass(frozen=True)
class ShowroomTaskSpec:
    """Identity and platform choices for one episodic showroom task."""

    task_id: str
    env_name: str
    instruction: str
    target_object: str
    target_side: str
    environment_usd_path: str
    policy_cameras: tuple[str, ...] = ("cam_head", "cam_wrist_left", "cam_wrist_right")
    robot_profile: str = DEFAULT_SG2_CAMERA_PROFILE
    control_hz: float = 15.0
    physics_hz: float = 30.0
    camera_hz: float = 15.0
    episode_length_s: float = 120.0


@configclass
class EpisodicShowroomSceneCfg(ShowroomSceneCfg):
    left_eef: FrameTransformerCfg = MISSING
    right_eef: FrameTransformerCfg = MISSING


@configclass
class EpisodicShowroomObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=showroom_obs.last_action)
        joint_pos = ObsTerm(
            func=showroom_obs.joint_pos_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=showroom_obs.joint_pos_target_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        left_eef_pose = ObsTerm(
            func=showroom_obs.eef_pose,
            params={"eef_cfg": SceneEntityCfg("left_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        right_eef_pose = ObsTerm(
            func=showroom_obs.eef_pose,
            params={"eef_cfg": SceneEntityCfg("right_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        cam_head = ObsTerm(
            func=showroom_obs.image,
            params={"sensor_cfg": SceneEntityCfg("cam_head"), "data_type": "rgb", "normalize": False},
        )
        cam_wrist_left = ObsTerm(
            func=showroom_obs.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_left"), "data_type": "rgb", "normalize": False},
        )
        cam_wrist_right = ObsTerm(
            func=showroom_obs.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_right"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class EpisodicShowroomTerminationsCfg:
    time_out = DoneTerm(func=showroom_obs.time_out, time_out=True)


def make_eef_frame(side: str) -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = f"/Visuals/ShowroomTask/{side.capitalize()}EEF"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/ffw_sg2_follower/arm_{side[0]}_link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, -0.2]),
            )
        ],
    )


@configclass
class EpisodicShowroomTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Reusable 19D episodic environment for HDF5 seed recording."""

    task_spec: ShowroomTaskSpec = MISSING
    env_name: str = ""
    task_id: str = ""
    task_instruction: str = ""
    target_object: str = ""
    target_side: str = ""
    policy_camera_names: tuple[str, ...] = ()
    robot_profile: str = ""
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""
    randomization: ShowroomRandomizationCfg | ShowroomGenerationRandomizationCfg = (
        NO_RANDOMIZATION
    )
    control_hz: float = 15.0
    physics_hz: float = 30.0
    camera_hz: float = 15.0
    recording_control_hz: float = 15.0
    operator_camera_rows: tuple = OPERATOR_CAMERA_ROWS
    operator_camera_rotations: tuple = OPERATOR_CAMERA_ROTATIONS
    operator_camera_title: str = "SG2 Task Operator Dashboard"
    operator_camera_window_size: int = 1800

    scene: EpisodicShowroomSceneCfg = EpisodicShowroomSceneCfg(
        num_envs=1, env_spacing=8.0, replicate_physics=False
    )
    observations: EpisodicShowroomObservationsCfg = EpisodicShowroomObservationsCfg()
    actions: EpisodicShowroomActionsCfg = EpisodicShowroomActionsCfg()
    events: DeterministicResetEventsCfg = DeterministicResetEventsCfg()
    terminations: EpisodicShowroomTerminationsCfg = EpisodicShowroomTerminationsCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        spec = self.task_spec
        self.env_name = spec.env_name
        self.task_id = spec.task_id
        self.task_instruction = spec.instruction
        self.target_object = spec.target_object
        self.target_side = spec.target_side
        self.policy_camera_names = spec.policy_cameras
        self.robot_profile = spec.robot_profile
        self.control_hz = spec.control_hz
        self.physics_hz = spec.physics_hz
        self.camera_hz = spec.camera_hz
        self.recording_control_hz = spec.control_hz
        self.operator_camera_title = f"SG2 Task {spec.task_id} Operator Dashboard"

        control_ratio = self.physics_hz / self.control_hz
        render_ratio = self.physics_hz / self.camera_hz
        if min(self.control_hz, self.physics_hz, self.camera_hz) <= 0.0:
            raise ValueError("Showroom task rates must be positive")
        if not control_ratio.is_integer() or not render_ratio.is_integer():
            raise ValueError("Showroom task physics/control and physics/render ratios must be integers")
        self.decimation = int(control_ratio)
        self.episode_length_s = spec.episode_length_s
        self.sim.dt = 1.0 / self.physics_hz
        self.sim.render_interval = int(render_ratio)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(spec.environment_usd_path)
        apply_sg2_showroom_camera_profile(self, self.robot_profile)
        for object_name, object_cfg in iter_robotis_showroom_object_cfgs():
            setattr(self.scene, object_name, object_cfg)
        if not hasattr(self.scene, self.target_object):
            raise ValueError(f"Task target object is missing from the showroom: {self.target_object}")
        self.scene.left_eef = make_eef_frame("left")
        self.scene.right_eef = make_eef_frame("right")
        self.apply_randomization_profile(self.randomization)

    def apply_randomization_profile(
        self,
        profile: ShowroomRandomizationCfg | ShowroomGenerationRandomizationCfg,
    ) -> None:
        """Rebuild optional reset terms from one selected profile."""
        self.randomization = profile
        validate_profile_scene_entities(
            self.scene, profile, target_object=self.target_object
        )
        self.events = configure_profiled_reset_events(
            type(self.events)(),
            profile,
            target_object=self.target_object,
        )

    def enable_operator_preview_cameras(self) -> None:
        enable_sg2_showroom_operator_cameras(self)

    def init_action_cfg(self, mode: str) -> None:
        if mode not in ("record", "inference"):
            raise ValueError(f"Episodic showroom task does not support action mode: {mode}")
