"""Task 000458 seed-demonstration environment in the ROBOTIS showroom.

This configuration intentionally keeps the existing continuous showroom task
unchanged. It adds the RL episode, recorder, end-effector, and camera
interfaces required by ``scripts/sim2real/imitation_learning/recorder/record_demos.py``.
"""

from __future__ import annotations

from dataclasses import MISSING

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
from cyclo_lab.manager_based.actions import FFWSG2JointPositionActionsCfg
from cyclo_lab.manager_based.manipulation.pick_place.config.ffw_sg2 import mdp as pick_place_mdp
from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_PUBLISHED_JOINT_NAMES

from . import mdp
from .joint_pos_env_cfg import EventCfg as ShowroomEventCfg
from .joint_pos_env_cfg import FFWSG2ShowroomEnvCfg, make_sg2_showroom_robot_cfg
from .seed_task_specs import TASK_000458_SPEC, SeedTaskSpec
from .showroom_env_cfg import ShowroomSceneCfg


@configclass
class SeedTaskSceneCfg(ShowroomSceneCfg):
    """Showroom scene extended with the EEF frames required by Mimic."""

    left_eef: FrameTransformerCfg = MISSING
    right_eef: FrameTransformerCfg = MISSING


@configclass
class SeedTaskActionsCfg(FFWSG2JointPositionActionsCfg):
    """Ordered 19D SG2 joint targets; the mobile base stays fixed per episode."""


@configclass
class SeedTaskObservationsCfg:
    """Recorder observations shared by raw seed capture and later IK conversion."""

    @configclass
    class PolicyCfg(ObsGroup):
        actions = ObsTerm(func=mdp.last_action)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        joint_pos_target = ObsTerm(
            func=mdp.joint_pos_target_name,
            params={"joint_names": FFW_SG2_PUBLISHED_JOINT_NAMES, "asset_name": "robot"},
        )
        left_eef_pose = ObsTerm(
            func=pick_place_mdp.eef_pose,
            params={"eef_cfg": SceneEntityCfg("left_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        right_eef_pose = ObsTerm(
            func=pick_place_mdp.eef_pose,
            params={"eef_cfg": SceneEntityCfg("right_eef"), "robot_cfg": SceneEntityCfg("robot")},
        )
        cam_head = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_head"), "data_type": "rgb", "normalize": False},
        )
        cam_wrist_left = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_left"), "data_type": "rgb", "normalize": False},
        )
        cam_wrist_right = ObsTerm(
            func=mdp.image,
            params={"sensor_cfg": SceneEntityCfg("cam_wrist_right"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class SeedTaskTerminationsCfg:
    """Minimal episode terms; success is selected manually with the N key."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


def _make_eef_frame(side: str) -> FrameTransformerCfg:
    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = f"/Visuals/SeedTask/{side.capitalize()}EEF"
    return FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/ffw_sg2_follower/arm_base_link",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/ffw_sg2_follower/arm_{side[0]}_link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, -0.2]),
            ),
        ],
    )


@configclass
class FFWSG2ShowroomSeedTaskEnvCfg(ManagerBasedRLEnvCfg):
    """Single-environment SG2 showroom configuration for raw HDF5 seed capture."""

    seed_task_spec: SeedTaskSpec = TASK_000458_SPEC
    env_name: str = TASK_000458_SPEC.env_name
    task_id: str = TASK_000458_SPEC.task_id
    task_instruction: str = TASK_000458_SPEC.instruction
    target_object: str = TASK_000458_SPEC.target_object
    target_side: str = TASK_000458_SPEC.target_side
    policy_camera_names: tuple[str, ...] = TASK_000458_SPEC.policy_cameras

    control_hz: float = TASK_000458_SPEC.control_hz
    physics_hz: float = TASK_000458_SPEC.physics_hz
    camera_hz: float = TASK_000458_SPEC.camera_hz
    recording_control_hz: float = TASK_000458_SPEC.control_hz

    robot_profile: str = TASK_000458_SPEC.robot_profile
    robot_profile_id: str = ""
    robot_profile_sha256: str = ""
    robot_profile_source: str = ""

    operator_camera_rows: tuple = (
        (
            ("cam_overhead_left", "External Left"),
            ("cam_overhead_center", "External Top"),
            ("cam_overhead_right", "External Right"),
        ),
        (
            ("cam_wrist_left", "Wrist Left"),
            ("cam_head", "Head"),
            ("cam_wrist_right", "Wrist Right"),
        ),
    )
    operator_camera_rotations: tuple = (
        ("cam_wrist_left", 1),
        ("cam_wrist_right", 1),
    )
    operator_camera_title: str = "SG2 Task 458 Operator Dashboard"
    operator_camera_window_size: int = 1800

    scene: SeedTaskSceneCfg = SeedTaskSceneCfg(num_envs=1, env_spacing=8.0, replicate_physics=False)
    observations: SeedTaskObservationsCfg = SeedTaskObservationsCfg()
    actions: SeedTaskActionsCfg = SeedTaskActionsCfg()
    events: ShowroomEventCfg = ShowroomEventCfg()
    terminations: SeedTaskTerminationsCfg = SeedTaskTerminationsCfg()
    recorders: ActionStateRecorderManagerCfg = ActionStateRecorderManagerCfg()

    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        spec = self.seed_task_spec
        self.env_name = spec.env_name
        self.task_id = spec.task_id
        self.task_instruction = spec.instruction
        self.target_object = spec.target_object
        self.target_side = spec.target_side
        self.policy_camera_names = spec.policy_cameras
        self.control_hz = spec.control_hz
        self.physics_hz = spec.physics_hz
        self.camera_hz = spec.camera_hz
        self.recording_control_hz = spec.control_hz
        self.robot_profile = spec.robot_profile
        self.operator_camera_title = f"SG2 Task {spec.task_id} Operator Dashboard"

        if min(self.control_hz, self.physics_hz, self.camera_hz) <= 0.0:
            raise ValueError("Task 458 control, physics, and camera rates must be positive.")
        if self.physics_hz < max(self.control_hz, self.camera_hz):
            raise ValueError("Task 458 physics rate must cover both control and camera rates.")

        physics_steps_per_control = self.physics_hz / self.control_hz
        physics_steps_per_render = self.physics_hz / self.camera_hz
        if not physics_steps_per_control.is_integer() or not physics_steps_per_render.is_integer():
            raise ValueError("Task 458 physics/control and physics/render ratios must be integers.")

        self.decimation = int(physics_steps_per_control)
        self.episode_length_s = self.seed_task_spec.episode_length_s
        self.sim.dt = 1.0 / self.physics_hz
        self.sim.render_interval = int(physics_steps_per_render)
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625

        self.scene.robot = make_sg2_showroom_robot_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.spawn.semantic_tags = [("class", "robot")]
        self.scene.environment = make_robotis_showroom_environment_cfg(self.seed_task_spec.environment_usd_path)

        # Reuse the calibrated camera-profile implementation used by the
        # continuous showroom instead of defining a second camera contract.
        FFWSG2ShowroomEnvCfg.apply_robot_profile(self, self.robot_profile)

        for object_name, object_cfg in iter_robotis_showroom_object_cfgs():
            setattr(self.scene, object_name, object_cfg)
        if not hasattr(self.scene, self.target_object):
            raise ValueError(f"Seed task target object is missing from the showroom: {self.target_object}")

        self.scene.left_eef = _make_eef_frame("left")
        self.scene.right_eef = _make_eef_frame("right")

    def enable_operator_preview_cameras(self) -> None:
        """Enable the robot-following cameras used only by the operator dashboard."""
        FFWSG2ShowroomEnvCfg.enable_operator_preview_cameras(self)

    def init_action_cfg(self, mode: str) -> None:
        """Satisfy the shared recorder contract without changing the 19D order."""
        if mode not in ("record", "inference"):
            raise ValueError(f"Seed task environment does not support action mode: {mode}")


@configclass
class FFWSG2ShowroomTask000458SeedEnvCfg(FFWSG2ShowroomSeedTaskEnvCfg):
    """Task 000458 seed-demonstration configuration."""

    seed_task_spec: SeedTaskSpec = TASK_000458_SPEC
