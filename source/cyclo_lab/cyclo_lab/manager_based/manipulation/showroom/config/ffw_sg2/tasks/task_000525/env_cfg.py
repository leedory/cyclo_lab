"""Runnable 22D environment presets for showroom Task000525."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from cyclo_lab.assets.environments.robotis_showroom import (
    iter_robotis_showroom_object_cfgs,
)

from ...platform import observations as showroom_obs
from ...platform.action_cfg import ContinuousShowroomActionsCfg
from ..common import (
    EpisodicShowroomObservationsCfg,
    EpisodicShowroomTaskEnvCfg,
    ShowroomTaskSpec,
)
from .appearance_events import randomize_coffee_can_visual_yaw
from .destination_mat import (
    attach_task000525_destination_mat_to_table,
    make_task000525_destination_mat_cfg,
)
from .home_pose import TASK000525_SAVE_POSE_3_JOINT_POSITIONS
from .layout import TASK000525_SELECTED_LAYOUT_KEY
from .object_cfg import iter_task000525_coffee_can_cfgs
from .profiles import (
    TASK000525_DETERMINISTIC,
    TASK000525_RECORD_RANDOMIZED,
    Task000525RandomizationCfg,
    validate_task000525_randomization_cfg,
)
from .reset_events import randomize_coffee_can_center_regions
from .robot_stability import apply_task000525_arm_hold_tuning
from .spec import TASK_000525_ROBOT_USD_PATH, TASK_000525_SPEC


# Task458's cabinet-relative pose translated by the measured cabinet_01 ->
# cabinet_02 displacement (0.0, -0.815072039386852, 0.0). Orientation and the
# initial upper-body joint pose remain identical to Task458.
TASK_000525_ROBOT_SPAWN_POS = (-1.47138, 0.775837960613148, 0.0)
TASK_000525_ROBOT_SPAWN_ROT = (0.0, 0.0, 0.0, 1.0)


@configclass
class Task000525ObservationsCfg(EpisodicShowroomObservationsCfg):
    """Policy data for joint19 + measured mobile state and SDG frames."""

    @configclass
    class PolicyCfg(EpisodicShowroomObservationsCfg.PolicyCfg):
        base_velocity_body = ObsTerm(
            func=showroom_obs.base_twist, params={"asset_name": "robot"}
        )
        robot_root_pose_world = ObsTerm(
            func=showroom_obs.asset_root_pose_world, params={"asset_name": "robot"}
        )
        target_object_pose_world = ObsTerm(
            func=showroom_obs.asset_root_pose_world,
            params={"asset_name": TASK_000525_SPEC.target_object},
        )
        left_eef_pose_world = ObsTerm(
            func=showroom_obs.eef_pose_world,
            params={"eef_cfg": SceneEntityCfg("left_eef")},
        )
        right_eef_pose_world = ObsTerm(
            func=showroom_obs.eef_pose_world,
            params={"eef_cfg": SceneEntityCfg("right_eef")},
        )

        def __post_init__(self):
            super().__post_init__()

    policy: PolicyCfg = PolicyCfg()


def _remove_authored_showroom_objects(scene_cfg) -> None:
    """Leave Task000525 with only its four task-local dynamic coffee cans."""

    for object_name, _ in iter_robotis_showroom_object_cfgs():
        if hasattr(scene_cfg, object_name):
            delattr(scene_cfg, object_name)


@configclass
class Task000525EnvCfg(EpisodicShowroomTaskEnvCfg):
    """Deterministic B-layout centers with 19 joint plus 3 base actions."""

    task_spec: ShowroomTaskSpec = TASK_000525_SPEC
    randomization: Task000525RandomizationCfg = TASK000525_DETERMINISTIC
    actions: ContinuousShowroomActionsCfg = ContinuousShowroomActionsCfg()
    observations: Task000525ObservationsCfg = Task000525ObservationsCfg()

    def __post_init__(self):
        # Add the task-local coffee entities before the common task shell checks
        # that its selected target exists.
        for object_name, object_cfg in iter_task000525_coffee_can_cfgs(
            TASK000525_SELECTED_LAYOUT_KEY
        ):
            setattr(self.scene, object_name, object_cfg)

        super().__post_init__()

        # The common shell adds authored jelly/peanut/chestnut/basket entities.
        # Task000525's freely dynamic object set is the four coffee cans.
        _remove_authored_showroom_objects(self.scene)

        # Keep the reviewed mat task-local and rigid, then attach it to the
        # authored dining-table rigid body before PhysX parses the stage.
        self.scene.destination_mat = make_task000525_destination_mat_cfg()
        self.events.attach_task000525_destination_mat_to_table = EventTerm(
            func=attach_task000525_destination_mat_to_table,
            mode="prestartup",
        )

        # Use the task-specific rigid soft-finger USD without changing SG2
        # consumers outside Task000525.
        self.scene.robot.spawn.usd_path = TASK_000525_ROBOT_USD_PATH
        self.scene.robot.init_state.pos = TASK_000525_ROBOT_SPAWN_POS
        self.scene.robot.init_state.rot = TASK_000525_ROBOT_SPAWN_ROT
        self.scene.robot.init_state.joint_pos.update(TASK000525_SAVE_POSE_3_JOINT_POSITIONS)
        # ``set_robot_joint_pose`` is the reset event used by R/N. Use a copy
        # to prevent a later task-local mutation from changing the named
        # constant or a shared event configuration.
        self.events.set_robot_joint_pose.params["joint_positions"] = dict(
            TASK000525_SAVE_POSE_3_JOINT_POSITIONS
        )

        # Online Dijkstra limits linear speed to 0.10 m/s.  The shared swerve
        # default deadband is also 0.10 m/s, which erases diagonal vx/vy path
        # components before they reach the wheels.  This task-only threshold
        # keeps the conservative command while preserving its direction.
        self.actions.base_action.linear_deadband = 0.01
        self.actions.base_action.angular_deadband = 0.01

        # At 30 Hz, PhysX explicitly recommends stabilization. The wheel
        # collision floor already coincides with root Z, so changing spawn Z
        # would introduce a real gap or penetration instead of fixing settling.
        self.sim.physx.enable_stabilization = True

        # The baseline arm can gravity-settle by about 1.05 degrees while
        # holding its initial targets. Task-local gain scaling reduced the
        # measured maximum to about 0.61 degrees without increasing effort
        # limits or changing Task458/shared SG2 dynamics.
        apply_task000525_arm_hold_tuning(self.scene.robot)

        validate_task000525_randomization_cfg(self.randomization)
        coffee_positions = self.randomization.coffee_positions
        if coffee_positions.enabled:
            self.events.randomize_task000525_coffee_positions = EventTerm(
                func=randomize_coffee_can_center_regions,
                mode="reset",
                params={"layout_key": coffee_positions.layout_key},
            )
        coffee_visual_yaw = self.randomization.coffee_visual_yaw
        if coffee_visual_yaw.enabled:
            self.events.randomize_task000525_coffee_visual_yaw = EventTerm(
                func=randomize_coffee_can_visual_yaw,
                mode="reset",
                params={
                    "object_names": coffee_visual_yaw.object_names,
                    "yaw_range_rad": coffee_visual_yaw.yaw_range_rad,
                },
            )


@configclass
class Task000525RandomEnvCfg(Task000525EnvCfg):
    """B-region X/Y plus collision-invariant visual-label yaw randomization."""

    env_name: str = "Cyclo-Real-Showroom-Task000525-Random-FFW-SG2-v0"
    randomization: Task000525RandomizationCfg = TASK000525_RECORD_RANDOMIZED

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
