"""Isaac Lab Mimic configuration for showroom peanut take-out generation."""

from __future__ import annotations

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.manager_based.actions.ffw_sg2 import (
    make_ffw_sg2_joint_position_action_cfg,
)
from cyclo_lab.manager_based.manipulation.common import configure_ffw_sg2_mimic_ik_actions
from cyclo_lab.robot_specs.ffw.sg2 import (
    FFW_SG2_HEAD_JOINT_NAMES,
    FFW_SG2_LEFT_ARM_JOINT_NAMES,
    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES,
    FFW_SG2_LIFT_JOINT_NAMES,
)

from ...platform.env_cfg import DeterministicResetEventsCfg as ShowroomEventCfg
from ...randomization.cfg import ShowroomGenerationRandomizationCfg
from ..common import EpisodicShowroomObservationsCfg
from . import takeout_terms
from .env_cfg import Task000458EnvCfg
from .profiles import TASK000458_MIMIC_GENERATION, TASK000458_MIMIC_SEED


@configclass
class Task458GenerationEventsCfg(ShowroomEventCfg):
    """Reset order: showroom, target support/pose, optional root, appearance."""

    refresh_shelf_support = None
    randomize_target_pose = None
    randomize_robot_root = None
    randomize_non_target_presence = None
    randomize_lighting = None
    randomize_shelf_appearance = None
    randomize_wall_color = None
    randomize_cameras = None


@configclass
class Task458MimicObservationsCfg(EpisodicShowroomObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        grasp_stable = None
        released_after_takeout = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class Task458MimicTerminationsCfg:
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    success = None


@configclass
class Task000458MimicSeedEnvCfg(Task000458EnvCfg, MimicEnvCfg):
    """Deterministic environment used to convert and annotate seed demos."""

    env_name: str = "Cyclo-Real-Showroom-Task000458-Mimic-Seed-FFW-SG2-v0"
    randomization: ShowroomGenerationRandomizationCfg = TASK000458_MIMIC_SEED
    hold_inactive_joint_targets: bool = False

    observations: Task458MimicObservationsCfg = Task458MimicObservationsCfg()
    events: Task458GenerationEventsCfg = Task458GenerationEventsCfg()
    terminations: Task458MimicTerminationsCfg = Task458MimicTerminationsCfg()

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name

        params = self.seed_success_metric_params
        self.observations.subtask_terms.grasp_stable = ObsTerm(
            func=takeout_terms.grasp_stable,
            params={"metric_params": params},
        )
        self.observations.subtask_terms.released_after_takeout = ObsTerm(
            func=takeout_terms.released_after_takeout,
            params={"metric_params": params},
        )
        self.terminations.success = DoneTerm(
            func=takeout_terms.takeout_success,
            params={"metric_params": params},
        )

        self.datagen_config.name = (
            f"task{self.task_id}_GENERATED_{self.target_object}_takeout_v1"
        )
        self.datagen_config.generation_guarantee = False
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 100
        self.datagen_config.generation_select_src_per_subtask = False
        self.datagen_config.generation_transform_first_robot_pose = False
        # Start each subtask blend from the achieved EEF pose, not the previous
        # command target. This reduces controller-lag jumps after grasp.
        self.datagen_config.generation_interpolate_from_last_target_pose = False
        self.datagen_config.max_num_failures = 100
        self.datagen_config.seed = 20260825

        self.subtask_configs.clear()
        self.subtask_configs["right_arm"] = [
            SubTaskConfig(
                object_ref=self.target_object,
                subtask_term_signal="grasp_stable",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description=f"Approach {self.target_object} and close the gripper",
                next_subtask_description=f"Pull {self.target_object} out of the shelf and release it",
            ),
            SubTaskConfig(
                object_ref=self.target_object,
                subtask_term_signal="released_after_takeout",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description=f"Pull {self.target_object} out of the shelf and release it",
                next_subtask_description="Return the right arm to the initial pose",
            ),
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Return the right arm to the initial pose",
                next_subtask_description="Task complete",
            ),
        ]

    def init_action_cfg(self, mode: str) -> None:
        if mode == "mimic_ik":
            configure_ffw_sg2_mimic_ik_actions(self.actions)
            if self.hold_inactive_joint_targets:
                self.actions.arm_l_action = make_ffw_sg2_joint_position_action_cfg(
                    FFW_SG2_LEFT_ARM_JOINT_NAMES
                )
                self.actions.gripper_l_action = make_ffw_sg2_joint_position_action_cfg(
                    FFW_SG2_LEFT_GRIPPER_JOINT_NAMES
                )
                self.actions.head_action = make_ffw_sg2_joint_position_action_cfg(
                    FFW_SG2_HEAD_JOINT_NAMES
                )
                self.actions.lift_action = make_ffw_sg2_joint_position_action_cfg(
                    FFW_SG2_LIFT_JOINT_NAMES
                )
            return
        super().init_action_cfg(mode)


@configclass
class Task000458MimicGenerateEnvCfg(Task000458MimicSeedEnvCfg):
    """Default generated-data task using the retained target and appearance ranges."""

    env_name: str = "Cyclo-Real-Showroom-Task000458-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomGenerationRandomizationCfg = TASK000458_MIMIC_GENERATION
    hold_inactive_joint_targets: bool = True
