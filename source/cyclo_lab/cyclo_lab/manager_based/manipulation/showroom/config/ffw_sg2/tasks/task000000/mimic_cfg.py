"""Task000000에 Isaac Lab Mimic을 연결하는 최소 구성 예시."""

from isaaclab.envs import mdp as isaaclab_mdp
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.common import (
    configure_ffw_sg2_mimic_ik_actions,
)

from ...platform.env_cfg import DeterministicResetEventsCfg
from ...randomization.cfg import ShowroomGenerationRandomizationCfg
from ..common import EpisodicShowroomObservationsCfg
from .env_cfg import Task000000EnvCfg
from .profiles import TASK000000_MIMIC_GENERATION, TASK000000_MIMIC_SEED


@configclass
class Task000000GenerationEventsCfg(DeterministicResetEventsCfg):
    """generation profile이 선택적으로 채울 reset event 자리."""

    refresh_shelf_support = None
    randomize_target_pose = None
    randomize_robot_root = None
    randomize_non_target_presence = None
    randomize_lighting = None
    randomize_shelf_appearance = None
    randomize_wall_color = None
    randomize_cameras = None


@configclass
class Task000000MimicObservationsCfg(EpisodicShowroomObservationsCfg):
    @configclass
    class SubtaskCfg(ObsGroup):
        # 예: grasped = ObsTerm(func=success_terms.grasped, params={...})
        # 아래 SubTaskConfig에서 쓰는 모든 신호를 여기에 같은 이름으로 연결한다.
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    subtask_terms: SubtaskCfg = SubtaskCfg()


@configclass
class Task000000MimicTerminationsCfg:
    time_out = DoneTerm(func=isaaclab_mdp.time_out, time_out=True)
    # 예: success = DoneTerm(func=success_terms.task_success, params={...})
    success = None


@configclass
class Task000000MimicSeedEnvCfg(Task000000EnvCfg, MimicEnvCfg):
    """원본 demo를 재생하고 subtask 구간을 확인하는 고정 환경."""

    env_name: str = "Cyclo-Real-Showroom-Task000000-Mimic-Seed-FFW-SG2-v0"
    randomization: ShowroomGenerationRandomizationCfg = TASK000000_MIMIC_SEED
    observations: Task000000MimicObservationsCfg = Task000000MimicObservationsCfg()
    events: Task000000GenerationEventsCfg = Task000000GenerationEventsCfg()
    terminations: Task000000MimicTerminationsCfg = Task000000MimicTerminationsCfg()

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name

        self.datagen_config.name = f"task{self.task_id}_GENERATED_v1"
        self.datagen_config.generation_num_trials = 10
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.max_num_failures = 25
        self.datagen_config.seed = 42

        self.subtask_configs.clear()
        self.subtask_configs[f"{self.target_side}_arm"] = [
            SubTaskConfig(
                object_ref=self.target_object,
                # 이 문자열과 같은 observation term을 위 SubtaskCfg에 구현한다.
                subtask_term_signal="<replace_with_grasp_signal>",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Approach and grasp the target object",
                next_subtask_description="Move and release the target object",
            ),
            SubTaskConfig(
                object_ref=self.target_object,
                subtask_term_signal="<replace_with_release_signal>",
                subtask_term_offset_range=(0, 0),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 1},
                action_noise=0.0,
                num_interpolation_steps=10,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Move and release the target object",
                next_subtask_description="Task complete",
            ),
            # 마지막 구간은 종료 신호가 없다. 필요 없으면 이 항목은 지운다.
            SubTaskConfig(
                object_ref=None,
                subtask_term_signal=None,
                subtask_term_offset_range=(0, 0),
                selection_strategy="random",
                selection_strategy_kwargs={},
                action_noise=0.0,
                num_interpolation_steps=5,
                num_fixed_steps=0,
                apply_noise_during_interpolation=False,
                description="Finish the task",
                next_subtask_description="Task complete",
            ),
        ]

    def init_action_cfg(self, mode: str) -> None:
        if mode == "mimic_ik":
            configure_ffw_sg2_mimic_ik_actions(self.actions)
            return
        super().init_action_cfg(mode)


@configclass
class Task000000MimicGenerateEnvCfg(Task000000MimicSeedEnvCfg):
    """검토된 randomization을 적용해 새 trajectory를 만드는 환경."""

    env_name: str = "Cyclo-Real-Showroom-Task000000-Mimic-Generate-FFW-SG2-v0"
    randomization: ShowroomGenerationRandomizationCfg = (
        TASK000000_MIMIC_GENERATION
    )
