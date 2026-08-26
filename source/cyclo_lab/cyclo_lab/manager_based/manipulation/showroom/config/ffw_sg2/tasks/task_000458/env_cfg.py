"""Episodic recording presets for showroom Task000458."""

from dataclasses import MISSING, dataclass

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from ...randomization.cfg import ShowroomRandomizationCfg
from ..common import EpisodicShowroomTaskEnvCfg, ShowroomTaskSpec
from .profiles import (
    TASK000458_RECORD_DETERMINISTIC,
    TASK000458_RECORD_RANDOM,
)
from .spec import TASK_000458_SPEC


@dataclass(frozen=True)
class TakeoutSuccessCfg:
    gripper_close_threshold: float = 0.8
    tcp_envelope_min_m: tuple[float, float, float] = (-0.040, -0.020, -0.050)
    tcp_envelope_max_m: tuple[float, float, float] = (0.020, 0.040, 0.020)
    relative_motion_tolerance_m: float = 0.010
    stable_control_steps: int = 2
    shelf_front_clearance_m: float = 0.0
    neighbor_translation_tolerance_m: float = 0.005
    neighbor_rotation_tolerance_rad: float = 0.0872664626
    neighbor_baseline_settle_steps: int = 1


TASK_458_TAKEOUT_SUCCESS = TakeoutSuccessCfg()
TARGET_EEF = "right_eef"
TARGET_GRIPPER_JOINT = "gripper_r_joint1"


def takeout_term_params(target_object: str) -> dict:
    return {
        "object_cfg": SceneEntityCfg(target_object),
        "robot_cfg": SceneEntityCfg("robot"),
        "eef_cfg": SceneEntityCfg(TARGET_EEF),
        "gripper_joint_name": TARGET_GRIPPER_JOINT,
        "gripper_close_threshold": TASK_458_TAKEOUT_SUCCESS.gripper_close_threshold,
        "tcp_envelope_min_m": TASK_458_TAKEOUT_SUCCESS.tcp_envelope_min_m,
        "tcp_envelope_max_m": TASK_458_TAKEOUT_SUCCESS.tcp_envelope_max_m,
        "relative_motion_tolerance_m": TASK_458_TAKEOUT_SUCCESS.relative_motion_tolerance_m,
        "stable_control_steps": TASK_458_TAKEOUT_SUCCESS.stable_control_steps,
        "shelf_front_clearance_m": TASK_458_TAKEOUT_SUCCESS.shelf_front_clearance_m,
        "neighbor_translation_tolerance_m": TASK_458_TAKEOUT_SUCCESS.neighbor_translation_tolerance_m,
        "neighbor_rotation_tolerance_rad": TASK_458_TAKEOUT_SUCCESS.neighbor_rotation_tolerance_rad,
        "neighbor_baseline_settle_steps": TASK_458_TAKEOUT_SUCCESS.neighbor_baseline_settle_steps,
        "shelf_prim_suffix": "/RobotisShowroom/robotis_showroom/kolbjorn_cabinet_1",
    }


@configclass
class Task000458EnvCfg(EpisodicShowroomTaskEnvCfg):
    """Deterministic 19D preset for operator-approved raw HDF5 seeds."""

    task_spec: ShowroomTaskSpec = TASK_000458_SPEC
    randomization: ShowroomRandomizationCfg = TASK000458_RECORD_DETERMINISTIC
    seed_success_metric_params: dict = MISSING
    shelf_prim_suffix: str = "/RobotisShowroom/robotis_showroom/kolbjorn_cabinet_1"

    def __post_init__(self):
        super().__post_init__()
        self.seed_success_metric_params = takeout_term_params(self.target_object)


@configclass
class Task000458RandomEnvCfg(Task000458EnvCfg):
    """19D raw-seed preset with the explicit record randomization profile."""

    env_name: str = "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0"
    randomization: ShowroomRandomizationCfg = TASK000458_RECORD_RANDOM

    def __post_init__(self):
        requested_env_name = self.env_name
        super().__post_init__()
        self.env_name = requested_env_name
