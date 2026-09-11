"""Trajectory-generation environment presets for coffee transport."""

from __future__ import annotations

from isaaclab.utils import configclass

from cyclo_lab.manager_based.manipulation.common.ffw_sg2_mimic_action_cfg import (
    configure_ffw_sg2_mimic_ik_actions,
)

from .task_000001.env_cfg import Task000001EnvCfg
from .task_000001.profiles import (
    TASK000001_PRODUCTION_TRAJECTORY_GENERATION,
    Task000001RandomizationCfg,
)
from .task_000002.env_cfg import Task000002EnvCfg
from .task_000002.profiles import (
    TASK000002_PRODUCTION_TRAJECTORY_GENERATION,
    Task000002RandomizationCfg,
)


def _configure_generation(cfg, env_name: str) -> None:
    cfg.env_name = env_name
    cfg.terminations.success = None
    configure_ffw_sg2_mimic_ik_actions(
        cfg.actions,
        posture_bias_sides=("l", "r"),
    )


@configclass
class Task000001TrajectoryGenerationEnvCfg(Task000001EnvCfg):
    """Random physical starts with dual-EEF19 control."""

    env_name: str = (
        "Cyclo-Real-Showroom-Task000001-Trajectory-Generation-FFW-SG2-v0"
    )
    randomization: Task000001RandomizationCfg = (
        TASK000001_PRODUCTION_TRAJECTORY_GENERATION
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_generation(
            self,
            "Cyclo-Real-Showroom-Task000001-Trajectory-Generation-FFW-SG2-v0",
        )


@configclass
class Task000002TrajectoryGenerationEnvCfg(Task000002EnvCfg):
    """Random HOME/route starts with dual-EEF19 plus body-velocity3 control."""

    env_name: str = (
        "Cyclo-Real-Showroom-Task000002-Trajectory-Generation-FFW-SG2-v0"
    )
    randomization: Task000002RandomizationCfg = (
        TASK000002_PRODUCTION_TRAJECTORY_GENERATION
    )

    def __post_init__(self):
        super().__post_init__()
        _configure_generation(
            self,
            "Cyclo-Real-Showroom-Task000002-Trajectory-Generation-FFW-SG2-v0",
        )
