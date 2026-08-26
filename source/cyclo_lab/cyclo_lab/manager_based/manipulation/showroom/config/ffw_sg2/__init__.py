"""Continuous SG2 showroom task registration."""

import gymnasium as gym


gym.register(
    id="Cyclo-Real-Showroom-FFW-SG2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.platform.env_cfg:ContinuousShowroomEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Showroom-Random-FFW-SG2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.platform.env_cfg:ContinuousRandomShowroomEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Showroom-Task000458-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000458.env_cfg:Task000458EnvCfg"
        ),
    },
    disable_env_checker=True,
)


gym.register(
    id="Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000458.env_cfg:Task000458RandomEnvCfg"
        ),
    },
    disable_env_checker=True,
)
_TASK458_MIMIC_ENTRY_POINT = f"{__name__}.tasks.task_000458.mimic_env:Task000458MimicEnv"

gym.register(
    id="Cyclo-Real-Showroom-Task000458-Mimic-Seed-FFW-SG2-v0",
    entry_point=_TASK458_MIMIC_ENTRY_POINT,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000458.mimic_cfg:Task000458MimicSeedEnvCfg"
        )
    },
    disable_env_checker=True,
)

gym.register(
    id="Cyclo-Real-Showroom-Task000458-Mimic-Generate-FFW-SG2-v0",
    entry_point=_TASK458_MIMIC_ENTRY_POINT,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.tasks.task_000458.mimic_cfg:Task000458MimicGenerateEnvCfg"
        )
    },
    disable_env_checker=True,
)
