"""Continuous F2 showroom task registration."""

import gymnasium as gym


gym.register(
    id="Cyclo-Real-Showroom-FFW-F2-v0",
    entry_point="cyclo_lab.manager_based.continuous_env:ContinuousManagerBasedEnv",
    kwargs={"env_cfg_entry_point": f"{__name__}.platform.env_cfg:ContinuousShowroomEnvCfg"},
    disable_env_checker=True,
)
