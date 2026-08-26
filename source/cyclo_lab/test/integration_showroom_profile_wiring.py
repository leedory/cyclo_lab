"""Isaac integration checks for SG2 showroom profile-to-event wiring."""

from __future__ import annotations

from dataclasses import replace

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True, enable_cameras=True).app

try:
    import gymnasium as gym
    import torch

    import cyclo_lab  # noqa: F401

    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.platform.env_cfg import (
        ContinuousRandomShowroomEnvCfg,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.randomization.cfg import (
        ObjectPoseRandomizationCfg,
        RobotRootRandomizationCfg,
        ShowroomRandomizationCfg,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000458.env_cfg import (
        Task000458RandomEnvCfg,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000458.mimic_cfg import (
        Task000458MimicGenerateEnvCfg,
        Task000458MimicSeedEnvCfg,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000458.profiles import (
        TASK000458_MIMIC_GENERATION,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000458.spec import (
        TASK_000458_SPEC,
    )

    sentinel = ShowroomRandomizationCfg(
        robot_root=RobotRootRandomizationCfg(
            enabled=True,
            depth_x_max_m=0.011,
            lateral_y_max_m=0.022,
            yaw_max_rad=0.033,
        ),
        objects=ObjectPoseRandomizationCfg(
            enabled=True,
            object_names=("peanut_mix_bag_01", "jelly_bag_03"),
            x_max_m=0.044,
            y_max_m=0.055,
            yaw_max_rad=0.066,
        ),
    )

    consumers = (
        ContinuousRandomShowroomEnvCfg(randomization=sentinel),
        Task000458RandomEnvCfg(randomization=sentinel),
    )
    for cfg in consumers:
        robot = cfg.events.randomize_robot_root_pose
        objects = cfg.events.randomize_selected_objects
        assert robot.params["x_max"] == 0.011
        assert robot.params["y_max"] == 0.022
        assert robot.params["yaw_max"] == 0.033
        assert robot.params["asset_cfg"].name == "robot"
        assert objects.params["object_names"] == (
            "peanut_mix_bag_01",
            "jelly_bag_03",
        )
        assert objects.params["x_max"] == 0.044
        assert objects.params["y_max"] == 0.055
        assert objects.params["yaw_max"] == 0.066

    disabled = ShowroomRandomizationCfg()
    for cfg in consumers:
        cfg.apply_randomization_profile(disabled)
        assert cfg.events.randomize_robot_root_pose is None
        assert cfg.events.randomize_selected_objects is None
        cfg.apply_randomization_profile(sentinel)

    assert consumers[0].events is not consumers[1].events
    assert (
        consumers[0].events.randomize_robot_root_pose.params
        is not consumers[1].events.randomize_robot_root_pose.params
    )
    consumers[0].events.randomize_robot_root_pose.params["x_max"] = 9.0
    assert consumers[1].events.randomize_robot_root_pose.params["x_max"] == 0.011

    seed = Task000458MimicSeedEnvCfg()
    for event_name in (
        "refresh_shelf_support",
        "randomize_target_pose",
        "randomize_robot_root",
        "randomize_non_target_presence",
        "randomize_lighting",
        "randomize_shelf_appearance",
        "randomize_wall_color",
        "randomize_cameras",
    ):
        assert getattr(seed.events, event_name) is None

    generation_profile = replace(
        TASK000458_MIMIC_GENERATION,
        robot_root=RobotRootRandomizationCfg(
            enabled=True,
            depth_x_max_m=0.071,
            lateral_y_max_m=0.072,
            yaw_max_rad=0.073,
        ),
    )
    generation = Task000458MimicGenerateEnvCfg(
        randomization=generation_profile
    )
    assert generation.target_object == TASK_000458_SPEC.target_object
    assert (
        generation.events.randomize_target_pose.params["asset_cfg"].name
        == TASK_000458_SPEC.target_object
    )
    assert generation.events.randomize_robot_root.params["depth_x_max_m"] == 0.071
    assert generation.events.randomize_robot_root.params["lateral_y_max_m"] == 0.072
    assert generation.events.randomize_robot_root.params["yaw_max_rad"] == 0.073
    assert (
        TASK_000458_SPEC.target_object
        not in generation.events.randomize_non_target_presence.params["object_names"]
    )
    assert (
        generation.events.randomize_cameras.params["camera_names"]
        == TASK_000458_SPEC.policy_cameras
    )

    runtime_cfg = Task000458RandomEnvCfg()
    runtime_cfg.init_action_cfg("record")
    env = gym.make(
        "Cyclo-Real-Showroom-Task000458-Random-FFW-SG2-v0",
        cfg=runtime_cfg,
    ).unwrapped
    env.reset()
    robot_xy_1 = env.scene["robot"].data.root_pos_w[:, :2].clone()
    target_xy_1 = env.scene[TASK_000458_SPEC.target_object].data.root_pos_w[:, :2].clone()
    env.reset()
    robot_xy_2 = env.scene["robot"].data.root_pos_w[:, :2].clone()
    target_xy_2 = env.scene[TASK_000458_SPEC.target_object].data.root_pos_w[:, :2].clone()
    assert not torch.allclose(robot_xy_1, robot_xy_2)
    assert not torch.allclose(target_xy_1, target_xy_2)
    env.close()
    print("SHOWROOM_PROFILE_WIRING_AND_RESET_OK", flush=True)
finally:
    app.close()
