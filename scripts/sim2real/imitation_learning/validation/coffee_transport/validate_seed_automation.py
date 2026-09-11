"""Validate the two-pickup coffee seed automation state machine in Isaac."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from isaaclab.app import AppLauncher


app = AppLauncher(headless=True).app

try:
    import torch

    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.coffee_transport_common import (
        CABINET_RIGHT_DELTA_Y_M,
        COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M,
        SHELF_LEVEL_DELTA_Z_M,
        SOURCE_SPAWN_SQUARES,
        SOURCE_UPPER_SUPPORT_Z_M,
    )
    from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.coffee_transport.seed_automation import (
        CoffeeSeedAutomationCfg,
        CoffeeTransportSeedAutomation,
        SeedAutomationState,
    )
    from cyclo_lab.robot_specs.ffw.sg2 import FFW_SG2_ACTION_JOINT_NAMES

    class Scene(dict):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.env_origins = torch.zeros((1, 3))

    def rigid_object(position):
        pose = torch.tensor([[*position, 1.0, 0.0, 0.0, 0.0]])
        return SimpleNamespace(
            data=SimpleNamespace(
                root_pose_w=pose,
                root_pos_w=pose[:, :3],
                root_quat_w=pose[:, 3:7],
                root_lin_vel_w=torch.zeros((1, 3)),
                root_ang_vel_w=torch.zeros((1, 3)),
            )
        )

    def make_env():
        robot_pose = torch.tensor([[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]])
        robot = SimpleNamespace(
            joint_names=list(FFW_SG2_ACTION_JOINT_NAMES),
            data=SimpleNamespace(
                joint_pos=torch.zeros((1, len(FFW_SG2_ACTION_JOINT_NAMES))),
                default_joint_pos=torch.zeros((1, len(FFW_SG2_ACTION_JOINT_NAMES))),
                root_pose_w=robot_pose,
                root_pos_w=robot_pose[:, :3],
            ),
        )
        scene = Scene(robot=robot)
        for square in SOURCE_SPAWN_SQUARES:
            scene[square.name] = rigid_object(square.default_position_m)
        for side in ("left", "right"):
            scene[f"{side}_eef"] = SimpleNamespace(
                data=SimpleNamespace(target_pos_w=torch.zeros((1, 1, 3)))
            )
        env = SimpleNamespace(device="cpu", scene=scene)
        env._coffee_transport_episode_home_root_pose_w = robot_pose.clone()
        return env

    def follow_perfect_plant(env, action, task_id, control_hz):
        env.scene["robot"].data.joint_pos[:, :] = action[:, :19]
        if task_id == "000002":
            env.scene["robot"].data.root_pose_w[0, 0] += float(action[0, 19]) / control_hz
            env.scene["robot"].data.root_pose_w[0, 1] += float(action[0, 20]) / control_hz

    def arm_operator(env, controller, task_id, step):
        assert controller.awaiting_operator_tact
        assert controller.enable_operator_cycle(recorded_step_index=step)
        assert controller.operator_enabled
        return step

    def put_current_can_at_destination(env, controller, task_id):
        cycle = controller.current_cycle
        square = next(item for item in SOURCE_SPAWN_SQUARES if item.name == cycle.can_name)
        position = list(square.default_position_m)
        if task_id == "000001":
            position[2] += SHELF_LEVEL_DELTA_Z_M
        else:
            position[1] += CABINET_RIGHT_DELTA_Y_M
        pose = env.scene[cycle.can_name].data.root_pose_w
        pose[0, :3] = torch.tensor(position)
        env.scene[f"{cycle.arm}_eef"].data.target_pos_w[0, 0, :] = torch.tensor(
            [position[0] + 0.20, position[1], position[2]]
        )

    def record_one_operator_pickup(env, controller, task_id, step):
        cycle = controller.current_cycle
        can = env.scene[cycle.can_name]
        gripper_index = controller._gripper_index[cycle.arm]
        initial = can.data.root_pose_w[0, :3].detach().clone()
        for local_step in range(16):
            action = torch.zeros((1, 19 if task_id == "000001" else 22))
            if local_step >= 5:
                action[0, gripper_index] = 1.0
                can.data.root_pose_w[0, :3] = initial + torch.tensor([0.25, 0.0, 0.0])
                env.scene[f"{cycle.arm}_eef"].data.target_pos_w[0, 0, :] = (
                    can.data.root_pos_w[0] + torch.tensor([0.07, 0.0, 0.02])
                )
            output = controller.apply(action, recorded_step_index=step)
            follow_perfect_plant(env, output, task_id, controller.cfg.control_hz)
            step += 1
        controller.request_g()
        return step

    def run_two_cycles(task_id):
        env = make_env()
        cfg = CoffeeSeedAutomationCfg(control_hz=15.0)
        controller = CoffeeTransportSeedAutomation(env, task_id=task_id, cfg=cfg)
        controller.begin_episode()
        step = 0
        tact_handoffs = []
        final_cycle_output = None
        while not controller.task_complete:
            cycle = controller.current_cycle
            assert cycle is not None
            tact_handoffs.append(cycle.arm)
            step = arm_operator(env, controller, task_id, step)
            step = record_one_operator_pickup(env, controller, task_id, step)
            for _ in range(700):
                if controller.state is SeedAutomationState.VALIDATE_PLACEMENT:
                    put_current_can_at_destination(env, controller, task_id)
                action = torch.zeros((1, 19 if task_id == "000001" else 22))
                output = controller.apply(action, recorded_step_index=step)
                follow_perfect_plant(env, output, task_id, cfg.control_hz)
                final_cycle_output = output
                step += 1
                if controller.failed:
                    raise AssertionError(controller.failure_reason)
                if controller.awaiting_operator_tact or controller.task_complete:
                    break
            else:
                raise AssertionError(f"Task{task_id} automatic cycle did not finish")

        assert controller.cycle_index == 2
        assert tact_handoffs == ["right", "left"]
        assert not controller.final_placement_failures()
        metadata = controller.metadata()
        assert metadata["coffee_completed_cycles"] == 2
        assert metadata["coffee_phase_contract_version"] == (
            7 if task_id == "000001" else 6
        )
        assert metadata["coffee_task_axis_action_contract"] == (
            "lift_absolute_endpoint_only_0.0_or_-0.28_m_v1"
            if task_id == "000001"
            else "base_body_velocity_feedback_v1"
        )
        assert "cycle_0_operator_enabled_step" in metadata
        assert "cycle_1_operator_enabled_step" in metadata
        assert final_cycle_output is not None
        assert torch.max(
            torch.abs(
                final_cycle_output[:, controller._home_settle_indices]
                - controller._nominal_action[:, controller._home_settle_indices]
            )
        ) <= 1.0e-6
        if task_id == "000002":
            home = env._coffee_transport_episode_home_root_pose_w[0]
            root = env.scene["robot"].data.root_pose_w[0]
            assert torch.linalg.vector_norm(root[:2] - home[:2]) <= cfg.base_position_tolerance_m

    for checked_task_id in ("000001", "000002"):
        run_two_cycles(checked_task_id)

    handoff_env = make_env()
    handoff_controller = CoffeeTransportSeedAutomation(
        handoff_env, task_id="000002"
    )
    handoff_controller.begin_episode()
    handoff_step = arm_operator(
        handoff_env, handoff_controller, "000002", 0
    )
    baseline_action = handoff_controller._nominal_action.clone()
    baseline_action[
        :, handoff_controller._gripper_index["right"]
    ] = 0.5
    baseline_action = torch.cat((baseline_action, torch.ones((1, 3))), dim=1)
    baseline_output = handoff_controller.apply(
        baseline_action, recorded_step_index=handoff_step
    )
    assert handoff_controller._operator_gripper_input_baseline == 0.5
    assert baseline_output[0, handoff_controller._gripper_index["right"]] == (
        handoff_controller._nominal_action[
            0, handoff_controller._gripper_index["right"]
        ]
    )
    assert torch.equal(baseline_output[:, 19:22], torch.zeros((1, 3)))

    jump_action = baseline_action.clone()
    right_gripper = handoff_controller._gripper_index["right"]
    right_arm_joints = [
        index
        for index in handoff_controller._arm_indices["right"]
        if index != right_gripper
    ]
    jump_action[:, right_arm_joints] += 1.0
    jump_action[:, right_gripper] += 1.0
    jump_output = handoff_controller.apply(
        jump_action, recorded_step_index=handoff_step + 1
    )
    assert torch.max(
        torch.abs(
            jump_output[:, right_arm_joints]
            - baseline_output[:, right_arm_joints]
        )
    ) <= handoff_controller.cfg.operator_arm_max_delta_rad + 1.0e-6
    assert torch.abs(
        jump_output[0, right_gripper] - baseline_output[0, right_gripper]
    ) <= handoff_controller.cfg.operator_gripper_max_delta_rad + 1.0e-6

    failed_env = make_env()
    failed_controller = CoffeeTransportSeedAutomation(failed_env, task_id="000001")
    failed_controller.begin_episode()
    failed_step = arm_operator(
        failed_env, failed_controller, "000001", 0
    )
    failed_controller.request_g()
    failed_controller.apply(
        torch.zeros((1, 19)), recorded_step_index=failed_step
    )
    assert failed_controller.failed
    assert "enough pickup/extract frames" in failed_controller.failure_reason
    failed_hold = failed_controller.apply(torch.zeros((1, 19)), recorded_step_index=1)
    failed_env.scene["robot"].data.joint_pos[:, :] = -0.4
    repeated_hold = failed_controller.apply(torch.zeros((1, 19)), recorded_step_index=2)
    assert torch.equal(failed_hold, repeated_hold)

    ungated_env = make_env()
    ungated_controller = CoffeeTransportSeedAutomation(ungated_env, task_id="000001")
    ungated_controller.begin_episode()
    ungated_start_step = arm_operator(
        ungated_env, ungated_controller, "000001", 0
    )
    gripper_index = ungated_controller._gripper_index["right"]
    for step in range(16):
        action = torch.zeros((1, 19))
        if step >= 5:
            action[0, gripper_index] = 1.0
        output = ungated_controller.apply(
            action, recorded_step_index=ungated_start_step + step
        )
        follow_perfect_plant(ungated_env, output, "000001", 15.0)
    assert ungated_controller._stable_steps == 0
    ungated_controller.request_g()
    ungated_controller.apply(
        torch.zeros((1, 19)), recorded_step_index=ungated_start_step + 16
    )
    assert ungated_controller.state is SeedAutomationState.RETURNING_HOME
    return_env = make_env()
    return_controller = CoffeeTransportSeedAutomation(return_env, task_id="000001")
    return_controller.begin_episode()
    return_controller._held_action = return_controller._nominal_action.clone()
    return_controller._enter(SeedAutomationState.VALIDATE_PLACEMENT)
    for step in range(round(return_controller.cfg.control_hz) + 1):
        return_output = return_controller.apply(
            torch.zeros((1, 19)), recorded_step_index=step
        )
        follow_perfect_plant(return_env, return_output, "000001", 15.0)
        if return_controller.state is SeedAutomationState.AXIS_TO_SOURCE:
            break
    assert return_controller.state is SeedAutomationState.AXIS_TO_SOURCE
    assert not return_controller.failed

    drop_env = make_env()
    drop_controller = CoffeeTransportSeedAutomation(drop_env, task_id="000001")
    drop_controller.begin_episode()
    drop_step = arm_operator(drop_env, drop_controller, "000001", 0)
    drop_step = record_one_operator_pickup(
        drop_env, drop_controller, "000001", drop_step
    )
    for offset in range(60):
        drop_output = drop_controller.apply(
            torch.zeros((1, 19)), recorded_step_index=drop_step + offset
        )
        follow_perfect_plant(drop_env, drop_output, "000001", 15.0)
        if drop_controller.state is SeedAutomationState.AXIS_TO_DESTINATION:
            break
    assert drop_controller.state is SeedAutomationState.AXIS_TO_DESTINATION
    drop_env.scene["right_eef"].data.target_pos_w[0, 0, 0] += 1.0
    drop_controller.apply(
        torch.zeros((1, 19)), recorded_step_index=drop_step + offset + 1
    )
    assert drop_controller.failed
    assert "held can separated" in drop_controller.failure_reason


    Path("/tmp/coffee_seed_automation_integration.fail").unlink(missing_ok=True)
    expected_z = SOURCE_UPPER_SUPPORT_Z_M + COFFEE_CAN_ORIGIN_ABOVE_SUPPORT_M
    Path("/tmp/coffee_seed_automation_integration.ok").write_text(
        "Task000001=2 cycles; Task000002=2 cycles; early-G=fail-closed; body-gate=diagnostic; latched-hold; tact-handoff; nonblocking-axis-return; "
        "dropped-can=fail-closed\n",
        encoding="utf-8",
    )
    assert expected_z > 0.0
    print("COFFEE_SEED_AUTOMATION_2X2_AND_DROP_FAIL_CLOSED_OK", flush=True)
except BaseException as exc:
    Path("/tmp/coffee_seed_automation_integration.ok").unlink(missing_ok=True)
    Path("/tmp/coffee_seed_automation_integration.fail").write_text(
        f"{type(exc).__name__}: {exc}\n",
        encoding="utf-8",
    )
    print(f"COFFEE_SEED_AUTOMATION_FAILED: {type(exc).__name__}: {exc}", flush=True)
    raise
finally:
    app.close()
