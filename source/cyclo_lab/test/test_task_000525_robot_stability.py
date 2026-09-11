"""Unit tests for Task000525-only SG2 arm hold tuning."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cyclo_lab.manager_based.manipulation.showroom.config.ffw_sg2.tasks.task_000525.robot_stability import (
    TASK000525_ARM_DAMPING_SCALE,
    TASK000525_ARM_HOLD_ACTUATOR_NAMES,
    TASK000525_ARM_STIFFNESS_SCALE,
    apply_task000525_arm_hold_tuning,
)


def make_robot_cfg():
    return SimpleNamespace(
        actuators={
            "DY_80": SimpleNamespace(stiffness=600.0, damping=30.0),
            "DY_70": SimpleNamespace(stiffness=600.0, damping=20.0),
            "DP-42": SimpleNamespace(stiffness=200.0, damping=3.0),
            "lift": SimpleNamespace(stiffness=250_000.0, damping=5_000.0),
        }
    )


def test_arm_hold_tuning_scales_only_task_arm_actuators() -> None:
    robot_cfg = make_robot_cfg()
    original = {
        name: (actuator.stiffness, actuator.damping)
        for name, actuator in robot_cfg.actuators.items()
    }

    apply_task000525_arm_hold_tuning(robot_cfg)

    for name in TASK000525_ARM_HOLD_ACTUATOR_NAMES:
        stiffness, damping = original[name]
        assert robot_cfg.actuators[name].stiffness == pytest.approx(
            stiffness * TASK000525_ARM_STIFFNESS_SCALE
        )
        assert robot_cfg.actuators[name].damping == pytest.approx(
            damping * TASK000525_ARM_DAMPING_SCALE
        )
    assert (
        robot_cfg.actuators["lift"].stiffness,
        robot_cfg.actuators["lift"].damping,
    ) == original["lift"]


def test_arm_hold_tuning_fails_if_expected_actuator_is_missing() -> None:
    robot_cfg = make_robot_cfg()
    del robot_cfg.actuators["DP-42"]

    with pytest.raises(KeyError, match="DP-42"):
        apply_task000525_arm_hold_tuning(robot_cfg)


def test_arm_hold_tuning_rejects_non_numeric_gain() -> None:
    robot_cfg = make_robot_cfg()
    robot_cfg.actuators["DY_80"].stiffness = {"arm_l_joint1": 600.0}

    with pytest.raises(TypeError, match="DY_80 stiffness"):
        apply_task000525_arm_hold_tuning(robot_cfg)
