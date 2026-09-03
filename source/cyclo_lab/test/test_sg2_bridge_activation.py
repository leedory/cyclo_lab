"""Regression tests for smooth SG2 live-command activation."""

from __future__ import annotations

import threading
from unittest.mock import Mock, patch
from types import SimpleNamespace

import pytest

from cyclo_lab.runtime.bridges.sg2.topic_bridge import FFWSG2TopicBridge


def make_bridge_for_target_test() -> FFWSG2TopicBridge:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge._lock = threading.Lock()
    bridge.joint_names = ["joint_a", "joint_b"]
    bridge._target_joint_state = {"joint_a": 0.0, "joint_b": 2.0}
    bridge._trajectory_commands = {
        "left_arm": {"joint_a": 1.0},
        "right_arm": None,
        "head": None,
        "lift": None,
    }
    bridge.active_trajectory_groups = ("left_arm", "right_arm", "head", "lift")
    bridge._activation_blend_anchor = {"joint_a": 0.0, "joint_b": 2.0}
    bridge._activation_blend_start_time = 10.0
    bridge._activation_blend_duration = 0.5
    return bridge


def test_activation_blends_cached_command_from_current_pose() -> None:
    bridge = make_bridge_for_target_test()
    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=10.25,
    ):
        targets = bridge._joint_targets()

    assert targets == pytest.approx({"joint_a": 0.5, "joint_b": 2.0})
    assert bridge._activation_blend_anchor is not None


def test_activation_releases_anchor_after_transition() -> None:
    bridge = make_bridge_for_target_test()
    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=10.5,
    ):
        targets = bridge._joint_targets()

    assert targets == pytest.approx({"joint_a": 1.0, "joint_b": 2.0})
    assert bridge._activation_blend_anchor is None


def test_inactive_left_group_cannot_move_right_only_recording() -> None:
    bridge = make_bridge_for_target_test()
    bridge.active_trajectory_groups = ("right_arm", "head", "lift")

    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=10.5,
    ):
        targets = bridge._joint_targets()

    assert targets == pytest.approx({"joint_a": 0.0, "joint_b": 2.0})
    assert bridge._activation_blend_anchor is None


def test_activation_holds_inactive_left_group_at_absolute_reset_target() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge._lock = threading.Lock()
    bridge.joint_names = ["arm_l_joint1", "arm_r_joint1"]
    bridge.active_trajectory_groups = ("right_arm", "head", "lift")
    bridge._trajectory_commands = {
        "left_arm": {"arm_l_joint1": 0.9},
        "right_arm": None,
        "head": None,
        "lift": None,
    }
    bridge._read_current_joint_state = lambda: {
        "arm_l_joint1": 0.1,
        "arm_r_joint1": -0.2,
    }
    bridge._read_default_joint_state = lambda: {
        "arm_l_joint1": 0.0,
        "arm_r_joint1": 0.0,
    }

    bridge.begin_control_activation(transition_seconds=0.0)
    targets = bridge._joint_targets()

    assert targets == pytest.approx({"arm_l_joint1": 0.0, "arm_r_joint1": -0.2})
    assert bridge._activation_blend_anchor is None


def test_arm_tact_counters_ignore_non_toggle_enable_messages() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge._lock = threading.Lock()
    bridge._arm_tact_generation = {"left": 0, "right": 0}

    bridge._on_arm_enable("left", SimpleNamespace(data=1))
    bridge._on_arm_enable("left", SimpleNamespace(data=2))
    bridge._on_arm_enable("right", SimpleNamespace(data=0))
    bridge._on_arm_enable("right", SimpleNamespace(data=2))

    assert bridge.arm_tact_generation("left") == 1
    assert bridge.arm_tact_generation("right") == 1


def test_publish_observations_returns_camera_batch_result() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge.state_publisher = Mock()
    bridge.camera_publishers = Mock()
    bridge.camera_publishers.publish.return_value = True

    assert bridge.publish_observations() is True
    bridge.state_publisher.publish_all.assert_called_once_with()
    bridge.camera_publishers.publish.assert_called_once_with()
