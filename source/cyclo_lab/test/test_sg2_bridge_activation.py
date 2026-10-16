"""Regression tests for smooth SG2 live-command activation."""

from __future__ import annotations

import math
import threading
from unittest.mock import Mock, call, patch
from types import SimpleNamespace

import pytest

from cyclo_lab.runtime.bridges.sg2.topic_bridge import (
    DEFAULT_ACTIVATION_BLEND_SECONDS,
    DEFAULT_CONTINUOUS_TRACKING_TIME_CONSTANT_SECONDS,
    FFWSG2TopicBridge,
)


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


def test_activation_blocks_stale_head_and_lift_until_the_command_changes() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge._lock = threading.Lock()
    bridge.joint_names = ["head_joint1", "head_joint2", "lift_joint"]
    bridge.active_trajectory_groups = ("left_arm", "right_arm", "head", "lift")
    bridge._trajectory_commands = {
        "left_arm": None,
        "right_arm": None,
        "head": {"head_joint1": -0.2, "head_joint2": 0.0},
        "lift": {"lift_joint": -0.1},
    }
    bridge._trajectory_command_generation = {label: 0 for label in bridge._trajectory_commands}
    bridge._activation_blocked_commands = {}
    bridge._read_current_joint_state = lambda: {
        "head_joint1": 0.1,
        "head_joint2": 0.0,
        "lift_joint": 0.0,
    }

    bridge.begin_control_activation(transition_seconds=0.0)

    assert bridge._trajectory_commands["head"] is None
    assert bridge._trajectory_commands["lift"] is None
    stale_lift = SimpleNamespace(
        joint_names=["lift_joint"],
        points=[SimpleNamespace(positions=[-0.1])],
    )
    bridge._on_joint_trajectory("lift", stale_lift)
    assert bridge._trajectory_commands["lift"] is None

    moved_lift = SimpleNamespace(
        joint_names=["lift_joint"],
        points=[SimpleNamespace(positions=[-0.11])],
    )
    bridge._on_joint_trajectory("lift", moved_lift)
    assert bridge._trajectory_commands["lift"] == {"lift_joint": -0.11}


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
    bridge._activation_blocked_commands = {}
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
    bridge._leader_activation_requested = threading.Event()

    bridge._on_arm_enable("left", SimpleNamespace(data=1))
    bridge._on_arm_enable("left", SimpleNamespace(data=2))
    bridge._on_arm_enable("right", SimpleNamespace(data=0))
    bridge._on_arm_enable("right", SimpleNamespace(data=2))

    assert bridge.arm_tact_generation("left") == 1
    assert bridge.arm_tact_generation("right") == 1


def test_arm_mode_can_be_set_explicitly_for_automation_ownership() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge.arm_enable_publishers = {"left": Mock(), "right": Mock()}

    bridge.set_all_arms_idle()
    bridge.set_arm_enabled("left", True)

    assert bridge.arm_enable_publishers["left"].publish.call_args_list == [
        call(data=0),
        call(data=1),
    ]
    bridge.arm_enable_publishers["right"].publish.assert_called_once_with(data=0)
    with pytest.raises(KeyError, match="Unknown FFW-SG2 arm side"):
        bridge.set_arm_enabled("center", True)


def test_publish_observations_returns_camera_batch_result() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge.state_publisher = Mock()
    bridge.camera_publishers = Mock()
    bridge.camera_publishers.publish.return_value = True

    assert bridge.publish_observations() is True
    bridge.state_publisher.publish_all.assert_called_once_with()
    bridge.camera_publishers.publish.assert_called_once_with()


def make_smoothing_bridge() -> FFWSG2TopicBridge:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge._lock = threading.Lock()
    bridge._smoothed_joint_targets = {"joint_a": 0.0, "joint_b": 2.0}
    bridge._tracking_filter_last_time = 10.0
    return bridge


def test_slow_start_and_continuous_tracking_defaults() -> None:
    assert DEFAULT_ACTIVATION_BLEND_SECONDS == 20.0
    assert DEFAULT_CONTINUOUS_TRACKING_TIME_CONSTANT_SECONDS == 0.4


def test_reset_pose_hold_uses_the_full_slow_start_blend() -> None:
    bridge = FFWSG2TopicBridge.__new__(FFWSG2TopicBridge)
    bridge.hold_reset_joint_targets = True
    bridge._lock = threading.Lock()
    bridge._target_joint_state = {"arm_l_joint1": 0.43, "arm_r_joint1": -0.43}
    bridge._trajectory_commands = {
        "left_arm": {"arm_l_joint1": 0.9},
        "right_arm": {"arm_r_joint1": -0.9},
        "head": None,
        "lift": None,
    }
    bridge._trajectory_command_generation = {label: 0 for label in bridge._trajectory_commands}
    bridge._activation_blocked_commands = {}
    bridge.active_trajectory_groups = ("left_arm", "right_arm", "head", "lift")

    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=30.0,
    ):
        bridge.begin_control_activation()

    assert bridge._trajectory_commands["left_arm"] is None
    assert bridge._trajectory_commands["right_arm"] is None
    assert bridge._activation_blend_anchor == {
        "arm_l_joint1": 0.43,
        "arm_r_joint1": -0.43,
    }
    assert bridge._activation_blend_duration == 20.0


def test_continuous_smoothing_uses_point_four_second_time_constant() -> None:
    bridge = make_smoothing_bridge()
    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=10.4,
    ):
        targets = bridge._smooth_joint_targets({"joint_a": 1.0, "joint_b": 2.0})

    expected_alpha = 1.0 - math.exp(-1.0)
    assert targets == pytest.approx({"joint_a": expected_alpha, "joint_b": 2.0})


def test_continuous_smoothing_tracks_latest_target_without_queuing_old_targets() -> None:
    bridge = make_smoothing_bridge()
    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        side_effect=(10.2, 10.4),
    ):
        first = bridge._smooth_joint_targets({"joint_a": 1.0, "joint_b": 2.0})
        second = bridge._smooth_joint_targets({"joint_a": -1.0, "joint_b": 2.0})

    alpha = 1.0 - math.exp(-0.5)
    assert first["joint_a"] == pytest.approx(alpha)
    assert second["joint_a"] == pytest.approx(alpha + alpha * (-1.0 - alpha))
    assert second["joint_a"] < first["joint_a"]


def test_first_target_after_filter_reset_is_used_as_the_safe_anchor() -> None:
    bridge = make_smoothing_bridge()
    bridge._smoothed_joint_targets = None
    bridge._tracking_filter_last_time = 0.0

    with patch(
        "cyclo_lab.runtime.bridges.sg2.topic_bridge.time.monotonic",
        return_value=20.0,
    ):
        targets = bridge._smooth_joint_targets({"joint_a": 0.25, "joint_b": 1.5})

    assert targets == pytest.approx({"joint_a": 0.25, "joint_b": 1.5})
    assert bridge._tracking_filter_last_time == 20.0
