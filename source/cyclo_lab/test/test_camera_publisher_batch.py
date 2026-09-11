"""Pure unit tests for camera-batch publication accounting."""

from __future__ import annotations

from unittest.mock import Mock, patch

from cyclo_lab.runtime.publishers.camera_publishers import CompressedCameraPublishers


def make_publishers(*, writers: dict, publish_hz: float | None = 15.0):
    publishers = CompressedCameraPublishers.__new__(CompressedCameraPublishers)
    publishers._scene = {camera_name: object() for camera_name in writers}
    publishers._publish_hz = publish_hz
    publishers._image_rotations = {}
    publishers._last_publish_time = 0.0
    publishers._warned_camera_publish_errors = set()
    publishers.writers = writers
    return publishers


def test_empty_camera_batch_is_not_reported_as_published() -> None:
    publishers = make_publishers(writers={})

    assert publishers.publish() is False


def test_rate_limited_camera_batch_is_not_reported_as_published() -> None:
    publishers = make_publishers(writers={"cam_head": Mock()})
    publishers._last_publish_time = 10.0

    with (
        patch(
            "cyclo_lab.runtime.publishers.camera_publishers.time.monotonic",
            return_value=10.01,
        ),
        patch(
            "cyclo_lab.runtime.publishers.camera_publishers.publish_compressed_camera"
        ) as publish_camera,
    ):
        assert publishers.publish() is False

    publish_camera.assert_not_called()


def test_due_successful_camera_batch_is_reported_as_published() -> None:
    publishers = make_publishers(
        writers={"cam_head": Mock(), "cam_overhead_center": Mock()}
    )

    with (
        patch(
            "cyclo_lab.runtime.publishers.camera_publishers.time.monotonic",
            return_value=10.0,
        ),
        patch(
            "cyclo_lab.runtime.publishers.camera_publishers.publish_compressed_camera"
        ) as publish_camera,
    ):
        assert publishers.publish() is True

    assert publish_camera.call_count == 2
    assert publishers._last_publish_time == 10.0


def test_partial_camera_failure_does_not_report_a_successful_batch() -> None:
    publishers = make_publishers(
        writers={"cam_head": Mock(), "cam_overhead_center": Mock()},
        publish_hz=None,
    )

    with patch(
        "cyclo_lab.runtime.publishers.camera_publishers.publish_compressed_camera",
        side_effect=(None, RuntimeError("encode failed")),
    ) as publish_camera:
        assert publishers.publish() is False

    assert publish_camera.call_count == 2
    assert publishers._warned_camera_publish_errors == {"cam_overhead_center"}
