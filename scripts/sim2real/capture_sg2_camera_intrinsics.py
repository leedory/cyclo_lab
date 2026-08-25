#!/usr/bin/env python3
"""Record canonical and raw SG2 CameraInfo contracts from a running robot."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import socket
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo


DEFAULT_TOPICS = {
    "head": "/head_camera/color/camera_info",
    "wrist_left": "/camera_left/camera_left/color/camera_info",
    "wrist_right": "/camera_right/camera_right/color/camera_info",
    "head_raw": "/zed/zed_node/left/camera_info",
}


def stamp_dict(stamp) -> dict[str, int]:
    return {"sec": int(stamp.sec), "nanosec": int(stamp.nanosec)}


def roi_dict(roi) -> dict:
    return {
        "x_offset": int(roi.x_offset),
        "y_offset": int(roi.y_offset),
        "height": int(roi.height),
        "width": int(roi.width),
        "do_rectify": bool(roi.do_rectify),
    }


def camera_info_dict(message: CameraInfo) -> dict:
    fx, fy = float(message.k[0]), float(message.k[4])
    width, height = int(message.width), int(message.height)
    return {
        "stamp": stamp_dict(message.header.stamp),
        "frame_id": message.header.frame_id,
        "width": width,
        "height": height,
        "distortion_model": message.distortion_model,
        "d": list(map(float, message.d)),
        "k": list(map(float, message.k)),
        "r": list(map(float, message.r)),
        "p": list(map(float, message.p)),
        "binning_x": int(message.binning_x),
        "binning_y": int(message.binning_y),
        "roi": roi_dict(message.roi),
        "derived": {
            "fx_px": fx,
            "fy_px": fy,
            "mean_focal_px": 0.5 * (fx + fy),
            "horizontal_fov_deg_from_fx": math.degrees(2.0 * math.atan(width / (2.0 * fx))),
            "vertical_fov_deg_from_fy": math.degrees(2.0 * math.atan(height / (2.0 * fy))),
        },
    }


def calibration_signature(message: CameraInfo) -> tuple:
    return (
        int(message.width),
        int(message.height),
        message.distortion_model,
        tuple(message.d),
        tuple(message.k),
        tuple(message.r),
        tuple(message.p),
        int(message.binning_x),
        int(message.binning_y),
        tuple(roi_dict(message.roi).values()),
    )


class IntrinsicsRecorder(Node):
    def __init__(self, topics: dict[str, str], required_samples: int) -> None:
        super().__init__("sg2_camera_intrinsics_recorder")
        self.required_samples = required_samples
        self.samples: dict[str, list[CameraInfo]] = {role: [] for role in topics}
        self._last_stamps: dict[str, tuple[int, int] | None] = {role: None for role in topics}
        self._camera_subscriptions = []
        for role, topic in topics.items():
            self._camera_subscriptions.append(
                self.create_subscription(
                    CameraInfo,
                    topic,
                    lambda message, role=role: self._callback(role, message),
                    qos_profile_sensor_data,
                )
            )

    def _callback(self, role: str, message: CameraInfo) -> None:
        stamp = (int(message.header.stamp.sec), int(message.header.stamp.nanosec))
        if stamp == self._last_stamps[role] or len(self.samples[role]) >= self.required_samples:
            return
        self._last_stamps[role] = stamp
        self.samples[role].append(message)

    @property
    def complete(self) -> bool:
        return all(len(samples) >= self.required_samples for samples in self.samples.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=20.0)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--head-topic", default=DEFAULT_TOPICS["head"])
    parser.add_argument("--wrist-left-topic", default=DEFAULT_TOPICS["wrist_left"])
    parser.add_argument("--wrist-right-topic", default=DEFAULT_TOPICS["wrist_right"])
    parser.add_argument("--head-raw-topic", default=DEFAULT_TOPICS["head_raw"])
    parser.add_argument("--skip-head-raw", action="store_true")
    args = parser.parse_args()
    if args.samples < 1:
        parser.error("--samples must be at least 1")

    topics = {
        "head": args.head_topic,
        "wrist_left": args.wrist_left_topic,
        "wrist_right": args.wrist_right_topic,
    }
    if not args.skip_head_raw:
        topics["head_raw"] = args.head_raw_topic

    rclpy.init()
    node = IntrinsicsRecorder(topics, args.samples)
    deadline = time.monotonic() + args.timeout
    try:
        while not node.complete and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.25)

        cameras = {}
        for role, samples in node.samples.items():
            signatures = {calibration_signature(message) for message in samples}
            cameras[role] = {
                "topic": topics[role],
                "sample_count": len(samples),
                "stable_across_samples": len(signatures) == 1 and len(samples) == args.samples,
                "sample_stamps": [stamp_dict(message.header.stamp) for message in samples],
                "camera_info": camera_info_dict(samples[0]) if samples else None,
            }

        complete = node.complete and all(item["stable_across_samples"] for item in cameras.values())
        record = {
            "schema": "sg2-camera-intrinsics-v1",
            "complete": complete,
            "captured_at_utc": datetime.now(timezone.utc).isoformat(),
            "hostname": socket.gethostname(),
            "ros": {
                "distro": os.environ.get("ROS_DISTRO"),
                "domain_id": os.environ.get("ROS_DOMAIN_ID"),
                "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION"),
            },
            "required_samples": args.samples,
            "cameras": cameras,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"complete": complete, "output": str(args.output)}))
        return 0 if complete else 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
