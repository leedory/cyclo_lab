#!/usr/bin/env python3
"""Record the live SG2 head-camera topic during a headless policy rollout."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import threading
import time

import cv2
import numpy as np

from cyclo_lab.runtime.transport.ros2_zenoh import (
    COMPRESSED_IMAGE,
    best_effort_qos,
    close_endpoint,
    create_subscriber,
)


HEAD_TOPIC = "/zed/zed_node/left/image_rect_color/compressed"
EXPECTED_SIZE = (672, 376)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _decode(data) -> np.ndarray:
    payload = bytes(data) if isinstance(data, (list, tuple)) else data
    frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("head-camera compressed frame could not be decoded")
    height, width = frame.shape[:2]
    if (width, height) != EXPECTED_SIZE:
        raise RuntimeError(
            f"head-camera frame must be {EXPECTED_SIZE}, got {(width, height)}"
        )
    return frame


def record(
    output: Path,
    duration_s: float,
    fps: float,
    *,
    status_file: Path | None = None,
    stop_event: threading.Event | None = None,
) -> dict:
    if duration_s <= 0.0 or fps <= 0.0:
        raise ValueError("duration and fps must be positive")
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    stop_event = stop_event or threading.Event()
    if status_file is not None:
        status_file = status_file.resolve()
        _write_json(
            status_file,
            {"state": "starting", "pid": os.getpid(), "frame_count": 0},
        )
    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, EXPECTED_SIZE
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open MP4 writer: {output}")

    lock = threading.Lock()
    timestamps: list[float] = []
    errors: list[str] = []
    first_frame_reported = False

    def on_message(message) -> None:
        nonlocal first_frame_reported
        try:
            frame = _decode(message.data)
            with lock:
                writer.write(frame)
                timestamps.append(time.monotonic())
                if status_file is not None and not first_frame_reported:
                    first_frame_reported = True
                    _write_json(
                        status_file,
                        {
                            "state": "recording",
                            "pid": os.getpid(),
                            "frame_count": len(timestamps),
                        },
                    )
        except Exception as error:
            with lock:
                if not errors:
                    errors.append(str(error))

    subscriber = create_subscriber(
        HEAD_TOPIC,
        COMPRESSED_IMAGE,
        on_message,
        best_effort_qos(depth=4),
    )
    started = time.monotonic()
    try:
        while time.monotonic() - started < duration_s and not stop_event.is_set():
            time.sleep(0.05)
            with lock:
                if errors:
                    raise RuntimeError(errors[0])
    finally:
        close_endpoint(subscriber)
        with lock:
            writer.release()

    frame_count = len(timestamps)
    if stop_event.is_set():
        if status_file is not None:
            _write_json(
                status_file,
                {"state": "stopped", "pid": os.getpid(), "frame_count": frame_count},
            )
        raise RuntimeError("head-camera recording interrupted")
    if frame_count == 0:
        if status_file is not None:
            _write_json(
                status_file,
                {"state": "error", "pid": os.getpid(), "frame_count": 0},
            )
        raise RuntimeError(f"no frames received from {HEAD_TOPIC}")
    observed_fps = None
    if frame_count > 1:
        observed_duration = timestamps[-1] - timestamps[0]
        if observed_duration > 0.0:
            observed_fps = (frame_count - 1) / observed_duration
    report = {
        "schema": "cyclo.coffee_policy_head_camera_capture.v1",
        "topic": HEAD_TOPIC,
        "width": EXPECTED_SIZE[0],
        "height": EXPECTED_SIZE[1],
        "encoded_fps": fps,
        "observed_fps": observed_fps,
        "requested_duration_s": duration_s,
        "frame_count": frame_count,
        "video": str(output),
        "sha256": _sha256(output),
    }
    report_path = output.with_suffix(".json")
    _write_json(report_path, report)
    if status_file is not None:
        _write_json(
            status_file,
            {"state": "complete", "pid": os.getpid(), "frame_count": frame_count},
        )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration-s", type=float, default=120.0)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--status-file", type=Path)
    args = parser.parse_args()
    stop_event = threading.Event()

    def request_stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    print(
        json.dumps(
            record(
                args.output,
                args.duration_s,
                args.fps,
                status_file=args.status_file,
                stop_event=stop_event,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
