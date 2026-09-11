#!/usr/bin/env python3
"""Control the LeRobot S2R lifecycle for a Coffee Transport checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import PurePosixPath


SERVICE = "/lerobot_s2r/inference_command"
SERVICE_TYPE = "interfaces/srv/InferenceCommand"
COMMANDS = {"load": 0, "start": 1, "pause": 2, "resume": 3, "stop": 4, "unload": 5}
INSTRUCTIONS = {
    "000001": (
        "Move the right and left orange coffee cans to the lower cabinet shelf "
        "using the matching hands."
    ),
    "000002": (
        "Move the right and left orange coffee cans to the matching cabinet on "
        "the right using the matching hands."
    ),
}
REQUEST_DEFINITION = """\
uint8 command
string model_path
string embodiment_tag
string robot_type
string task_instruction
bool publish_to_robot
string action_request_mode
string acceleration_mode
string acceleration_engine_path
"""
RESPONSE_DEFINITION = """\
bool success
string message
string[] action_keys
"""


def _checkpoint(value: str) -> str:
    path = PurePosixPath(value)
    allowed = PurePosixPath("/workspace/model/lerobot")
    if not path.is_absolute() or (path != allowed and allowed not in path.parents):
        raise ValueError("checkpoint must be below /workspace/model/lerobot")
    return str(path)


def build_request(args: argparse.Namespace) -> dict:
    command = COMMANDS[args.command]
    if args.command == "load":
        if args.task_id is None or args.checkpoint is None:
            raise ValueError("load requires --task-id and --checkpoint")
        checkpoint = _checkpoint(args.checkpoint)
        instruction = INSTRUCTIONS[args.task_id]
    else:
        if args.checkpoint is not None or args.task_id is not None:
            raise ValueError("--task-id/--checkpoint are valid only for load")
        checkpoint = ""
        instruction = ""
    return {
        "command": command,
        "model_path": checkpoint,
        "embodiment_tag": "",
        "robot_type": "ffw_sg2_rev1" if args.command == "load" else "",
        "task_instruction": instruction,
        "publish_to_robot": bool(args.publish_actions),
        "action_request_mode": "async" if args.command == "load" else "",
        "acceleration_mode": "pytorch" if args.command == "load" else "",
        "acceleration_engine_path": "",
    }


def call(request: dict, timeout_s: float) -> dict:
    from zenoh_ros2_sdk import ROS2ServiceClient

    client = ROS2ServiceClient(
        service_name=SERVICE,
        srv_type=SERVICE_TYPE,
        request_definition=REQUEST_DEFINITION,
        response_definition=RESPONSE_DEFINITION,
        timeout=timeout_s,
    )
    try:
        response = client.call(**request)
    finally:
        client.close()
    if response is None:
        raise RuntimeError(f"no response from {SERVICE}")
    result = {
        "success": bool(response.success),
        "message": str(response.message),
        "action_keys": list(response.action_keys),
    }
    if not result["success"]:
        raise RuntimeError(f"inference command failed: {result['message']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=tuple(COMMANDS))
    parser.add_argument("--task-id", choices=tuple(INSTRUCTIONS))
    parser.add_argument("--checkpoint")
    parser.add_argument(
        "--publish-actions",
        action="store_true",
        help="Permit command publication; required for the Isaac bridge to move.",
    )
    parser.add_argument("--timeout-s", type=float, default=7200.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.timeout_s <= 0.0:
        parser.error("--timeout-s must be positive")
    try:
        request = build_request(args)
    except ValueError as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps({"service": SERVICE, "request": request}, indent=2))
        return
    print(json.dumps(call(request, args.timeout_s), indent=2))


if __name__ == "__main__":
    main()
