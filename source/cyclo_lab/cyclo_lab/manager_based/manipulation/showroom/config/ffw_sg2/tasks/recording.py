"""Pre-launch-safe registry for task-owned demonstration recorders.

The common recorder loads only this module. Task-specific modules declare their
own CLI trigger and validate the selected environment before Isaac runtime
objects are constructed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


_SUPPORT_SPECS = (
    (("Task000001", "Task000002"), "coffee_transport/recording.py"),
    (("Task000525",), "task_000525/recording.py"),
)


def _load_support(relative_path: str):
    module_path = Path(__file__).resolve().parent / relative_path
    module_name = f"_cyclo_recording_{module_path.parent.name}"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load task recording support: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_SUPPORTS = tuple(
    (task_tokens, _load_support(relative_path))
    for task_tokens, relative_path in _SUPPORT_SPECS
)


def add_cli_arguments(parser: Any) -> None:
    """Add CLI options owned by every registered task recorder."""

    for _task_tokens, support in _SUPPORTS:
        support.add_cli_arguments(parser)


def _support_for_task(task_name: str):
    matches = {
        support
        for task_tokens, support in _SUPPORTS
        if any(task_token in task_name for task_token in task_tokens)
    }
    if len(matches) > 1:
        raise ValueError(f"Multiple task recording supports match {task_name!r}")
    return next(iter(matches), None)


def build_recording_factory(args: Any):
    """Build the explicitly requested controller for the selected task."""

    requested = [
        support for _task_tokens, support in _SUPPORTS if support.is_requested(args)
    ]
    if len(requested) > 1:
        raise ValueError("Task recording modes are mutually exclusive.")
    if not requested:
        return None

    requested_support = requested[0]
    selected_support = _support_for_task(args.task or "")
    if selected_support is not None and selected_support is not requested_support:
        raise ValueError(
            f"The requested recording mode does not support task {args.task!r}."
        )
    # Let the requested module produce its more specific wrong-task/CLI error
    # when the task name has no registered support.
    return requested_support.build_recording_factory(args)
